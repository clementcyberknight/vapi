from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import random
import re
import shutil
import threading
import uuid
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.schemas.transcription import TranscriptSegment
from app.services.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

ArchiveSource = Literal["http", "sse", "openai", "websocket"]

_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 409, 422})
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(slots=True)
class R2UploadResult:
    bucket: str
    object_key: str
    url: str | None
    file_size_bytes: int
    audio_sha256: str


@dataclass(slots=True)
class ArchiveMetadata:
    request_id: str
    source: ArchiveSource
    original_filename: str
    content_type: str | None
    model: str
    engine: str
    language: str | None
    transcript_text: str
    duration_seconds: float
    processing_time_seconds: float
    rtf: float
    segments: list[TranscriptSegment] | None = None
    session_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _sanitize_token(value: str, fallback: str, max_len: int = 100) -> str:
    cleaned = _SAFE_TOKEN_RE.sub("_", Path(value).name).strip("._-")
    return cleaned[:max_len] or fallback


def _compute_file_sha256_and_size(file_path: Path) -> tuple[str, int]:
    size = file_path.stat().st_size
    with file_path.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    return digest, size


def _write_pcm16_wav(destination: Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _extract_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from PostgREST / httpx exceptions if present."""
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val <= 599:
            return val
        if isinstance(val, str) and val.isdigit():
            parsed = int(val)
            if 100 <= parsed <= 599:
                return parsed
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _is_non_retryable_supabase_error(exc: BaseException) -> bool:
    """Return True for permanent 4xx schema, auth, or RLS errors that should fail fast."""
    status_code = _extract_status_code(exc)
    if status_code in _NON_RETRYABLE_STATUS_CODES:
        return True
    # PostgREST Postgres SQLSTATE codes (e.g. '42P01' undefined_table, '42501' insufficient_privilege)
    pg_code = getattr(exc, "code", None)
    return isinstance(pg_code, str) and pg_code.startswith(("22", "23", "42"))


class R2StorageClient:
    """Idempotent, pooled Cloudflare R2 S3v4 client with bounded thread concurrency."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._s3_client: Any | None = None
        self._transfer_config: Any | None = None

    def _get_client_and_transfer_config(self) -> tuple[Any, Any]:
        if self._s3_client is not None and self._transfer_config is not None:
            return self._s3_client, self._transfer_config

        with self._lock:
            if self._s3_client is not None and self._transfer_config is not None:
                return self._s3_client, self._transfer_config

            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config

            endpoint = self._settings.resolved_r2_endpoint
            if not endpoint:
                raise RuntimeError("Cloudflare R2 endpoint is not configured.")

            pool_size = max(10, self._settings.archive_concurrency * 2)
            self._s3_client = boto3.client(
                service_name="s3",
                endpoint_url=endpoint,
                aws_access_key_id=self._settings.r2_access_key_id,
                aws_secret_access_key=self._settings.r2_secret_access_key,
                region_name=self._settings.r2_region,
                config=Config(
                    signature_version="s3v4",
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                    connect_timeout=5,
                    read_timeout=30,
                    max_pool_connections=pool_size,
                    retries={
                        "max_attempts": self._settings.archive_max_retries,
                        "mode": "standard",
                    },
                ),
            )
            # Limit per-file multipart worker threads to avoid CPU contention on 2-vCPU containers
            self._transfer_config = TransferConfig(
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                max_concurrency=2,
            )
            return self._s3_client, self._transfer_config

    def build_object_key(
        self,
        request_id: str,
        original_filename: str,
        created_at: datetime,
    ) -> str:
        """Build a deterministic R2 key so retries overwrite the same object idempotently."""
        date_path = created_at.astimezone(UTC).strftime("%Y/%m/%d")
        prefix = self._settings.r2_key_prefix.strip("/")
        safe_req = _sanitize_token(request_id, fallback="req", max_len=64)
        safe_name = _sanitize_token(original_filename, fallback="audio.bin", max_len=100)
        base_key = f"{date_path}/{safe_req}_{safe_name}"
        return f"{prefix}/{base_key}" if prefix else base_key

    def _upload_sync(
        self,
        file_path: Path,
        object_key: str,
        request_id: str,
        content_type: str | None,
        audio_sha256: str,
        file_size: int,
    ) -> R2UploadResult:
        bucket = self._settings.r2_bucket_name
        if not bucket:
            raise RuntimeError("R2_BUCKET_NAME is not configured.")

        resolved_ct = content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        client, transfer_cfg = self._get_client_and_transfer_config()

        client.upload_file(
            Filename=str(file_path),
            Bucket=bucket,
            Key=object_key,
            ExtraArgs={
                "ContentType": resolved_ct,
                "Metadata": {
                    "request-id": _sanitize_token(request_id, fallback="req", max_len=64),
                    "sha256": audio_sha256,
                },
            },
            Config=transfer_cfg,
        )

        public_url: str | None = None
        if self._settings.r2_public_base_url:
            base = self._settings.r2_public_base_url.rstrip("/")
            public_url = f"{base}/{object_key}"

        return R2UploadResult(
            bucket=bucket,
            object_key=object_key,
            url=public_url,
            file_size_bytes=file_size,
            audio_sha256=audio_sha256,
        )

    async def upload_file(
        self,
        file_path: Path,
        *,
        request_id: str,
        original_filename: str,
        created_at: datetime,
        content_type: str | None = None,
        audio_sha256: str | None = None,
        file_size_bytes: int | None = None,
    ) -> R2UploadResult:
        object_key = self.build_object_key(request_id, original_filename, created_at)

        def _run() -> R2UploadResult:
            if audio_sha256 is not None and file_size_bytes is not None:
                sha, size = audio_sha256, file_size_bytes
            else:
                sha, size = _compute_file_sha256_and_size(file_path)
            return self._upload_sync(file_path, object_key, request_id, content_type, sha, size)

        return await asyncio.to_thread(_run)


class SupabaseAuditLogger:
    """Idempotent async Supabase logger with connection reuse and transient backoff."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._client: Any | None = None

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client

            from supabase import acreate_client

            if not self._settings.supabase_url or not self._settings.supabase_secret_key:
                raise RuntimeError("Supabase URL or secret key is not configured.")

            self._client = await acreate_client(
                self._settings.supabase_url,
                self._settings.supabase_secret_key,
            )
            return self._client

    async def _reset_client(self) -> None:
        async with self._lock:
            self._client = None

    async def upsert_record(self, row: dict[str, Any]) -> None:
        """Idempotently upsert a transcription row by `request_id` with transient retries."""
        max_attempts = max(1, self._settings.archive_max_retries)
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                client = await self._get_client()
                await asyncio.wait_for(
                    client.table(self._settings.supabase_table)
                    .upsert(row, on_conflict="request_id")
                    .execute(),
                    timeout=10.0,
                )
                return
            except Exception as exc:
                last_exc = exc
                if _is_non_retryable_supabase_error(exc) or attempt >= max_attempts:
                    raise

                # Reset client on transport/timeout failures so stale sockets aren't reused
                if _extract_status_code(exc) is None:
                    await self._reset_client()

                backoff_s = min(4.0, 0.4 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.2)
                logger.warning(
                    "Transient Supabase upsert error on attempt %d/%d (retrying in %.2fs): %s",
                    attempt,
                    max_attempts,
                    backoff_s,
                    exc,
                )
                await asyncio.sleep(backoff_s)

        if last_exc is not None:
            raise last_exc


class ArchivalService:
    """Bounded, idempotent background service for R2 voice backup and Supabase logging."""

    def __init__(
        self,
        settings: Settings,
        *,
        metrics: MetricsRegistry | None = None,
        r2_client: R2StorageClient | None = None,
        supabase_logger: SupabaseAuditLogger | None = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics or MetricsRegistry()
        self._r2 = r2_client or (R2StorageClient(settings) if settings.r2_enabled else None)
        self._supabase = supabase_logger or (
            SupabaseAuditLogger(settings) if settings.supabase_enabled else None
        )
        self._semaphore = asyncio.Semaphore(settings.archive_concurrency)
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def is_enabled(self) -> bool:
        return self._r2 is not None or self._supabase is not None

    def _normalize_meta(self, meta: ArchiveMetadata) -> ArchiveMetadata:
        if not meta.request_id or meta.request_id == "-":
            meta.request_id = uuid.uuid4().hex[:16]
        return meta

    def _try_spawn(self, coro: Any, *, staged_path: Path | None, request_id: str) -> bool:
        """Spawn a background archival task or shed load cleanly if the queue is full."""
        if len(self._pending_tasks) >= self.settings.archive_max_queue_size:
            coro.close()
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            self.metrics.record_archive_dropped()
            logger.warning(
                "Archival queue full (%d pending); dropping background archive for request_id=%s.",
                len(self._pending_tasks),
                request_id,
            )
            return False

        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return True

    async def aclose(self, timeout: float = 15.0) -> None:
        """Wait bounded time for in-flight R2 uploads and Supabase upserts on shutdown."""
        if not self._pending_tasks:
            return
        pending = list(self._pending_tasks)
        logger.info("Draining %d pending voice archival tasks...", len(pending))
        _, still_running = await asyncio.wait(pending, timeout=timeout)
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)

    async def _archive_staged_file(
        self,
        staged_path: Path | None,
        meta: ArchiveMetadata,
        initial_size_bytes: int | None,
        initial_sha256: str | None = None,
    ) -> None:
        try:
            async with self._semaphore:
                r2_result: R2UploadResult | None = None
                file_size = initial_size_bytes
                audio_sha256 = initial_sha256
                r2_error_msg: str | None = None

                if staged_path is not None and staged_path.exists():
                    try:
                        audio_sha256, file_size = await asyncio.to_thread(
                            _compute_file_sha256_and_size,
                            staged_path,
                        )
                    except OSError as exc:
                        r2_error_msg = f"file_stat_failed: {exc}"

                    if self._r2 is not None and r2_error_msg is None:
                        try:
                            r2_result = await self._r2.upload_file(
                                staged_path,
                                request_id=meta.request_id,
                                original_filename=meta.original_filename,
                                created_at=meta.created_at,
                                content_type=meta.content_type,
                                audio_sha256=audio_sha256,
                                file_size_bytes=file_size,
                            )
                            self.metrics.record_r2_upload()
                        except Exception as exc:
                            r2_error_msg = f"r2_upload_failed: {type(exc).__name__}: {exc}"[:500]
                            self.metrics.record_archive_error()
                            logger.exception(
                                "Cloudflare R2 backup failed for '%s' (request_id=%s)",
                                meta.original_filename,
                                meta.request_id,
                            )

                if self._supabase is not None:
                    row: dict[str, Any] = {
                        "request_id": meta.request_id,
                        "session_id": meta.session_id,
                        "source": meta.source,
                        "original_filename": meta.original_filename,
                        "content_type": meta.content_type,
                        "file_size_bytes": r2_result.file_size_bytes if r2_result else file_size,
                        "audio_sha256": r2_result.audio_sha256 if r2_result else audio_sha256,
                        "r2_bucket": r2_result.bucket if r2_result else None,
                        "r2_object_key": r2_result.object_key if r2_result else None,
                        "r2_url": r2_result.url if r2_result else None,
                        "archive_error": r2_error_msg,
                        "model": meta.model,
                        "engine": meta.engine,
                        "language": meta.language,
                        "transcript_text": meta.transcript_text,
                        "duration_seconds": meta.duration_seconds,
                        "processing_time_seconds": meta.processing_time_seconds,
                        "rtf": meta.rtf,
                        "segments": [seg.model_dump(exclude_none=True) for seg in meta.segments]
                        if meta.segments is not None
                        else None,
                    }
                    try:
                        await self._supabase.upsert_record(row)
                        self.metrics.record_supabase_log()
                    except Exception:
                        self.metrics.record_archive_error()
                        logger.exception(
                            "Supabase audit upsert failed (request_id=%s)",
                            meta.request_id,
                        )
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    def enqueue_file(self, source_path: Path, meta: ArchiveMetadata) -> None:
        """Stage `source_path` via O(1) rename and schedule bounded idempotent archival."""
        if not self.is_enabled:
            return

        normalized = self._normalize_meta(meta)
        if len(self._pending_tasks) >= self.settings.archive_max_queue_size:
            self.metrics.record_archive_dropped()
            logger.warning(
                "Archival queue full (%d pending); skipping archive for request_id=%s.",
                len(self._pending_tasks),
                normalized.request_id,
            )
            return

        staged_path: Path | None = None
        initial_size: int | None = None

        if source_path.exists():
            initial_size = source_path.stat().st_size
            self.settings.upload_tmp_dir.mkdir(parents=True, exist_ok=True)
            staged_path = self.settings.upload_tmp_dir / f"arch_{uuid.uuid4().hex}{source_path.suffix}"
            try:
                shutil.move(str(source_path), str(staged_path))
            except OSError:
                logger.warning("Could not stage '%s' for archival.", source_path)
                staged_path = None

        self._try_spawn(
            self._archive_staged_file(staged_path, normalized, initial_size),
            staged_path=staged_path,
            request_id=normalized.request_id,
        )

    async def _encode_and_archive_waveform(self, waveform: np.ndarray, meta: ArchiveMetadata) -> None:
        staged_path: Path | None = None
        size_bytes = waveform.size * 2 + 44
        self.settings.upload_tmp_dir.mkdir(parents=True, exist_ok=True)
        candidate = self.settings.upload_tmp_dir / f"arch_{uuid.uuid4().hex}.wav"
        try:
            await asyncio.to_thread(_write_pcm16_wav, candidate, waveform, SAMPLE_RATE)
            staged_path = candidate
        except Exception:
            self.metrics.record_archive_error()
            logger.exception("Failed to encode WebSocket segment WAV for archival")
            candidate.unlink(missing_ok=True)

        await self._archive_staged_file(staged_path, meta, size_bytes)

    def enqueue_waveform(self, waveform: np.ndarray, meta: ArchiveMetadata) -> None:
        """Schedule bounded WAV encoding + R2 backup + Supabase upsert for stream segments."""
        if not self.is_enabled or not self.settings.archive_websocket_segments:
            return
        normalized = self._normalize_meta(meta)
        self._try_spawn(
            self._encode_and_archive_waveform(waveform.copy(), normalized),
            staged_path=None,
            request_id=normalized.request_id,
        )
