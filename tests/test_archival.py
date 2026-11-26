from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.main import create_app
from app.services.archival import (
    ArchivalService,
    ArchiveMetadata,
    R2StorageClient,
    R2UploadResult,
    SupabaseAuditLogger,
)
from app.services.transcription import TranscriptionService
from tests.conftest import FakeASRBackend


async def test_r2_storage_client_deterministic_key_and_pooled_s3v4(tmp_path: Path) -> None:
    audio_file = tmp_path / "sample.m4a"
    audio_file.write_bytes(b"fake-m4a-payload")

    settings = Settings(
        _env_file=None,
        preload_model=False,
        r2_account_id="cf_acct_123",
        r2_access_key_id="r2_key",
        r2_secret_access_key="r2_secret",
        r2_bucket_name="voice-backups",
        r2_public_base_url="https://cdn.example.com",
        archive_concurrency=4,
        archive_max_retries=3,
    )

    mock_s3 = MagicMock()
    mock_boto3 = SimpleNamespace(client=MagicMock(return_value=mock_s3))
    mock_config_cls = MagicMock(side_effect=lambda **kwargs: kwargs)
    mock_transfer_cls = MagicMock(side_effect=lambda **kwargs: kwargs)
    fixed_ts = datetime(2026, 11, 26, 23, 59, 59, tzinfo=UTC)

    with patch.dict(
        "sys.modules",
        {
            "boto3": mock_boto3,
            "boto3.s3.transfer": SimpleNamespace(TransferConfig=mock_transfer_cls),
            "botocore.config": SimpleNamespace(Config=mock_config_cls),
        },
    ):
        r2 = R2StorageClient(settings)
        res = await r2.upload_file(
            audio_file,
            request_id="../../req123",
            original_filename="voice note.m4a",
            created_at=fixed_ts,
            content_type="audio/mp4",
        )

    mock_config_cls.assert_called_once_with(
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        connect_timeout=5,
        read_timeout=30,
        max_pool_connections=10,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    mock_transfer_cls.assert_called_once_with(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=2,
    )
    assert res.bucket == "voice-backups"
    # Path traversal characters stripped, deterministic UTC date preserved
    assert res.object_key == "voices/2026/11/26/req123_voice_note.m4a"
    assert res.url == "https://cdn.example.com/voices/2026/11/26/req123_voice_note.m4a"
    assert len(res.audio_sha256) == 64


async def test_supabase_logger_idempotent_upsert_and_transient_retry() -> None:
    settings = Settings(
        _env_file=None,
        preload_model=False,
        supabase_url="https://proj.supabase.co",
        supabase_secret_key="sb_secret_test",
        supabase_table="voice_transcriptions",
        archive_max_retries=3,
    )

    mock_query = MagicMock()
    # First attempt fails transiently with 503; second succeeds idempotently
    transient_err = RuntimeError("upstream gateway timeout")
    transient_err.status_code = 503  # type: ignore[attr-defined]
    mock_query.execute = AsyncMock(side_effect=[transient_err, SimpleNamespace(data=[])])

    mock_table = MagicMock()
    mock_table.upsert = MagicMock(return_value=mock_query)
    mock_client = MagicMock()
    mock_client.table = MagicMock(return_value=mock_table)
    mock_acreate = AsyncMock(return_value=mock_client)

    with (
        patch.dict("sys.modules", {"supabase": SimpleNamespace(acreate_client=mock_acreate)}),
        patch("app.services.archival.asyncio.sleep", new=AsyncMock()),
    ):
        audit = SupabaseAuditLogger(settings)
        await audit.upsert_record({"request_id": "r1", "transcript_text": "Hello"})

    mock_acreate.assert_awaited_once_with("https://proj.supabase.co", "sb_secret_test")
    assert mock_table.upsert.call_count == 2
    mock_table.upsert.assert_called_with(
        {"request_id": "r1", "transcript_text": "Hello"},
        on_conflict="request_id",
    )


async def test_supabase_logger_fails_fast_on_4xx_schema_error() -> None:
    settings = Settings(
        _env_file=None,
        preload_model=False,
        supabase_url="https://proj.supabase.co",
        supabase_secret_key="sb_secret_test",
        archive_max_retries=3,
    )

    bad_req = RuntimeError("relation voice_transcriptions does not exist")
    bad_req.code = "42P01"  # type: ignore[attr-defined]
    mock_query = MagicMock()
    mock_query.execute = AsyncMock(side_effect=bad_req)
    mock_table = MagicMock()
    mock_table.upsert = MagicMock(return_value=mock_query)
    mock_client = MagicMock()
    mock_client.table = MagicMock(return_value=mock_table)

    with (
        patch.dict(
            "sys.modules",
            {"supabase": SimpleNamespace(acreate_client=AsyncMock(return_value=mock_client))},
        ),
        pytest.raises(RuntimeError, match="does not exist"),
    ):
        audit = SupabaseAuditLogger(settings)
        await audit.upsert_record({"request_id": "r1"})

    # Must not retry permanent 4xx / SQLSTATE 42P01 errors
    assert mock_table.upsert.call_count == 1


def test_transcribe_endpoint_r2_failure_still_logs_to_supabase(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    fake_waveform = np.zeros(16_000, dtype=np.float32)
    settings = Settings(
        _env_file=None,
        preload_model=False,
        upload_tmp_dir=tmp_path,
        log_format="text",
    )
    metrics = MetricsRegistry()
    backend = FakeASRBackend(settings)
    transcription_service = TranscriptionService(settings, backend=backend, metrics=metrics)

    mock_r2 = MagicMock(spec=R2StorageClient)
    mock_r2.upload_file = AsyncMock(side_effect=ConnectionError("R2 unreachable"))
    mock_supa = MagicMock(spec=SupabaseAuditLogger)
    mock_supa.upsert_record = AsyncMock()

    archival_service = ArchivalService(
        settings,
        metrics=metrics,
        r2_client=mock_r2,
        supabase_logger=mock_supa,
    )
    app = create_app(
        app_settings=settings,
        transcription_service=transcription_service,
        archival_service=archival_service,
        metrics=metrics,
    )

    with (
        patch("app.services.transcription.decode_audio_file", return_value=(fake_waveform, 1.0)),
        TestClient(app) as client,
    ):
        # Also verify X-Request-ID sanitization strips path traversal characters
        res = client.post(
            "/api/v1/transcribe",
            headers={"X-Request-ID": "../../req-audit-001"},
            files={"file": ("clip.wav", b"123456789012", "audio/wav")},
        )
        assert res.status_code == 200
        assert res.headers["X-Request-ID"] == "req-audit-001"

    mock_r2.upload_file.assert_awaited_once()
    mock_supa.upsert_record.assert_awaited_once()
    row = mock_supa.upsert_record.call_args.args[0]
    assert row["request_id"] == "req-audit-001"
    assert row["r2_object_key"] is None
    assert "r2_upload_failed: ConnectionError" in row["archive_error"]
    assert len(row["audio_sha256"]) == 64
    assert metrics.archive_errors_total == 1
    assert metrics.supabase_logs_total == 1
    assert list(tmp_path.glob("*")) == []


def test_transcribe_endpoint_happy_path_r2_and_supabase(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    fake_waveform = np.zeros(16_000, dtype=np.float32)
    settings = Settings(
        _env_file=None,
        preload_model=False,
        upload_tmp_dir=tmp_path,
        log_format="text",
    )
    metrics = MetricsRegistry()
    backend = FakeASRBackend(settings)
    transcription_service = TranscriptionService(settings, backend=backend, metrics=metrics)

    mock_r2 = MagicMock(spec=R2StorageClient)
    mock_r2.upload_file = AsyncMock(
        return_value=R2UploadResult(
            bucket="uba-voice",
            object_key="uba-voices/2026/11/26/req-ok_clip.wav",
            url=None,
            file_size_bytes=12,
            audio_sha256="a" * 64,
        )
    )
    mock_supa = MagicMock(spec=SupabaseAuditLogger)
    mock_supa.upsert_record = AsyncMock()

    archival_service = ArchivalService(
        settings,
        metrics=metrics,
        r2_client=mock_r2,
        supabase_logger=mock_supa,
    )
    app = create_app(
        app_settings=settings,
        transcription_service=transcription_service,
        archival_service=archival_service,
        metrics=metrics,
    )

    with (
        patch("app.services.transcription.decode_audio_file", return_value=(fake_waveform, 1.0)),
        TestClient(app) as client,
    ):
        res = client.post(
            "/api/v1/transcribe",
            headers={"X-Request-ID": "req-ok"},
            files={"file": ("clip.wav", b"123456789012", "audio/wav")},
        )
        assert res.status_code == 200

    row = mock_supa.upsert_record.call_args.args[0]
    assert row["request_id"] == "req-ok"
    assert row["r2_bucket"] == "uba-voice"
    assert row["archive_error"] is None
    assert metrics.r2_uploads_total == 1
    assert metrics.supabase_logs_total == 1
    assert list(tmp_path.glob("*")) == []


async def test_archival_queue_sheds_load_when_saturated(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, preload_model=False, archive_max_queue_size=4)
    metrics = MetricsRegistry()
    mock_supa = MagicMock(spec=SupabaseAuditLogger)
    archival = ArchivalService(settings, metrics=metrics, supabase_logger=mock_supa)

    # Simulate 4 in-flight tasks saturating the queue
    blocker = [MagicMock() for _ in range(4)]
    archival._pending_tasks.update(blocker)  # type: ignore[arg-type]

    src = tmp_path / "voice.wav"
    src.write_bytes(b"audio")
    archival.enqueue_file(
        src,
        ArchiveMetadata(
            request_id="shed-1",
            source="http",
            original_filename="voice.wav",
            content_type="audio/wav",
            model="NCAIR1/NigerianAccentedEnglish",
            engine="ctranslate2",
            language="en",
            transcript_text="test",
            duration_seconds=1.0,
            processing_time_seconds=0.1,
            rtf=0.1,
        ),
    )
    assert metrics.archive_dropped_total == 1
