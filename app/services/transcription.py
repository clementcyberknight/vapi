from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.exceptions import (
    ASRError,
    ModelNotReadyError,
    ServiceBusyError,
    TranscriptionFailedError,
)
from app.core.metrics import MetricsRegistry
from app.schemas.transcription import TranscriptSegment
from app.services.audio import SAMPLE_RATE, decode_audio_file
from app.services.backends import (
    ASRBackend,
    TranscriptionOptions,
    TranscriptionResult,
    create_backend,
)
from app.services.backends.faster_whisper import reclaim_process_memory

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Orchestrates bounded-queue admission control, JIT audio decoding, and inference.

    Key memory guarantee:
    Uploaded files stay on disk until an inference slot is acquired. Audio decoding
    (`ffmpeg` -> float32 waveform) happens strictly inside the bounded semaphore so
    10-20 simultaneous 20 MB uploads never inflate RAM with idle decoded waveforms.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        backend: ASRBackend | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or create_backend(settings)
        self.metrics = metrics or MetricsRegistry()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_transcriptions)
        self._in_flight = 0

    @property
    def is_ready(self) -> bool:
        return self.backend.is_ready

    @property
    def status(self) -> str:
        return self.backend.status

    @property
    def last_error(self) -> str | None:
        return self.backend.last_error

    @property
    def resolved_device(self) -> str | None:
        return self.backend.resolved_device

    @property
    def resolved_compute_type(self) -> str | None:
        return self.backend.resolved_compute_type

    def ensure_accepting_requests(self) -> None:
        if self.backend.status == "loading":
            raise ModelNotReadyError("Model is still initializing on the server. Please retry shortly.")

    def load_model(self) -> None:
        self.backend.load()

    @asynccontextmanager
    async def _acquire_inference_slot(self) -> AsyncIterator[None]:
        self.ensure_accepting_requests()

        if self._in_flight >= self.settings.max_queued_requests:
            self.metrics.record_rejected()
            raise ServiceBusyError(
                f"Server transcription queue is full ({self.settings.max_queued_requests} requests in flight).",
                retry_after=5,
            )

        self._in_flight += 1
        self.metrics.inc_queued()
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.settings.queue_timeout_seconds,
                )
                acquired = True
            except TimeoutError as exc:
                self.metrics.record_rejected()
                raise ServiceBusyError(
                    "Timed out waiting for an available transcription slot.",
                    retry_after=5,
                ) from exc
            finally:
                self.metrics.dec_queued()

            self.metrics.inc_active_transcriptions()
            try:
                yield
            finally:
                self.metrics.dec_active_transcriptions()
        finally:
            if acquired:
                self._semaphore.release()
            self._in_flight = max(0, self._in_flight - 1)

    def _run_waveform_sync(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        duration_s = round(float(waveform.size) / float(SAMPLE_RATE), 3)
        try:
            seg_iter, detected_lang = self.backend.iter_segments(waveform, options)
            segments = list(seg_iter)
        finally:
            if duration_s >= 15.0:
                reclaim_process_memory()
        elapsed = round(time.perf_counter() - t0, 3)
        full_text = " ".join(seg.text for seg in segments).strip()
        rtf = round(elapsed / duration_s, 4) if duration_s > 0 else 0.0
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=detected_lang,
            duration_seconds=duration_s,
            processing_time_seconds=elapsed,
            rtf=rtf,
        )

    def _decode_and_transcribe_file_sync(
        self,
        file_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        waveform, _ = decode_audio_file(file_path, normalize=self.settings.normalize_audio)
        return self._run_waveform_sync(waveform, options)

    async def transcribe_file(
        self,
        file_path: Path,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        """Queue on disk, then decode and transcribe JIT inside a bounded inference slot."""
        opts = options or TranscriptionOptions()
        async with self._acquire_inference_slot():
            try:
                result = await asyncio.to_thread(
                    self._decode_and_transcribe_file_sync,
                    file_path,
                    opts,
                )
            except ASRError:
                self.metrics.record_error()
                raise
            except Exception as exc:
                self.metrics.record_error()
                logger.exception("File transcription failed for '%s'", file_path.name)
                raise TranscriptionFailedError(f"Transcription failed: {exc}") from exc

        self.metrics.record_transcription(result.duration_seconds, result.processing_time_seconds)
        return result

    async def transcribe_waveform(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        """Transcribe an in-memory 16 kHz float32 waveform (used by WebSocket streaming)."""
        opts = options or TranscriptionOptions()
        async with self._acquire_inference_slot():
            try:
                return await asyncio.to_thread(self._run_waveform_sync, waveform, opts)
            except ASRError:
                self.metrics.record_error()
                raise
            except Exception as exc:
                self.metrics.record_error()
                logger.exception("Waveform transcription failed")
                raise TranscriptionFailedError(f"Transcription failed: {exc}") from exc

    async def stream_file_segments(
        self,
        file_path: Path,
        options: TranscriptionOptions | None = None,
    ) -> AsyncIterator[tuple[TranscriptSegment | None, float]]:
        """Decode JIT inside a bounded slot and stream decoded segments as they arrive.

        Yields `(segment, duration_seconds)` for each segment.
        """
        opts = options or TranscriptionOptions(return_timestamps=True, use_batched=False)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[TranscriptSegment | None, float] | BaseException | None] = (
            asyncio.Queue(maxsize=32)
        )

        def _producer() -> None:
            try:
                waveform, duration_s = decode_audio_file(
                    file_path,
                    normalize=self.settings.normalize_audio,
                )
                seg_iter, _ = self.backend.iter_segments(waveform, opts)
                emitted = False
                for seg in seg_iter:
                    emitted = True
                    asyncio.run_coroutine_threadsafe(queue.put((seg, duration_s)), loop).result()
                if not emitted:
                    asyncio.run_coroutine_threadsafe(queue.put((None, duration_s)), loop).result()
            except BaseException as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        async with self._acquire_inference_slot():
            worker = asyncio.create_task(asyncio.to_thread(_producer))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        self.metrics.record_error()
                        if isinstance(item, ASRError):
                            raise item
                        raise TranscriptionFailedError(f"Streaming transcription failed: {item}") from item
                    yield item
            finally:
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
