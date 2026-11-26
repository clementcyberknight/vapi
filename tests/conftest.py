from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.main import create_app
from app.schemas.transcription import TranscriptSegment
from app.services.audio import SAMPLE_RATE
from app.services.engine import ASREngine, TranscriptionOptions, TranscriptionResult


class FakeASREngine(ASREngine):
    """In-memory mock engine so CI and local tests never download model weights."""

    def __init__(self, settings: Settings, metrics: MetricsRegistry) -> None:
        super().__init__(settings, metrics=metrics)
        self.status = "ready"
        self.resolved_device = "cpu"
        self.resolved_compute_type = "int8"
        self._fw_model = object()

    async def transcribe(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        opts = options or TranscriptionOptions()
        duration = round(float(waveform.size) / float(SAMPLE_RATE), 3)
        text = "How far, wetin dey happen today?"
        segments = [TranscriptSegment(id=0, start=0.0, end=duration, text=text)]
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=opts.language or "en",
            duration_seconds=duration,
            processing_time_seconds=0.05,
            rtf=round(0.05 / duration, 4) if duration > 0 else 0.0,
        )

    async def stream_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        yield TranscriptSegment(id=0, start=0.0, end=1.2, text="Good morning Lagos.")
        yield TranscriptSegment(id=1, start=1.2, end=2.5, text="We are live and streaming.")


def make_test_client(
    *,
    api_key: str | None = None,
    max_upload_size_mb: int = 25,
    stream_partial_interval_ms: int = 150,
    stream_silence_ms: int = 250,
    stream_min_speech_ms: int = 100,
) -> TestClient:
    settings = Settings(
        preload_model=False,
        api_key=api_key,
        max_upload_size_mb=max_upload_size_mb,
        log_format="text",
        stream_partial_interval_ms=stream_partial_interval_ms,
        stream_silence_ms=stream_silence_ms,
        stream_min_speech_ms=stream_min_speech_ms,
    )
    metrics = MetricsRegistry()
    engine = FakeASREngine(settings, metrics=metrics)
    app = create_app(app_settings=settings, engine=engine, metrics=metrics)
    return TestClient(app)
