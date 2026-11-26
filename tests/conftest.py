from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.main import create_app
from app.schemas.transcription import TranscriptSegment
from app.services.audio import SAMPLE_RATE
from app.services.backends.base import ASRBackend, TranscriptionOptions
from app.services.transcription import TranscriptionService


class FakeASRBackend(ASRBackend):
    """In-memory mock backend so CI and local tests never download model weights."""

    name = "ctranslate2"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.status = "ready"
        self.resolved_device = "cpu"
        self.resolved_compute_type = "int8"

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def load(self) -> None:
        self.status = "ready"

    def iter_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> tuple[Iterator[TranscriptSegment], str | None]:
        duration = round(float(waveform.size) / float(SAMPLE_RATE), 3)

        def _gen() -> Iterator[TranscriptSegment]:
            if duration >= 2.4:
                yield TranscriptSegment(id=0, start=0.0, end=1.2, text="Good morning Lagos.")
                yield TranscriptSegment(id=1, start=1.2, end=duration, text="We are live and streaming.")
            else:
                yield TranscriptSegment(
                    id=0,
                    start=0.0,
                    end=duration,
                    text="How far, wetin dey happen today?",
                )

        return _gen(), (options.language or "en")


def make_test_client(
    *,
    api_key: str | None = None,
    max_upload_size_mb: int = 25,
    max_queued_requests: int = 20,
    stream_partial_interval_ms: int = 150,
    stream_silence_ms: int = 250,
    stream_min_speech_ms: int = 100,
    tmp_dir: Path | None = None,
) -> TestClient:
    upload_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="asr_test_"))
    settings = Settings(
        _env_file=None,
        preload_model=False,
        api_key=api_key,
        max_upload_size_mb=max_upload_size_mb,
        max_queued_requests=max_queued_requests,
        upload_tmp_dir=upload_dir,
        log_format="text",
        stream_partial_interval_ms=stream_partial_interval_ms,
        stream_silence_ms=stream_silence_ms,
        stream_min_speech_ms=stream_min_speech_ms,
    )
    metrics = MetricsRegistry()
    backend = FakeASRBackend(settings)
    service = TranscriptionService(settings, backend=backend, metrics=metrics)
    app = create_app(app_settings=settings, transcription_service=service, metrics=metrics)
    return TestClient(app)
