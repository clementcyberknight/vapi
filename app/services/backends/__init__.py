from __future__ import annotations

from app.core.config import Settings
from app.services.backends.base import ASRBackend, TranscriptionOptions, TranscriptionResult
from app.services.backends.faster_whisper import FasterWhisperBackend
from app.services.backends.transformers import TransformersBackend


def create_backend(settings: Settings) -> ASRBackend:
    """Instantiate the configured ASR inference backend."""
    if settings.engine == "ctranslate2":
        return FasterWhisperBackend(settings)
    return TransformersBackend(settings)


__all__ = [
    "ASRBackend",
    "FasterWhisperBackend",
    "TranscriptionOptions",
    "TranscriptionResult",
    "TransformersBackend",
    "create_backend",
]
