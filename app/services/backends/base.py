from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.schemas.transcription import TranscriptSegment


@dataclass(slots=True)
class TranscriptionOptions:
    language: str | None = None
    beam_size: int | None = None
    patience: float | None = None
    vad_filter: bool | None = None
    return_timestamps: bool = False
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float | tuple[float, ...] | None = None
    use_batched: bool = True


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    segments: list[TranscriptSegment]
    language: str | None
    duration_seconds: float
    processing_time_seconds: float
    rtf: float


class ASRBackend(ABC):
    """Abstract base class for pluggable speech-to-text inference backends."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_lock = threading.Lock()
        self.status: str = "idle"
        self.last_error: str | None = None
        self.resolved_device: str | None = None
        self.resolved_compute_type: str | None = None

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True once the underlying model weights are loaded in memory."""

    @abstractmethod
    def load(self) -> None:
        """Download/convert (if needed) and load the model into memory."""

    @abstractmethod
    def iter_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> tuple[Iterator[TranscriptSegment], str | None]:
        """Yield decoded segments lazily along with the resolved language code."""
