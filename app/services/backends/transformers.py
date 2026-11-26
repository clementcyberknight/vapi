from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import TranscriptionFailedError
from app.schemas.transcription import TranscriptSegment
from app.services.audio import SAMPLE_RATE
from app.services.backends.base import ASRBackend, TranscriptionOptions

logger = logging.getLogger(__name__)


class TransformersBackend(ASRBackend):
    """Fallback PyTorch / Hugging Face Transformers ASR pipeline backend."""

    name = "transformers"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and self._pipeline is not None

    def load(self) -> None:
        if self.is_ready:
            return

        with self._load_lock:
            if self.is_ready:
                return

            self.status = "loading"
            self.last_error = None
            t0 = time.perf_counter()

            try:
                import torch
                from transformers import pipeline

                use_cuda = self.settings.device == "cuda" or (
                    self.settings.device == "auto" and torch.cuda.is_available()
                )
                device = "cuda:0" if use_cuda else "cpu"
                dtype = torch.float16 if use_cuda else torch.float32

                if not use_cuda:
                    torch.set_num_threads(self.settings.cpu_threads)

                self.resolved_device = "cuda" if use_cuda else "cpu"
                self.resolved_compute_type = "float16" if use_cuda else "float32"

                pipe_kwargs: dict[str, Any] = {
                    "task": "automatic-speech-recognition",
                    "model": self.settings.model_id,
                    "device": device,
                    "dtype": dtype,
                    "model_kwargs": {"low_cpu_mem_usage": True},
                }
                if self.settings.hf_token:
                    pipe_kwargs["token"] = self.settings.hf_token

                self._pipeline = pipeline(**pipe_kwargs)
                self.status = "ready"
                logger.info(
                    "TransformersBackend ready for '%s' (device=%s, dtype=%s) in %.2fs.",
                    self.settings.model_id,
                    device,
                    self.resolved_compute_type,
                    time.perf_counter() - t0,
                )
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                logger.exception("Failed to load TransformersBackend for '%s'", self.settings.model_id)
                raise

    def iter_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> tuple[Iterator[TranscriptSegment], str | None]:
        if not self.is_ready:
            self.load()

        pipe = self._pipeline
        if pipe is None:
            raise TranscriptionFailedError("Transformers pipeline is not loaded.")

        import torch

        duration_s = float(waveform.size) / float(SAMPLE_RATE)
        language = options.language if options.language is not None else self.settings.default_language

        call_kwargs: dict[str, Any] = {
            "chunk_length_s": self.settings.chunk_length_s,
            "batch_size": self.settings.batch_size,
        }
        if options.return_timestamps or options.word_timestamps:
            call_kwargs["return_timestamps"] = "word" if options.word_timestamps else True

        with torch.inference_mode():
            output = pipe({"array": waveform, "sampling_rate": SAMPLE_RATE}, **call_kwargs)

        def _segment_generator() -> Iterator[TranscriptSegment]:
            chunks = output.get("chunks")
            if chunks:
                for idx, chunk in enumerate(chunks):
                    ts = chunk.get("timestamp") or (0.0, duration_s)
                    start = float(ts[0] or 0.0)
                    end = float(ts[1] if ts[1] is not None else duration_s)
                    text = str(chunk.get("text", "")).strip()
                    if text:
                        yield TranscriptSegment(id=idx, start=round(start, 3), end=round(end, 3), text=text)
            else:
                text = str(output.get("text", "")).strip()
                if text:
                    yield TranscriptSegment(id=0, start=0.0, end=round(duration_s, 3), text=text)

        return _segment_generator(), language
