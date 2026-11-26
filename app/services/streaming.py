from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.services.audio import SAMPLE_RATE, compute_rms_energy, pcm16_bytes_to_float32


@dataclass(slots=True)
class UtteranceChunk:
    """Audio slice ready for either interim partial or finalized transcription."""

    kind: str  # "partial" | "final"
    segment_id: int
    waveform: np.ndarray
    duration_seconds: float


class StreamingSession:
    """Bounded-memory real-time speech buffer with energy VAD & silence endpointing.

    Design goals:
    - Zero clipping on speech onset via a 250 ms pre-roll ring buffer.
    - Bounded RAM (~640 KB max per connection at 20 s max utterance).
    - Coalesced interim partials rate-limited by `stream_partial_interval_ms`.
    - Clean finalization on trailing silence, max utterance length, or explicit flush/end.
    """

    PRE_ROLL_MS = 250

    def __init__(
        self,
        settings: Settings,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        interim_results: bool = True,
        initial_prompt: str | None = None,
    ) -> None:
        self.settings = settings
        self.sample_rate = sample_rate
        self.language = language if language is not None else settings.default_language
        self.interim_results = interim_results
        self.initial_prompt = initial_prompt

        self._pre_roll_samples = int(SAMPLE_RATE * self.PRE_ROLL_MS / 1000)
        self._min_speech_samples = int(SAMPLE_RATE * settings.stream_min_speech_ms / 1000)
        self._silence_samples_limit = int(SAMPLE_RATE * settings.stream_silence_ms / 1000)
        self._max_utterance_samples = int(SAMPLE_RATE * settings.stream_max_utterance_s)
        self._partial_interval_s = settings.stream_partial_interval_ms / 1000.0
        self._energy_threshold = settings.stream_energy_threshold

        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_len = 0

        self._speech_chunks: list[np.ndarray] = []
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self._in_speech = False

        self._last_partial_at = 0.0
        self.last_partial_text = ""
        self.segment_id = 0
        self.total_audio_seconds = 0.0
        self.total_final_segments = 0

    def update_config(
        self,
        *,
        sample_rate: int | None = None,
        language: str | None = None,
        interim_results: bool | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if language is not None:
            self.language = language
        if interim_results is not None:
            self.interim_results = interim_results
        if initial_prompt is not None:
            self.initial_prompt = initial_prompt

    def _append_pre_roll(self, chunk: np.ndarray) -> None:
        self._pre_roll.append(chunk)
        self._pre_roll_len += chunk.size
        while self._pre_roll_len > self._pre_roll_samples and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_len -= dropped.size

    def _concat_speech(self) -> np.ndarray:
        if not self._speech_chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._speech_chunks) == 1:
            return self._speech_chunks[0]
        return np.concatenate(self._speech_chunks)

    def _finalize_current(self) -> UtteranceChunk | None:
        if self._speech_samples < self._min_speech_samples:
            self._reset_utterance()
            return None

        waveform = self._concat_speech()
        seg_id = self.segment_id
        duration = round(float(waveform.size) / float(SAMPLE_RATE), 3)

        self.segment_id += 1
        self.total_final_segments += 1
        self._reset_utterance()

        return UtteranceChunk(
            kind="final",
            segment_id=seg_id,
            waveform=waveform,
            duration_seconds=duration,
        )

    def _reset_utterance(self) -> None:
        self._speech_chunks.clear()
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self._in_speech = False
        self.last_partial_text = ""
        self._last_partial_at = 0.0

    def push_pcm16(
        self,
        pcm_bytes: bytes,
        *,
        now: float | None = None,
    ) -> list[UtteranceChunk]:
        """Ingest a raw PCM16 chunk and return any partial/final chunks ready for inference."""
        samples = pcm16_bytes_to_float32(
            pcm_bytes,
            source_sample_rate=self.sample_rate,
            target_sample_rate=SAMPLE_RATE,
        )
        if samples.size == 0:
            return []

        self.total_audio_seconds += float(samples.size) / float(SAMPLE_RATE)
        rms = compute_rms_energy(samples)
        is_voiced = rms >= self._energy_threshold
        actions: list[UtteranceChunk] = []
        ts = time.monotonic() if now is None else now

        if not self._in_speech:
            if is_voiced:
                self._in_speech = True
                self._trailing_silence_samples = 0
                if self._pre_roll:
                    self._speech_chunks.extend(self._pre_roll)
                    self._speech_samples += self._pre_roll_len
                    self._pre_roll.clear()
                    self._pre_roll_len = 0
                self._speech_chunks.append(samples)
                self._speech_samples += samples.size
                self._last_partial_at = ts
            else:
                self._append_pre_roll(samples)
            return actions

        # Currently inside an active speech utterance
        self._speech_chunks.append(samples)
        self._speech_samples += samples.size

        if is_voiced:
            self._trailing_silence_samples = 0
        else:
            self._trailing_silence_samples += samples.size

        # Endpoint on trailing silence or max utterance duration cap
        if (
            self._trailing_silence_samples >= self._silence_samples_limit
            or self._speech_samples >= self._max_utterance_samples
        ):
            finalized = self._finalize_current()
            if finalized is not None:
                actions.append(finalized)
            return actions

        # Emit rate-limited interim partial snapshot while speaking
        if (
            self.interim_results
            and self._speech_samples >= self._min_speech_samples
            and (ts - self._last_partial_at) >= self._partial_interval_s
        ):
            self._last_partial_at = ts
            waveform = self._concat_speech()
            actions.append(
                UtteranceChunk(
                    kind="partial",
                    segment_id=self.segment_id,
                    waveform=waveform,
                    duration_seconds=round(float(waveform.size) / float(SAMPLE_RATE), 3),
                )
            )

        return actions

    def flush(self) -> UtteranceChunk | None:
        """Force-finalize any buffered speech (used on client `flush` or `end`)."""
        return self._finalize_current()
