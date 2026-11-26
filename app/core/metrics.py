from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    """Zero-dependency thread-safe Prometheus text-format metrics collector."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    transcriptions_total: int = 0
    transcription_errors_total: int = 0
    rejected_requests_total: int = 0
    stream_segments_total: int = 0
    audio_seconds_total: float = 0.0
    processing_seconds_total: float = 0.0
    active_transcriptions: int = 0
    queued_transcriptions: int = 0
    active_ws_connections: int = 0

    def inc_queued(self) -> None:
        with self._lock:
            self.queued_transcriptions += 1

    def dec_queued(self) -> None:
        with self._lock:
            self.queued_transcriptions = max(0, self.queued_transcriptions - 1)

    def inc_active_transcriptions(self) -> None:
        with self._lock:
            self.active_transcriptions += 1

    def dec_active_transcriptions(self) -> None:
        with self._lock:
            self.active_transcriptions = max(0, self.active_transcriptions - 1)

    def inc_active_ws(self) -> None:
        with self._lock:
            self.active_ws_connections += 1

    def dec_active_ws(self) -> None:
        with self._lock:
            self.active_ws_connections = max(0, self.active_ws_connections - 1)

    def record_transcription(self, audio_seconds: float, processing_seconds: float) -> None:
        with self._lock:
            self.transcriptions_total += 1
            self.audio_seconds_total += max(0.0, audio_seconds)
            self.processing_seconds_total += max(0.0, processing_seconds)

    def record_stream_segment(self, audio_seconds: float, processing_seconds: float) -> None:
        with self._lock:
            self.stream_segments_total += 1
            self.audio_seconds_total += max(0.0, audio_seconds)
            self.processing_seconds_total += max(0.0, processing_seconds)

    def record_error(self) -> None:
        with self._lock:
            self.transcription_errors_total += 1

    def record_rejected(self) -> None:
        with self._lock:
            self.rejected_requests_total += 1

    def render_prometheus(self, model_ready: bool) -> str:
        with self._lock:
            avg_rtf = (
                self.processing_seconds_total / self.audio_seconds_total
                if self.audio_seconds_total > 0
                else 0.0
            )
            lines = [
                "# HELP asr_model_ready Whether the ASR model is loaded and ready (1=ready, 0=not ready)",
                "# TYPE asr_model_ready gauge",
                f"asr_model_ready {1 if model_ready else 0}",
                "# HELP asr_transcriptions_total Total completed HTTP transcription requests",
                "# TYPE asr_transcriptions_total counter",
                f"asr_transcriptions_total {self.transcriptions_total}",
                "# HELP asr_stream_segments_total Total finalized WebSocket streaming segments",
                "# TYPE asr_stream_segments_total counter",
                f"asr_stream_segments_total {self.stream_segments_total}",
                "# HELP asr_transcription_errors_total Total failed transcription attempts",
                "# TYPE asr_transcription_errors_total counter",
                f"asr_transcription_errors_total {self.transcription_errors_total}",
                "# HELP asr_rejected_requests_total Total requests rejected due to queue backpressure",
                "# TYPE asr_rejected_requests_total counter",
                f"asr_rejected_requests_total {self.rejected_requests_total}",
                "# HELP asr_audio_seconds_total Cumulative audio duration transcribed in seconds",
                "# TYPE asr_audio_seconds_total counter",
                f"asr_audio_seconds_total {self.audio_seconds_total:.3f}",
                "# HELP asr_processing_seconds_total Cumulative model inference time in seconds",
                "# TYPE asr_processing_seconds_total counter",
                f"asr_processing_seconds_total {self.processing_seconds_total:.3f}",
                "# HELP asr_realtime_factor_average Average real-time factor (processing_time / audio_duration)",
                "# TYPE asr_realtime_factor_average gauge",
                f"asr_realtime_factor_average {avg_rtf:.4f}",
                "# HELP asr_active_transcriptions Currently running inference tasks",
                "# TYPE asr_active_transcriptions gauge",
                f"asr_active_transcriptions {self.active_transcriptions}",
                "# HELP asr_queued_transcriptions Uploads currently waiting on disk for an inference slot",
                "# TYPE asr_queued_transcriptions gauge",
                f"asr_queued_transcriptions {self.queued_transcriptions}",
                "# HELP asr_active_ws_connections Currently active real-time WebSocket streams",
                "# TYPE asr_active_ws_connections gauge",
                f"asr_active_ws_connections {self.active_ws_connections}",
            ]
            return "\n".join(lines) + "\n"
