from __future__ import annotations

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] | None = None


class TranscriptionResponse(BaseModel):
    text: str
    model: str
    engine: str
    language: str | None = None
    filename: str
    duration_seconds: float
    processing_time_seconds: float
    rtf: float = Field(description="Real-time factor (processing_time / audio_duration; <1.0 is faster than real time)")
    segments: list[TranscriptSegment] | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    engine: str
    model_status: str
    device: str | None = None
    compute_type: str | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None


def _format_timestamp_srt(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp_srt(seconds).replace(",", ".")


def format_srt(segments: list[TranscriptSegment]) -> str:
    blocks: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        blocks.append(
            f"{idx}\n"
            f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def format_vtt(segments: list[TranscriptSegment]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)
