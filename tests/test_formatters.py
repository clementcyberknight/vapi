from __future__ import annotations

import numpy as np

from app.schemas.transcription import TranscriptSegment, format_srt, format_vtt
from app.services.audio import compute_rms_energy, pcm16_bytes_to_float32


def test_srt_and_vtt_formatters() -> None:
    segments = [
        TranscriptSegment(id=0, start=0.0, end=1.25, text="Hello Lagos"),
        TranscriptSegment(id=1, start=61.5, end=65.0, text="How body?"),
    ]

    srt = format_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:01,250\nHello Lagos" in srt
    assert "2\n00:01:01,500 --> 00:01:05,000\nHow body?" in srt

    vtt = format_vtt(segments)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.250\nHello Lagos" in vtt
    assert "00:01:01.500 --> 00:01:05.000\nHow body?" in vtt


def test_pcm16_conversion_and_resampling() -> None:
    raw = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    floats = pcm16_bytes_to_float32(raw, source_sample_rate=16_000, target_sample_rate=16_000)
    assert floats.dtype == np.float32
    assert floats.size == 4
    assert np.isclose(floats[1], 0.5, atol=1e-3)
    assert compute_rms_energy(floats) > 0.4

    # Resample 48 kHz -> 16 kHz (3x downsampling)
    tone_48k = np.zeros(4800, dtype="<i2").tobytes()
    resampled = pcm16_bytes_to_float32(tone_48k, source_sample_rate=48_000, target_sample_rate=16_000)
    assert resampled.size == 1600
