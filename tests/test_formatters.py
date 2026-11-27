from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.schemas.transcription import TranscriptSegment, format_srt, format_vtt
from app.services.audio import (
    compute_rms_energy,
    normalize_speech_waveform,
    pcm16_bytes_to_float32,
)
from app.services.backends.faster_whisper import _detect_trained_without_timestamps
from app.services.streaming import StreamingSession


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

    tone_48k = np.zeros(4800, dtype="<i2").tobytes()
    resampled = pcm16_bytes_to_float32(tone_48k, source_sample_rate=48_000, target_sample_rate=16_000)
    assert resampled.size == 1600


def test_normalize_speech_boosts_quiet_speech_but_not_silence() -> None:
    # Quiet speech (ref level = 0.08) gets boosted cleanly toward ~0.80
    quiet = np.array([0.0, 0.08, -0.08, 0.04] * 100, dtype=np.float32)
    boosted = normalize_speech_waveform(quiet)
    assert float(np.percentile(np.abs(boosted), 99.0)) > 0.6

    # A single loud mic click/pop must NOT prevent quiet speech from being normalized
    quiet_with_click = quiet.copy()
    quiet_with_click[0] = 0.98
    boosted_click = normalize_speech_waveform(quiet_with_click)
    assert float(np.percentile(np.abs(boosted_click), 95.0)) > 0.5

    # Near-silent noise floor (< 0.008) is NOT amplified into loud hiss
    hiss = np.array([0.001, -0.002, 0.0015] * 100, dtype=np.float32)
    untouched = normalize_speech_waveform(hiss)
    assert float(np.max(np.abs(untouched))) < 0.01


def test_streaming_session_context_carryover() -> None:
    settings = Settings(_env_file=None, preload_model=False)
    session = StreamingSession(settings, initial_prompt="Lagos, Naira")
    assert session.effective_prompt() == "Lagos, Naira"

    session.record_final_text("Good morning, send five thousand Naira to Chinedu.")
    assert "Chinedu." in (session.effective_prompt() or "")
    assert (session.effective_prompt() or "").startswith("Lagos, Naira")

    # Context tail remains strictly bounded
    session.record_final_text(" ".join(f"word{i}" for i in range(60)))
    assert len(session._recent_context) <= StreamingSession.MAX_CONTEXT_CHARS


def test_detect_trained_without_timestamps(tmp_path: Path) -> None:
    cfg = {"forced_decoder_ids": [[1, 50259], [2, 50359], [3, 50363]]}
    (tmp_path / "hf_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert _detect_trained_without_timestamps(tmp_path) is True

    legacy = tmp_path / "legacy_cfg"
    legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert _detect_trained_without_timestamps(legacy) is True

    other = tmp_path / "with_ts"
    other.mkdir()
    (other / "hf_config.json").write_text(json.dumps({"forced_decoder_ids": [[1, 50259]]}), encoding="utf-8")
    assert _detect_trained_without_timestamps(other) is False
