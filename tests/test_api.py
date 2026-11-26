from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.conftest import make_test_client


def test_health_and_metrics() -> None:
    with make_test_client() as client:
        live = client.get("/health/live")
        assert live.status_code == 200
        assert live.json()["status"] == "ok"
        assert "X-Request-ID" in live.headers

        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["engine"] == "ctranslate2"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "asr_model_ready 1" in metrics.text


def test_transcribe_v1_success_and_cleanup() -> None:
    fake_waveform = np.zeros(16_000, dtype=np.float32)

    with patch("app.api.v1.transcribe.decode_audio_file", return_value=(fake_waveform, 1.0)) as mock_decode:
        with make_test_client() as client:
            res = client.post(
                "/api/v1/transcribe?return_timestamps=true",
                files={"file": ("voice_note.m4a", b"fake-audio-bytes", "audio/mp4")},
            )

    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "How far, wetin dey happen today?"
    assert body["filename"] == "voice_note.m4a"
    assert body["duration_seconds"] == 1.0
    assert body["engine"] == "ctranslate2"
    assert body["rtf"] >= 0.0
    assert len(body["segments"]) == 1

    temp_path: Path = mock_decode.call_args.args[0]
    assert not temp_path.exists()


def test_transcribe_sse_streaming() -> None:
    fake_waveform = np.zeros(40_000, dtype=np.float32)

    with patch("app.api.v1.transcribe.decode_audio_file", return_value=(fake_waveform, 2.5)):
        with make_test_client() as client:
            res = client.post(
                "/api/v1/transcribe/stream",
                files={"file": ("meeting.mp3", b"fake-mp3", "audio/mpeg")},
            )

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert "event: segment" in res.text
    assert "Good morning Lagos." in res.text
    assert "We are live and streaming." in res.text
    assert "event: done" in res.text


def test_openai_compatible_formats() -> None:
    fake_waveform = np.zeros(32_000, dtype=np.float32)

    with patch("app.api.v1.openai.decode_audio_file", return_value=(fake_waveform, 2.0)):
        with make_test_client() as client:
            res_json = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", b"wav", "audio/wav")},
                data={"model": "NCAIR1/NigerianAccentedEnglish", "response_format": "json"},
            )
            assert res_json.status_code == 200
            assert res_json.json() == {"text": "How far, wetin dey happen today?"}

            res_text = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", b"wav", "audio/wav")},
                data={"response_format": "text"},
            )
            assert res_text.status_code == 200
            assert res_text.text == "How far, wetin dey happen today?"

            res_srt = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", b"wav", "audio/wav")},
                data={"response_format": "srt"},
            )
            assert res_srt.status_code == 200
            assert "00:00:00,000 --> 00:00:02,000" in res_srt.text

            res_vtt = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", b"wav", "audio/wav")},
                data={"response_format": "vtt"},
            )
            assert res_vtt.status_code == 200
            assert res_vtt.text.startswith("WEBVTT")


def test_validation_and_api_key_auth() -> None:
    fake_waveform = np.zeros(16_000, dtype=np.float32)

    with patch("app.api.v1.transcribe.decode_audio_file", return_value=(fake_waveform, 1.0)):
        with make_test_client(api_key="top-secret") as client:
            unauth = client.post(
                "/api/v1/transcribe",
                files={"file": ("voice.wav", b"abc", "audio/wav")},
            )
            assert unauth.status_code == 401

            bad_ext = client.post(
                "/api/v1/transcribe",
                headers={"Authorization": "Bearer top-secret"},
                files={"file": ("doc.pdf", b"abc", "application/pdf")},
            )
            assert bad_ext.status_code == 415

            empty = client.post(
                "/api/v1/transcribe",
                headers={"X-API-Key": "top-secret"},
                files={"file": ("empty.wav", b"", "audio/wav")},
            )
            assert empty.status_code == 400

            ok = client.post(
                "/api/v1/transcribe",
                headers={"X-API-Key": "top-secret"},
                files={"file": ("voice.wav", b"abc", "audio/wav")},
            )
            assert ok.status_code == 200
