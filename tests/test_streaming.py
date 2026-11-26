from __future__ import annotations

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.services.audio import SAMPLE_RATE
from app.services.streaming import StreamingSession
from tests.conftest import make_test_client


def _tone_pcm16(duration_s: float, amplitude: float = 0.25, sample_rate: int = SAMPLE_RATE) -> bytes:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    wave = (amplitude * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    return (wave * 32767.0).astype("<i2").tobytes()


def _silence_pcm16(duration_s: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    return np.zeros(int(sample_rate * duration_s), dtype="<i2").tobytes()


def test_streaming_session_endpointing_and_preroll() -> None:
    settings = Settings(
        preload_model=False,
        stream_min_speech_ms=200,
        stream_silence_ms=300,
        stream_partial_interval_ms=200,
        stream_energy_threshold=0.01,
    )
    session = StreamingSession(settings)

    # Pre-roll silence should not trigger actions
    assert session.push_pcm16(_silence_pcm16(0.2), now=0.0) == []

    # Speech onset + continuation triggers interim partial
    assert session.push_pcm16(_tone_pcm16(0.2), now=0.1) == []
    partials = session.push_pcm16(_tone_pcm16(0.2), now=0.4)
    assert len(partials) == 1
    assert partials[0].kind == "partial"
    assert partials[0].segment_id == 0

    # Trailing silence triggers finalization
    finals = session.push_pcm16(_silence_pcm16(0.35), now=0.8)
    assert len(finals) == 1
    assert finals[0].kind == "final"
    assert finals[0].segment_id == 0
    # Includes pre-roll + speech + trailing silence
    assert finals[0].duration_seconds >= 0.7


def test_websocket_realtime_streaming_protocol() -> None:
    with (
        make_test_client() as client,
        client.websocket_connect("/api/v1/stream?sample_rate=16000") as ws,
    ):
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["encoding"] == "pcm_s16le"

        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

        ws.send_json({"type": "config", "sample_rate": 16000, "language": "en", "interim_results": True})
        cfg_ack = ws.receive_json()
        assert cfg_ack["type"] == "configured"

        # Stream voiced audio followed by end control frame
        ws.send_bytes(_tone_pcm16(0.35))
        ws.send_json({"type": "end"})

        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["type"] == "done":
                break

        event_types = [e["type"] for e in events]
        assert "final" in event_types
        assert event_types[-1] == "done"


def test_websocket_rejects_unauthorized_when_api_key_set() -> None:
    with make_test_client(api_key="ws-secret") as client:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/stream"):
            pass

        with client.websocket_connect("/api/v1/stream?api_key=ws-secret") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            ws.send_json({"type": "end"})
            done = ws.receive_json()
            assert done["type"] == "done"
