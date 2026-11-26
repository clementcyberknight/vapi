from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.core.security import verify_ws_api_key
from app.schemas.streaming import StreamConfigMessage, StreamControlMessage
from app.services.engine import ASREngine, TranscriptionOptions
from app.services.streaming import StreamingSession, UtteranceChunk

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Real-Time Streaming"])


@router.websocket("/api/v1/stream")
async def websocket_stream_audio(websocket: WebSocket) -> None:
    """Real-time bidirectional audio streaming over WebSocket.

    Protocol:
    - Connect to `ws://<host>/api/v1/stream`
      (optional query params: `sample_rate`, `language`, `interim_results`, `api_key`).
    - Send binary frames containing raw 16-bit signed little-endian mono PCM audio.
    - Optionally send JSON control frames:
        - `{"type": "config", "sample_rate": 16000, "language": "en", "interim_results": true}`
        - `{"type": "flush"}` (force finalize current utterance)
        - `{"type": "end"}` (finalize remaining audio and close cleanly)
        - `{"type": "ping"}` (replies `{"type": "pong"}`)
    - Receive JSON events: `ready`, `partial`, `final`, `done`, `pong`, `error`.
    """
    settings: Settings = websocket.app.state.settings
    engine: ASREngine = websocket.app.state.engine
    metrics: MetricsRegistry = websocket.app.state.metrics

    if not verify_ws_api_key(websocket, settings):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing API key")
        return

    if metrics.active_ws_connections >= settings.max_ws_connections:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server at max streaming capacity")
        return

    if engine.status == "loading":
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Model is still initializing")
        return

    await websocket.accept()
    metrics.inc_active_ws()

    session_id = uuid.uuid4().hex[:12]
    try:
        qp = websocket.query_params
        sample_rate = int(qp.get("sample_rate", "16000"))
        sample_rate = max(8_000, min(48_000, sample_rate))
        language = qp.get("language") or settings.default_language
        interim_results = qp.get("interim_results", "true").lower() not in {"0", "false", "no"}
        initial_prompt = qp.get("initial_prompt") or None
    except ValueError:
        sample_rate = 16_000
        language = settings.default_language
        interim_results = True
        initial_prompt = None

    session = StreamingSession(
        settings,
        sample_rate=sample_rate,
        language=language,
        interim_results=interim_results,
        initial_prompt=initial_prompt,
    )

    send_lock = asyncio.Lock()
    final_queue: asyncio.Queue[UtteranceChunk | None] = asyncio.Queue()
    latest_partial: list[UtteranceChunk | None] = [None]
    work_event = asyncio.Event()
    closed = False

    async def safe_send(payload: dict[str, Any]) -> None:
        nonlocal closed
        if closed:
            return
        async with send_lock:
            if closed:
                return
            try:
                await websocket.send_json(payload)
            except Exception:
                closed = True

    async def inference_worker() -> None:
        """Process finalized utterances in strict order; coalesce interim partials."""
        while True:
            await work_event.wait()
            work_event.clear()

            # Always drain finalized utterances first so ordering is guaranteed
            while not final_queue.empty():
                chunk = final_queue.get_nowait()
                if chunk is None:
                    final_queue.task_done()
                    return

                # Discard any stale partial from the same or earlier segment
                if latest_partial[0] is not None and latest_partial[0].segment_id <= chunk.segment_id:
                    latest_partial[0] = None

                try:
                    result = await engine.transcribe(
                        chunk.waveform,
                        TranscriptionOptions(
                            language=session.language,
                            beam_size=settings.beam_size,
                            vad_filter=settings.vad_filter,
                            return_timestamps=False,
                            initial_prompt=session.initial_prompt,
                            use_batched=False,
                        ),
                    )
                    metrics.record_stream_segment(result.duration_seconds, result.processing_time_seconds)
                    if result.text:
                        await safe_send(
                            {
                                "type": "final",
                                "segment_id": chunk.segment_id,
                                "text": result.text,
                                "language": result.language,
                                "duration_seconds": result.duration_seconds,
                                "processing_time_seconds": result.processing_time_seconds,
                            }
                        )
                except Exception as exc:
                    metrics.record_error()
                    logger.exception("WS session %s final transcription error", session_id)
                    await safe_send({"type": "error", "code": "inference_error", "message": str(exc)})
                finally:
                    final_queue.task_done()

            # Process only the newest pending interim partial (fast greedy beam_size=1)
            partial_chunk = latest_partial[0]
            latest_partial[0] = None
            if partial_chunk is not None and not closed:
                try:
                    result = await engine.transcribe(
                        partial_chunk.waveform,
                        TranscriptionOptions(
                            language=session.language,
                            beam_size=1,
                            vad_filter=False,
                            return_timestamps=False,
                            initial_prompt=session.initial_prompt,
                            use_batched=False,
                        ),
                    )
                    if result.text and result.text != session.last_partial_text:
                        session.last_partial_text = result.text
                        await safe_send(
                            {
                                "type": "partial",
                                "segment_id": partial_chunk.segment_id,
                                "text": result.text,
                                "duration_seconds": partial_chunk.duration_seconds,
                            }
                        )
                except Exception as exc:
                    logger.warning("WS session %s partial error: %s", session_id, exc)

            if not final_queue.empty() or latest_partial[0] is not None:
                work_event.set()

    worker_task = asyncio.create_task(inference_worker())

    try:
        await safe_send(
            {
                "type": "ready",
                "session_id": session_id,
                "model": settings.model_id,
                "engine": settings.engine,
                "sample_rate": session.sample_rate,
                "encoding": "pcm_s16le",
            }
        )

        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            audio_bytes = message.get("bytes")
            if audio_bytes is not None:
                actions = session.push_pcm16(audio_bytes)
                for action in actions:
                    if action.kind == "final":
                        await final_queue.put(action)
                    else:
                        latest_partial[0] = action
                    work_event.set()
                continue

            text_payload = message.get("text")
            if text_payload is not None:
                try:
                    data = json.loads(text_payload)
                except json.JSONDecodeError:
                    await safe_send({"type": "error", "code": "invalid_json", "message": "Malformed JSON frame."})
                    continue

                ctrl_type = data.get("type")
                if ctrl_type == "config":
                    try:
                        cfg = StreamConfigMessage.model_validate(data)
                    except Exception as exc:
                        await safe_send({"type": "error", "code": "invalid_config", "message": str(exc)})
                        continue
                    session.update_config(
                        sample_rate=cfg.sample_rate,
                        language=cfg.language,
                        interim_results=cfg.interim_results,
                        initial_prompt=cfg.initial_prompt,
                    )
                    await safe_send(
                        {
                            "type": "configured",
                            "sample_rate": session.sample_rate,
                            "language": session.language,
                            "interim_results": session.interim_results,
                        }
                    )
                elif ctrl_type in {"flush", "end", "ping"}:
                    StreamControlMessage.model_validate({"type": ctrl_type})
                    if ctrl_type == "ping":
                        await safe_send({"type": "pong"})
                    elif ctrl_type == "flush":
                        flushed = session.flush()
                        if flushed is not None:
                            await final_queue.put(flushed)
                            work_event.set()
                        await final_queue.join()
                    elif ctrl_type == "end":
                        flushed = session.flush()
                        if flushed is not None:
                            await final_queue.put(flushed)
                        await final_queue.put(None)
                        work_event.set()
                        await worker_task
                        await safe_send(
                            {
                                "type": "done",
                                "total_segments": session.total_final_segments,
                                "total_audio_seconds": round(session.total_audio_seconds, 3),
                            }
                        )
                        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                        break
                else:
                    await safe_send(
                        {
                            "type": "error",
                            "code": "unknown_control",
                            "message": f"Unsupported control message type: {ctrl_type!r}",
                        }
                    )
    except WebSocketDisconnect:
        logger.debug("WebSocket session %s disconnected by client.", session_id)
    finally:
        closed = True
        if not worker_task.done():
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        metrics.dec_active_ws()
