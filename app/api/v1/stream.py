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
from app.services.archival import ArchivalService, ArchiveMetadata
from app.services.backends import TranscriptionOptions
from app.services.streaming import StreamingSession, UtteranceChunk
from app.services.transcription import TranscriptionService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Real-Time Streaming"])


class WebSocketStreamController:
    """Manages a single real-time audio WebSocket connection with coalesced partials."""

    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        service: TranscriptionService,
        archival: ArchivalService,
        metrics: MetricsRegistry,
    ) -> None:
        self._ws = websocket
        self._settings = settings
        self._service = service
        self._archival = archival
        self._metrics = metrics
        self._session_id = uuid.uuid4().hex[:12]
        self._session = self._build_session_from_query()

        self._send_lock = asyncio.Lock()
        self._final_queue: asyncio.Queue[UtteranceChunk | None] = asyncio.Queue()
        self._latest_partial: UtteranceChunk | None = None
        self._work_event = asyncio.Event()
        self._closed = False

    def _build_session_from_query(self) -> StreamingSession:
        qp = self._ws.query_params
        try:
            sample_rate = max(8_000, min(48_000, int(qp.get("sample_rate", "16000"))))
        except ValueError:
            sample_rate = 16_000

        language = qp.get("language") or self._settings.default_language
        interim_results = qp.get("interim_results", "true").lower() not in {"0", "false", "no"}
        initial_prompt = qp.get("initial_prompt") or None

        return StreamingSession(
            self._settings,
            sample_rate=sample_rate,
            language=language,
            interim_results=interim_results,
            initial_prompt=initial_prompt,
        )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        async with self._send_lock:
            if self._closed:
                return
            try:
                await self._ws.send_json(payload)
            except Exception:
                self._closed = True

    async def _process_final_chunk(self, chunk: UtteranceChunk) -> None:
        if self._latest_partial is not None and self._latest_partial.segment_id <= chunk.segment_id:
            self._latest_partial = None

        try:
            result = await self._service.transcribe_waveform(
                chunk.waveform,
                TranscriptionOptions(
                    language=self._session.language,
                    beam_size=self._settings.beam_size,
                    vad_filter=self._settings.vad_filter,
                    return_timestamps=False,
                    initial_prompt=self._session.initial_prompt,
                    use_batched=False,
                ),
            )
            self._metrics.record_stream_segment(result.duration_seconds, result.processing_time_seconds)
            if result.text:
                self._archival.enqueue_waveform(
                    chunk.waveform,
                    ArchiveMetadata(
                        request_id=f"{self._session_id}-{chunk.segment_id}",
                        session_id=self._session_id,
                        source="websocket",
                        original_filename=f"stream_{self._session_id}_seg{chunk.segment_id}.wav",
                        content_type="audio/wav",
                        model=self._settings.model_id,
                        engine=self._settings.engine,
                        language=result.language,
                        transcript_text=result.text,
                        duration_seconds=result.duration_seconds,
                        processing_time_seconds=result.processing_time_seconds,
                        rtf=result.rtf,
                    ),
                )
                await self._send_json(
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
            logger.exception("WS session %s final transcription error", self._session_id)
            await self._send_json({"type": "error", "code": "inference_error", "message": str(exc)})

    async def _process_partial_chunk(self, chunk: UtteranceChunk) -> None:
        try:
            result = await self._service.transcribe_waveform(
                chunk.waveform,
                TranscriptionOptions(
                    language=self._session.language,
                    beam_size=1,
                    vad_filter=False,
                    return_timestamps=False,
                    initial_prompt=self._session.initial_prompt,
                    use_batched=False,
                ),
            )
            if result.text and result.text != self._session.last_partial_text:
                self._session.last_partial_text = result.text
                await self._send_json(
                    {
                        "type": "partial",
                        "segment_id": chunk.segment_id,
                        "text": result.text,
                        "duration_seconds": chunk.duration_seconds,
                    }
                )
        except Exception as exc:
            logger.warning("WS session %s partial inference skipped: %s", self._session_id, exc)

    async def _inference_worker(self) -> None:
        while True:
            await self._work_event.wait()
            self._work_event.clear()

            while not self._final_queue.empty():
                chunk = self._final_queue.get_nowait()
                if chunk is None:
                    self._final_queue.task_done()
                    return
                try:
                    await self._process_final_chunk(chunk)
                finally:
                    self._final_queue.task_done()

            partial = self._latest_partial
            self._latest_partial = None
            if partial is not None and not self._closed:
                await self._process_partial_chunk(partial)

            if not self._final_queue.empty() or self._latest_partial is not None:
                self._work_event.set()

    async def _handle_binary_frame(self, audio_bytes: bytes) -> None:
        actions = self._session.push_pcm16(audio_bytes)
        for action in actions:
            if action.kind == "final":
                await self._final_queue.put(action)
            else:
                self._latest_partial = action
            self._work_event.set()

    async def _handle_text_frame(self, text_payload: str, worker_task: asyncio.Task[None]) -> bool:
        """Process a JSON control frame. Returns True if the session should terminate."""
        try:
            data = json.loads(text_payload)
        except json.JSONDecodeError:
            await self._send_json({"type": "error", "code": "invalid_json", "message": "Malformed JSON frame."})
            return False

        ctrl_type = data.get("type")
        if ctrl_type == "config":
            try:
                cfg = StreamConfigMessage.model_validate(data)
            except Exception as exc:
                await self._send_json({"type": "error", "code": "invalid_config", "message": str(exc)})
                return False
            self._session.update_config(
                sample_rate=cfg.sample_rate,
                language=cfg.language,
                interim_results=cfg.interim_results,
                initial_prompt=cfg.initial_prompt,
            )
            await self._send_json(
                {
                    "type": "configured",
                    "sample_rate": self._session.sample_rate,
                    "language": self._session.language,
                    "interim_results": self._session.interim_results,
                }
            )
            return False

        if ctrl_type not in {"flush", "end", "ping"}:
            await self._send_json(
                {
                    "type": "error",
                    "code": "unknown_control",
                    "message": f"Unsupported control message type: {ctrl_type!r}",
                }
            )
            return False

        StreamControlMessage.model_validate({"type": ctrl_type})
        if ctrl_type == "ping":
            await self._send_json({"type": "pong"})
            return False

        if ctrl_type == "flush":
            flushed = self._session.flush()
            if flushed is not None:
                await self._final_queue.put(flushed)
                self._work_event.set()
            await self._final_queue.join()
            return False

        # ctrl_type == "end"
        flushed = self._session.flush()
        if flushed is not None:
            await self._final_queue.put(flushed)
        await self._final_queue.put(None)
        self._work_event.set()
        await worker_task
        await self._send_json(
            {
                "type": "done",
                "total_segments": self._session.total_final_segments,
                "total_audio_seconds": round(self._session.total_audio_seconds, 3),
            }
        )
        await self._ws.close(code=status.WS_1000_NORMAL_CLOSURE)
        return True

    async def run(self) -> None:
        worker_task = asyncio.create_task(self._inference_worker())
        try:
            await self._send_json(
                {
                    "type": "ready",
                    "session_id": self._session_id,
                    "model": self._settings.model_id,
                    "engine": self._settings.engine,
                    "sample_rate": self._session.sample_rate,
                    "encoding": "pcm_s16le",
                }
            )

            while True:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                audio_bytes = message.get("bytes")
                if audio_bytes is not None:
                    await self._handle_binary_frame(audio_bytes)
                    continue

                text_payload = message.get("text")
                if text_payload is not None:
                    should_stop = await self._handle_text_frame(text_payload, worker_task)
                    if should_stop:
                        break
        except WebSocketDisconnect:
            logger.debug("WebSocket session %s disconnected by client.", self._session_id)
        finally:
            self._closed = True
            if not worker_task.done():
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)


@router.websocket("/api/v1/stream")
async def websocket_stream_audio(websocket: WebSocket) -> None:
    """Real-time bidirectional audio streaming over WebSocket.

    Protocol:
    - Connect to `ws://<host>/api/v1/stream`
      (optional query params: `sample_rate`, `language`, `interim_results`, `api_key`).
    - Send binary frames containing raw 16-bit signed little-endian mono PCM audio.
    - Send JSON control frames: `config`, `flush`, `end`, `ping`.
    - Receive JSON events: `ready`, `partial`, `final`, `done`, `pong`, `error`.
    """
    settings: Settings = websocket.app.state.settings
    service: TranscriptionService = websocket.app.state.transcription_service
    archival: ArchivalService = websocket.app.state.archival_service
    metrics: MetricsRegistry = websocket.app.state.metrics

    if not verify_ws_api_key(websocket, settings):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing API key")
        return

    if metrics.active_ws_connections >= settings.max_ws_connections:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server at max streaming capacity")
        return

    if service.status == "loading":
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Model is still initializing")
        return

    await websocket.accept()
    metrics.inc_active_ws()
    try:
        controller = WebSocketStreamController(websocket, settings, service, archival, metrics)
        await controller.run()
    finally:
        metrics.dec_active_ws()
