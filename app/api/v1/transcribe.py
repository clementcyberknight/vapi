from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.core.security import verify_api_key
from app.schemas.transcription import TranscriptionResponse
from app.services.audio import decode_audio_file, save_upload_to_tempfile
from app.services.engine import ASREngine, TranscriptionOptions

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


async def _run_file_transcription(
    request: Request,
    file: UploadFile,
    *,
    language: str | None,
    return_timestamps: bool,
    word_timestamps: bool,
    beam_size: int | None,
    vad_filter: bool | None,
    initial_prompt: str | None,
) -> TranscriptionResponse:
    settings: Settings = request.app.state.settings
    engine: ASREngine = request.app.state.engine
    metrics: MetricsRegistry = request.app.state.metrics

    if engine.status == "loading":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing on the server. Please retry shortly.",
        )

    tmp_path = await save_upload_to_tempfile(file, max_bytes=settings.max_upload_bytes)
    try:
        waveform, _ = await asyncio.to_thread(decode_audio_file, tmp_path)
        options = TranscriptionOptions(
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            return_timestamps=return_timestamps,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )
        try:
            result = await engine.transcribe(waveform, options)
        except Exception as exc:
            metrics.record_error()
            logger.exception("Transcription failed for '%s'", file.filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {exc}",
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()

    metrics.record_transcription(result.duration_seconds, result.processing_time_seconds)

    return TranscriptionResponse(
        text=result.text,
        model=settings.model_id,
        engine=settings.engine,
        language=result.language,
        filename=file.filename or "audio",
        duration_seconds=result.duration_seconds,
        processing_time_seconds=result.processing_time_seconds,
        rtf=result.rtf,
        segments=result.segments if (return_timestamps or word_timestamps) else None,
    )


@router.post(
    "/api/v1/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Transcribe an uploaded voice recording",
)
async def transcribe_v1(
    request: Request,
    file: UploadFile = File(..., description="Voice recording (wav, mp3, m4a, ogg, webm, opus, flac, etc.)"),
    language: str | None = Query(default=None, description="ISO language code (defaults to server setting, e.g. 'en')"),
    return_timestamps: bool = Query(default=False, description="Include segment timestamps"),
    word_timestamps: bool = Query(default=False, description="Include word-level timestamps"),
    beam_size: int | None = Query(default=None, ge=1, le=10, description="Override beam size (1=fastest)"),
    vad_filter: bool | None = Query(default=None, description="Override Silero VAD silence filtering"),
    initial_prompt: str | None = Query(default=None, max_length=500, description="Optional vocabulary hint prompt"),
) -> TranscriptionResponse:
    return await _run_file_transcription(
        request,
        file,
        language=language,
        return_timestamps=return_timestamps,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Alias for /api/v1/transcribe",
)
async def transcribe_alias(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Query(default=None),
    return_timestamps: bool = Query(default=False),
    word_timestamps: bool = Query(default=False),
    beam_size: int | None = Query(default=None, ge=1, le=10),
    vad_filter: bool | None = Query(default=None),
    initial_prompt: str | None = Query(default=None, max_length=500),
) -> TranscriptionResponse:
    return await _run_file_transcription(
        request,
        file,
        language=language,
        return_timestamps=return_timestamps,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )


@router.post(
    "/api/v1/transcribe/stream",
    dependencies=[Depends(verify_api_key)],
    summary="Stream transcription segments incrementally via Server-Sent Events (SSE)",
)
async def transcribe_sse_stream(
    request: Request,
    file: UploadFile = File(..., description="Voice recording to transcribe incrementally"),
    language: str | None = Query(default=None),
    word_timestamps: bool = Query(default=False),
    beam_size: int | None = Query(default=None, ge=1, le=10),
    vad_filter: bool | None = Query(default=None),
    initial_prompt: str | None = Query(default=None, max_length=500),
) -> StreamingResponse:
    settings: Settings = request.app.state.settings
    engine: ASREngine = request.app.state.engine
    metrics: MetricsRegistry = request.app.state.metrics

    if engine.status == "loading":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing on the server. Please retry shortly.",
        )

    tmp_path = await save_upload_to_tempfile(file, max_bytes=settings.max_upload_bytes)
    try:
        waveform, duration_seconds = await asyncio.to_thread(decode_audio_file, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()

    options = TranscriptionOptions(
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        return_timestamps=True,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        use_batched=False,  # Direct sequential decoding yields first segment immediately
    )

    async def _event_generator() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        collected_texts: list[str] = []
        try:
            async for seg in engine.stream_segments(waveform, options):
                collected_texts.append(seg.text)
                yield f"event: segment\ndata: {seg.model_dump_json()}\n\n"

            elapsed = round(time.perf_counter() - t0, 3)
            rtf = round(elapsed / duration_seconds, 4) if duration_seconds > 0 else 0.0
            metrics.record_transcription(duration_seconds, elapsed)
            done_payload = {
                "text": " ".join(collected_texts).strip(),
                "model": settings.model_id,
                "engine": settings.engine,
                "filename": file.filename or "audio",
                "duration_seconds": duration_seconds,
                "processing_time_seconds": elapsed,
                "rtf": rtf,
            }
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            metrics.record_error()
            logger.exception("SSE streaming transcription failed")
            err_payload = {"error": "transcription_failed", "detail": str(exc)}
            yield f"event: error\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
