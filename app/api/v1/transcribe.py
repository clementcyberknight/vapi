from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    MetricsDep,
    SettingsDep,
    TranscriptionServiceDep,
    UploadedAudio,
    UploadedAudioDep,
)
from app.core.security import verify_api_key
from app.schemas.transcription import TranscriptionResponse
from app.services.audio import stream_to_disk
from app.services.backends import TranscriptionOptions

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


async def _transcribe_uploaded_audio(
    audio: UploadedAudio,
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    *,
    language: str | None,
    return_timestamps: bool,
    word_timestamps: bool,
    beam_size: int | None,
    vad_filter: bool | None,
    initial_prompt: str | None,
) -> TranscriptionResponse:
    options = TranscriptionOptions(
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        return_timestamps=return_timestamps,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
    )
    result = await service.transcribe_file(audio.path, options)
    return TranscriptionResponse(
        text=result.text,
        model=settings.model_id,
        engine=settings.engine,
        language=result.language,
        filename=audio.filename,
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
    audio: UploadedAudioDep,
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    language: Annotated[
        str | None,
        Query(description="ISO language code (defaults to server setting, e.g. 'en')"),
    ] = None,
    return_timestamps: Annotated[bool, Query(description="Include segment timestamps")] = False,
    word_timestamps: Annotated[bool, Query(description="Include word-level timestamps")] = False,
    beam_size: Annotated[int | None, Query(ge=1, le=10, description="Override beam size (1=fastest)")] = None,
    vad_filter: Annotated[bool | None, Query(description="Override Silero VAD silence filtering")] = None,
    initial_prompt: Annotated[
        str | None,
        Query(max_length=500, description="Optional vocabulary hint prompt"),
    ] = None,
) -> TranscriptionResponse:
    return await _transcribe_uploaded_audio(
        audio,
        settings,
        service,
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
    audio: UploadedAudioDep,
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    language: str | None = Query(default=None),
    return_timestamps: bool = Query(default=False),
    word_timestamps: bool = Query(default=False),
    beam_size: int | None = Query(default=None, ge=1, le=10),
    vad_filter: bool | None = Query(default=None),
    initial_prompt: str | None = Query(default=None, max_length=500),
) -> TranscriptionResponse:
    return await _transcribe_uploaded_audio(
        audio,
        settings,
        service,
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
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    metrics: MetricsDep,
    file: UploadFile = File(..., description="Voice recording to transcribe incrementally"),
    language: str | None = Query(default=None),
    word_timestamps: bool = Query(default=False),
    beam_size: int | None = Query(default=None, ge=1, le=10),
    vad_filter: bool | None = Query(default=None),
    initial_prompt: str | None = Query(default=None, max_length=500),
) -> StreamingResponse:
    """Stream upload to disk, then decode JIT and yield segments live as SSE events.

    Note: The StreamingResponse generator owns temp-file cleanup so the file stays
    valid on disk until the async SSE stream completes or the client disconnects.
    """
    service.ensure_accepting_requests()
    tmp_path = await stream_to_disk(
        file.read,
        filename=file.filename,
        upload_dir=settings.upload_tmp_dir,
        max_bytes=settings.max_upload_bytes,
    )
    filename = file.filename or "audio"
    await file.close()

    options = TranscriptionOptions(
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        return_timestamps=True,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        use_batched=False,
    )

    async def _event_generator() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        collected_texts: list[str] = []
        duration_seconds = 0.0
        try:
            async for seg, duration_s in service.stream_file_segments(tmp_path, options):
                duration_seconds = duration_s
                if seg is not None:
                    collected_texts.append(seg.text)
                    yield f"event: segment\ndata: {seg.model_dump_json()}\n\n"

            elapsed = round(time.perf_counter() - t0, 3)
            rtf = round(elapsed / duration_seconds, 4) if duration_seconds > 0 else 0.0
            metrics.record_transcription(duration_seconds, elapsed)
            done_payload = {
                "text": " ".join(collected_texts).strip(),
                "model": settings.model_id,
                "engine": settings.engine,
                "filename": filename,
                "duration_seconds": duration_seconds,
                "processing_time_seconds": elapsed,
                "rtf": rtf,
            }
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("SSE streaming transcription failed for '%s'", filename)
            err_payload = {"error": "transcription_failed", "detail": str(exc)}
            yield f"event: error\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
