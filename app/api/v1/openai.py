from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from app.api.dependencies import (
    ArchivalServiceDep,
    SettingsDep,
    TranscriptionServiceDep,
    UploadedAudioDep,
)
from app.core.logging import request_id_ctx
from app.core.security import verify_api_key
from app.schemas.transcription import format_srt, format_vtt
from app.services.archival import ArchiveMetadata
from app.services.backends import TranscriptionOptions

router = APIRouter(tags=["OpenAI Compatible"])


@router.post(
    "/v1/audio/transcriptions",
    response_model=None,
    dependencies=[Depends(verify_api_key)],
    summary="OpenAI-compatible speech-to-text endpoint",
)
async def openai_audio_transcriptions(
    audio: UploadedAudioDep,
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    archival: ArchivalServiceDep,
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(default="json"),
    temperature: float | None = Form(default=None, ge=0.0, le=1.0),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
) -> Response:
    """Drop-in compatible with the OpenAI Audio Transcriptions API."""
    want_segments = response_format in {"verbose_json", "srt", "vtt"}
    want_words = bool(timestamp_granularities and "word" in timestamp_granularities)

    options = TranscriptionOptions(
        language=language,
        return_timestamps=want_segments or want_words,
        word_timestamps=want_words,
        initial_prompt=prompt,
        temperature=temperature,
    )
    result = await service.transcribe_file(audio.path, options)

    archival.enqueue_file(
        audio.path,
        ArchiveMetadata(
            request_id=request_id_ctx.get(),
            source="openai",
            original_filename=audio.filename,
            content_type=audio.content_type,
            model=settings.model_id,
            engine=settings.engine,
            language=result.language,
            transcript_text=result.text,
            duration_seconds=result.duration_seconds,
            processing_time_seconds=result.processing_time_seconds,
            rtf=result.rtf,
            segments=result.segments if (want_segments or want_words) else None,
        ),
    )

    if response_format == "text":
        return PlainTextResponse(content=result.text)

    if response_format == "srt":
        return PlainTextResponse(content=format_srt(result.segments), media_type="text/plain; charset=utf-8")

    if response_format == "vtt":
        return PlainTextResponse(content=format_vtt(result.segments), media_type="text/vtt; charset=utf-8")

    if response_format == "verbose_json":
        payload = {
            "task": "transcribe",
            "language": result.language or "en",
            "duration": result.duration_seconds,
            "text": result.text,
            "model": model or settings.model_id,
            "segments": [seg.model_dump(exclude_none=True) for seg in result.segments],
        }
        return JSONResponse(content=payload)

    return JSONResponse(content={"text": result.text})
