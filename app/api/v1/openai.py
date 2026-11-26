from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.core.security import verify_api_key
from app.schemas.transcription import format_srt, format_vtt
from app.services.audio import decode_audio_file, save_upload_to_tempfile
from app.services.engine import ASREngine, TranscriptionOptions

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OpenAI Compatible"])


@router.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(verify_api_key)],
    summary="OpenAI-compatible speech-to-text endpoint",
)
async def openai_audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(default="json"),
    temperature: float = Form(default=0.0, ge=0.0, le=1.0),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
) -> JSONResponse | PlainTextResponse:
    """Drop-in compatible with the OpenAI Audio Transcriptions API."""
    settings: Settings = request.app.state.settings
    engine: ASREngine = request.app.state.engine
    metrics: MetricsRegistry = request.app.state.metrics

    if engine.status == "loading":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing on the server. Please retry shortly.",
        )

    want_segments = response_format in {"verbose_json", "srt", "vtt"}
    want_words = bool(timestamp_granularities and "word" in timestamp_granularities)

    tmp_path = await save_upload_to_tempfile(file, max_bytes=settings.max_upload_bytes)
    try:
        waveform, _ = await asyncio.to_thread(decode_audio_file, tmp_path)
        options = TranscriptionOptions(
            language=language,
            return_timestamps=want_segments or want_words,
            word_timestamps=want_words,
            initial_prompt=prompt,
            temperature=temperature,
        )
        try:
            result = await engine.transcribe(waveform, options)
        except Exception as exc:
            metrics.record_error()
            logger.exception("OpenAI-compatible transcription failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {exc}",
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()

    metrics.record_transcription(result.duration_seconds, result.processing_time_seconds)

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
