from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, File, Request, UploadFile

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.services.archival import ArchivalService
from app.services.audio import stream_to_disk
from app.services.transcription import TranscriptionService


@dataclass(slots=True)
class UploadedAudio:
    """Request-scoped handle for a disk-backed uploaded audio file."""

    path: Path
    filename: str
    content_type: str | None


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_transcription_service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


def get_archival_service(request: Request) -> ArchivalService:
    return request.app.state.archival_service


def get_metrics(request: Request) -> MetricsRegistry:
    return request.app.state.metrics


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
ArchivalServiceDep = Annotated[ArchivalService, Depends(get_archival_service)]
MetricsDep = Annotated[MetricsRegistry, Depends(get_metrics)]


async def receive_uploaded_audio(
    settings: SettingsDep,
    service: TranscriptionServiceDep,
    file: UploadFile = File(
        ...,
        description="Voice recording (wav, mp3, m4a, ogg, webm, opus, flac, etc.)",
    ),
) -> AsyncIterator[UploadedAudio]:
    """Standard FastAPI yield dependency that streams an upload to disk and guarantees cleanup."""
    service.ensure_accepting_requests()
    tmp_path = await stream_to_disk(
        file.read,
        filename=file.filename,
        upload_dir=settings.upload_tmp_dir,
        max_bytes=settings.max_upload_bytes,
    )
    try:
        yield UploadedAudio(
            path=tmp_path,
            filename=file.filename or "audio",
            content_type=file.content_type,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()


UploadedAudioDep = Annotated[UploadedAudio, Depends(receive_uploaded_audio)]
