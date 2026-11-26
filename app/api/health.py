from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.dependencies import MetricsDep, SettingsDep, TranscriptionServiceDep
from app.schemas.transcription import HealthResponse

router = APIRouter(tags=["Observability"])


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def health_live(
    settings: SettingsDep,
    service: TranscriptionServiceDep,
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.model_id,
        engine=settings.engine,
        model_status=service.status,
        device=service.resolved_device,
        compute_type=service.resolved_compute_type,
        error=service.last_error,
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def health_ready(
    settings: SettingsDep,
    service: TranscriptionServiceDep,
) -> JSONResponse:
    if service.is_ready or (not settings.preload_model and service.status == "idle"):
        payload = HealthResponse(
            status="ready",
            model=settings.model_id,
            engine=settings.engine,
            model_status=service.status,
            device=service.resolved_device,
            compute_type=service.resolved_compute_type,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload.model_dump())

    payload = HealthResponse(
        status="not_ready",
        model=settings.model_id,
        engine=settings.engine,
        model_status=service.status,
        device=service.resolved_device,
        compute_type=service.resolved_compute_type,
        error=service.last_error,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
async def metrics_endpoint(
    metrics: MetricsDep,
    service: TranscriptionServiceDep,
) -> PlainTextResponse:
    return PlainTextResponse(
        content=metrics.render_prometheus(model_ready=service.is_ready),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
