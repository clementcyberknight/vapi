from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.schemas.transcription import HealthResponse
from app.services.engine import ASREngine

router = APIRouter(tags=["Observability"])


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def health_live(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    engine: ASREngine = request.app.state.engine
    return HealthResponse(
        status="ok",
        model=settings.model_id,
        engine=settings.engine,
        model_status=engine.status,
        device=engine.resolved_device,
        compute_type=engine.resolved_compute_type,
        error=engine.last_error,
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def health_ready(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    engine: ASREngine = request.app.state.engine

    if engine.is_ready or (not settings.preload_model and engine.status == "idle"):
        payload = HealthResponse(
            status="ready",
            model=settings.model_id,
            engine=settings.engine,
            model_status=engine.status,
            device=engine.resolved_device,
            compute_type=engine.resolved_compute_type,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload.model_dump())

    payload = HealthResponse(
        status="not_ready",
        model=settings.model_id,
        engine=settings.engine,
        model_status=engine.status,
        device=engine.resolved_device,
        compute_type=engine.resolved_compute_type,
        error=engine.last_error,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
)
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    metrics: MetricsRegistry = request.app.state.metrics
    engine: ASREngine = request.app.state.engine
    return PlainTextResponse(
        content=metrics.render_prometheus(model_ready=engine.is_ready),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
