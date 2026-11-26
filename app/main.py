from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.v1.router import v1_router
from app.core.config import Settings, get_settings
from app.core.exceptions import ASRError, ServiceBusyError
from app.core.logging import configure_logging, request_id_ctx
from app.core.metrics import MetricsRegistry
from app.schemas.transcription import ErrorResponse
from app.services.audio import cleanup_stale_uploads
from app.services.transcription import TranscriptionService

logger = logging.getLogger(__name__)


def create_app(
    app_settings: Settings | None = None,
    transcription_service: TranscriptionService | None = None,
    metrics: MetricsRegistry | None = None,
) -> FastAPI:
    cfg = app_settings or get_settings()
    configure_logging(level=cfg.log_level, log_format=cfg.log_format)

    metrics_registry = metrics or MetricsRegistry()
    service = transcription_service or TranscriptionService(cfg, metrics=metrics_registry)

    async def _preload_safely() -> None:
        # Details are logged by the backend and exposed via /health/ready
        with suppress(Exception):
            await asyncio.to_thread(service.load_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.transcription_service = service
        app.state.metrics = metrics_registry

        removed = cleanup_stale_uploads(cfg.upload_tmp_dir)
        if removed:
            logger.info("Cleaned up %d orphaned upload files in '%s'.", removed, cfg.upload_tmp_dir)

        preload_task: asyncio.Task[None] | None = None
        if cfg.preload_model:
            logger.info(
                "Starting background server-side model preload (model='%s', engine='%s')...",
                cfg.model_id,
                cfg.engine,
            )
            preload_task = asyncio.create_task(_preload_safely())

        yield

        if preload_task and not preload_task.done():
            preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)

    app = FastAPI(
        title=cfg.app_name,
        description=(
            "High-performance, production-ready Speech-to-Text & real-time audio streaming API "
            f"powered by `{cfg.model_id}` and CTranslate2 (`faster-whisper`)."
        ),
        version=cfg.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_ctx.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx.reset(token)

    @app.exception_handler(ASRError)
    async def asr_exception_handler(request: Request, exc: ASRError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(exc, ServiceBusyError):
            headers["Retry-After"] = str(exc.retry_after)
        body = ErrorResponse(
            error=exc.code,
            detail=exc.message,
            request_id=request_id_ctx.get(),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        body = ErrorResponse(
            error=f"http_{exc.status_code}",
            detail=str(exc.detail),
            request_id=request_id_ctx.get(),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error processing request %s", request.url.path)
        body = ErrorResponse(
            error="internal_server_error",
            detail="An unexpected server error occurred.",
            request_id=request_id_ctx.get(),
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get("/", tags=["Info"])
    async def root() -> dict[str, Any]:
        return {
            "service": cfg.app_name,
            "version": cfg.app_version,
            "model": cfg.model_id,
            "engine": cfg.engine,
            "docs": "/docs",
            "endpoints": {
                "transcribe": "/api/v1/transcribe",
                "transcribe_sse": "/api/v1/transcribe/stream",
                "realtime_websocket": "/api/v1/stream",
                "openai_compatible": "/v1/audio/transcriptions",
                "liveness": "/health/live",
                "readiness": "/health/ready",
                "metrics": "/metrics",
            },
        }

    app.include_router(health_router)
    app.include_router(v1_router)

    return app


app = create_app()
