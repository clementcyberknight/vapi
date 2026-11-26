from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.openai import router as openai_router
from app.api.v1.stream import router as stream_router
from app.api.v1.transcribe import router as transcribe_router

v1_router = APIRouter()
v1_router.include_router(transcribe_router)
v1_router.include_router(stream_router)
v1_router.include_router(openai_router)
