from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StreamConfigMessage(BaseModel):
    """Optional client JSON message sent over WS /api/v1/stream to configure session."""

    type: Literal["config"] = "config"
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    language: str | None = Field(default=None)
    interim_results: bool = Field(default=True)
    initial_prompt: str | None = Field(default=None, max_length=500)


class StreamControlMessage(BaseModel):
    type: Literal["flush", "end", "ping"]
