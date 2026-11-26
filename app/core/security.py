from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, WebSocket, status

from app.core.config import Settings


def _extract_token(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def is_valid_api_key(expected: str | None, provided: str | None) -> bool:
    if not expected:
        return True
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Validate optional API key for HTTP endpoints."""
    settings: Settings = request.app.state.settings
    provided = _extract_token(x_api_key, authorization)
    if not is_valid_api_key(settings.api_key, provided):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def verify_ws_api_key(websocket: WebSocket, settings: Settings) -> bool:
    """Validate optional API key for WebSocket connections (header or `?api_key=`)."""
    if not settings.api_key:
        return True
    provided = (
        websocket.query_params.get("api_key")
        or _extract_token(
            websocket.headers.get("x-api-key"),
            websocket.headers.get("authorization"),
        )
    )
    return is_valid_api_key(settings.api_key, provided)
