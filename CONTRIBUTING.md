# Contributing

Thank you for your interest in contributing to the **Nigerian Accented English ASR API**!

## Development Workflow

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`).
2. Keep pull requests focused, typed, and well-tested.
3. Run linting and offline unit tests (no model download required):
   ```bash
   ruff check app tests examples
   pytest -q
   ```
4. Never commit `.env` files, Hugging Face tokens, or downloaded model weights.

## Architecture Overview

- `app/core/` — Configuration, structured logging, API key authentication, Prometheus metrics.
- `app/services/` — Audio decoding (`audio.py`), CTranslate2 / Transformers inference (`engine.py`), and bounded-memory live streaming (`streaming.py`).
- `app/api/` — Versioned HTTP, SSE, WebSocket, OpenAI-compatible, and health endpoints.
- `app/schemas/` — Pydantic v2 schemas and SRT/VTT formatters.
