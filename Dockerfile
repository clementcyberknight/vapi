# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# ==============================================================================
# Stage 1: Dependency Builder (Astral uv + BuildKit cache mounts)
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml ./

# Install lean PyTorch wheel first (used for one-time server CT2 conversion or fallback),
# then install faster-whisper / CTranslate2 / FastAPI dependencies without upgrading torch.
# Model weights are NEVER baked into the image — they download on the server into /cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "https://pypi.org/simple" \
        "torch>=2.6.0" \
    && uv pip install --python /app/.venv/bin/python \
        --no-upgrade \
        -r pyproject.toml

# ==============================================================================
# Stage 2: Minimal Hardened Runtime Image
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="Nigerian Accented English ASR API" \
      org.opencontainers.image.description="Fast, production-ready Speech-to-Text & real-time streaming API powered by NCAIR1/NigerianAccentedEnglish and CTranslate2" \
      org.opencontainers.image.licenses="MIT"

# ffmpeg decodes all voice recording formats; libgomp1 provides OpenMP for CTranslate2
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1

# Create non-root runtime user and persistent cache/upload scratch directories
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /cache/huggingface /cache/ct2 /cache/torch /cache/tmp \
    && chown -R appuser:appuser /cache

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser app /app/app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR="/tmp" \
    CACHE_DIR="/cache" \
    UPLOAD_TMP_DIR="/cache/tmp" \
    XDG_CACHE_HOME="/cache" \
    HF_HOME="/cache/huggingface" \
    TORCH_HOME="/cache/torch" \
    CT2_VERBOSE=0 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]

CMD [ \
    "uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--proxy-headers", \
    "--forwarded-allow-ips", "*", \
    "--ws-ping-interval", "20", \
    "--ws-ping-timeout", "20" \
]
