# Nigerian Accented English ASR API

A **fast, production-ready, open-source Speech-to-Text & Real-Time Audio Streaming API** powered by [`NCAIR1/NigerianAccentedEnglish`](https://huggingface.co/NCAIR1/NigerianAccentedEnglish) (Whisper Small fine-tune) and **CTranslate2 (`faster-whisper`)**.

## Key Features

| Feature | Implementation |
| --- | --- |
| **4x–6x Faster Inference** | Downloads only inference weights and tokenizer files on the server (skipping the ~1.9 GB `optimizer.pt` training state), converts once into quantized **CTranslate2** (`int8` on CPU / `float16` on GPU), and runs via `faster-whisper` + Silero VAD + batched decoding. |
| **10–20 Concurrent 20 MB Uploads in 3 GB RAM** | Streams uploads asynchronously via `aiofiles` to disk (`/cache/tmp`), enforces bounded queue admission (`MAX_QUEUED_REQUESTS=20`), and decodes audio with `ffmpeg` **just-in-time inside the inference semaphore** so queued uploads use ~0 MB RAM. |
| **Real-Time Audio Streaming** | Bidirectional WebSocket (`WS /api/v1/stream`) with a 250 ms pre-roll ring buffer (zero clipped word starts), RMS energy VAD endpointing, coalesced greedy interim partials, ordered finalization, and ~640 KB max audio buffer per stream. |
| **Incremental SSE Streaming** | `POST /api/v1/transcribe/stream` streams decoded segments over Server-Sent Events (`text/event-stream`) as soon as each segment completes. |
| **OpenAI Drop-In Compatible** | `POST /v1/audio/transcriptions` supports `json`, `text`, `verbose_json`, `srt`, `vtt`, and `timestamp_granularities[]=word` — works directly with official OpenAI SDKs. |
| **Zero Local Model Downloads** | No weights are baked into the Docker image or downloaded locally. Models download and convert once on your server into the persistent `model_cache` Docker volume. |
| **Hardened Modern Docker** | BuildKit multi-stage `uv` build, non-root `appuser` (UID `10001`), `read_only` rootfs, `cap_drop: [ALL]`, `no-new-privileges`, `pids_limit`, strict CPU/RAM limits, and readiness healthchecks. |
| **Production Observability** | Structured JSON logs with `X-Request-ID`, `/health/live`, `/health/ready`, and Prometheus `/metrics` (RTF, active/queued counts, errors, and WebSocket streams). |
| **Optional R2 Backup & Supabase Audit Log** | Zero-latency background backup of voice recordings to **Cloudflare R2** (`boto3` S3v4) and structured audit logging into **Supabase Postgres** (`acreate_client` + RLS-hardened schema). |

---

## Architecture

```text
app/
├── main.py                         # FastAPI app factory, lifespan, middleware, domain exception handlers
├── core/                           # Infrastructure: config, domain exceptions, JSON logging, API-key auth, Prometheus metrics
├── schemas/                        # Pydantic v2 DTOs, WebSocket schemas, SRT/VTT formatters
├── services/                       # Framework-free business logic (zero FastAPI imports)
│   ├── audio.py                    # aiofiles disk streaming, stale cleanup, ffmpeg decode, PCM16 + resampling
│   ├── streaming.py                # Bounded-memory real-time speech session with pre-roll & silence endpointing
│   ├── transcription.py            # Bounded-queue admission control + JIT decoding + metrics orchestration
│   └── backends/                   # Pluggable Strategy pattern (ASRBackend ABC)
│       ├── base.py
│       ├── faster_whisper.py       # Default: CTranslate2 auto-converter + faster-whisper + Silero VAD
│       └── transformers.py         # Optional PyTorch / Hugging Face Transformers fallback
└── api/                            # Thin transport layer using standard FastAPI Depends
    ├── dependencies.py             # Annotated Depends + yield UploadedAudioDep with guaranteed cleanup
    ├── health.py                   # GET /health/live, GET /health/ready, GET /metrics
    └── v1/
        ├── router.py               # v1 router aggregator
        ├── transcribe.py           # POST /api/v1/transcribe, /transcribe, /api/v1/transcribe/stream (SSE)
        ├── stream.py               # WS /api/v1/stream (WebSocketStreamController)
        └── openai.py               # POST /v1/audio/transcriptions
```

---

## Quick Start (Server Deployment)

### 1. Accept Model Access on Hugging Face

`NCAIR1/NigerianAccentedEnglish` is an auto-gated model repository:

1. Sign in to Hugging Face and accept terms at [https://huggingface.co/NCAIR1/NigerianAccentedEnglish](https://huggingface.co/NCAIR1/NigerianAccentedEnglish).
2. Generate a Read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` on your server and set at minimum `HF_TOKEN` (and `API_KEY` for public deployments):

```env
HF_TOKEN=hf_your_real_token_here
API_KEY=your-strong-secret-key
CPU_LIMIT=2.0
MEMORY_LIMIT=3G
CPU_THREADS=2
NUM_WORKERS=1
MAX_CONCURRENT_TRANSCRIPTIONS=2
MAX_QUEUED_REQUESTS=20
```

> **Security**: `.env` is excluded by both `.gitignore` and `.dockerignore`. Never commit real tokens or secrets.

### 3. Build & Run with Docker Compose

```bash
# Coolify / Dokploy / Traefik reverse-proxy server (uses internal `expose: "8000"` — zero host port conflicts)
docker compose up -d --build

# Standalone VPS or local machine without a reverse proxy (publishes HOST_PORT, default 8080 -> 8000)
docker compose -f compose.yaml -f compose.local.yaml up -d --build

# NVIDIA GPU server (requires NVIDIA Container Toolkit — CUDA + float16 CTranslate2)
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

> **Coolify Domain Routing**: In Coolify, open **Domains → Add domain**, select `asr-api`, and enter `https://your-domain.com:8000` (the `:8000` suffix tells Coolify's proxy which internal container port to forward to while public traffic uses standard HTTPS `443`).

Monitor first-boot download and one-time CTranslate2 conversion:

```bash
docker compose logs -f asr-api
```

Verify readiness:

```bash
curl http://localhost:8000/health/ready
```

Interactive OpenAPI docs: `http://<server>:8000/docs`.

---

## API Reference

> **Authentication**: When `API_KEY` is configured, pass either `X-API-Key: <key>` or `Authorization: Bearer <key>` on HTTP requests, and `?api_key=<key>` (or headers) on WebSocket connections.

### 1. Upload Voice Recording — `POST /api/v1/transcribe` (alias: `POST /transcribe`)

Supported formats: `.wav`, `.mp3`, `.m4a`, `.mp4`, `.ogg`, `.opus`, `.webm`, `.flac`, `.aac`, `.amr`, `.wma`, `.3gp`, `.mpeg`, `.mpga`.

**Query parameters**:
- `language` (`string`, optional) — ISO language code (defaults to `DEFAULT_LANGUAGE`, e.g. `en`)
- `return_timestamps` (`bool`, default `false`) — include segment start/end timestamps
- `word_timestamps` (`bool`, default `false`) — include per-word timestamps
- `beam_size` (`int`, `1–10`, optional) — override server `BEAM_SIZE` (`1` = fastest)
- `vad_filter` (`bool`, optional) — override server `VAD_FILTER`
- `initial_prompt` (`string`, max 500 chars, optional) — domain vocabulary hint

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe?return_timestamps=true" \
  -H "X-API-Key: your-secret-key" \
  -F "file=@voice_note.m4a"
```

```json
{
  "text": "Good morning, please help me check my account balance.",
  "model": "NCAIR1/NigerianAccentedEnglish",
  "engine": "ctranslate2",
  "language": "en",
  "filename": "voice_note.m4a",
  "duration_seconds": 4.25,
  "processing_time_seconds": 0.62,
  "rtf": 0.1459,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.25,
      "text": "Good morning, please help me check my account balance.",
      "words": null
    }
  ]
}
```

### 2. Incremental SSE Streaming — `POST /api/v1/transcribe/stream`

Streams decoded segments immediately via Server-Sent Events:

```bash
curl -N -X POST "http://localhost:8000/api/v1/transcribe/stream" \
  -H "X-API-Key: your-secret-key" \
  -F "file=@long_recording.mp3"
```

```text
event: segment
data: {"id":0,"start":0.0,"end":3.8,"text":"Welcome everybody to today's meeting.","words":null}

event: done
data: {"text":"Welcome everybody to today's meeting.","model":"NCAIR1/NigerianAccentedEnglish","engine":"ctranslate2","filename":"long_recording.mp3","duration_seconds":3.8,"processing_time_seconds":0.54,"rtf":0.1421}
```

### 3. Real-Time Audio Streaming — `WS /api/v1/stream`

Connect to `ws://<server>:8000/api/v1/stream` (optional query params: `sample_rate`, `language`, `interim_results`, `initial_prompt`, `api_key`).

- **Client → Server**:
  - **Binary frames**: Raw **16-bit signed little-endian mono PCM** (`pcm_s16le`) audio chunks (recommended 100–300 ms per frame). Rates between `8000` and `48000` Hz are supported and resampled automatically to 16 kHz.
  - **JSON control frames**:
    - `{"type": "config", "sample_rate": 16000, "language": "en", "interim_results": true, "initial_prompt": null}`
    - `{"type": "flush"}` — force-finalize the active utterance without closing
    - `{"type": "end"}` — finalize remaining audio, emit `done`, and close cleanly
    - `{"type": "ping"}` — replies `{"type": "pong"}`
- **Server → Client JSON events**:
  - `ready` — `{type, session_id, model, engine, sample_rate, encoding}`
  - `configured` — `{type, sample_rate, language, interim_results}`
  - `partial` — `{type, segment_id, text, duration_seconds}`
  - `final` — `{type, segment_id, text, language, duration_seconds, processing_time_seconds}`
  - `done` — `{type, total_segments, total_audio_seconds}`
  - `pong` / `error` (`{type, code, message}`)

Ready-to-run streaming clients are included in [`examples/`](./examples):
- Open [`examples/live_mic.html`](./examples/live_mic.html) in a browser to stream your microphone live.
- Run [`examples/stream_client.py`](./examples/stream_client.py) to stream a 16-bit mono WAV file over WebSocket.

### 4. OpenAI-Compatible Endpoint — `POST /v1/audio/transcriptions`

Drop-in compatible with official OpenAI SDKs (`response_format`: `json`, `text`, `verbose_json`, `srt`, `vtt`; optional `timestamp_granularities[]=word`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-key")

with open("voice_note.m4a", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="NCAIR1/NigerianAccentedEnglish",
        file=audio,
        response_format="verbose_json",
    )

print(transcript.text)
```

### 5. Health, Errors & Prometheus Metrics

- `GET /health/live` — Liveness probe (`200`)
- `GET /health/ready` — Readiness probe (`200` when model is loaded; `503` while downloading/converting/loading)
- `GET /metrics` — Prometheus text format (`asr_model_ready`, `asr_transcriptions_total`, `asr_stream_segments_total`, `asr_transcription_errors_total`, `asr_rejected_requests_total`, `asr_audio_seconds_total`, `asr_processing_seconds_total`, `asr_realtime_factor_average`, `asr_active_transcriptions`, `asr_queued_transcriptions`, `asr_active_ws_connections`)
- **Standard error envelope** (`400`, `401`, `413`, `415`, `500`, `503` with `Retry-After` on queue saturation):
  ```json
  {
    "error": "service_busy",
    "detail": "Server transcription queue is full (20 requests in flight).",
    "request_id": "a1b2c3d4e5f67890"
  }
  ```

---

## Environment Variables Reference

All variables match [app/core/config.py](./app/core/config.py) and [compose.yaml](./compose.yaml).

### Model & Inference

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | *(required)* | Hugging Face Read token for the gated `NCAIR1/NigerianAccentedEnglish` repository. |
| `MODEL_ID` | `NCAIR1/NigerianAccentedEnglish` | Hugging Face Whisper model repository ID. |
| `ENGINE` | `ctranslate2` | `ctranslate2` (`faster-whisper`, recommended) or `transformers` (PyTorch fallback). |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda`. |
| `COMPUTE_TYPE` | `auto` | `auto` (`int8` on CPU, `float16` on CUDA), `int8`, `int8_float16`, `float16`, or `float32`. |
| `PRELOAD_MODEL` | `true` | Download, convert, and load the model during startup (`false` loads on first request). |
| `DELETE_HF_SNAPSHOT_AFTER_CONVERT` | `false` | Set `true` to remove the raw ~1 GB HF snapshot after CTranslate2 conversion. |
| `CACHE_DIR` | `/cache` | Persistent model cache root (`huggingface/`, `ct2/`, `torch/`). |
| `UPLOAD_TMP_DIR` | `/cache/tmp` | Disk-backed scratch directory for incoming uploads (keeps queued files out of RAM). |

### Decoding & Speed

| Variable | Default | Description |
| --- | --- | --- |
| `DEFAULT_LANGUAGE` | `en` | Skips Whisper's language-detection pass; set empty string to auto-detect. |
| `BEAM_SIZE` | `2` | Beam search width (`1` = fastest greedy, `2` = balanced, `5` = max accuracy). |
| `VAD_FILTER` | `true` | Enables Silero VAD to skip silence and avoid hallucinations. |
| `VAD_MIN_SILENCE_MS` | `500` | Minimum silence duration (ms) used by Silero VAD segmentation. |
| `ENABLE_BATCHED_INFERENCE` | `true` | Uses `BatchedInferencePipeline` for recordings `>= 30` seconds. |
| `BATCH_SIZE` | `4` | Batch size for chunked/batched inference (`4` is memory-safe for 3 GB RAM; `16` on GPU). |
| `CHUNK_LENGTH_S` | `30` | Chunk window (seconds) for the `transformers` fallback backend. |

### Docker Resources, Concurrency & Queueing

| Variable | Default | Description |
| --- | --- | --- |
| `HOST_PORT` | `8000` | Host port mapped to container port `8000`. |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | Build arg for PyTorch wheel index (`cu126` in `compose.gpu.yaml`). |
| `CPU_LIMIT` / `CPU_RESERVATION` | `2.0` / `0.5` | Docker Compose CPU limit and reservation. |
| `MEMORY_LIMIT` / `MEMORY_RESERVATION` | `3G` / `1G` | Docker Compose RAM limit and reservation (covers ~2.2 GB first-boot CT2 conversion; steady state ~600 MB idle). |
| `CPU_THREADS` | `2` | Intra-op CPU threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`). |
| `NUM_WORKERS` | `1` | Parallel CTranslate2 model replicas inside the process (keep `NUM_WORKERS * CPU_THREADS <= CPU_LIMIT`). |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | Max simultaneous active decodes + inferences in RAM. Keep `2` on 3 GB RAM so 10–20 concurrent uploads queue cleanly on disk without OOM. |
| `MAX_QUEUED_REQUESTS` | `20` | Max total in-flight + queued requests before returning HTTP `503` + `Retry-After`. |
| `QUEUE_TIMEOUT_SECONDS` | `300` | Max seconds a queued upload waits for an inference slot. |
| `MAX_UPLOAD_SIZE_MB` | `25` | Max uploaded audio file size in MB (`413` if exceeded). |

### Real-Time WebSocket Streaming & Security

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_WS_CONNECTIONS` | `16` | Max concurrent live WebSocket connections. |
| `STREAM_PARTIAL_INTERVAL_MS` | `600` | Minimum interval (ms) between live interim `partial` updates. |
| `STREAM_SILENCE_MS` | `800` | Trailing silence (ms) that finalizes a spoken segment. |
| `STREAM_MIN_SPEECH_MS` | `300` | Minimum speech duration (ms) before transcribing. |
| `STREAM_MAX_UTTERANCE_S` | `20.0` | Hard utterance cap (seconds) bounding RAM to ~640 KB per stream. |
| `STREAM_ENERGY_THRESHOLD` | `0.012` | RMS energy threshold for real-time speech onset detection. |
| `API_KEY` | *(empty)* | Optional secret enforced via `X-API-Key`, `Authorization: Bearer`, or `?api_key=`. |
| `CORS_ORIGINS` | `["*"]` | JSON array of allowed CORS origins. |
| `ENVIRONMENT` | `production` | `production` or `development`. |
| `LOG_LEVEL` | `INFO` | Standard Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FORMAT` | `json` | `json` (structured logs with `request_id`) or `text`. |

### Optional Cloudflare R2 Voice Backup & Supabase Logging

Both integrations are **optional** and execute in the background after an `O(1)` file stage inside `/cache/tmp`, adding **0 ms** to user-facing transcription responses.

1. **Supabase Table Setup**: Run [`supabase/migrations/20261126000000_create_voice_transcriptions.sql`](./supabase/migrations/20261126000000_create_voice_transcriptions.sql) in the Supabase SQL Editor (creates `public.voice_transcriptions` with `bigint identity` PK, indexes, and RLS enabled).
2. **Cloudflare R2 Setup**: Create an R2 bucket and an R2 API token with **Object Write** permissions in the Cloudflare Dashboard.

| Variable | Default | Description |
| --- | --- | --- |
| `R2_ACCOUNT_ID` | *(empty)* | Cloudflare Account ID (builds `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`). |
| `R2_ENDPOINT_URL` | *(empty)* | Optional explicit S3 endpoint URL override. |
| `R2_ACCESS_KEY_ID` | *(empty)* | Cloudflare R2 S3 Access Key ID. |
| `R2_SECRET_ACCESS_KEY` | *(empty)* | Cloudflare R2 S3 Secret Access Key. |
| `R2_BUCKET_NAME` | *(empty)* | Target R2 bucket name. |
| `R2_REGION` | `auto` | R2 region (`auto`, or jurisdictional hint such as `eu`). |
| `R2_KEY_PREFIX` | `voices` | Object key prefix (`voices/YYYY/MM/DD/<request_id>_<filename>`). |
| `R2_PUBLIC_BASE_URL` | *(empty)* | Optional custom domain / CDN base URL stored in `r2_url`. |
| `SUPABASE_URL` | *(empty)* | Supabase project URL (`https://<project_ref>.supabase.co`). |
| `SUPABASE_SECRET_KEY` | *(empty)* | Server-side Supabase Secret key (`sb_secret_...`) or `service_role` key (never expose to browsers). |
| `SUPABASE_TABLE` | `voice_transcriptions` | Target Supabase table name. |
| `ARCHIVE_CONCURRENCY` | `4` | Max concurrent background R2 uploads / Supabase upserts. |
| `ARCHIVE_MAX_QUEUE_SIZE` | `64` | Max queued background archival tasks before shedding load during cloud outages. |
| `ARCHIVE_MAX_RETRIES` | `3` | Retry attempts with exponential backoff + jitter for transient R2 / Supabase errors. |
| `ARCHIVE_WEBSOCKET_SEGMENTS` | `true` | Also encode, upload, and log finalized `WS /api/v1/stream` segments. |

---

## Sizing Profiles

| Server Profile | `CPU_LIMIT` | `MEMORY_LIMIT` | `NUM_WORKERS` | `CPU_THREADS` | `MAX_CONCURRENT_TRANSCRIPTIONS` | `MAX_QUEUED_REQUESTS` | `BATCH_SIZE` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **3 GB RAM / 2 vCPU (Default)** | `2.0` | `3G` | `1` | `2` | `2` | `20` | `4` |
| **3 GB RAM / 4 vCPU** | `4.0` | `3G` | `2` | `2` | `2`–`3` | `20` | `4` |
| **NVIDIA GPU (6 GB+ RAM)** | `4.0` | `6G` | `2` | `2` | `4` | `40` | `16` |

---

## Development & Testing

Run linting and offline unit tests (no model download required):

```bash
make lint   # ruff check app tests examples
make test   # pytest -q
```

---

## License

Code is released under the [MIT License](./LICENSE). The `NCAIR1/NigerianAccentedEnglish` model weights are downloaded at runtime from Hugging Face and remain subject to their respective model license and attribution terms (NCAIR / Awarri Technologies / Federal Ministry of Communications, Innovation and Digital Economy).
