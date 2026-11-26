# Nigerian Accented English ASR API

A **fast, production-ready, open-source Speech-to-Text & Real-Time Audio Streaming API** powered by [`NCAIR1/NigerianAccentedEnglish`](https://huggingface.co/NCAIR1/NigerianAccentedEnglish) and **CTranslate2 (`faster-whisper`)**.

## Why It's Fast & Production-Ready

| Feature | Implementation |
| --- | --- |
| **4x–6x Faster Inference** | Automatically downloads only the inference weights on the server (skipping the 1.9 GB `optimizer.pt` training checkpoint), converts once into quantized **CTranslate2** (`int8` on CPU / `float16` on GPU), and runs via `faster-whisper` + Silero VAD + batched decoding. |
| **Real-Time Audio Streaming** | Bidirectional WebSocket (`WS /api/v1/stream`) with pre-roll ring buffer (zero word-onset clipping), energy VAD endpointing, coalesced interim partials, and bounded ~640 KB audio RAM per stream. |
| **Incremental SSE Streaming** | `POST /api/v1/transcribe/stream` streams decoded segments via Server-Sent Events as soon as each segment completes. |
| **OpenAI Drop-In Compatible** | `POST /v1/audio/transcriptions` supports `json`, `text`, `verbose_json`, `srt`, and `vtt` formats — works directly with the official OpenAI SDK. |
| **Server-Side Model Caching** | Zero model weights baked into the Docker image or downloaded locally. Model weights persist in a Docker volume (`model_cache:/cache`). |
| **High-Concurrency Low-RAM Uploads** | Streams uploads asynchronously via `aiofiles` to disk (`/cache/tmp`), queues up to `MAX_QUEUED_REQUESTS` (default `20`), and decodes audio **just-in-time inside the inference semaphore** — safely handling 10+ simultaneous 20 MB voice notes in 3 GB RAM. |
| **Hardened Modern Docker** | BuildKit multi-stage `uv` build, non-root UID `10001`, `read_only` rootfs, disk-backed upload scratch, `cap_drop: [ALL]`, strict CPU/RAM/PID limits, and readiness probes. |
| **Production Observability** | Structured JSON logs with `X-Request-ID`, `/health/live`, `/health/ready`, and Prometheus `/metrics` (including live Real-Time Factor). |

---

## Quick Start (Server Deployment)

### 1. Accept Model Access on Hugging Face

1. Sign in to Hugging Face and accept access terms at [https://huggingface.co/NCAIR1/NigerianAccentedEnglish](https://huggingface.co/NCAIR1/NigerianAccentedEnglish).
2. Create a Read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 2. Configure Environment

```bash
cp .env.example .env
```

Set your token and server resource limits in `.env`:

```env
HF_TOKEN=hf_your_real_token_here
CPU_LIMIT=2.0
MEMORY_LIMIT=3G
CPU_THREADS=2
API_KEY=your-strong-optional-secret
```

### 3. Build & Run with Docker Compose

```bash
# CPU server (default — lean CPU PyTorch + int8 CTranslate2)
docker compose up -d --build

# NVIDIA GPU server (CUDA + float16 CTranslate2)
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

Monitor first-boot server download & CTranslate2 conversion:

```bash
docker compose logs -f asr-api
```

Check readiness:

```bash
curl http://localhost:8000/health/ready
```

Swagger UI is available at `http://<server>:8000/docs`.

---

## API Reference

### 1. Upload Voice Recording — `POST /api/v1/transcribe`

Supports `.wav`, `.mp3`, `.m4a`, `.mp4`, `.ogg`, `.opus`, `.webm`, `.flac`, `.aac`, `.amr`, `.wma`, and `.3gp`.

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe?return_timestamps=true" \
  -H "X-API-Key: your-secret" \
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

Streams each segment immediately as it is decoded:

```bash
curl -N -X POST "http://localhost:8000/api/v1/transcribe/stream" \
  -F "file=@long_recording.mp3"
```

```text
event: segment
data: {"id":0,"start":0.0,"end":3.8,"text":"Welcome everybody to today's meeting.","words":null}

event: done
data: {"text":"Welcome everybody to today's meeting.","model":"NCAIR1/NigerianAccentedEnglish",...}
```

### 3. Real-Time Audio Streaming — `WS /api/v1/stream`

Connect to `ws://<server>:8000/api/v1/stream` (optional query params: `sample_rate=16000`, `language=en`, `interim_results=true`, `api_key=...`).

- **Send binary frames**: Raw **16-bit signed little-endian mono PCM** (`pcm_s16le`) audio chunks (recommended 100–300 ms per frame).
- **Send JSON control frames**:
  - `{"type": "config", "sample_rate": 16000, "language": "en", "interim_results": true}`
  - `{"type": "flush"}` — force finalize the active utterance
  - `{"type": "end"}` — finalize remaining audio and close cleanly
  - `{"type": "ping"}` — replies `{"type": "pong"}`
- **Receive JSON events**:
  - `ready` — session initialized
  - `partial` — live interim transcript while speaking
  - `final` — finalized transcript segment after silence or flush
  - `done` — stream summary before clean close
  - `error` — structured error event

Try the ready-made clients in [`examples/`](./examples):
- Open [`examples/live_mic.html`](./examples/live_mic.html) in any modern browser to stream your microphone live.
- Run [`examples/stream_client.py`](./examples/stream_client.py) to stream a WAV file in real time.

### 4. OpenAI-Compatible Endpoint — `POST /v1/audio/transcriptions`

Works out of the box with the official OpenAI Python/JS SDKs:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-or-unused")

with open("voice_note.m4a", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="NCAIR1/NigerianAccentedEnglish",
        file=audio,
        response_format="verbose_json",
    )

print(transcript.text)
```

Supported `response_format` values: `json`, `text`, `verbose_json`, `srt`, `vtt`.

### 5. Health & Prometheus Metrics

- `GET /health/live` — Liveness probe
- `GET /health/ready` — Readiness probe (`200` once model is loaded, `503` during initialization)
- `GET /metrics` — Prometheus text metrics (`asr_transcriptions_total`, `asr_stream_segments_total`, `asr_realtime_factor_average`, `asr_active_ws_connections`, etc.)

---

## Tuning Performance & Resource Limits

| Environment Variable | Default | Effect |
| --- | --- | --- |
| `ENGINE` | `ctranslate2` | `ctranslate2` (`faster-whisper`, recommended) or `transformers`. |
| `COMPUTE_TYPE` | `auto` | `auto` picks `int8` on CPU and `float16` on CUDA. |
| `BEAM_SIZE` | `2` | Set `1` for lowest latency (greedy), `5` for maximum accuracy. |
| `VAD_FILTER` | `true` | Skips silence via Silero VAD (huge speedup + avoids Whisper hallucinations). |
| `ENABLE_BATCHED_INFERENCE` | `true` | Uses `BatchedInferencePipeline` for files >= 30 seconds. |
| `CPU_LIMIT` / `CPU_THREADS` | `2.0` / `2` | Match `CPU_THREADS` to `CPU_LIMIT` to eliminate thread contention. |
| `MEMORY_LIMIT` | `3G` | Covers first-boot CT2 conversion (~2 GB peak); steady-state `int8` uses ~500–800 MB. |
| `DELETE_HF_SNAPSHOT_AFTER_CONVERT` | `false` | Set `true` to reclaim ~1 GB disk after one-time CT2 conversion. |

---

## License

Code is released under the [MIT License](./LICENSE). The `NCAIR1/NigerianAccentedEnglish` model weights are downloaded at runtime from Hugging Face and remain subject to their respective model license and attribution terms.
