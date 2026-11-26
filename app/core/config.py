from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service metadata
    app_name: str = Field(default="Nigerian Accented English ASR API")
    app_version: str = Field(default="2.1.0")
    environment: Literal["development", "production"] = Field(default="production")

    # Model & Inference Backend
    model_id: str = Field(
        default="NCAIR1/NigerianAccentedEnglish",
        description="Hugging Face model repository ID",
    )
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face access token (required for gated repositories)",
    )
    engine: Literal["ctranslate2", "transformers"] = Field(
        default="ctranslate2",
        description="Inference backend: ctranslate2 (faster-whisper, 4-6x faster) or transformers",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Target execution device",
    )
    compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = Field(
        default="auto",
        description="Quantization/compute precision (auto -> int8 on CPU, float16 on CUDA)",
    )
    cache_dir: Path = Field(
        default=Path("/cache"),
        description="Persistent directory for Hugging Face & CTranslate2 model cache",
    )
    upload_tmp_dir: Path = Field(
        default=Path("/cache/tmp"),
        description="Disk-backed directory for streaming uploaded audio files (avoids RAM pressure)",
    )
    delete_hf_snapshot_after_convert: bool = Field(
        default=False,
        description="Delete raw HF weights after CTranslate2 conversion to save disk space",
    )
    preload_model: bool = Field(
        default=True,
        description="Download, convert, and load the model during server startup",
    )

    # Decoding & Speed Tuning
    default_language: str | None = Field(
        default="en",
        description="Default language code (setting 'en' skips language-detection overhead)",
    )
    beam_size: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Beam search width for final transcriptions (1=fastest greedy, 5=max accuracy)",
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Silero VAD to strip silence and speed up transcription",
    )
    vad_min_silence_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Minimum silence duration (ms) for VAD segmentation",
    )
    enable_batched_inference: bool = Field(
        default=True,
        description="Use faster-whisper BatchedInferencePipeline for long audio files",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Batch size for batched/chunked inference",
    )
    chunk_length_s: int = Field(
        default=30,
        ge=5,
        le=30,
        description="Chunk size in seconds for long audio",
    )

    # Concurrency, Queuing & Resource Guardrails
    cpu_threads: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Intra-op CPU threads (match your Docker CPU_LIMIT)",
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="CTranslate2 parallel worker replicas inside the process",
    )
    max_concurrent_transcriptions: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Maximum simultaneous active audio decodes & model inferences in RAM",
    )
    max_queued_requests: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum total in-flight + queued transcription requests before HTTP 503 backpressure",
    )
    queue_timeout_seconds: float = Field(
        default=300.0,
        ge=5.0,
        le=3600.0,
        description="Maximum seconds a queued upload waits for an inference slot",
    )
    max_upload_size_mb: int = Field(
        default=25,
        ge=1,
        le=250,
        description="Maximum uploaded audio file size in MB",
    )

    # Real-Time WebSocket Streaming Settings
    max_ws_connections: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Maximum concurrent live WebSocket streaming sessions",
    )
    stream_partial_interval_ms: int = Field(
        default=600,
        ge=150,
        le=5000,
        description="Minimum interval between interim partial transcripts during live speech",
    )
    stream_silence_ms: int = Field(
        default=800,
        ge=250,
        le=5000,
        description="Trailing silence duration (ms) that triggers segment finalization",
    )
    stream_min_speech_ms: int = Field(
        default=300,
        ge=100,
        le=3000,
        description="Minimum speech duration (ms) before emitting a transcript",
    )
    stream_max_utterance_s: float = Field(
        default=20.0,
        ge=5.0,
        le=60.0,
        description="Maximum buffered speech duration (s) before forcing finalization",
    )
    stream_energy_threshold: float = Field(
        default=0.012,
        ge=0.001,
        le=0.5,
        description="RMS energy threshold for real-time speech activity detection",
    )

    # Security, CORS, Logging
    api_key: str | None = Field(
        default=None,
        description="Optional API key enforced on transcription & streaming endpoints",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Structured JSON logs for production or readable text logs for dev",
    )

    # Optional Cloudflare R2 Voice Backup (S3-compatible API)
    r2_account_id: str | None = Field(
        default=None,
        description="Cloudflare account ID (used to build https://<ACCOUNT_ID>.r2.cloudflarestorage.com)",
    )
    r2_endpoint_url: str | None = Field(
        default=None,
        description="Optional explicit R2 S3 endpoint URL (overrides r2_account_id when set)",
    )
    r2_access_key_id: str | None = Field(
        default=None,
        description="Cloudflare R2 S3 API Access Key ID",
    )
    r2_secret_access_key: str | None = Field(
        default=None,
        description="Cloudflare R2 S3 API Secret Access Key",
    )
    r2_bucket_name: str | None = Field(
        default=None,
        description="Target Cloudflare R2 bucket name for voice backups",
    )
    r2_region: str = Field(
        default="auto",
        description="Cloudflare R2 region (must be 'auto' unless using jurisdictional buckets like 'eu')",
    )
    r2_key_prefix: str = Field(
        default="voices",
        description="Object key prefix inside the R2 bucket",
    )
    r2_public_base_url: str | None = Field(
        default=None,
        description="Optional custom public/CDN base URL for stored R2 objects",
    )

    # Optional Supabase Database Audit Logging
    supabase_url: str | None = Field(
        default=None,
        description="Supabase project URL (e.g. https://<project_ref>.supabase.co)",
    )
    supabase_secret_key: str | None = Field(
        default=None,
        description="Supabase server-side Secret key (sb_secret_...) or service_role key",
    )
    supabase_table: str = Field(
        default="voice_transcriptions",
        description="Target Postgres table in Supabase for logging transcriptions",
    )

    # Background Archival Controls
    archive_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max concurrent background R2 uploads / Supabase upserts",
    )
    archive_max_queue_size: int = Field(
        default=64,
        ge=4,
        le=1000,
        description="Max pending background archival tasks before shedding load during cloud outages",
    )
    archive_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts for transient R2 / Supabase network or 5xx errors",
    )
    archive_websocket_segments: bool = Field(
        default=True,
        description="Also back up and log finalized real-time WebSocket audio segments",
    )

    @property
    def resolved_r2_endpoint(self) -> str | None:
        if self.r2_endpoint_url:
            return self.r2_endpoint_url.rstrip("/")
        if self.r2_account_id:
            return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
        return None

    @property
    def r2_enabled(self) -> bool:
        return bool(
            self.resolved_r2_endpoint
            and self.r2_access_key_id
            and self.r2_secret_access_key
            and self.r2_bucket_name
        )

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key and self.supabase_table)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def ct2_cache_dir(self) -> Path:
        return self.cache_dir / "ct2"

    @property
    def hf_cache_dir(self) -> Path:
        return self.cache_dir / "huggingface"


@lru_cache
def get_settings() -> Settings:
    return Settings()
