from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.metrics import MetricsRegistry
from app.schemas.transcription import TranscriptSegment, WordTimestamp
from app.services.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    segments: list[TranscriptSegment]
    language: str | None
    duration_seconds: float
    processing_time_seconds: float
    rtf: float


@dataclass(slots=True)
class TranscriptionOptions:
    language: str | None = None
    beam_size: int | None = None
    vad_filter: bool | None = None
    return_timestamps: bool = False
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float = 0.0
    use_batched: bool = True


class ASREngine:
    """High-performance ASR inference manager with CTranslate2 (default) and Transformers support.

    When `engine="ctranslate2"` (default):
    - Downloads only the inference weights & tokenizer from Hugging Face on the server
      (skipping multi-GB training optimizer checkpoints such as `optimizer.pt`).
    - Performs a one-time conversion to quantized CTranslate2 format inside `cache_dir/ct2`.
    - Runs inference with `faster-whisper` (4x-6x faster, ~60% lower RAM, Silero VAD,
      lazy segment streaming, and optional batched decoding for long files).
    """

    def __init__(self, settings: Settings, metrics: MetricsRegistry | None = None) -> None:
        self.settings = settings
        self.metrics = metrics or MetricsRegistry()
        self._load_lock = threading.Lock()
        self._inference_semaphore = asyncio.Semaphore(settings.max_concurrent_transcriptions)

        self.status: str = "idle"
        self.last_error: str | None = None
        self.resolved_device: str | None = None
        self.resolved_compute_type: str | None = None

        # Backend handles
        self._fw_model: Any | None = None
        self._fw_batched: Any | None = None
        self._hf_pipeline: Any | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and (
            self._fw_model is not None or self._hf_pipeline is not None
        )

    def _detect_cuda(self) -> bool:
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return True
        except Exception:
            pass
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_device_and_compute(self) -> tuple[str, str]:
        if self.settings.device == "cuda" or (
            self.settings.device == "auto" and self._detect_cuda()
        ):
            device = "cuda"
        else:
            device = "cpu"

        if self.settings.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.settings.compute_type

        return device, compute_type

    def _ct2_model_dir(self, quantization: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.settings.model_id)
        return self.settings.ct2_cache_dir / f"{safe_name}__{quantization}"

    def _ensure_ct2_model(self, quantization: str) -> Path:
        """Download HF weights once on server and convert atomically to CTranslate2."""
        target_dir = self._ct2_model_dir(quantization)
        ready_marker = target_dir / ".ready"
        if ready_marker.exists() and (target_dir / "model.bin").exists():
            logger.info("Using cached CTranslate2 model at '%s'.", target_dir)
            return target_dir

        import ctranslate2
        from huggingface_hub import snapshot_download

        self.settings.ct2_cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading inference files for '%s' from Hugging Face (skipping training states)...",
            self.settings.model_id,
        )
        snapshot_dir = Path(
            snapshot_download(
                repo_id=self.settings.model_id,
                token=self.settings.hf_token or None,
                cache_dir=str(self.settings.hf_cache_dir),
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "pytorch_model.bin",
                    "*.safetensors",
                ],
            )
        )

        copy_candidates = [
            "tokenizer.json",
            "preprocessor_config.json",
            "vocabulary.json",
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "normalizer.json",
        ]
        copy_files = [name for name in copy_candidates if (snapshot_dir / name).exists()]

        tmp_dir = target_dir.parent / f".{target_dir.name}.tmp-{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            "Converting '%s' to CTranslate2 (quantization=%s) at '%s'...",
            self.settings.model_id,
            quantization,
            target_dir,
        )
        t0 = time.perf_counter()
        try:
            converter = ctranslate2.converters.TransformersConverter(
                str(snapshot_dir),
                copy_files=copy_files,
            )
            converter.convert(
                output_dir=str(tmp_dir),
                quantization=quantization,
                force=True,
            )
            (tmp_dir / ".ready").write_text(
                f"model={self.settings.model_id}\nquantization={quantization}\n",
                encoding="utf-8",
            )
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.move(str(tmp_dir), str(target_dir))
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        logger.info(
            "CTranslate2 conversion complete in %.2fs.",
            time.perf_counter() - t0,
        )

        if self.settings.delete_hf_snapshot_after_convert:
            logger.info("Cleaning up raw Hugging Face snapshot at '%s'...", snapshot_dir)
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        return target_dir

    def load_model(self) -> None:
        """Download/convert (if not cached) and load the configured ASR backend."""
        if self.is_ready:
            return

        with self._load_lock:
            if self.is_ready:
                return

            self.status = "loading"
            self.last_error = None
            t0 = time.perf_counter()

            try:
                device, compute_type = self._resolve_device_and_compute()
                self.resolved_device = device
                self.resolved_compute_type = compute_type

                if self.settings.engine == "ctranslate2":
                    self._load_faster_whisper(device, compute_type)
                else:
                    self._load_transformers(device)

                self.status = "ready"
                logger.info(
                    "ASR engine '%s' ready for '%s' on device=%s compute_type=%s in %.2fs.",
                    self.settings.engine,
                    self.settings.model_id,
                    device,
                    compute_type,
                    time.perf_counter() - t0,
                )
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                logger.exception("Failed to initialize ASR model '%s'", self.settings.model_id)
                raise

    def _load_faster_whisper(self, device: str, compute_type: str) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        ct2_dir = self._ensure_ct2_model(compute_type)
        self._fw_model = WhisperModel(
            str(ct2_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=self.settings.cpu_threads,
            num_workers=self.settings.num_workers,
        )
        if self.settings.enable_batched_inference:
            self._fw_batched = BatchedInferencePipeline(model=self._fw_model)

    def _load_transformers(self, device: str) -> None:
        import torch
        from transformers import pipeline

        if device == "cpu":
            torch.set_num_threads(self.settings.cpu_threads)

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.resolved_compute_type = "float16" if device == "cuda" else "float32"

        pipe_kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.settings.model_id,
            "device": "cuda:0" if device == "cuda" else "cpu",
            "dtype": dtype,
            "model_kwargs": {"low_cpu_mem_usage": True},
        }
        if self.settings.hf_token:
            pipe_kwargs["token"] = self.settings.hf_token

        self._hf_pipeline = pipeline(**pipe_kwargs)

    def _iter_segments_sync(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> tuple[Iterator[TranscriptSegment], str | None]:
        """Return a lazy iterator of decoded segments for true streaming output."""
        if not self.is_ready:
            self.load_model()

        language = options.language if options.language is not None else self.settings.default_language
        language = language or None
        beam_size = options.beam_size or self.settings.beam_size
        vad_filter = self.settings.vad_filter if options.vad_filter is None else options.vad_filter
        duration_s = float(waveform.size) / float(SAMPLE_RATE)

        if self.settings.engine == "ctranslate2":
            assert self._fw_model is not None
            use_batched = (
                options.use_batched
                and self._fw_batched is not None
                and self.settings.batch_size > 1
                and duration_s >= 30.0
            )

            fw_kwargs: dict[str, Any] = {
                "language": language,
                "task": "transcribe",
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "vad_parameters": {"min_silence_duration_ms": self.settings.vad_min_silence_ms},
                "word_timestamps": options.word_timestamps,
                "without_timestamps": not (options.return_timestamps or options.word_timestamps),
                "condition_on_previous_text": False,
                "temperature": options.temperature,
            }
            if options.initial_prompt:
                fw_kwargs["initial_prompt"] = options.initial_prompt

            if use_batched:
                assert self._fw_batched is not None
                fw_kwargs["batch_size"] = self.settings.batch_size
                raw_segments, info = self._fw_batched.transcribe(waveform, **fw_kwargs)
            else:
                raw_segments, info = self._fw_model.transcribe(waveform, **fw_kwargs)

            detected_lang = getattr(info, "language", language)

            def _gen() -> Iterator[TranscriptSegment]:
                for idx, seg in enumerate(raw_segments):
                    text = seg.text.strip()
                    if not text:
                        continue
                    words = None
                    if options.word_timestamps and getattr(seg, "words", None):
                        words = [
                            WordTimestamp(
                                start=round(float(w.start), 3),
                                end=round(float(w.end), 3),
                                word=w.word,
                                probability=round(float(w.probability), 4)
                                if getattr(w, "probability", None) is not None
                                else None,
                            )
                            for w in seg.words
                        ]
                    yield TranscriptSegment(
                        id=idx,
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        text=text,
                        words=words,
                    )

            return _gen(), detected_lang

        # Fallback: Hugging Face Transformers pipeline
        assert self._hf_pipeline is not None
        import torch

        call_kwargs: dict[str, Any] = {
            "chunk_length_s": self.settings.chunk_length_s,
            "batch_size": self.settings.batch_size,
        }
        if options.return_timestamps or options.word_timestamps:
            call_kwargs["return_timestamps"] = "word" if options.word_timestamps else True

        with torch.inference_mode():
            output = self._hf_pipeline(
                {"array": waveform, "sampling_rate": SAMPLE_RATE},
                **call_kwargs,
            )

        def _hf_gen() -> Iterator[TranscriptSegment]:
            chunks = output.get("chunks")
            if chunks:
                for idx, ch in enumerate(chunks):
                    ts = ch.get("timestamp") or (0.0, duration_s)
                    start = float(ts[0] or 0.0)
                    end = float(ts[1] if ts[1] is not None else duration_s)
                    text = str(ch.get("text", "")).strip()
                    if text:
                        yield TranscriptSegment(id=idx, start=round(start, 3), end=round(end, 3), text=text)
            else:
                text = str(output.get("text", "")).strip()
                if text:
                    yield TranscriptSegment(id=0, start=0.0, end=round(duration_s, 3), text=text)

        return _hf_gen(), language

    def _transcribe_sync(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        duration_s = round(float(waveform.size) / float(SAMPLE_RATE), 3)
        seg_iter, detected_lang = self._iter_segments_sync(waveform, options)
        segments = list(seg_iter)
        elapsed = round(time.perf_counter() - t0, 3)
        full_text = " ".join(seg.text for seg in segments).strip()
        rtf = round(elapsed / duration_s, 4) if duration_s > 0 else 0.0
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=detected_lang,
            duration_seconds=duration_s,
            processing_time_seconds=elapsed,
            rtf=rtf,
        )

    async def transcribe(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        """Run bounded async transcription off the event loop."""
        opts = options or TranscriptionOptions()
        async with self._inference_semaphore:
            self.metrics.inc_active_transcriptions()
            try:
                return await asyncio.to_thread(self._transcribe_sync, waveform, opts)
            finally:
                self.metrics.dec_active_transcriptions()

    async def stream_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream decoded segments asynchronously as soon as CTranslate2 yields each one."""
        opts = options or TranscriptionOptions(return_timestamps=True)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[TranscriptSegment | BaseException | None] = asyncio.Queue(maxsize=32)

        def _producer() -> None:
            try:
                seg_iter, _ = self._iter_segments_sync(waveform, opts)
                for seg in seg_iter:
                    asyncio.run_coroutine_threadsafe(queue.put(seg), loop).result()
            except BaseException as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        async with self._inference_semaphore:
            self.metrics.inc_active_transcriptions()
            worker = asyncio.create_task(asyncio.to_thread(_producer))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                self.metrics.dec_active_transcriptions()
