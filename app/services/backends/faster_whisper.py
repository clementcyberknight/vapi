from __future__ import annotations

import logging
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import TranscriptionFailedError
from app.schemas.transcription import TranscriptSegment, WordTimestamp
from app.services.audio import SAMPLE_RATE
from app.services.backends.base import ASRBackend, TranscriptionOptions

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """CTranslate2 (`faster-whisper`) backend with server-side one-time quantization."""

    name = "ctranslate2"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model: Any | None = None
        self._batched: Any | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and self._model is not None

    def _detect_cuda(self) -> bool:
        try:
            import ctranslate2

            return int(ctranslate2.get_cuda_device_count()) > 0
        except Exception:
            return False

    def _resolve_device_and_compute(self) -> tuple[str, str]:
        if self.settings.device == "cuda" or (self.settings.device == "auto" and self._detect_cuda()):
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
        """Download HF inference files once on the server and convert atomically to CTranslate2."""
        target_dir = self._ct2_model_dir(quantization)
        if (target_dir / ".ready").exists() and (target_dir / "model.bin").exists():
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
                allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "*.safetensors"],
            )
        )

        copy_candidates = (
            "tokenizer.json",
            "preprocessor_config.json",
            "vocabulary.json",
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "normalizer.json",
        )
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
            converter.convert(output_dir=str(tmp_dir), quantization=quantization, force=True)
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

        logger.info("CTranslate2 conversion complete in %.2fs.", time.perf_counter() - t0)

        if self.settings.delete_hf_snapshot_after_convert:
            logger.info("Removing raw HF snapshot at '%s' to reclaim disk space.", snapshot_dir)
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        return target_dir

    def load(self) -> None:
        if self.is_ready:
            return

        with self._load_lock:
            if self.is_ready:
                return

            self.status = "loading"
            self.last_error = None
            t0 = time.perf_counter()

            try:
                from faster_whisper import BatchedInferencePipeline, WhisperModel

                device, compute_type = self._resolve_device_and_compute()
                self.resolved_device = device
                self.resolved_compute_type = compute_type

                ct2_dir = self._ensure_ct2_model(compute_type)
                self._model = WhisperModel(
                    str(ct2_dir),
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self.settings.cpu_threads,
                    num_workers=self.settings.num_workers,
                )
                if self.settings.enable_batched_inference:
                    self._batched = BatchedInferencePipeline(model=self._model)

                self.status = "ready"
                logger.info(
                    "FasterWhisperBackend ready for '%s' "
                    "(device=%s, compute_type=%s, workers=%d, threads=%d) in %.2fs.",
                    self.settings.model_id,
                    device,
                    compute_type,
                    self.settings.num_workers,
                    self.settings.cpu_threads,
                    time.perf_counter() - t0,
                )
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                logger.exception("Failed to load FasterWhisperBackend for '%s'", self.settings.model_id)
                raise

    def iter_segments(
        self,
        waveform: np.ndarray,
        options: TranscriptionOptions,
    ) -> tuple[Iterator[TranscriptSegment], str | None]:
        if not self.is_ready:
            self.load()

        model = self._model
        if model is None:
            raise TranscriptionFailedError("FasterWhisper model is not loaded.")

        language = options.language if options.language is not None else self.settings.default_language
        language = language or None
        beam_size = options.beam_size or self.settings.beam_size
        vad_filter = self.settings.vad_filter if options.vad_filter is None else options.vad_filter
        duration_s = float(waveform.size) / float(SAMPLE_RATE)

        use_batched = (
            options.use_batched
            and self._batched is not None
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

        if use_batched and self._batched is not None:
            fw_kwargs["batch_size"] = self.settings.batch_size
            raw_segments, info = self._batched.transcribe(waveform, **fw_kwargs)
        else:
            raw_segments, info = model.transcribe(waveform, **fw_kwargs)

        detected_lang = getattr(info, "language", language)

        def _segment_generator() -> Iterator[TranscriptSegment]:
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

        return _segment_generator(), detected_lang
