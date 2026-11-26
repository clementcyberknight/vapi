from __future__ import annotations

import logging
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiofiles
import numpy as np

from app.core.exceptions import (
    InvalidAudioError,
    PayloadTooLargeError,
    UnsupportedAudioFormatError,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB

ALLOWED_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".ogg",
        ".opus",
        ".webm",
        ".flac",
        ".aac",
        ".amr",
        ".wma",
        ".3gp",
        ".mpeg",
        ".mpga",
    }
)


def validate_audio_filename(filename: str | None) -> str:
    """Validate and return the normalized file extension (or '.bin' when omitted)."""
    original_name = filename or "audio"
    suffix = Path(original_name).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio file extension '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    return suffix or ".bin"


async def stream_to_disk(
    read_chunk: Callable[[int], Awaitable[bytes]],
    *,
    filename: str | None,
    upload_dir: Path,
    max_bytes: int,
) -> Path:
    """Stream uploaded bytes asynchronously to a disk-backed temporary file.

    Writing directly to disk in 1 MiB chunks via `aiofiles` keeps RAM overhead
    near zero even when 10+ clients simultaneously upload 20 MB voice notes.
    """
    suffix = validate_audio_filename(filename)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"asr_{uuid.uuid4().hex}{suffix}"
    total_bytes = 0

    try:
        async with aiofiles.open(destination, "wb") as out_file:
            while chunk := await read_chunk(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise PayloadTooLargeError(
                        f"Audio file exceeds the {max_bytes // (1024 * 1024)} MB limit."
                    )
                await out_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise InvalidAudioError("Uploaded audio file is empty.")

    return destination


def cleanup_stale_uploads(upload_dir: Path) -> int:
    """Remove orphaned upload and archive scratch files left behind by an abrupt crash."""
    if not upload_dir.exists():
        return 0
    removed = 0
    for pattern in ("asr_*", "arch_*"):
        for path in upload_dir.glob(pattern):
            if path.is_file():
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    logger.warning("Failed to remove stale scratch file '%s'.", path)
    return removed


def decode_audio_file(file_path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Decode any ffmpeg-supported media file into a contiguous 16 kHz mono float32 array."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(file_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise InvalidAudioError("ffmpeg is not installed on the server.") from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidAudioError("Timed out while decoding audio file.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise InvalidAudioError(
            f"Unable to decode audio file: {stderr or 'invalid or corrupted media'}"
        ) from exc

    waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if waveform.size == 0:
        raise InvalidAudioError("No audio samples found in the uploaded file.")

    duration_seconds = round(float(waveform.size) / float(sample_rate), 3)
    return waveform, duration_seconds


def pcm16_bytes_to_float32(
    pcm_bytes: bytes,
    source_sample_rate: int = SAMPLE_RATE,
    target_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert raw signed 16-bit little-endian mono PCM bytes into normalized float32 [-1, 1]."""
    valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if valid_len <= 0:
        return np.empty(0, dtype=np.float32)

    samples = np.frombuffer(pcm_bytes[:valid_len], dtype="<i2").astype(np.float32) / 32768.0
    if source_sample_rate != target_sample_rate and samples.size > 1:
        samples = resample_linear(samples, source_sample_rate, target_sample_rate)
    return samples


def resample_linear(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Fast linear resampling for live PCM streams when client rate differs from 16 kHz."""
    if source_rate == target_rate or waveform.size == 0:
        return waveform
    ratio = float(target_rate) / float(source_rate)
    out_len = max(1, round(waveform.size * ratio))
    x_old = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


def compute_rms_energy(waveform: np.ndarray) -> float:
    """Compute Root Mean Square energy for lightweight real-time speech gating."""
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32))))
