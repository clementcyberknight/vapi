from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile, status

SAMPLE_RATE = 16_000
READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB

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


async def save_upload_to_tempfile(upload: UploadFile, max_bytes: int) -> Path:
    """Stream an uploaded audio file to temporary storage while enforcing max size."""
    original_name = upload.filename or "audio"
    suffix = Path(original_name).suffix.lower()

    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio file extension '{suffix}'. "
                f"Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    fd, tmp_path_str = tempfile.mkstemp(prefix="asr_", suffix=suffix or ".bin")
    tmp_path = Path(tmp_path_str)
    total_bytes = 0

    try:
        with os.fdopen(fd, "wb") as out_file:
            while True:
                chunk = await upload.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Audio file exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    )
                out_file.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    return tmp_path


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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ffmpeg is not installed on the server.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Timed out while decoding audio file.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to decode audio file: {stderr or 'invalid or corrupted media'}",
        ) from exc

    waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if waveform.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio samples found in the uploaded file.",
        )

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
    out_len = max(1, int(round(waveform.size * ratio)))
    x_old = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


def compute_rms_energy(waveform: np.ndarray) -> float:
    """Compute Root Mean Square energy for lightweight real-time speech gating."""
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32))))
