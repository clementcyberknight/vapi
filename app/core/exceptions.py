from __future__ import annotations


class ASRError(Exception):
    """Base domain exception for the ASR service."""

    code: str = "asr_error"
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotReadyError(ASRError):
    """Raised when the model is still downloading, converting, or loading."""

    code = "model_not_ready"
    status_code = 503


class ServiceBusyError(ASRError):
    """Raised when backpressure queue capacity or wait timeout is exceeded."""

    code = "service_busy"
    status_code = 503

    def __init__(self, message: str, retry_after: int = 5) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class UnsupportedAudioFormatError(ASRError):
    """Raised when an uploaded file has an unsupported media extension."""

    code = "unsupported_media_type"
    status_code = 415


class PayloadTooLargeError(ASRError):
    """Raised when an uploaded audio file exceeds the configured size cap."""

    code = "payload_too_large"
    status_code = 413


class InvalidAudioError(ASRError):
    """Raised when uploaded audio is empty, corrupted, or cannot be decoded."""

    code = "invalid_audio"
    status_code = 400


class TranscriptionFailedError(ASRError):
    """Raised when the underlying inference backend fails unexpectedly."""

    code = "transcription_failed"
    status_code = 500
