"""Request and response Pydantic models for the TTS API."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class AudioFormat(StrEnum):
    wav = "wav"
    pcm = "pcm"   # raw int16 PCM, no header


class SpeechRequest(BaseModel):
    """
    OpenAI-compatible TTS synthesis request.
    Compatible with: POST /v1/audio/speech
    """

    model: str = Field(default="kokoro", description="TTS model identifier")
    input: str = Field(..., description="Text to synthesize", min_length=1)
    voice: str = Field(default="af_heart", description="Voice name (see GET /v1/voices)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    response_format: AudioFormat = Field(
        default=AudioFormat.wav,
        description="Output audio format. 'wav' returns a complete file; 'pcm' returns raw int16.",
    )
    stream: bool = Field(
        default=False,
        description=(
            "If true, audio is streamed as chunked transfer encoding. "
            "First audio arrives before the full text is synthesized."
        ),
    )

    @field_validator("input")
    @classmethod
    def check_length(cls, v: str) -> str:
        # Hard limit is enforced here; the per-settings soft limit is checked in the handler.
        if len(v) > 50_000:
            raise ValueError("Input text exceeds absolute maximum of 50,000 characters.")
        return v


class VoiceInfo(BaseModel):
    id: str
    language: str
    gender: str


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: str        # "healthy" | "degraded"
    backend: str
    tts_ready: bool
    cache_size: int
    cache_max_size: int
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
