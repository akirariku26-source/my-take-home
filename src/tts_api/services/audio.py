"""
Audio encoding utilities.

All TTS backends produce float32 numpy arrays at SAMPLE_RATE Hz.
These helpers convert them to byte streams suitable for HTTP responses.
"""

import struct

import numpy as np

SAMPLE_RATE = 24_000  # Hz — Kokoro native sample rate
CHANNELS = 1
BIT_DEPTH = 16  # int16 PCM


# ── PCM helpers ───────────────────────────────────────────────────────────────


def float32_to_pcm16(audio) -> bytes:
    """Convert a float32 [-1, 1] numpy array or torch Tensor to signed 16-bit PCM bytes."""
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32_767).astype(np.int16).tobytes()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes back to a float32 numpy array."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_767.0


# ── WAV encoding ──────────────────────────────────────────────────────────────


def _wav_header(pcm_size: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Build a standard 44-byte RIFF/WAV header."""
    byte_rate = sample_rate * CHANNELS * BIT_DEPTH // 8
    block_align = CHANNELS * BIT_DEPTH // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + pcm_size,      # total file size - 8
        b"WAVE",
        b"fmt ",
        16,                  # fmt chunk size
        1,                   # PCM = 1
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        BIT_DEPTH,
        b"data",
        pcm_size,
    )


def audio_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 audio array to a complete WAV file (bytes)."""
    pcm = float32_to_pcm16(audio)
    return _wav_header(len(pcm), sample_rate) + pcm


def make_streaming_wav_header(sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    WAV header suitable for streaming: data-chunk size set to 0xFFFFFFFF
    (unknown / max).  Most decoders treat this as "play until EOF".
    """
    return _wav_header(0xFFFF_FFFF - 36, sample_rate)


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    """Strip the 44-byte WAV header and return raw PCM bytes."""
    return wav_bytes[44:]
