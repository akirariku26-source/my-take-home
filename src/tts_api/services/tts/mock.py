"""
Mock TTS service — returns silent audio instantly.

Used in tests and CI so we never hit the real model.
Activated via TTS_BACKEND=mock.
"""

import threading

import numpy as np

from tts_api.services.audio import SAMPLE_RATE, audio_to_wav, float32_to_pcm16
from tts_api.services.tts.base import TTSServiceBase

_CHUNK_DURATION = 0.05  # seconds of silence per chunk


class MockTTSService(TTSServiceBase):
    sample_rate = SAMPLE_RATE

    async def _synthesize(self, text: str, voice: str, speed: float) -> bytes:
        # Produce ~100 ms of silence proportional to text length
        duration = max(0.1, len(text) * 0.01)
        audio = np.zeros(int(self.sample_rate * duration), dtype=np.float32)
        return audio_to_wav(audio, self.sample_rate)

    async def synthesize_streaming(
        self, text: str, voice: str, speed: float,
        cancel: threading.Event | None = None,
    ):
        # Yield 3 chunks of silence so streaming tests can validate chunk flow
        chunk_samples = int(self.sample_rate * _CHUNK_DURATION)
        audio = np.zeros(chunk_samples, dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        for _ in range(3):
            if cancel and cancel.is_set():
                return
            yield pcm

    async def get_voices(self) -> list[str]:
        from tts_api.services.tts.base import DEFAULT_VOICES

        return list(DEFAULT_VOICES)

    async def health_check(self) -> bool:
        return True
