"""
Abstract TTS service interface.

To add a new backend
────────────────────
1. Create a new file, e.g. services/tts/mytts.py
2. Subclass TTSServiceBase and implement the three abstract methods:

       class MyTTSService(TTSServiceBase):
           sample_rate = 24_000           # Hz of audio produced

           async def _synthesize(self, text, voice, speed) -> bytes:
               # Return a complete WAV file.
               # CPU-bound work should be offloaded via run_in_executor.
               ...

           async def get_voices(self) -> list[str]:
               return ["voice_a", "voice_b"]

           async def health_check(self) -> bool:
               return True

3. Optionally override synthesize_streaming() for sentence-level streaming.
   The default yields the full PCM in one shot (correct, just not low-latency).
   For streaming, record MODEL_LATENCY yourself:

       from tts_api.services.tts.metrics import MODEL_LATENCY

4. Register the backend in services/tts/factory.py.

Buffered inference latency (tts_model_inference_seconds{mode="buffered"}) is
recorded automatically by the synthesize() wrapper below — no action needed.
Streaming latency must be timed by the subclass inside its own generator.
"""

import threading
import time
from abc import ABC, abstractmethod

from tts_api.services.audio import SAMPLE_RATE
from tts_api.services.tts.metrics import MODEL_LATENCY, REAL_TIME_FACTOR

# Canonical voice list shared by all backends.  When adding a new voice to
# Kokoro update this list — every backend (including Mock and Celery) reads
# it so callers always see a consistent catalogue.
DEFAULT_VOICES: list[str] = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]


class TTSServiceBase(ABC):
    # Sample rate of audio produced by this backend.
    # Subclasses must override this if they differ from the default.
    sample_rate: int = SAMPLE_RATE

    # ── Public API ────────────────────────────────────────────────────────────

    async def synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """
        Return a complete WAV file as bytes.

        Times the call and records tts_model_inference_seconds{mode="buffered"}
        automatically — subclasses implement _synthesize(), not this method.
        """
        start = time.perf_counter()
        wav = await self._synthesize(text, voice, speed)
        elapsed = time.perf_counter() - start
        MODEL_LATENCY.labels(voice=voice, mode="buffered").observe(elapsed)
        # RTF = inference_time / audio_duration.  WAV has a 44-byte header;
        # remaining bytes are int16 PCM (2 bytes per sample, mono).
        pcm_bytes = len(wav) - 44
        if pcm_bytes > 0:
            audio_duration = pcm_bytes / (self.sample_rate * 2)
            REAL_TIME_FACTOR.labels(voice=voice, mode="buffered").observe(
                elapsed / audio_duration
            )
        return wav

    async def synthesize_streaming(
        self, text: str, voice: str, speed: float,
        cancel: threading.Event | None = None,
    ):
        """
        Async generator that yields raw int16 PCM byte chunks as they are
        synthesized.

        Default implementation calls synthesize() and yields the PCM portion
        in one shot.  Override for true sentence-level streaming.
        When overriding, record MODEL_LATENCY{mode="streaming"} yourself.

        *cancel*: if set by the caller, the generator should stop producing
        new chunks as soon as possible.  Backends should check cancel.is_set()
        between sentences to release the worker thread promptly.
        """
        from tts_api.services.audio import wav_to_pcm

        wav = await self.synthesize(text, voice, speed)
        if cancel and cancel.is_set():
            return
        yield wav_to_pcm(wav)

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    async def _synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """Return a complete WAV file as bytes. Called by synthesize()."""
        ...

    @abstractmethod
    async def get_voices(self) -> list[str]:
        """Return sorted list of available voice identifiers."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is ready to accept requests."""
        ...

    async def shutdown(self) -> None:  # noqa: B027 — intentional no-op default
        """Release resources (thread pools, etc.).  Override if needed."""
