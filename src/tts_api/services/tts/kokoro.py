"""
Kokoro TTS backend.

Architecture
────────────
Kokoro's KPipeline wraps a PyTorch model.  Inference is CPU-bound (or
GPU-bound), so we must not call it from the asyncio event loop directly.

Strategy:
  • A ThreadPoolExecutor with `max_workers` threads handles all inference.
  • Each thread owns its own KPipeline instance via threading.local() to avoid
    shared mutable state while still sharing underlying model weights.
  • For streaming: the worker thread pushes PCM chunks into an asyncio.Queue;
    the async generator consumes them. Queue maxsize provides back-pressure so
    a slow client can't exhaust memory.

Scaling beyond a single process
────────────────────────────────
  • Run multiple container replicas behind a load balancer (stateless design).
  • GPU: set TTS_MAX_WORKERS=1 (one GPU per container, batched internally).
  • CPU: set TTS_MAX_WORKERS to match physical core count.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tts_api.core.logging import get_logger
from tts_api.services.audio import SAMPLE_RATE, audio_to_wav, float32_to_pcm16
from tts_api.services.tts.base import TTSServiceBase
from tts_api.services.tts.metrics import MODEL_LATENCY, REAL_TIME_FACTOR

logger = get_logger(__name__)

_STREAM_QUEUE_SIZE = 16  # max buffered PCM chunks per streaming request


class KokoroTTSService(TTSServiceBase):
    sample_rate = SAMPLE_RATE

    def __init__(self, lang_code: str = "a", max_workers: int = 4) -> None:
        self._lang_code = lang_code
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tts-worker"
        )
        # Thread-local storage: each worker thread gets its own KPipeline.
        self._local = threading.local()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pipeline(self):
        """Return the thread-local KPipeline, creating it on first access."""
        if not hasattr(self._local, "pipeline"):
            from kokoro import KPipeline  # lazy import — model download happens here

            logger.info("Initialising Kokoro pipeline", thread=threading.current_thread().name)
            self._local.pipeline = KPipeline(lang_code=self._lang_code)
        return self._local.pipeline

    def _synth_sync(self, text: str, voice: str, speed: float) -> np.ndarray:
        """Run Kokoro synchronously (called inside thread pool)."""
        pipe = self._pipeline()
        chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in pipe(text, voice=voice, speed=speed):
            chunks.append(audio)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    async def _synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize the full text and return a complete WAV file."""
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(self._executor, self._synth_sync, text, voice, speed)
        return audio_to_wav(audio, self.sample_rate)

    async def synthesize_streaming(
        self, text: str, voice: str, speed: float,
        cancel: threading.Event | None = None,
    ):
        """
        Async generator — yields PCM int16 byte chunks as Kokoro processes
        each sentence, enabling sub-sentence first-audio latency.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_SIZE)
        _cancel = cancel or threading.Event()

        def _produce() -> None:
            try:
                pipe = self._pipeline()
                start = time.perf_counter()
                total_pcm_bytes = 0
                for _g, _p, audio in pipe(text, voice=voice, speed=speed):
                    if _cancel.is_set():
                        break
                    pcm = float32_to_pcm16(audio)
                    total_pcm_bytes += len(pcm)
                    # Back-pressure: block the worker thread if the consumer
                    # (client) is too slow, preventing unbounded memory use.
                    future = asyncio.run_coroutine_threadsafe(queue.put(pcm), loop)
                    try:
                        future.result(timeout=10)
                    except Exception:
                        break
                elapsed = time.perf_counter() - start
                if not _cancel.is_set():
                    MODEL_LATENCY.labels(voice=voice, mode="streaming").observe(elapsed)
                    if total_pcm_bytes > 0:
                        audio_duration = total_pcm_bytes / (SAMPLE_RATE * 2)
                        REAL_TIME_FACTOR.labels(voice=voice, mode="streaming").observe(
                            elapsed / audio_duration
                        )
            except Exception as exc:
                if not _cancel.is_set():
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result(timeout=5)
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)
                except Exception:
                    pass

        self._executor.submit(_produce)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Consumer is done (client disconnect or normal end) — signal
            # the producer thread to stop as soon as possible.
            _cancel.set()

    async def get_voices(self) -> list[str]:
        from tts_api.services.tts.base import DEFAULT_VOICES

        return list(DEFAULT_VOICES)

    async def health_check(self) -> bool:
        """Verify the model can be loaded (triggers download on cold start)."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._pipeline)
            return True
        except Exception as exc:
            logger.warning("Kokoro health check failed", error=str(exc))
            return False

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Kokoro worker pool shut down")
