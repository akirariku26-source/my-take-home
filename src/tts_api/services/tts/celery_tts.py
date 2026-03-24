"""
CeleryTTSService — TTSServiceBase implementation that dispatches synthesis
to Celery workers via a Redis broker.

Used only in the FastAPI process (tasks are submitted here, never executed).
The WebSocket handler continues to use KokoroTTSService directly.

Buffered path
─────────────
  submit synthesize_buffered_task.delay()
  → await AsyncResult via run_in_executor (non-blocking)
  → decode hex → return WAV bytes

Streaming path
──────────────
  generate unique Pub/Sub channel
  → submit synthesize_streaming_task.delay()
  → subscribe to channel via async Redis client
  → yield PCM chunks as they arrive; stop on b"__done__"
"""

import asyncio
from uuid import uuid4

import redis.asyncio as aioredis

from tts_api.services.audio import SAMPLE_RATE
from tts_api.services.tts.base import TTSServiceBase
from tts_api.workers.tasks import synthesize_buffered_task, synthesize_streaming_task


class CeleryTTSService(TTSServiceBase):
    sample_rate = SAMPLE_RATE

    def __init__(self, broker_url: str) -> None:
        self._broker_url = broker_url

    # ── TTSServiceBase implementation ─────────────────────────────────────────

    async def _synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """Submit a buffered synthesis task and await the result."""
        task_result = synthesize_buffered_task.delay(text, voice, speed)
        loop = asyncio.get_running_loop()
        # run_in_executor offloads the blocking Celery .get() call so the
        # asyncio event loop remains free to handle other requests.
        hex_wav: str = await loop.run_in_executor(
            None, lambda: task_result.get(timeout=120)
        )
        return bytes.fromhex(hex_wav)

    async def synthesize_streaming(self, text: str, voice: str, speed: float):
        """
        Submit a streaming synthesis task and yield PCM chunks via Redis
        Pub/Sub as the worker produces them sentence by sentence.
        """
        channel = f"tts:stream:{uuid4().hex}"
        synthesize_streaming_task.delay(channel, text, voice, speed)

        async with aioredis.from_url(self._broker_url) as r:
            pubsub = r.pubsub()
            await pubsub.subscribe(channel)
            try:
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    data: bytes = message["data"]
                    if data == b"__done__":
                        break
                    if data.startswith(b"__error__:"):
                        error_msg = data[len(b"__error__:"):].decode()
                        raise RuntimeError(f"Worker synthesis failed: {error_msg}")
                    yield data
            finally:
                await pubsub.unsubscribe(channel)

    async def get_voices(self) -> list[str]:
        from tts_api.services.tts.base import DEFAULT_VOICES

        return list(DEFAULT_VOICES)

    async def health_check(self) -> bool:
        """Ping Redis to verify broker connectivity."""
        try:
            async with aioredis.from_url(self._broker_url) as r:
                await r.ping()
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        pass  # connections are opened/closed per-request; nothing to tear down
