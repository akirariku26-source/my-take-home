"""
TTS service factory.

Public API
──────────
  create_service_bundle(settings) → ServiceBundle
      The single entry point for all service wiring.  Callers (main.py,
      conftest.py) never import concrete service classes directly.

ServiceBundle
─────────────
  .buffered   — POST /v1/audio/speech (non-streaming)
                Celery in queue mode; direct KokoroTTSService locally.
  .streaming  — POST /v1/audio/speech (stream=True) and WebSocket
                GrpcTTSService in queue mode (direct async gRPC, no Redis
                hops); same as buffered locally.

Adding a new backend
────────────────────
1. Create services/tts/mybackend.py subclassing TTSServiceBase.
2. Add a branch in _create_direct_service().
3. No other files need to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from tts_api.core.config import Settings
from tts_api.services.tts.base import TTSServiceBase


@dataclass
class ServiceBundle:
    """
    Pair of TTS services used by the API layer.

    In queue mode these are different objects:
      buffered  → CeleryTTSService (job queue, retries, backpressure)
      streaming → GrpcTTSService   (direct async gRPC, zero Redis hops)

    In local/dev mode both slots point to the same instance.
    """

    buffered: TTSServiceBase
    streaming: TTSServiceBase

    async def shutdown(self) -> None:
        await self.buffered.shutdown()
        if self.streaming is not self.buffered:
            await self.streaming.shutdown()


def create_tts_service(settings: Settings) -> TTSServiceBase:
    """Instantiate the local TTS backend (Kokoro or Mock).

    Useful in synchronous contexts (tests, CLI tools) that need a single
    backend instance without the async bundle setup.
    """
    backend = settings.backend.lower()

    if backend == "kokoro":
        from tts_api.services.tts.kokoro import KokoroTTSService

        return KokoroTTSService(
            lang_code=settings.kokoro_lang_code,
            max_workers=settings.max_workers,
        )

    if backend == "mock":
        from tts_api.services.tts.mock import MockTTSService

        return MockTTSService()

    raise ValueError(f"Unknown TTS backend: {backend!r}. Choose 'kokoro' or 'mock'.")


async def create_service_bundle(settings: Settings) -> ServiceBundle:
    """
    Build and return the ServiceBundle for the given settings.

    In queue mode, also opens the async gRPC channel so it is ready to
    accept RPCs immediately after this call returns.
    """
    if settings.queue_enabled:
        from tts_api.services.tts.celery_tts import CeleryTTSService
        from tts_api.services.tts.grpc_tts import GrpcTTSService

        buffered = CeleryTTSService(broker_url=settings.celery_broker_url)
        streaming = GrpcTTSService(
            host=settings.inference_host, port=settings.inference_port
        )
        await streaming.start()
        return ServiceBundle(buffered=buffered, streaming=streaming)

    direct = create_tts_service(settings)
    return ServiceBundle(buffered=direct, streaming=direct)
