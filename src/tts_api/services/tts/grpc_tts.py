"""
GrpcTTSService — async gRPC client for the tts-inference container.

Used by FastAPI directly for streaming paths (WebSocket and HTTP streaming)
when queue_enabled=True.  Bypasses the Celery+Redis intermediary entirely,
cutting the 4 Redis round-trips-per-sentence overhead to zero.

The buffered HTTP path continues to use Celery for its queue, retry, and
backpressure semantics.  This service is purely for low-latency streaming.
"""

import threading
import time

import grpc.aio

from tts_api.inference import tts_pb2, tts_pb2_grpc
from tts_api.services.audio import SAMPLE_RATE
from tts_api.services.tts.base import TTSServiceBase
from tts_api.services.tts.metrics import (
    MODEL_ERRORS,
    MODEL_LATENCY,
    REAL_TIME_FACTOR,
    classify_error,
)

_MAX_MSG = 64 * 1024 * 1024  # 64 MB


class GrpcTTSService(TTSServiceBase):
    """
    Persistent async gRPC client for the inference server.

    A single channel is shared across all concurrent requests; gRPC
    multiplexes streaming calls over it transparently.  Call start() once
    at app startup and shutdown() at shutdown.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, host: str = "localhost", port: int = 50051) -> None:
        self._target = f"{host}:{port}"
        self._channel: grpc.aio.Channel | None = None
        self._stub: tts_pb2_grpc.TTSInferenceStub | None = None

    async def start(self) -> None:
        """Open the persistent async gRPC channel.  Call once at app startup."""
        self._channel = grpc.aio.insecure_channel(
            self._target,
            options=[
                ("grpc.max_send_message_length", _MAX_MSG),
                ("grpc.max_receive_message_length", _MAX_MSG),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self._stub = tts_pb2_grpc.TTSInferenceStub(self._channel)

    async def _synthesize(self, text: str, voice: str, speed: float) -> bytes:
        response = await self._stub.Synthesize(
            tts_pb2.SynthesizeRequest(text=text, voice=voice, speed=speed),
            timeout=120,
        )
        return response.wav_bytes

    async def synthesize_streaming(
        self,
        text: str,
        voice: str,
        speed: float,
        cancel: threading.Event | None = None,
    ):
        _cancel = cancel or threading.Event()
        start = time.perf_counter()
        total_pcm_bytes = 0
        try:
            async for chunk in self._stub.SynthesizeStream(
                tts_pb2.SynthesizeRequest(text=text, voice=voice, speed=speed),
                timeout=30,
            ):
                if _cancel.is_set():
                    return
                pcm = chunk.pcm_bytes
                total_pcm_bytes += len(pcm)
                yield pcm
        except Exception as exc:
            if not _cancel.is_set():
                MODEL_ERRORS.labels(
                    voice=voice, mode="streaming", error_type=classify_error(exc)
                ).inc()
                raise
            return

        elapsed = time.perf_counter() - start
        if not _cancel.is_set():
            MODEL_LATENCY.labels(voice=voice, mode="streaming").observe(elapsed)
            if total_pcm_bytes > 0:
                audio_duration = total_pcm_bytes / (SAMPLE_RATE * 2)
                REAL_TIME_FACTOR.labels(voice=voice, mode="streaming").observe(
                    elapsed / audio_duration
                )

    async def get_voices(self) -> list[str]:
        from tts_api.services.tts.base import DEFAULT_VOICES

        return list(DEFAULT_VOICES)

    async def health_check(self) -> bool:
        if self._stub is None:
            return False
        try:
            response = await self._stub.HealthCheck(
                tts_pb2.HealthRequest(), timeout=5
            )
            return response.ready
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
