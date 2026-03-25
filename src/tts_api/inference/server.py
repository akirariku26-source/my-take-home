"""
gRPC inference server — entry point for the tts-inference container.

Responsibilities
────────────────
• Load KokoroTTSService (one ThreadPoolExecutor per container).
• Warm up the model before accepting RPCs so the first request pays no
  cold-start cost.
• Expose three RPCs:
    Synthesize       — buffered, returns a complete WAV file.
    SynthesizeStream — sentence-level streaming, yields PCM chunks.
    HealthCheck      — returns ready=true only after warm-up completes.

Scaling
───────
Scale horizontally by running more replicas of this container.  Each
replica owns its own model weights copy in memory.  GPU deployment:
set TTS_MAX_WORKERS=1 and attach a GPU device to the container.

Environment variables
─────────────────────
TTS_INFERENCE_PORT   gRPC listen port (default 50051)
TTS_MAX_WORKERS      Kokoro thread-pool size (default 2)
TTS_KOKORO_LANG_CODE Kokoro language code (default "a" = American English)
HF_HOME              HuggingFace model cache directory
"""

import asyncio
import os
import signal

import grpc
import grpc.aio

from tts_api.core.logging import get_logger
from tts_api.inference import tts_pb2, tts_pb2_grpc
from tts_api.services.tts.kokoro import KokoroTTSService

logger = get_logger(__name__)


class TTSInferenceServicer(tts_pb2_grpc.TTSInferenceServicer):
    def __init__(self, tts_service: KokoroTTSService) -> None:
        self._svc = tts_service

    async def Synthesize(
        self,
        request: tts_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tts_pb2.SynthesizeResponse:
        try:
            wav = await self._svc.synthesize(request.text, request.voice, request.speed)
            return tts_pb2.SynthesizeResponse(wav_bytes=wav)
        except Exception as exc:
            logger.error("Synthesize RPC failed", error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def SynthesizeStream(
        self,
        request: tts_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Stream PCM chunks sentence-by-sentence as Kokoro produces them."""
        try:
            async for pcm_chunk in self._svc.synthesize_streaming(
                request.text, request.voice, request.speed
            ):
                yield tts_pb2.AudioChunk(pcm_bytes=pcm_chunk)
        except Exception as exc:
            logger.error("SynthesizeStream RPC failed", error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def HealthCheck(
        self,
        request: tts_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> tts_pb2.HealthResponse:
        ready = await self._svc.health_check()
        return tts_pb2.HealthResponse(ready=ready)


async def serve() -> None:
    port = int(os.environ.get("TTS_INFERENCE_PORT", "50051"))
    max_workers = int(os.environ.get("TTS_MAX_WORKERS", "2"))
    lang_code = os.environ.get("TTS_KOKORO_LANG_CODE", "a")

    logger.info("Initialising Kokoro model", lang_code=lang_code, max_workers=max_workers)
    tts_service = KokoroTTSService(lang_code=lang_code, max_workers=max_workers)

    # Warm up eagerly — run one inference pass so the model weights are fully
    # loaded before the first real RPC arrives.  HealthCheck returns ready=true
    # only after this completes.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, tts_service._pipeline)
    logger.info("Model warm-up complete, ready to serve")

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),     # 64 MB
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
    )
    tts_pb2_grpc.add_TTSInferenceServicer_to_server(
        TTSInferenceServicer(tts_service), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    logger.info("gRPC inference server listening", port=port)

    # Graceful shutdown on SIGTERM / SIGINT.
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    logger.info("Shutting down gRPC inference server")
    await server.stop(grace=10)
    await tts_service.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
