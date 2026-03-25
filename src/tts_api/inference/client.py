"""
gRPC client for the tts-inference container.

Usage (Celery worker)
─────────────────────
Create a channel and stub once per worker process in worker_process_init,
then reuse for all tasks.  gRPC channels are thread-safe and handle
transparent reconnection internally.

    channel = create_channel()
    stub    = create_stub(channel)

    # buffered synthesis
    response = stub.Synthesize(SynthesizeRequest(...), timeout=120)
    wav_bytes = response.wav_bytes

    # streaming synthesis
    for chunk in stub.SynthesizeStream(SynthesizeRequest(...), timeout=120):
        pcm_bytes = chunk.pcm_bytes

Environment variables
─────────────────────
TTS_INFERENCE_HOST  hostname of the inference container (default "localhost")
TTS_INFERENCE_PORT  gRPC port (default 50051)
"""

import os

import grpc

from tts_api.inference import tts_pb2, tts_pb2_grpc  # noqa: F401 — re-exported for callers

_MAX_MSG = 64 * 1024 * 1024  # 64 MB — WAV files for long texts can be large


def create_channel() -> grpc.Channel:
    """Create a persistent gRPC channel to the inference server.

    Call once per process; the channel manages its own connection pool and
    handles reconnection on transient failures.
    """
    host = os.environ.get("TTS_INFERENCE_HOST", "localhost")
    port = int(os.environ.get("TTS_INFERENCE_PORT", "50051"))
    target = f"{host}:{port}"
    return grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
            # Keep-alive: detect dead connections and re-establish promptly.
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )


def create_stub(channel: grpc.Channel) -> tts_pb2_grpc.TTSInferenceStub:
    return tts_pb2_grpc.TTSInferenceStub(channel)
