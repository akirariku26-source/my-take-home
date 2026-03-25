"""
Celery tasks for TTS synthesis.

Worker process lifecycle
────────────────────────
Celery prefork spawns a separate OS process per concurrency slot.
_init_tts_worker() runs once per forked process (via worker_process_init
signal) and creates a persistent gRPC channel + stub to the tts-inference
container.  No model weights are loaded in this process — all inference is
delegated to the inference server over gRPC.

Task flow
─────────
  synthesize_buffered_task  →  stub.Synthesize()       →  inference server
  synthesize_streaming_task →  stub.SynthesizeStream() →  inference server
                             →  publish PCM chunks to Redis Pub/Sub channel
                             →  CeleryTTSService (FastAPI) consumes via subscribe

The Redis Pub/Sub interface on the FastAPI side (CeleryTTSService) is
unchanged — gRPC replaces the previous in-process Kokoro calls while
keeping the same message protocol on the broker.
"""

import grpc
from celery.signals import worker_process_init, worker_shutdown

from tts_api.inference import tts_pb2
from tts_api.inference.client import create_channel, create_stub
from tts_api.workers.celery_app import broker_url, celery_app

_channel: grpc.Channel | None = None
_stub = None


@worker_process_init.connect
def _init_tts_worker(**kwargs):
    """Create a gRPC channel to the inference server, one per worker process."""
    global _channel, _stub
    _channel = create_channel()
    _stub = create_stub(_channel)


@worker_shutdown.connect
def _shutdown_tts_worker(**kwargs):
    """Drain in-flight RPCs and close the channel on worker shutdown."""
    global _channel
    if _channel is not None:
        _channel.close()


@celery_app.task(name="tts.synthesize_buffered")
def synthesize_buffered_task(text: str, voice: str, speed: float) -> str:
    """
    Synthesize text via the inference server and return a complete WAV file
    encoded as a hex string.

    Returns hex (not raw bytes) so Celery's JSON serialiser can transport it.
    The caller decodes with bytes.fromhex().
    """
    if _stub is None:
        raise RuntimeError("gRPC stub not initialised (worker_process_init not fired)")

    response = _stub.Synthesize(
        tts_pb2.SynthesizeRequest(text=text, voice=voice, speed=speed),
        timeout=120,
    )
    return response.wav_bytes.hex()


@celery_app.task(name="tts.synthesize_streaming")
def synthesize_streaming_task(channel: str, text: str, voice: str, speed: float) -> None:
    """
    Stream synthesis from the inference server sentence-by-sentence, publishing
    each PCM chunk to a Redis Pub/Sub channel as it arrives.

    Publishes b"__done__" as the final sentinel to signal completion.
    Publishes b"__error__:<message>" on failure.
    """
    if _stub is None:
        raise RuntimeError("gRPC stub not initialised (worker_process_init not fired)")

    import redis as sync_redis

    r = sync_redis.Redis.from_url(broker_url)
    try:
        for chunk in _stub.SynthesizeStream(
            tts_pb2.SynthesizeRequest(text=text, voice=voice, speed=speed),
            timeout=120,
        ):
            r.publish(channel, chunk.pcm_bytes)
        r.publish(channel, b"__done__")
    except Exception as exc:
        r.publish(channel, f"__error__:{exc}".encode())
    finally:
        r.close()
