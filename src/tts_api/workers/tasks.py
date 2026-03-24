"""
Celery tasks for TTS synthesis.

Worker process lifecycle
────────────────────────
Celery prefork spawns a separate OS process per concurrency slot.
_init_tts_worker() runs once per forked process (via worker_process_init
signal) and stores a KokoroTTSService in the process-global _tts_service.
This mirrors the thread-local KPipeline pattern from kokoro.py, adapted for
process-level isolation.

Tasks call _tts_service._pipeline() and _tts_service._synth_sync() directly
— the ThreadPoolExecutor inside KokoroTTSService is not used here because
each Celery worker process IS a dedicated execution slot.
"""

import os

import redis as sync_redis
from celery.signals import worker_process_init

from tts_api.workers.celery_app import broker_url, celery_app

_tts_service = None  # set by _init_tts_worker; None in the FastAPI process


@worker_process_init.connect
def _init_tts_worker(**kwargs):
    """Initialize one KokoroTTSService per Celery worker process."""
    global _tts_service
    from tts_api.services.tts.kokoro import KokoroTTSService

    lang_code = os.environ.get("TTS_KOKORO_LANG_CODE", "a")
    _tts_service = KokoroTTSService(lang_code=lang_code, max_workers=1)
    # Eagerly warm up so the first task doesn't pay the model-load cost.
    _tts_service._pipeline()


@celery_app.task(name="tts.synthesize_buffered")
def synthesize_buffered_task(text: str, voice: str, speed: float) -> str:
    """
    Synthesize text and return a complete WAV file encoded as a hex string.

    Returns hex (not raw bytes) so Celery's JSON serializer can transport it.
    The caller decodes with bytes.fromhex().
    """
    if _tts_service is None:
        raise RuntimeError("TTS service not initialized (worker_process_init not fired)")

    from tts_api.services.audio import audio_to_wav

    audio = _tts_service._synth_sync(text, voice, speed)
    wav = audio_to_wav(audio, _tts_service.sample_rate)
    return wav.hex()


@celery_app.task(name="tts.synthesize_streaming")
def synthesize_streaming_task(channel: str, text: str, voice: str, speed: float) -> None:
    """
    Synthesize text sentence-by-sentence, publishing PCM chunks to a Redis
    Pub/Sub channel as each sentence is produced.

    Publishes b"__done__" as the final message to signal completion.
    Publishes b"__error__:<message>" on failure.
    """
    if _tts_service is None:
        raise RuntimeError("TTS service not initialized (worker_process_init not fired)")

    from tts_api.services.audio import float32_to_pcm16

    r = sync_redis.Redis.from_url(broker_url)
    try:
        pipe = _tts_service._pipeline()
        for _g, _p, audio in pipe(text, voice=voice, speed=speed):
            pcm = float32_to_pcm16(audio)
            r.publish(channel, pcm)
        r.publish(channel, b"__done__")
    except Exception as exc:
        r.publish(channel, f"__error__:{exc}".encode())
    finally:
        r.close()
