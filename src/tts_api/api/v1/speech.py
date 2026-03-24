"""
TTS synthesis endpoints.

POST /v1/audio/speech          – synthesize (blocking or streaming)
GET  /v1/voices                – list available voices
GET  /health                   – readiness probe
GET  /metrics                  – Prometheus metrics
WS   /v1/audio/speech/ws       – real-time streaming for voice agents
"""

import json
import threading
import time

from fastapi import (  # noqa: E501
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from tts_api.api.deps import check_ws_api_key, verify_api_key
from tts_api.api.v1.schemas import (
    AudioFormat,
    HealthResponse,
    SpeechRequest,
    VoiceInfo,
    VoicesResponse,
)
from tts_api.core.logging import get_logger
from tts_api.services.audio import SAMPLE_RATE, make_streaming_wav_header
from tts_api.services.sentence import SentenceAccumulator

logger = get_logger(__name__)
router = APIRouter()       # auth-protected: /audio/speech, /voices, /audio/speech/ws
open_router = APIRouter()  # unauthenticated: /health, /metrics

# ── Prometheus metrics ────────────────────────────────────────────────────────

_REQUESTS = Counter(
    "tts_requests_total", "Total TTS synthesis requests", ["voice", "format", "status"]
)
_LATENCY = Histogram(
    "tts_request_duration_seconds",
    "End-to-end TTS request latency (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
_BYTES_OUT = Counter("tts_audio_bytes_total", "Total audio bytes sent to clients")
_CACHE_HITS = Counter("tts_cache_hits_total", "Cache hits")
_CACHE_MISSES = Counter("tts_cache_misses_total", "Cache misses")
_ACTIVE_WS = Gauge("tts_active_websocket_connections", "Active WebSocket connections")

_INPUT_LENGTH = Histogram(
    "tts_input_length_chars",
    "Distribution of TTS input text length in characters",
    buckets=(10, 50, 100, 300, 500, 1_000, 3_000, 10_000),
)
_STREAM_FIRST_CHUNK = Histogram(
    "tts_stream_first_chunk_seconds",
    "Latency to first audio chunk for streaming requests",
    ["voice"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
_STREAM_DURATION = Histogram(
    "tts_stream_duration_seconds",
    "Total duration from stream open to last audio chunk",
    ["voice"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
_WS_SESSION_DURATION = Histogram(
    "tts_websocket_session_duration_seconds",
    "Duration of individual WebSocket sessions",
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
)
_WS_TURNS = Counter(
    "tts_websocket_turns_total",
    "Completed voice-agent turns (flush events) over WebSocket",
)


# ── Dependency helpers ────────────────────────────────────────────────────────


def _tts(request: Request):
    settings = request.app.state.settings
    if settings.queue_enabled and hasattr(request.app.state, "queue_tts_service"):
        return request.app.state.queue_tts_service
    return request.app.state.tts_service


def _cache(request: Request):
    return request.app.state.audio_cache


def _settings(request: Request):
    return request.app.state.settings


def _limiter(request: Request):
    return request.app.state.concurrency_limiter


# ── REST endpoints ────────────────────────────────────────────────────────────


@router.post(
    "/audio/speech",
    summary="Synthesize speech",
    response_description="WAV or PCM audio bytes",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"content": {"audio/wav": {}, "audio/pcm": {}}},
        401: {"description": "Missing API key"},
        403: {"description": "Invalid API key"},
        413: {"description": "Input text too long"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "TTS engine error"},
    },
)
async def create_speech(body: SpeechRequest, request: Request):
    """
    Convert text to speech.  Compatible with the OpenAI audio/speech API shape.

    - **stream=false** (default): waits for full synthesis, returns a complete WAV file.
    - **stream=true**: begins streaming immediately; first audio chunk arrives after
      the *first sentence* is synthesized rather than the full input.
    """
    tts = _tts(request)
    cache = _cache(request)
    settings = _settings(request)
    limiter = _limiter(request)

    text = body.input.strip()
    voice = body.voice or settings.default_voice
    speed = body.speed

    if len(text) > settings.max_text_length:
        raise HTTPException(
            413, f"Input too long: {len(text)} chars (max {settings.max_text_length})"
        )

    _INPUT_LENGTH.observe(len(text))

    # ── Auto-upgrade to streaming for long text ───────────────────────────────
    # Blocking mode buffers all audio in RAM and blocks a worker until synthesis
    # is complete.  For inputs above the threshold we transparently switch to
    # streaming so the client gets progressive audio and the server stays
    # memory-bounded regardless of text length.
    threshold = settings.auto_stream_threshold
    auto_streamed = (
        not body.stream
        and threshold > 0
        and len(text) > threshold
    )
    if body.stream or auto_streamed:
        extra = {"X-Auto-Streamed": "true"} if auto_streamed else {}
        return _stream_response(text, voice, speed, body.response_format, tts, limiter, extra)

    # ── Non-streaming ─────────────────────────────────────────────────────────
    with _LATENCY.time():
        # Cache lookup — hits bypass the concurrency window (no synthesis work)
        cached = await cache.get(text, voice, speed)
        if cached is not None:
            _CACHE_HITS.inc()
            _REQUESTS.labels(voice=voice, format=body.response_format.value, status="cached").inc()
            _BYTES_OUT.inc(len(cached))
            payload = cached if body.response_format == AudioFormat.wav else cached[44:]
            media_type = "audio/wav" if body.response_format == AudioFormat.wav else "audio/pcm"
            return Response(
                content=payload,
                media_type=media_type,
                headers={"X-From-Cache": "true"},
            )

        _CACHE_MISSES.inc()

        # Acquire a concurrency slot — 503 if window is full
        async with limiter.acquire():
            try:
                wav_bytes = await tts.synthesize(text, voice, speed)
            except Exception as exc:
                logger.error("synthesis_failed", voice=voice, error=str(exc))
                _REQUESTS.labels(voice=voice, format=body.response_format.value, status="error").inc()
                raise HTTPException(500, "TTS synthesis failed") from exc

        await cache.set(text, voice, speed, wav_bytes)
        _REQUESTS.labels(voice=voice, format=body.response_format.value, status="ok").inc()
        _BYTES_OUT.inc(len(wav_bytes))

        if body.response_format == AudioFormat.pcm:
            return Response(
                content=wav_bytes[44:],
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Bit-Depth": "16",
                },
            )

        return Response(content=wav_bytes, media_type="audio/wav")


def _stream_response(
    text: str,
    voice: str,
    speed: float,
    fmt: AudioFormat,
    tts,
    limiter,
    extra_headers: dict | None = None,
):
    """Build a StreamingResponse that sends audio as it's synthesized."""

    async def _generate():
        # Acquire the concurrency slot BEFORE yielding any bytes.
        # If the window is full, HTTPException(503) propagates cleanly here
        # because no bytes have been sent yet — Starlette can still return a
        # proper error response.  Once we start yielding, headers are committed.
        async with limiter.acquire():
            if fmt == AudioFormat.wav:
                # WAV header with unknown data size — players treat it as "play until EOF"
                yield make_streaming_wav_header(tts.sample_rate)

            byte_count = 0
            first_chunk = True
            start = time.perf_counter()
            try:
                async for pcm_chunk in tts.synthesize_streaming(text, voice, speed):
                    if first_chunk:
                        _STREAM_FIRST_CHUNK.labels(voice=voice).observe(
                            time.perf_counter() - start
                        )
                        first_chunk = False
                    yield pcm_chunk
                    byte_count += len(pcm_chunk)
            except Exception as exc:
                logger.error("stream_error", voice=voice, error=str(exc))
                raise

            _STREAM_DURATION.labels(voice=voice).observe(time.perf_counter() - start)
            _BYTES_OUT.inc(byte_count)
            _REQUESTS.labels(voice=voice, format=f"{fmt.value}-stream", status="ok").inc()
            logger.info("stream_complete", voice=voice, bytes=byte_count)

    media_type = "audio/wav" if fmt == AudioFormat.wav else "audio/pcm"
    headers = {
        "Cache-Control": "no-cache, no-store",
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Channels": "1",
        "X-Bit-Depth": "16",
        **(extra_headers or {}),
    }
    return StreamingResponse(_generate(), media_type=media_type, headers=headers)


@router.get(
    "/voices",
    response_model=VoicesResponse,
    summary="List available voices",
    dependencies=[Depends(verify_api_key)],
)
async def list_voices(request: Request):
    """Return all voices supported by the active TTS backend."""
    tts = _tts(request)
    voice_ids = await tts.get_voices()

    # Heuristic metadata from voice ID naming convention
    def _meta(vid: str) -> VoiceInfo:
        lang = "British English" if vid.startswith("b") else "American English"
        gender = "female" if vid[1] in ("f",) else "male"
        return VoiceInfo(id=vid, language=lang, gender=gender)

    return VoicesResponse(voices=[_meta(v) for v in voice_ids])


# ── Health & metrics ──────────────────────────────────────────────────────────


@open_router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request):
    """
    Returns 200 when the service is ready to handle requests.
    Returns 503 when the TTS model has not loaded yet.
    """
    tts = _tts(request)
    cache = _cache(request)
    settings = _settings(request)

    ready = await tts.health_check()
    status = "healthy" if ready else "degraded"

    from tts_api import __version__

    resp = HealthResponse(
        status=status,
        backend=settings.backend,
        tts_ready=ready,
        cache_size=cache.size,
        cache_max_size=cache.max_size,
        version=__version__,
    )

    if not ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=resp.model_dump())

    return resp


@open_router.get("/metrics", summary="Prometheus metrics")
async def metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@router.websocket("/audio/speech/ws")
async def websocket_speech(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, alias="api_key"),
):
    """
    Real-time streaming TTS for voice agents.

    Authentication: pass ?api_key=<key> in the URL (HTTP headers are not
    available during the WebSocket upgrade handshake in most clients).

    Protocol (client → server, JSON frames):
      {"type": "config", "voice": "af_heart", "speed": 1.0}
      {"type": "text",   "text": "Hello, "}
      {"type": "text",   "text": "world!"}
      {"type": "flush"}    ← signals end of turn, flush remaining buffer

    Protocol (server → client):
      First frame: JSON {"type":"ready","sample_rate":24000,"channels":1,"bit_depth":16}
      Subsequent:  binary frames — raw int16 PCM at sample_rate Hz
      Final frame: JSON {"type":"done"}

    Design note
    ───────────
    Text chunks are accumulated in a SentenceAccumulator.  TTS is triggered on
    each complete sentence, so audio starts arriving after the *first sentence*
    rather than after the full turn — critical for low-latency voice agents.
    """
    tts = websocket.app.state.tts_service
    settings = websocket.app.state.settings

    # Auth check before accepting — reject at the handshake level
    if not check_ws_api_key(api_key, settings):
        await websocket.close(code=4001)  # 4001 = custom "unauthorized"
        return

    await websocket.accept()
    _ACTIVE_WS.inc()
    _ws_start = time.perf_counter()

    voice = settings.default_voice
    speed = settings.default_speed
    accumulator = SentenceAccumulator()
    cancel = threading.Event()

    # Send session metadata so the client knows how to decode PCM frames
    await websocket.send_json(
        {
            "type": "ready",
            "sample_rate": tts.sample_rate,
            "channels": 1,
            "bit_depth": 16,
            "format": "pcm",
        }
    )

    async def _synthesize_and_send(text: str) -> None:
        async for pcm_chunk in tts.synthesize_streaming(text, voice, speed, cancel=cancel):
            await websocket.send_bytes(pcm_chunk)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message["type"] != "websocket.receive":
                continue

            raw = message.get("text") or message.get("bytes")
            if raw is None:
                continue

            if isinstance(raw, bytes):
                raw = raw.decode()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")

            if msg_type == "config":
                voice = data.get("voice", voice)
                speed = float(data.get("speed", speed))

            elif msg_type == "text":
                chunk = data.get("text", "")
                sentences = accumulator.push(chunk)
                for sentence in sentences:
                    await _synthesize_and_send(sentence)

            elif msg_type == "flush":
                # End of LLM turn — synthesize any remaining buffered text
                for sentence in accumulator.flush():
                    await _synthesize_and_send(sentence)
                await websocket.send_json({"type": "done"})
                _WS_TURNS.inc()

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )

    except WebSocketDisconnect:
        cancel.set()
    except Exception as exc:
        cancel.set()
        logger.error("websocket_error", error=str(exc))
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        cancel.set()
        accumulator.reset()
        _WS_SESSION_DURATION.observe(time.perf_counter() - _ws_start)
        _ACTIVE_WS.dec()
