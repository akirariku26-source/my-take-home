"""
FastAPI application factory.

Lifecycle
─────────
  startup  → configure logging → create TTS service → warm-up model
           → create audio cache → attach everything to app.state
  shutdown → graceful TTS worker pool shutdown

Scalability design
──────────────────
  • Stateless HTTP layer — scale horizontally by adding replicas.
  • Worker threads for CPU-bound TTS inference (see services/tts/kokoro.py).
  • In-process LRU cache — mount a Redis adapter for cross-replica caching.
  • Token-bucket rate limiting middleware (in-process; Redis for multi-replica).
  • Prometheus metrics at /metrics for scraping by Grafana / Victoria Metrics.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from tts_api.api.v1.router import router as v1_router
from tts_api.core.config import Settings, get_settings
from tts_api.core.logging import configure_logging, get_logger
from tts_api.middleware.metrics import MetricsMiddleware
from tts_api.middleware.rate_limit import RateLimitMiddleware
from tts_api.services.cache import AudioCache
from tts_api.services.concurrency import AdaptiveConcurrencyLimiter
from tts_api.services.tts.factory import create_tts_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging()

    logger.info("tts_api_starting", backend=settings.backend, workers=settings.max_workers)

    # ── TTS service ───────────────────────────────────────────────────────────
    tts_service = create_tts_service(settings)

    # Warm up: trigger model download / thread-local init on all workers.
    # This makes the first real request fast instead of slow.
    logger.info("warming_up_tts_model")
    ready = await tts_service.health_check()
    if ready:
        logger.info("tts_model_ready")
    else:
        logger.warning("tts_model_not_ready", note="Will retry on first request")

    # ── Audio cache ───────────────────────────────────────────────────────────
    audio_cache = AudioCache(
        max_size=settings.cache_max_size,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )

    app.state.tts_service = tts_service
    app.state.audio_cache = audio_cache

    # ── Adaptive concurrency limiter ──────────────────────────────────────────
    initial = settings.adaptive_concurrency_initial or settings.max_workers * 2
    concurrency_limiter = AdaptiveConcurrencyLimiter(
        initial=initial,
        min_limit=1,
        max_limit=settings.max_workers * 8,
        target_latency_s=settings.adaptive_concurrency_target_latency_s,
        enabled=settings.adaptive_concurrency_enabled,
    )
    app.state.concurrency_limiter = concurrency_limiter

    # ── Queue TTS service (buffered HTTP when queue_enabled=True) ────────────
    if settings.queue_enabled:
        from tts_api.services.tts.celery_tts import CeleryTTSService
        from tts_api.services.tts.grpc_tts import GrpcTTSService

        queue_tts = CeleryTTSService(broker_url=settings.celery_broker_url)
        queue_ready = await queue_tts.health_check()
        if queue_ready:
            logger.info("celery_queue_ready", broker=settings.celery_broker_url)
        else:
            logger.warning("celery_queue_not_ready", broker=settings.celery_broker_url)
        app.state.queue_tts_service = queue_tts

        # Direct async gRPC for streaming paths (WebSocket + HTTP streaming).
        # Bypasses Celery+Redis, cutting 4 Redis round-trips per sentence to zero.
        grpc_tts = GrpcTTSService(
            host=settings.inference_host, port=settings.inference_port
        )
        await grpc_tts.start()
        grpc_ready = await grpc_tts.health_check()
        if grpc_ready:
            logger.info(
                "grpc_streaming_ready",
                target=f"{settings.inference_host}:{settings.inference_port}",
            )
        else:
            logger.warning(
                "grpc_streaming_not_ready",
                target=f"{settings.inference_host}:{settings.inference_port}",
            )
        app.state.grpc_tts_service = grpc_tts

    logger.info("tts_api_ready", port=settings.port)

    yield  # ── serve ──────────────────────────────────────────────────────────

    logger.info("tts_api_shutting_down")
    await tts_service.shutdown()
    if settings.queue_enabled and hasattr(app.state, "queue_tts_service"):
        await app.state.queue_tts_service.shutdown()
    if settings.queue_enabled and hasattr(app.state, "grpc_tts_service"):
        await app.state.grpc_tts_service.shutdown()
    logger.info("tts_api_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="TTS API",
        description=(
            "High-scale Text-to-Speech API.\n\n"
            "Powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) "
            "— a fast, high-quality open-source TTS model."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Attach settings before lifespan runs so lifespan can read them
    app.state.settings = settings

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # restrict in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            rpm=settings.rate_limit_rpm,
            burst=settings.rate_limit_burst,
        )

    # MetricsMiddleware must be added last so it becomes the outermost wrapper
    # and sees every response including 429s from RateLimitMiddleware.
    app.add_middleware(MetricsMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────────
    # v1 routes: /v1/audio/speech, /v1/voices, /health, /metrics, /v1/audio/speech/ws
    app.include_router(v1_router, prefix="/v1")

    # Top-level /health alias — unauthenticated, for k8s liveness / readiness probes
    from tts_api.api.v1.speech import health as _health_handler

    app.add_api_route("/health", _health_handler, methods=["GET"], tags=["ops"])

    # ── Dev console UI ────────────────────────────────────────────────────────
    _ui = Path(__file__).parents[2] / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(_ui)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
