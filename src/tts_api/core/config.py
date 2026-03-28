from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All vars are prefixed with TTS_ (e.g., TTS_BACKEND, TTS_MAX_WORKERS).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TTS_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── TTS engine ────────────────────────────────────────────────────────────
    # "kokoro" uses the local Kokoro-82M model via PyTorch.
    # "mock"   returns silent audio — useful for tests and CI.
    backend: str = "kokoro"

    # Kokoro language code: "a" = American English, "b" = British English
    kokoro_lang_code: str = "a"

    default_voice: str = "af_heart"
    default_speed: float = 1.0
    max_text_length: int = 10_000

    # ── Auto-streaming threshold ───────────────────────────────────────────
    # Requests with stream=False whose text exceeds this character count are
    # automatically upgraded to streaming.  This prevents large in-memory
    # audio buffers and client-side HTTP timeouts on long inputs.
    # Set to 0 to disable (always respect the client's stream flag).
    auto_stream_threshold: int = 300

    # ── Worker pool ───────────────────────────────────────────────────────────
    # Number of threads dedicated to TTS inference (CPU-bound work).
    # Rule of thumb: match physical CPU cores or set to 2x for IO-heavy workloads.
    max_workers: int = 4

    # ── gRPC inference server ─────────────────────────────────────────────────
    # Host/port of the tts-inference container.  Used by GrpcTTSService when
    # queue_enabled=True for zero-overhead streaming paths (WebSocket and HTTP
    # streaming), bypassing Celery+Redis entirely.
    inference_host: str = "localhost"
    inference_port: int = 50051

    # ── Job queue (Celery + Redis) ─────────────────────────────────────────────
    # Set TTS_QUEUE_ENABLED=true to route synthesis through Celery workers.
    # Both HTTP and WebSocket paths respect this flag.
    queue_enabled: bool = False
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend_url: str = "redis://localhost:6379/0"

    # ── Adaptive concurrency control ──────────────────────────────────────────
    # AIMD window that limits simultaneous in-flight synthesis requests.
    # Shrinks when observed latency exceeds the target; grows when healthy.
    # Excess requests receive an immediate 503 (fail-fast, not queue-and-wait).
    adaptive_concurrency_enabled: bool = True
    # Window size at startup.  0 = auto: max_workers * 2.
    adaptive_concurrency_initial: int = 0
    # Target EWMA latency (seconds).  Window shrinks when exceeded.
    adaptive_concurrency_target_latency_s: float = 10.0

    # ── Audio cache ───────────────────────────────────────────────────────────
    # LRU cache keyed by (text, voice, speed).
    # Greatly reduces latency for repeated requests (e.g. common phrases).
    cache_enabled: bool = True
    cache_max_size: int = 1_000   # entries
    cache_ttl_seconds: int = 3_600  # 1 hour

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Token-bucket per client IP. Uses in-process state — add Redis for
    # distributed rate limiting when running multiple replicas.
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60    # sustained requests/minute per client
    rate_limit_burst: int = 10  # burst capacity (tokens)
    # Only honour X-Forwarded-For when the service sits behind a trusted reverse
    # proxy that overwrites the header.  Leave False for direct exposure.
    rate_limit_trust_proxy: bool = False
    # Maximum number of distinct client IDs tracked in-memory.  Oldest-seen
    # entries are evicted when the limit is reached (LRU semantics).
    rate_limit_max_clients: int = 10_000

    # ── Authentication ────────────────────────────────────────────────────────
    # Comma-separated list of valid API keys.
    # If empty (default), authentication is disabled — useful for local dev.
    # Example: TTS_API_KEYS=sk-abc123,sk-def456
    #
    # Stored as a plain string to avoid pydantic-settings JSON-parsing list[str]
    # before our validator can handle comma-separated input.
    api_keys: str = ""

    @property
    def api_key_set(self) -> frozenset[str]:
        """Parsed set of valid API keys (empty = auth disabled)."""
        if not self.api_keys:
            return frozenset()
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
