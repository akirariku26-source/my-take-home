"""
Pytest configuration and shared fixtures.

All tests run with TTS_BACKEND=mock so no model is ever downloaded.
The mock service returns silent audio instantly, making the suite fast.

Note on ASGI lifespan
─────────────────────
httpx's ASGITransport does NOT trigger the ASGI lifespan protocol, so
app.state would normally be empty.  We work around this by pre-seeding
the state in the `app` fixture, which mirrors exactly what the lifespan does.
"""

import os

# Must be set BEFORE any app imports so pydantic-settings reads the right value.
os.environ.setdefault("TTS_BACKEND", "mock")
os.environ.setdefault("TTS_RATE_LIMIT_ENABLED", "false")  # disable rate limiting in tests

import pytest
from httpx import ASGITransport, AsyncClient

from tts_api.core.config import Settings, get_settings
from tts_api.main import create_app
from tts_api.services.cache import AudioCache
from tts_api.services.concurrency import AdaptiveConcurrencyLimiter
from tts_api.services.tts.factory import create_service_bundle

# Clear the lru_cache so settings re-read the env vars we just set
get_settings.cache_clear()


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def app(settings: Settings):
    """
    Create the FastAPI app and pre-seed app.state so tests don't need
    the ASGI lifespan to run.
    """
    application = create_app(settings)

    services = await create_service_bundle(settings)
    cache = AudioCache(
        max_size=settings.cache_max_size,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    application.state.services = services
    application.state.audio_cache = cache
    application.state.concurrency_limiter = AdaptiveConcurrencyLimiter(
        initial=settings.max_workers * 2,
        enabled=settings.adaptive_concurrency_enabled,
    )

    yield application

    await services.shutdown()


@pytest.fixture
async def client(app):
    """Async HTTP test client backed by the ASGI app (no lifespan)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
