"""
Tests for API key authentication.

Three scenarios are exercised:
  1. Auth disabled (TTS_API_KEYS empty)   — all routes open
  2. Auth enabled, valid key presented    — request succeeds
  3. Auth enabled, missing/wrong key      — 401 / 403 returned
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure mock backend is used (may already be set from conftest.py)
os.environ.setdefault("TTS_BACKEND", "mock")
os.environ.setdefault("TTS_RATE_LIMIT_ENABLED", "false")

from tts_api.core.config import Settings, get_settings  # noqa: E402
from tts_api.main import create_app  # noqa: E402
from tts_api.services.cache import AudioCache  # noqa: E402
from tts_api.services.tts.factory import create_tts_service  # noqa: E402

_VALID_KEY = "sk-test-valid-key"
_OTHER_KEY = "sk-test-other-key"


def _make_app(api_keys: str = ""):
    """Create an app instance with the given api_keys setting."""
    get_settings.cache_clear()
    os.environ["TTS_API_KEYS"] = api_keys
    try:
        settings = Settings()
        app = create_app(settings)
        app.state.tts_service = create_tts_service(settings)
        app.state.audio_cache = AudioCache(max_size=100, enabled=True)
        return app
    finally:
        # Restore clean state for other tests
        os.environ.pop("TTS_API_KEYS", None)
        get_settings.cache_clear()


# ── Auth disabled ─────────────────────────────────────────────────────────────


class TestAuthDisabled:
    """When TTS_API_KEYS is empty, no key is required."""

    async def test_speech_no_key_succeeds(self):
        app = _make_app(api_keys="")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/v1/audio/speech", json={"input": "hello"})
        assert r.status_code == 200

    async def test_health_no_key_succeeds(self):
        app = _make_app(api_keys="")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/v1/health")
        assert r.status_code == 200

    async def test_voices_no_key_succeeds(self):
        app = _make_app(api_keys="")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/v1/voices")
        assert r.status_code == 200


# ── Auth enabled — valid key ──────────────────────────────────────────────────


class TestAuthEnabledValidKey:
    async def test_bearer_token_accepted(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/v1/audio/speech",
                json={"input": "authenticated"},
                headers={"Authorization": f"Bearer {_VALID_KEY}"},
            )
        assert r.status_code == 200

    async def test_x_api_key_header_accepted(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/v1/audio/speech",
                json={"input": "authenticated"},
                headers={"X-API-Key": _VALID_KEY},
            )
        assert r.status_code == 200

    async def test_multiple_keys_all_valid(self):
        """Any key in the comma-separated list is accepted."""
        app = _make_app(api_keys=f"{_VALID_KEY},{_OTHER_KEY}")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r1 = await c.post(
                "/v1/audio/speech",
                json={"input": "key1"},
                headers={"Authorization": f"Bearer {_VALID_KEY}"},
            )
            r2 = await c.post(
                "/v1/audio/speech",
                json={"input": "key2"},
                headers={"Authorization": f"Bearer {_OTHER_KEY}"},
            )
        assert r1.status_code == 200
        assert r2.status_code == 200

    async def test_health_needs_no_key_when_auth_enabled(self):
        """/health is always open regardless of auth config."""
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/v1/health")
        assert r.status_code == 200

    async def test_metrics_needs_no_key_when_auth_enabled(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/v1/metrics")
        assert r.status_code == 200


# ── Auth enabled — missing / wrong key ────────────────────────────────────────


class TestAuthEnabledRejected:
    async def test_missing_key_returns_401(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/v1/audio/speech", json={"input": "no key"})
        assert r.status_code == 401

    async def test_wrong_bearer_returns_403(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/v1/audio/speech",
                json={"input": "bad key"},
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert r.status_code == 403

    async def test_wrong_x_api_key_returns_403(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/v1/audio/speech",
                json={"input": "bad key"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert r.status_code == 403

    async def test_voices_requires_key(self):
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/v1/voices")
        assert r.status_code == 401

    async def test_401_has_www_authenticate_header(self):
        """Standard 401 must include WWW-Authenticate for OAuth2 compliance."""
        app = _make_app(api_keys=_VALID_KEY)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/v1/audio/speech", json={"input": "no key"})
        assert r.status_code == 401
        assert "www-authenticate" in r.headers


# ── WebSocket auth ─────────────────────────────────────────────────────────────


class TestWebSocketAuth:
    def test_ws_rejected_without_key(self):
        app = _make_app(api_keys=_VALID_KEY)
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        with TestClient(app) as tc:
            with pytest.raises((WebSocketDisconnect, Exception)):
                with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                    ws.receive_json()  # should not get here

    def test_ws_accepted_with_valid_key(self):
        app = _make_app(api_keys=_VALID_KEY)
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect(f"/v1/audio/speech/ws?api_key={_VALID_KEY}") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready"

    def test_ws_open_without_key_when_auth_disabled(self):
        app = _make_app(api_keys="")
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready"
