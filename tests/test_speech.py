"""
Tests for POST /v1/audio/speech and related REST endpoints.
"""

import struct

# ── Helper ────────────────────────────────────────────────────────────────────


def is_valid_wav(data: bytes) -> bool:
    """Basic WAV header validation."""
    if len(data) < 44:
        return False
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


# ── Speech synthesis ──────────────────────────────────────────────────────────


class TestCreateSpeech:
    async def test_returns_wav_bytes(self, client):
        resp = await client.post("/v1/audio/speech", json={"input": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert is_valid_wav(resp.content)

    async def test_wav_header_correct_structure(self, client):
        resp = await client.post("/v1/audio/speech", json={"input": "Test"})
        data = resp.content
        # RIFF marker
        assert data[:4] == b"RIFF"
        # WAVE marker
        assert data[8:12] == b"WAVE"
        # fmt chunk
        assert data[12:16] == b"fmt "
        # PCM format = 1
        audio_fmt = struct.unpack_from("<H", data, 20)[0]
        assert audio_fmt == 1
        # Channels = 1 (mono)
        channels = struct.unpack_from("<H", data, 22)[0]
        assert channels == 1

    async def test_pcm_format_returns_raw_bytes(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        # PCM should NOT start with RIFF
        assert resp.content[:4] != b"RIFF"
        # PCM headers present
        assert "x-sample-rate" in resp.headers

    async def test_custom_voice_accepted(self, client):
        resp = await client.post(
            "/v1/audio/speech", json={"input": "Hi", "voice": "am_adam"}
        )
        assert resp.status_code == 200

    async def test_speed_in_range_accepted(self, client):
        resp = await client.post(
            "/v1/audio/speech", json={"input": "Fast", "speed": 2.0}
        )
        assert resp.status_code == 200

    async def test_speed_out_of_range_returns_422(self, client):
        resp = await client.post(
            "/v1/audio/speech", json={"input": "Fast", "speed": 10.0}
        )
        assert resp.status_code == 422

    async def test_empty_input_returns_422(self, client):
        resp = await client.post("/v1/audio/speech", json={"input": ""})
        assert resp.status_code == 422

    async def test_missing_input_returns_422(self, client):
        resp = await client.post("/v1/audio/speech", json={"voice": "af_heart"})
        assert resp.status_code == 422

    async def test_text_too_long_returns_413(self, client):
        # Mock allows up to max_text_length; settings default is 10_000
        huge_text = "a" * 11_000
        resp = await client.post("/v1/audio/speech", json={"input": huge_text})
        assert resp.status_code == 413

    async def test_openai_compatible_shape(self, client):
        """Verify the request body matches the OpenAI audio/speech API."""
        resp = await client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Hello from the API",
                "voice": "af_heart",
                "speed": 1.0,
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200


# ── Auto-stream threshold ─────────────────────────────────────────────────────


class TestAutoStream:
    async def test_short_text_not_auto_streamed(self, client):
        """Text under threshold returns a complete WAV (non-streaming)."""
        short = "Hello world"  # well under 300 chars
        resp = await client.post("/v1/audio/speech", json={"input": short, "stream": False})
        assert resp.status_code == 200
        assert resp.headers.get("x-auto-streamed") is None

    async def test_long_text_auto_upgraded_to_stream(self, client):
        """Text over threshold is transparently upgraded to streaming."""
        long_text = "word " * 80  # ~400 chars, above default threshold of 300
        resp = await client.post("/v1/audio/speech", json={"input": long_text, "stream": False})
        assert resp.status_code == 200
        assert resp.headers.get("x-auto-streamed") == "true"

    async def test_explicit_stream_has_no_auto_stream_header(self, client):
        """Explicitly requested streaming never sets X-Auto-Streamed."""
        long_text = "word " * 80
        resp = await client.post("/v1/audio/speech", json={"input": long_text, "stream": True})
        assert resp.status_code == 200
        assert resp.headers.get("x-auto-streamed") is None

    async def test_auto_streamed_response_is_wav(self, client):
        """Auto-streamed WAV response begins with the correct RIFF header."""
        long_text = "word " * 80
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": long_text, "stream": False, "response_format": "wav"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-auto-streamed") == "true"
        # Streaming WAV starts with RIFF header
        assert resp.content[:4] == b"RIFF"

    async def test_auto_streamed_pcm_no_header(self, client):
        """Auto-streamed PCM response has no WAV header."""
        long_text = "word " * 80
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": long_text, "stream": False, "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-auto-streamed") == "true"
        assert resp.content[:4] != b"RIFF"


# ── Cache behaviour ───────────────────────────────────────────────────────────


class TestCache:
    async def test_second_request_returns_from_cache(self, client):
        payload = {"input": "Cache me please", "voice": "af_heart", "speed": 1.0}

        r1 = await client.post("/v1/audio/speech", json=payload)
        r2 = await client.post("/v1/audio/speech", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Second response should be flagged as cached
        assert r2.headers.get("x-from-cache") == "true"
        # Content must be identical
        assert r1.content == r2.content

    async def test_different_voice_is_different_cache_entry(self, client):
        r1 = await client.post(
            "/v1/audio/speech", json={"input": "Same text", "voice": "af_heart"}
        )
        r2 = await client.post(
            "/v1/audio/speech", json={"input": "Same text", "voice": "am_adam"}
        )
        # Both should succeed; second should NOT be a cache hit for first
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.headers.get("x-from-cache") != "true"


# ── Voices ────────────────────────────────────────────────────────────────────


class TestVoices:
    async def test_list_voices_returns_200(self, client):
        resp = await client.get("/v1/voices")
        assert resp.status_code == 200

    async def test_voices_response_shape(self, client):
        resp = await client.get("/v1/voices")
        data = resp.json()
        assert "voices" in data
        assert len(data["voices"]) > 0
        first = data["voices"][0]
        assert "id" in first
        assert "language" in first
        assert "gender" in first

    async def test_voices_include_expected_defaults(self, client):
        resp = await client.get("/v1/voices")
        ids = {v["id"] for v in resp.json()["voices"]}
        assert "af_heart" in ids
        assert "am_adam" in ids


# ── Health & metrics ──────────────────────────────────────────────────────────


class TestOps:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_response_shape(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["tts_ready"] is True
        assert "version" in data
        assert "cache_size" in data

    async def test_v1_health_alias(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200

    async def test_metrics_returns_prometheus_text(self, client):
        resp = await client.get("/v1/metrics")
        assert resp.status_code == 200
        # Prometheus text format starts with # HELP or # TYPE or a metric name
        body = resp.text
        assert "tts_requests_total" in body or "tts_cache" in body or "#" in body
