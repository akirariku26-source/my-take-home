"""
Tests for streaming endpoints (HTTP chunked + WebSocket).
"""

import json
import struct

import pytest

from tts_api.services.audio import SAMPLE_RATE

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_streaming_wav_header(data: bytes) -> dict:
    """Parse the first 44 bytes of a streaming WAV."""
    assert data[:4] == b"RIFF", "Missing RIFF marker"
    assert data[8:12] == b"WAVE", "Missing WAVE marker"
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    channels = struct.unpack_from("<H", data, 22)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    return {"sample_rate": sample_rate, "channels": channels, "bit_depth": bits}


# ── Streaming HTTP ────────────────────────────────────────────────────────────


class TestStreamingHTTP:
    async def test_stream_flag_returns_200(self, client):
        resp = await client.post(
            "/v1/audio/speech", json={"input": "Hello streaming world", "stream": True}
        )
        assert resp.status_code == 200

    async def test_stream_wav_content_type(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Streaming test", "stream": True, "response_format": "wav"},
        )
        assert "audio/wav" in resp.headers["content-type"]

    async def test_stream_wav_header_valid(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Check wav header", "stream": True, "response_format": "wav"},
        )
        assert resp.status_code == 200
        body = resp.content
        assert len(body) >= 44, "Response too short to contain WAV header"
        info = parse_streaming_wav_header(body[:44])
        assert info["sample_rate"] == SAMPLE_RATE
        assert info["channels"] == 1
        assert info["bit_depth"] == 16

    async def test_stream_pcm_no_wav_header(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Raw PCM please", "stream": True, "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert "audio/pcm" in resp.headers["content-type"]
        # Should NOT have RIFF marker
        assert resp.content[:4] != b"RIFF"

    async def test_stream_response_non_empty(self, client):
        resp = await client.post(
            "/v1/audio/speech", json={"input": "Non-empty stream", "stream": True}
        )
        # WAV header (44) + at least one PCM chunk
        assert len(resp.content) > 44

    async def test_stream_sample_rate_header_present(self, client):
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": "Headers test", "stream": True, "response_format": "pcm"},
        )
        assert resp.headers.get("x-sample-rate") == str(SAMPLE_RATE)
        assert resp.headers.get("x-channels") == "1"
        assert resp.headers.get("x-bit-depth") == "16"


# ── WebSocket ─────────────────────────────────────────────────────────────────


def _make_ws_app(max_text_length: int):
    """Create a test app with a custom max_text_length for limit tests."""
    from tts_api.core.config import Settings
    from tts_api.main import create_app
    from tts_api.services.cache import AudioCache
    from tts_api.services.concurrency import AdaptiveConcurrencyLimiter
    from tts_api.services.tts.factory import ServiceBundle, create_tts_service

    s = Settings(backend="mock", max_text_length=max_text_length, rate_limit_enabled=False)
    application = create_app(s)
    svc = create_tts_service(s)
    application.state.services = ServiceBundle(buffered=svc, streaming=svc)
    application.state.audio_cache = AudioCache(max_size=100, ttl_seconds=60, enabled=False)
    application.state.concurrency_limiter = AdaptiveConcurrencyLimiter(
        initial=4, enabled=False
    )
    return application


class TestWebSocket:
    async def test_ws_accepts_connection(self, app):
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                # First frame must be the ready metadata
                ready = ws.receive_json()
                assert ready["type"] == "ready"
                assert ready["sample_rate"] == SAMPLE_RATE
                assert ready["channels"] == 1
                assert ready["bit_depth"] == 16

    async def test_ws_synthesizes_text_and_returns_done(self, app):
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                # Consume ready frame
                ws.receive_json()

                # Send a short text
                ws.send_json({"type": "text", "text": "Hello world."})
                ws.send_json({"type": "flush"})

                # Collect all frames until "done"
                # ws.receive() returns the raw ASGI event dict:
                # {"type": "websocket.send", "bytes": b"...", "text": None}
                frames = []
                while True:
                    raw = ws.receive()
                    payload_bytes = raw.get("bytes")
                    payload_text = raw.get("text")
                    if payload_bytes:
                        frames.append(payload_bytes)
                    elif payload_text:
                        msg = json.loads(payload_text)
                        if msg.get("type") == "done":
                            break
                        if msg.get("type") == "error":
                            pytest.fail(f"WS error: {msg}")

                # We should have received at least some audio bytes
                assert len(frames) > 0

    async def test_ws_config_message_changes_voice(self, app):
        """Config message should be accepted without error."""
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready
                ws.send_json({"type": "config", "voice": "am_adam", "speed": 1.2})
                ws.send_json({"type": "text", "text": "Testing config."})
                ws.send_json({"type": "flush"})

                # Collect until done
                while True:
                    raw = ws.receive()
                    if raw.get("text"):
                        msg = json.loads(raw["text"])
                        if msg.get("type") in ("done", "error"):
                            assert msg["type"] == "done"
                            break

    async def test_ws_unknown_message_type_returns_error_frame(self, app):
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready
                ws.send_json({"type": "bogus"})
                resp = ws.receive_json()
                assert resp["type"] == "error"

    def test_ws_per_message_limit_returns_error_keeps_session(self):
        """A single oversized frame returns an error frame; session stays open."""
        from starlette.testclient import TestClient

        app = _make_ws_app(max_text_length=20)
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready
                ws.send_json({"type": "text", "text": "x" * 21})  # one char over
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "21" in err["message"] and "20" in err["message"]
                # Session is still live: a normal message works afterwards
                ws.send_json({"type": "text", "text": "Hi."})
                ws.send_json({"type": "flush"})
                frames = []
                while True:
                    raw = ws.receive()
                    if raw.get("bytes"):
                        frames.append(raw["bytes"])
                    elif raw.get("text"):
                        msg = json.loads(raw["text"])
                        if msg["type"] == "done":
                            break
                        if msg["type"] == "error":
                            pytest.fail(f"Unexpected error after recovery: {msg}")
                assert len(frames) > 0, "Expected audio after recovering from limit error"

    def test_ws_per_turn_limit_returns_error_keeps_session(self):
        """Accumulated text exceeding the per-turn limit returns an error frame."""
        from starlette.testclient import TestClient

        app = _make_ws_app(max_text_length=30)
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready
                # 23 chars, no sentence boundary — stays in accumulator buffer
                ws.send_json({"type": "text", "text": "Hello world and friends"})
                # 12 more chars → total 35 > 30 → error frame expected
                ws.send_json({"type": "text", "text": " greet them all"})
                # Drain frames (possible audio from first message if a boundary fired)
                # until we find the error JSON; fail if we hit "done" first.
                err = None
                for _ in range(20):
                    raw = ws.receive()
                    if raw.get("text"):
                        msg = json.loads(raw["text"])
                        if msg["type"] == "error":
                            err = msg
                            break
                        if msg["type"] == "done":
                            pytest.fail("Received 'done' before expected error frame")
                assert err is not None, "Expected error frame for per-turn limit"
                assert "too long" in err["message"].lower()

    def test_ws_turn_counter_resets_after_flush(self):
        """After a flush the per-turn counter resets; the next turn gets a fresh budget."""
        from starlette.testclient import TestClient

        app = _make_ws_app(max_text_length=30)
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready

                # Turn 1: send 25 chars, flush
                ws.send_json({"type": "text", "text": "Hello world, how are you?"})  # 25 chars
                ws.send_json({"type": "flush"})
                while True:
                    raw = ws.receive()
                    if raw.get("text") and json.loads(raw["text"])["type"] == "done":
                        break

                # Turn 2: another 25 chars — should succeed (counter was reset)
                ws.send_json({"type": "text", "text": "Hello world, how are you?"})  # 25 chars
                ws.send_json({"type": "flush"})
                frames = []
                while True:
                    raw = ws.receive()
                    if raw.get("bytes"):
                        frames.append(raw["bytes"])
                    elif raw.get("text"):
                        msg = json.loads(raw["text"])
                        if msg["type"] == "done":
                            break
                        if msg["type"] == "error":
                            pytest.fail(f"Turn 2 should not hit limit: {msg}")
                assert len(frames) > 0, "Expected audio in turn 2 after counter reset"

    def test_ws_normal_flow_no_text_loss(self):
        """Multiple text messages within limits produce audio without loss or duplication."""
        from starlette.testclient import TestClient

        # Large enough limit so everything fits
        app = _make_ws_app(max_text_length=10_000)
        with TestClient(app) as tc:
            with tc.websocket_connect("/v1/audio/speech/ws") as ws:
                ws.receive_json()  # ready
                ws.send_json({"type": "text", "text": "First sentence. "})
                ws.send_json({"type": "text", "text": "Second sentence. "})
                ws.send_json({"type": "flush"})
                frames = []
                while True:
                    raw = ws.receive()
                    if raw.get("bytes"):
                        frames.append(raw["bytes"])
                    elif raw.get("text"):
                        msg = json.loads(raw["text"])
                        if msg["type"] == "done":
                            break
                        if msg["type"] == "error":
                            pytest.fail(f"Unexpected WS error: {msg}")
                assert len(frames) > 0


# ── Sentence accumulator unit tests ──────────────────────────────────────────


class TestSentenceAccumulator:
    def test_complete_sentence_emitted(self):
        from tts_api.services.sentence import SentenceAccumulator

        acc = SentenceAccumulator()
        result = acc.push("Hello world. ")
        assert result == ["Hello world."]
        assert acc.flush() == []

    def test_incomplete_sentence_buffered(self):
        from tts_api.services.sentence import SentenceAccumulator

        acc = SentenceAccumulator()
        result = acc.push("Hello")
        assert result == []
        remaining = acc.flush()
        assert remaining == ["Hello"]

    def test_multiple_sentences_in_one_push(self):
        from tts_api.services.sentence import SentenceAccumulator

        acc = SentenceAccumulator()
        result = acc.push("First sentence. Second sentence. ")
        assert len(result) == 2
        assert "First sentence." in result[0]
        assert "Second sentence." in result[1]

    def test_incremental_tokens_accumulate(self):
        from tts_api.services.sentence import SentenceAccumulator

        acc = SentenceAccumulator()
        tokens = ["He", "llo", " wor", "ld!"]
        sentences = []
        for t in tokens:
            sentences.extend(acc.push(t))
        sentences.extend(acc.flush())
        # All tokens combined should produce one sentence
        assert len(sentences) == 1
        assert "Hello world!" in sentences[0]

    def test_reset_clears_buffer(self):
        from tts_api.services.sentence import SentenceAccumulator

        acc = SentenceAccumulator()
        acc.push("Partial text")
        acc.reset()
        assert acc.flush() == []
