"""
Unit tests for the TTS service module.

Tests the abstract interface contract, MockTTSService behaviour, factory
routing, audio utilities, and provides a reusable conformance suite that
any new backend can subclass to verify it satisfies the TTSServiceBase API.
"""

import asyncio
import struct
from abc import ABC

import numpy as np
import pytest

from tts_api.services.audio import (
    SAMPLE_RATE,
    audio_to_wav,
    float32_to_pcm16,
    make_streaming_wav_header,
    pcm16_to_float32,
    wav_to_pcm,
)
from tts_api.services.tts.base import DEFAULT_VOICES, TTSServiceBase
from tts_api.services.tts.mock import MockTTSService


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — TTSServiceBase abstract contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestTTSServiceBaseContract:
    """Verify the ABC enforces the contract and provides sensible defaults."""

    def test_cannot_instantiate_directly(self):
        """TTSServiceBase is abstract — instantiation must fail."""
        with pytest.raises(TypeError, match="abstract"):
            TTSServiceBase()  # type: ignore[abstract]

    def test_subclass_must_implement_synthesize(self):
        """A subclass that omits _synthesize cannot be instantiated."""

        class Incomplete(TTSServiceBase):
            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_get_voices(self):
        class Incomplete(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return b""

            async def health_check(self):
                return True

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_health_check(self):
        class Incomplete(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return b""

            async def get_voices(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_minimal_complete_subclass_instantiates(self):
        """A subclass implementing all three abstract methods can be created."""

        class Minimal(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return audio_to_wav(np.zeros(100, dtype=np.float32))

            async def get_voices(self):
                return ["voice_a"]

            async def health_check(self):
                return True

        svc = Minimal()
        assert isinstance(svc, TTSServiceBase)

    def test_default_sample_rate(self):
        """Default sample_rate matches the global SAMPLE_RATE constant."""

        class Svc(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return b""

            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        svc = Svc()
        assert svc.sample_rate == SAMPLE_RATE

    def test_custom_sample_rate(self):
        """Subclasses can override sample_rate."""

        class CustomRate(TTSServiceBase):
            sample_rate = 48_000

            async def _synthesize(self, text, voice, speed):
                return b""

            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        assert CustomRate().sample_rate == 48_000

    async def test_synthesize_calls_private_and_returns_result(self):
        """synthesize() delegates to _synthesize() and returns its bytes."""
        expected = audio_to_wav(np.zeros(100, dtype=np.float32))

        class Svc(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return expected

            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        result = await Svc().synthesize("hi", "v", 1.0)
        assert result == expected

    async def test_default_streaming_yields_pcm_from_synthesize(self):
        """Default synthesize_streaming() wraps synthesize() into one PCM chunk."""
        audio = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
        wav = audio_to_wav(audio)

        class Svc(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return wav

            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        svc = Svc()
        chunks = [c async for c in svc.synthesize_streaming("hi", "v", 1.0)]
        assert len(chunks) == 1
        # Should be raw PCM (no WAV header)
        assert chunks[0][:4] != b"RIFF"
        assert len(chunks[0]) > 0

    async def test_shutdown_default_is_noop(self):
        """Default shutdown() does nothing and does not raise."""

        class Svc(TTSServiceBase):
            async def _synthesize(self, text, voice, speed):
                return b""

            async def get_voices(self):
                return []

            async def health_check(self):
                return True

        await Svc().shutdown()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — DEFAULT_VOICES constant
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultVoices:
    def test_not_empty(self):
        assert len(DEFAULT_VOICES) > 0

    def test_contains_expected_voices(self):
        for v in ("af_heart", "am_adam", "bf_emma", "bm_george"):
            assert v in DEFAULT_VOICES

    def test_no_duplicates(self):
        assert len(DEFAULT_VOICES) == len(set(DEFAULT_VOICES))

    def test_all_strings(self):
        assert all(isinstance(v, str) for v in DEFAULT_VOICES)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — MockTTSService unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMockTTSService:
    @pytest.fixture
    def mock_svc(self):
        return MockTTSService()

    async def test_is_subclass_of_base(self, mock_svc):
        assert isinstance(mock_svc, TTSServiceBase)

    async def test_synthesize_returns_valid_wav(self, mock_svc):
        wav = await mock_svc.synthesize("Hello world", "af_heart", 1.0)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        # Should have a data section
        assert len(wav) > 44

    async def test_synthesize_wav_header_fields(self, mock_svc):
        wav = await mock_svc.synthesize("Test", "af_heart", 1.0)
        # PCM format
        assert struct.unpack_from("<H", wav, 20)[0] == 1
        # Mono
        assert struct.unpack_from("<H", wav, 22)[0] == 1
        # Sample rate
        assert struct.unpack_from("<I", wav, 24)[0] == SAMPLE_RATE
        # 16-bit
        assert struct.unpack_from("<H", wav, 34)[0] == 16

    async def test_synthesize_longer_text_produces_more_audio(self, mock_svc):
        short = await mock_svc.synthesize("Hi", "af_heart", 1.0)
        long = await mock_svc.synthesize("A much longer sentence here", "af_heart", 1.0)
        # Longer text → more PCM data
        assert len(long) > len(short)

    async def test_streaming_yields_multiple_chunks(self, mock_svc):
        chunks = [c async for c in mock_svc.synthesize_streaming("Hello", "af_heart", 1.0)]
        assert len(chunks) == 3  # Mock yields exactly 3 chunks
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

    async def test_streaming_chunks_are_pcm(self, mock_svc):
        """Streaming chunks should be raw PCM, not WAV."""
        chunks = [c async for c in mock_svc.synthesize_streaming("Hello", "af_heart", 1.0)]
        for chunk in chunks:
            assert chunk[:4] != b"RIFF"

    async def test_get_voices_returns_default_list(self, mock_svc):
        voices = await mock_svc.get_voices()
        assert voices == DEFAULT_VOICES

    async def test_health_check_returns_true(self, mock_svc):
        assert await mock_svc.health_check() is True

    async def test_shutdown_is_safe(self, mock_svc):
        await mock_svc.shutdown()

    async def test_sample_rate(self, mock_svc):
        assert mock_svc.sample_rate == SAMPLE_RATE


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactory:
    def test_mock_backend(self):
        from tts_api.core.config import Settings
        from tts_api.services.tts.factory import create_tts_service

        settings = Settings(backend="mock")
        svc = create_tts_service(settings)
        assert isinstance(svc, MockTTSService)

    def test_unknown_backend_raises(self):
        from tts_api.core.config import Settings
        from tts_api.services.tts.factory import create_tts_service

        settings = Settings(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown TTS backend"):
            create_tts_service(settings)

    def test_backend_name_case_insensitive(self):
        from tts_api.core.config import Settings
        from tts_api.services.tts.factory import create_tts_service

        for name in ("Mock", "MOCK", "mock"):
            svc = create_tts_service(Settings(backend=name))
            assert isinstance(svc, MockTTSService)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Audio utilities
# ═══════════════════════════════════════════════════════════════════════════════


class TestAudioUtilities:
    def test_float32_to_pcm16_basic(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        assert isinstance(pcm, bytes)
        assert len(pcm) == len(audio) * 2  # 2 bytes per int16 sample

    def test_float32_to_pcm16_clipping(self):
        """Values outside [-1, 1] are clipped, not wrapped."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767

    def test_pcm16_to_float32_roundtrip(self):
        """float32 → PCM16 → float32 round-trips within quantization error."""
        original = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        pcm = float32_to_pcm16(original)
        recovered = pcm16_to_float32(pcm)
        np.testing.assert_allclose(recovered, original, atol=1 / 32767)

    def test_audio_to_wav_produces_valid_header(self):
        audio = np.zeros(1000, dtype=np.float32)
        wav = audio_to_wav(audio)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        pcm_size = struct.unpack_from("<I", wav, 40)[0]
        assert pcm_size == 1000 * 2  # 1000 samples × 2 bytes

    def test_audio_to_wav_custom_sample_rate(self):
        audio = np.zeros(100, dtype=np.float32)
        wav = audio_to_wav(audio, sample_rate=48_000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 48_000

    def test_wav_to_pcm_strips_header(self):
        audio = np.zeros(500, dtype=np.float32)
        wav = audio_to_wav(audio)
        pcm = wav_to_pcm(wav)
        assert pcm == wav[44:]
        assert len(pcm) == 500 * 2

    def test_make_streaming_wav_header_format(self):
        hdr = make_streaming_wav_header()
        assert len(hdr) == 44
        assert hdr[:4] == b"RIFF"
        assert hdr[8:12] == b"WAVE"
        sr = struct.unpack_from("<I", hdr, 24)[0]
        assert sr == SAMPLE_RATE

    def test_make_streaming_wav_header_custom_rate(self):
        hdr = make_streaming_wav_header(sample_rate=16_000)
        sr = struct.unpack_from("<I", hdr, 24)[0]
        assert sr == 16_000

    def test_empty_audio_produces_valid_wav(self):
        audio = np.zeros(0, dtype=np.float32)
        wav = audio_to_wav(audio)
        assert wav[:4] == b"RIFF"
        pcm = wav_to_pcm(wav)
        assert len(pcm) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Conformance test suite
#
# Any new TTSServiceBase backend can verify contract compliance by subclassing
# TTSBackendConformance and providing a `service` fixture.
# ═══════════════════════════════════════════════════════════════════════════════


class TTSBackendConformance(ABC):
    """
    Reusable conformance test suite for TTSServiceBase implementations.

    To test a new backend::

        class TestMyBackend(TTSBackendConformance):
            @pytest.fixture
            def service(self):
                return MyTTSService(...)
    """

    @pytest.fixture
    def service(self) -> TTSServiceBase:
        raise NotImplementedError

    async def test_is_tts_service(self, service):
        assert isinstance(service, TTSServiceBase)

    async def test_has_sample_rate(self, service):
        assert isinstance(service.sample_rate, int)
        assert service.sample_rate > 0

    async def test_synthesize_returns_wav(self, service):
        wav = await service.synthesize("Hello world.", "af_heart", 1.0)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert len(wav) > 44

    async def test_synthesize_wav_has_correct_sample_rate(self, service):
        wav = await service.synthesize("Test.", "af_heart", 1.0)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == service.sample_rate

    async def test_streaming_yields_bytes(self, service):
        chunks = [c async for c in service.synthesize_streaming("Hello.", "af_heart", 1.0)]
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

    async def test_streaming_chunks_are_pcm(self, service):
        chunks = [c async for c in service.synthesize_streaming("Hello.", "af_heart", 1.0)]
        for chunk in chunks:
            # PCM int16 — length must be even (2 bytes per sample)
            assert len(chunk) % 2 == 0
            # Not a WAV file
            assert chunk[:4] != b"RIFF"

    async def test_get_voices_returns_list_of_strings(self, service):
        voices = await service.get_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert all(isinstance(v, str) for v in voices)

    async def test_health_check_returns_bool(self, service):
        result = await service.health_check()
        assert isinstance(result, bool)

    async def test_shutdown_does_not_raise(self, service):
        await service.shutdown()


class TestMockConformance(TTSBackendConformance):
    """Run the full conformance suite against MockTTSService."""

    @pytest.fixture
    def service(self):
        return MockTTSService()
