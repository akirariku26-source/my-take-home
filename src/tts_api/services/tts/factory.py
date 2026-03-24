"""TTS backend factory."""

from tts_api.core.config import Settings
from tts_api.services.tts.base import TTSServiceBase


def create_tts_service(settings: Settings) -> TTSServiceBase:
    backend = settings.backend.lower()

    if backend == "kokoro":
        from tts_api.services.tts.kokoro import KokoroTTSService

        return KokoroTTSService(
            lang_code=settings.kokoro_lang_code,
            max_workers=settings.max_workers,
        )

    if backend == "mock":
        from tts_api.services.tts.mock import MockTTSService

        return MockTTSService()

    raise ValueError(f"Unknown TTS backend: {backend!r}. Choose 'kokoro' or 'mock'.")
