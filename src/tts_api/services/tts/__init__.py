"""TTS service module — pluggable speech synthesis backends."""

from tts_api.services.tts.base import DEFAULT_VOICES, TTSServiceBase
from tts_api.services.tts.factory import ServiceBundle, create_service_bundle, create_tts_service

__all__ = ["TTSServiceBase", "DEFAULT_VOICES", "ServiceBundle", "create_service_bundle", "create_tts_service"]
