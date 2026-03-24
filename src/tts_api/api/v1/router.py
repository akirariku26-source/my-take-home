"""Assemble the v1 API router."""

from fastapi import APIRouter

from tts_api.api.v1.speech import open_router
from tts_api.api.v1.speech import router as speech_router

router = APIRouter()

# Unauthenticated: /health, /metrics — infra tooling must reach these freely.
router.include_router(open_router, tags=["ops"])

# Auth-protected: /audio/speech, /voices.
# verify_api_key is applied per-route in speech.py (not here) so that the
# WebSocket endpoint — which handles auth via ?api_key query param — is
# excluded.  HTTPBearer/APIKeyHeader are HTTP-only and cannot run on WS.
router.include_router(speech_router, tags=["speech"])
