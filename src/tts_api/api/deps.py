"""
Shared FastAPI dependencies.

verify_api_key
──────────────
Accepts credentials in two ways (OpenAI-compatible):
  • Authorization: Bearer <key>   (standard HTTP auth header)
  • X-API-Key: <key>              (simple header, common in API products)

If TTS_API_KEYS is empty the check is skipped entirely, so local dev /
test environments work without any configuration.

WebSocket note
──────────────
The HTTP security primitives (HTTPBearer, APIKeyHeader) don't apply to
the WS upgrade handshake.  The WS endpoint reads `?api_key=` from the
query string instead and calls _check_key() directly.
"""

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from tts_api.core.config import Settings

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_key(token: str | None, settings: Settings) -> None:
    """Raise 401/403 if the token is not in the allowed key set."""
    if not settings.api_key_set:
        return  # auth disabled

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Pass 'Authorization: Bearer <key>' "
            "or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token not in settings.api_key_set:
        raise HTTPException(status_code=403, detail="Invalid API key.")


def verify_api_key(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    api_key_header: str | None = Security(_api_key_header),
) -> None:
    """
    FastAPI dependency — reads settings from app.state so the live
    configuration (not a cached snapshot) is always used.
    """
    settings: Settings = request.app.state.settings
    token = bearer.credentials if bearer else api_key_header
    _check_key(token, settings)


def check_ws_api_key(api_key: str | None, settings: Settings) -> bool:
    """
    Validate a WebSocket API key (from query param).
    Returns True if auth passes, False if key is wrong/missing.
    Unlike the HTTP dependency this doesn't raise — the WS handler
    closes the socket with a custom code instead.
    """
    if not settings.api_key_set:
        return True
    if not api_key or api_key not in settings.api_key_set:
        return False
    return True
