"""
Token-bucket rate limiter middleware.

Each client IP gets an independent bucket that refills at `rpm/60` tokens/second
up to a max of `burst` tokens.  A request costs 1 token.

Limitations
───────────
State is in-process, so this only works correctly with a single replica
(or sticky sessions).  For multi-replica deployments, replace the dict with
a Redis-backed atomic increment (e.g., INCR + EXPIRE).
"""

import asyncio
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = 60, burst: int = 10) -> None:
        super().__init__(app)
        self._rate = rpm / 60.0          # tokens refilled per second
        self._capacity = float(burst)
        self._tokens: dict[str, float] = defaultdict(lambda: float(burst))
        self._last_refill: dict[str, float] = defaultdict(time.monotonic)
        self._lock = asyncio.Lock()

    def _client_id(self, request: Request) -> str:
        # Respect X-Forwarded-For when behind a proxy / load balancer
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health / metrics endpoints
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        client_id = self._client_id(request)
        allowed = await self._check(client_id)

        if not allowed:
            retry_after = int(1.0 / self._rate)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    async def _check(self, client_id: str) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill[client_id]
            self._tokens[client_id] = min(
                self._capacity,
                self._tokens[client_id] + elapsed * self._rate,
            )
            self._last_refill[client_id] = now

            if self._tokens[client_id] >= 1.0:
                self._tokens[client_id] -= 1.0
                return True
            return False
