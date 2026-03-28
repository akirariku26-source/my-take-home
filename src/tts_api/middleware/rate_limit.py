"""
Token-bucket rate limiter middleware.

Each client IP gets an independent bucket that refills at `rpm/60` tokens/second
up to a max of `burst` tokens.  A request costs 1 token.

Trust-proxy mode
────────────────
`X-Forwarded-For` is only read when `trust_proxy=True`.  The header is trivially
spoofable by any client, so blindly trusting it lets attackers pick arbitrary
client IDs and circumvent per-IP limits.  Enable only when the service sits
behind a reverse proxy you control (nginx, AWS ALB, Cloudflare, etc.) that
strips or overwrites the header before forwarding.

Bounded state
─────────────
Client state is kept in an OrderedDict used as an LRU map.  When the number of
tracked clients exceeds `max_clients`, the least-recently-seen entry is evicted.
This bounds memory use regardless of how many distinct source IPs are observed
(e.g. during a distributed flood with randomised source addresses).

An evicted client starts with a full token bucket on their next request — the
same as a brand-new client.  Any in-flight request that holds a reference to the
evicted _ClientState continues safely; Python's reference counting keeps the
object alive until all holders release it.

Limitations
───────────
State is in-process, so this only works correctly with a single replica
(or sticky sessions).  For multi-replica deployments, replace the OrderedDict
with a Redis-backed atomic increment (e.g., INCR + EXPIRE).
"""

import asyncio
import time
from collections import OrderedDict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class _ClientState:
    """Per-client token-bucket state and its associated async lock."""

    __slots__ = ("tokens", "last_refill", "lock")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        rpm: int = 60,
        burst: int = 10,
        trust_proxy: bool = False,
        max_clients: int = 10_000,
    ) -> None:
        super().__init__(app)
        self._rate = rpm / 60.0          # tokens refilled per second
        self._capacity = float(burst)
        self._trust_proxy = trust_proxy
        self._max_clients = max_clients
        # OrderedDict as LRU: rightmost = most recently used.
        self._clients: OrderedDict[str, _ClientState] = OrderedDict()

    def _client_id(self, request: Request) -> str:
        if self._trust_proxy:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_or_create(self, client_id: str) -> _ClientState:
        """Return the client's state, creating it if absent.

        Synchronous (no awaits) so it runs atomically on the asyncio event loop —
        no outer lock is needed for the OrderedDict operations.
        """
        if client_id in self._clients:
            self._clients.move_to_end(client_id)
            return self._clients[client_id]

        state = _ClientState(self._capacity)
        self._clients[client_id] = state
        if len(self._clients) > self._max_clients:
            # Evict the least-recently-seen client.
            self._clients.popitem(last=False)
        return state

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
        state = self._get_or_create(client_id)
        async with state.lock:
            now = time.monotonic()
            elapsed = now - state.last_refill
            state.tokens = min(
                self._capacity,
                state.tokens + elapsed * self._rate,
            )
            state.last_refill = now

            if state.tokens >= 1.0:
                state.tokens -= 1.0
                return True
            return False
