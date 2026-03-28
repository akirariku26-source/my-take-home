"""
Adaptive concurrency limiter for TTS synthesis requests.

Algorithm: AIMD (Additive Increase / Multiplicative Decrease)
──────────────────────────────────────────────────────────────
Inspired by TCP congestion control and Netflix's concurrency-limits library.

• After each completed request, update an EWMA of observed synthesis latency.
• If EWMA > target_latency  → multiplicative decrease (limit *= 0.9)
• If EWMA < target_latency  → additive increase     (limit += 1)
• Requests that arrive when in_flight ≥ limit are rejected immediately (503).

This prevents the request queue from growing to a depth where tail latency
becomes unacceptable, while still probing for spare capacity when the system
is healthy.

Usage
─────
    limiter = AdaptiveConcurrencyLimiter(initial=8, target_latency_s=10.0)

    async with limiter.acquire():
        # synthesis work here — slot released on exit, EWMA updated
        ...

    # or in a streaming generator:
    async def _generate():
        async with limiter.acquire():
            async for chunk in tts.synthesize_streaming(...):
                yield chunk
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import HTTPException
from prometheus_client import Counter, Gauge

from tts_api.core.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

_LIMIT = Gauge(
    "tts_concurrency_limit",
    "Current adaptive concurrency window (max simultaneous synthesis requests)",
)
_IN_FLIGHT = Gauge(
    "tts_concurrency_in_flight",
    "Number of synthesis requests currently being processed",
)
_REJECTED = Counter(
    "tts_concurrency_rejected_total",
    "Requests rejected because the concurrency window was full",
)
_EWMA_LATENCY = Gauge(
    "tts_concurrency_ewma_latency_seconds",
    "EWMA of observed synthesis latency used by the adaptive algorithm",
)


class _AcquireToken:
    """Yielded by AdaptiveConcurrencyLimiter.acquire().

    Callers that can separate synthesis time from network-delivery time should
    call set_elapsed() once synthesis is complete so that slow-client
    back-pressure does not inflate the EWMA and unfairly shrink the window.

    If set_elapsed() is never called the full wall-clock time is used (the
    original behaviour, and correct for non-streaming callers).
    """

    __slots__ = ("elapsed",)

    def __init__(self) -> None:
        self.elapsed: float | None = None

    def set_elapsed(self, elapsed: float) -> None:
        """Override the latency sample reported to the AIMD algorithm.

        Call this after synthesis finishes but before the slot is released —
        i.e. before the ``async with limiter.acquire()`` block exits.
        """
        self.elapsed = elapsed


class AdaptiveConcurrencyLimiter:
    """
    AIMD-based adaptive concurrency limiter.

    Parameters
    ──────────
    initial         Starting concurrency window size.
                    Rule of thumb: TTS_MAX_WORKERS × 2 (allows some queuing).
    min_limit       Floor for the window.  Must be ≥ 1.
    max_limit       Ceiling for the window.  Rule of thumb: TTS_MAX_WORKERS × 8.
    target_latency_s
                    EWMA latency target (seconds).  When observed latency
                    exceeds this the window shrinks; below it the window grows.
                    Default 10 s is conservative for typical TTS synthesis
                    (2-5 s per request).  Tune lower for stricter SLA.
    ewma_alpha      Smoothing factor for the EWMA.  Higher = more reactive.
    enabled         Set False to disable limiting (pass-through mode).
    """

    def __init__(
        self,
        initial: int = 8,
        min_limit: int = 1,
        max_limit: int = 32,
        target_latency_s: float = 10.0,
        ewma_alpha: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self._limit: float = float(initial)
        self._min = float(min_limit)
        self._max = float(max_limit)
        self._target = target_latency_s
        self._alpha = ewma_alpha
        self._enabled = enabled
        self._in_flight: int = 0
        self._ewma: float = 0.0  # initialised on first sample
        self._ewma_ready = False
        self._lock = asyncio.Lock()

        _LIMIT.set(initial)
        _IN_FLIGHT.set(0)

        logger.info(
            "adaptive_concurrency_init",
            initial=initial,
            min=min_limit,
            max=max_limit,
            target_latency_s=target_latency_s,
            enabled=enabled,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self):
        """
        Async context manager that occupies one concurrency slot for the
        duration of the ``async with`` block.

        Raises HTTPException(503) immediately if the window is full.
        Updates the EWMA and adjusts the window on exit.
        """
        if not self._enabled:
            yield _AcquireToken()
            return

        async with self._lock:
            if self._in_flight >= int(self._limit):
                _REJECTED.inc()
                logger.warning(
                    "concurrency_limit_reached",
                    in_flight=self._in_flight,
                    limit=int(self._limit),
                )
                raise HTTPException(
                    status_code=503,
                    detail="Server at capacity — retry shortly",
                    headers={"Retry-After": "1"},
                )
            self._in_flight += 1
            _IN_FLIGHT.set(self._in_flight)

        token = _AcquireToken()
        start = time.perf_counter()
        try:
            yield token
        finally:
            # Use caller-supplied synthesis time if available; fall back to
            # full wall-clock elapsed so non-streaming callers need no changes.
            elapsed = token.elapsed if token.elapsed is not None else time.perf_counter() - start
            async with self._lock:
                self._in_flight -= 1
                _IN_FLIGHT.set(self._in_flight)
                self._adjust(elapsed)

    @property
    def limit(self) -> int:
        return int(self._limit)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    # ── Internal ──────────────────────────────────────────────────────────────

    def _adjust(self, latency: float) -> None:
        """Update EWMA and resize the concurrency window (called under lock)."""
        # Bootstrap the EWMA from the first real sample
        if not self._ewma_ready:
            self._ewma = latency
            self._ewma_ready = True
        else:
            self._ewma = self._alpha * latency + (1 - self._alpha) * self._ewma

        _EWMA_LATENCY.set(self._ewma)

        old_limit = self._limit

        if self._ewma > self._target:
            # Congestion: multiplicative decrease — react quickly
            self._limit = max(self._min, self._limit * 0.9)
        elif self._ewma < self._target * 0.8:
            # Healthy headroom: additive increase — probe slowly
            self._limit = min(self._max, self._limit + 1.0)
        # else: within [0.8×target, target] — hold steady

        _LIMIT.set(self._limit)

        if int(self._limit) != int(old_limit):
            logger.info(
                "concurrency_limit_adjusted",
                old=int(old_limit),
                new=int(self._limit),
                ewma_latency_s=round(self._ewma, 3),
                target_s=self._target,
            )
