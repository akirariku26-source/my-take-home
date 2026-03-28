"""
Deterministic unit tests for AdaptiveConcurrencyLimiter.

Tests are split into two groups:
  TestAIMDAlgorithm  — exercises _adjust() directly; no async machinery needed.
  TestAcquireBehavior — exercises acquire() and the _AcquireToken mechanism.
"""

import asyncio

import pytest
from fastapi import HTTPException

from tts_api.services.concurrency import AdaptiveConcurrencyLimiter


# ── AIMD algorithm (synchronous) ─────────────────────────────────────────────


class TestAIMDAlgorithm:
    def _limiter(self, **kw) -> AdaptiveConcurrencyLimiter:
        defaults = dict(initial=8, min_limit=1, max_limit=20, target_latency_s=10.0,
                        ewma_alpha=1.0, enabled=True)
        return AdaptiveConcurrencyLimiter(**{**defaults, **kw})

    def test_first_sample_bootstraps_ewma_directly(self):
        """First _adjust() sets EWMA = sample (no smoothing on bootstrap)."""
        lim = self._limiter()
        assert not lim._ewma_ready
        lim._adjust(3.0)
        assert lim._ewma == pytest.approx(3.0)
        assert lim._ewma_ready

    def test_subsequent_samples_apply_ewma_smoothing(self):
        """With alpha=0.5, EWMA converges toward new samples."""
        lim = self._limiter(ewma_alpha=0.5)
        lim._adjust(4.0)   # bootstrap → EWMA = 4.0
        lim._adjust(8.0)   # 0.5*8 + 0.5*4 = 6.0
        assert lim._ewma == pytest.approx(6.0)

    def test_window_shrinks_when_ewma_exceeds_target(self):
        """EWMA > target → multiplicative decrease (× 0.9)."""
        lim = self._limiter(initial=10)
        lim._adjust(15.0)  # 15 > 10 → shrink
        assert lim._limit == pytest.approx(10 * 0.9)

    def test_window_grows_when_ewma_below_80pct_target(self):
        """EWMA < 0.8 × target → additive increase (+ 1)."""
        lim = self._limiter(initial=8)
        lim._adjust(7.0)   # 7 < 8.0 (= 0.8 × 10) → grow
        assert lim._limit == pytest.approx(9.0)

    def test_window_holds_in_healthy_zone(self):
        """0.8 × target ≤ EWMA ≤ target → no change."""
        lim = self._limiter(initial=8)
        lim._adjust(9.0)   # 8 ≤ 9 ≤ 10 → hold
        assert lim._limit == pytest.approx(8.0)

    def test_window_never_falls_below_min_limit(self):
        """Repeated high-latency samples cannot push limit below min_limit."""
        lim = self._limiter(initial=1, min_limit=1)
        for _ in range(50):
            lim._adjust(100.0)
        assert lim._limit >= 1.0

    def test_window_never_exceeds_max_limit(self):
        """Repeated low-latency samples cannot push limit above max_limit."""
        lim = self._limiter(initial=8, max_limit=10)
        for _ in range(50):
            lim._adjust(0.1)
        assert lim._limit <= 10.0

    def test_window_boundary_at_exactly_target(self):
        """EWMA == target exactly falls in the 'hold' zone (not decrease)."""
        lim = self._limiter(initial=8)
        lim._adjust(10.0)   # == target → hold
        assert lim._limit == pytest.approx(8.0)

    def test_window_boundary_at_exactly_80pct_target(self):
        """EWMA == 0.8 × target exactly still falls in the 'hold' zone."""
        lim = self._limiter(initial=8)
        lim._adjust(8.0)    # == 0.8 × 10 → hold (condition is strictly <)
        assert lim._limit == pytest.approx(8.0)


# ── acquire() and _AcquireToken ───────────────────────────────────────────────


class TestAcquireBehavior:
    async def test_slot_increments_and_releases(self):
        lim = AdaptiveConcurrencyLimiter(initial=4, enabled=True)
        async with lim.acquire():
            assert lim.in_flight == 1
        assert lim.in_flight == 0

    async def test_slot_released_on_exception(self):
        lim = AdaptiveConcurrencyLimiter(initial=4, enabled=True)
        with pytest.raises(ValueError):
            async with lim.acquire():
                raise ValueError("boom")
        assert lim.in_flight == 0

    async def test_503_when_window_full(self):
        lim = AdaptiveConcurrencyLimiter(initial=1, enabled=True)
        lim._in_flight = 1  # saturate the single slot
        with pytest.raises(HTTPException) as exc_info:
            async with lim.acquire():
                pass
        assert exc_info.value.status_code == 503
        assert "Retry-After" in exc_info.value.headers

    async def test_disabled_mode_never_rejects(self):
        lim = AdaptiveConcurrencyLimiter(initial=1, enabled=False)
        lim._in_flight = 999  # would reject in enabled mode
        # Must not raise
        async with lim.acquire():
            pass

    async def test_set_elapsed_overrides_wall_clock(self):
        """slot.set_elapsed() is used instead of actual wall-clock time."""
        lim = AdaptiveConcurrencyLimiter(
            initial=8, target_latency_s=10.0, ewma_alpha=1.0, enabled=True
        )
        async with lim.acquire() as slot:
            # Pretend synthesis took 2 s regardless of how long we actually held the slot.
            slot.set_elapsed(2.0)
        # alpha=1.0 → EWMA = last sample = 2.0 (well below target=10 → window grows)
        assert lim._ewma == pytest.approx(2.0)
        assert lim._limit == pytest.approx(9.0)  # grew by +1

    async def test_wall_clock_used_when_set_elapsed_not_called(self):
        """Without set_elapsed(), actual elapsed time flows into the EWMA."""
        lim = AdaptiveConcurrencyLimiter(
            initial=8, target_latency_s=10.0, ewma_alpha=1.0, enabled=True
        )
        async with lim.acquire():
            await asyncio.sleep(0.05)   # 50 ms — well below 10 s target
        # EWMA should be around 50 ms (not 0 and not 10+ s)
        assert 0.01 < lim._ewma < 1.0

    async def test_set_elapsed_separates_synthesis_from_delivery(self):
        """Simulates slow client: synthesis is fast, delivery is slow.

        Without set_elapsed the EWMA would see the full delivery latency and
        shrink the window.  With set_elapsed it sees only synthesis time.
        """
        lim = AdaptiveConcurrencyLimiter(
            initial=8, target_latency_s=10.0, ewma_alpha=1.0, enabled=True
        )
        async with lim.acquire() as slot:
            # Synthesis done quickly
            slot.set_elapsed(1.0)
            # Simulate slow network delivery after synthesis is complete
            await asyncio.sleep(0.1)
        # EWMA reflects synthesis time (1.0 s), not synthesis + delivery (~1.1 s)
        assert lim._ewma == pytest.approx(1.0)
        assert lim._limit == pytest.approx(9.0)  # grew, not shrank

    async def test_disabled_token_set_elapsed_is_noop(self):
        """set_elapsed() on a disabled-mode token does not crash."""
        lim = AdaptiveConcurrencyLimiter(initial=8, enabled=False)
        async with lim.acquire() as slot:
            slot.set_elapsed(99.0)   # should be silently ignored

    async def test_multiple_concurrent_slots(self):
        """Concurrent acquires each track independently."""
        lim = AdaptiveConcurrencyLimiter(initial=4, enabled=True)

        async def _hold(duration: float):
            async with lim.acquire():
                await asyncio.sleep(duration)

        await asyncio.gather(_hold(0.02), _hold(0.02), _hold(0.02))
        assert lim.in_flight == 0
        assert lim._ewma_ready
