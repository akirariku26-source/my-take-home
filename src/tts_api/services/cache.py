"""
LRU audio cache.

Keys are (text, voice, speed) tuples hashed to MD5 for compactness.
Thread-safe via asyncio.Lock (single-process).  For multi-replica deployments,
swap the _store for a Redis-backed implementation.
"""

import asyncio
import time
from hashlib import md5

from cachetools import LRUCache
from prometheus_client import Gauge

from tts_api.core.logging import get_logger

_CACHE_SIZE = Gauge("tts_cache_size", "Current number of entries in the audio cache")

logger = get_logger(__name__)


class AudioCache:
    def __init__(self, max_size: int = 1_000, ttl_seconds: int = 3_600, enabled: bool = True):
        self._store: LRUCache = LRUCache(maxsize=max_size)
        # {key: expiry_timestamp}
        self._expiry: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._enabled = enabled
        self._lock = asyncio.Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str, voice: str, speed: float) -> str:
        return md5(f"{text}\x00{voice}\x00{speed:.4f}".encode()).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, text: str, voice: str, speed: float) -> bytes | None:
        if not self._enabled:
            return None
        key = self._key(text, voice, speed)
        async with self._lock:
            if key not in self._store:
                return None
            if time.monotonic() > self._expiry.get(key, 0.0):
                del self._store[key]
                self._expiry.pop(key, None)
                _CACHE_SIZE.set(len(self._store))
                return None
            return self._store[key]

    async def set(self, text: str, voice: str, speed: float, audio: bytes) -> None:
        if not self._enabled:
            return
        key = self._key(text, voice, speed)
        async with self._lock:
            self._store[key] = audio
            self._expiry[key] = time.monotonic() + self._ttl
            _CACHE_SIZE.set(len(self._store))

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def max_size(self) -> int:
        return self._store.maxsize
