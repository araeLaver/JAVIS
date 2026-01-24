"""In-memory cache implementation.

Provides a thread-safe in-memory cache with TTL support.
Suitable for development and single-instance deployments.
"""

import asyncio
import fnmatch
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from javis.cache.base import CacheInterface, CacheSerializer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with optional expiration."""

    value: Any
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class InMemoryCache(CacheInterface):
    """Thread-safe in-memory cache with TTL support.

    Features:
    - Automatic expiration of entries
    - Pattern-based key lookup
    - Periodic cleanup of expired entries
    - Thread-safe operations

    Example:
        cache = InMemoryCache()
        await cache.set("user:123", {"name": "John"}, ttl=300)
        user = await cache.get("user:123")
    """

    def __init__(
        self,
        default_ttl: Optional[int] = None,
        max_size: Optional[int] = None,
        cleanup_interval: int = 60,
    ):
        """Initialize in-memory cache.

        Args:
            default_ttl: Default TTL for entries (None = no expiration)
            max_size: Maximum number of entries (None = unlimited)
            cleanup_interval: Seconds between cleanup runs
        """
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup >= self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Remove expired entries (must hold lock)."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _enforce_max_size(self) -> None:
        """Remove oldest entries if over max size (must hold lock)."""
        if self._max_size is None:
            return

        while len(self._cache) >= self._max_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry due to size limit: {oldest_key}")

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                del self._cache[key]
                return None

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache."""
        with self._lock:
            self._maybe_cleanup()
            self._enforce_max_size()

            # Calculate expiration time
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = None
            if effective_ttl is not None:
                expires_at = time.time() + effective_ttl

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            return True

    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    async def clear(self) -> int:
        """Clear all keys from the cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching a pattern."""
        with self._lock:
            self._maybe_cleanup()

            if pattern == "*":
                return list(self._cache.keys())

            return [
                key for key in self._cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value atomically."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None or entry.is_expired():
                new_value = amount
                self._cache[key] = CacheEntry(value=new_value)
            else:
                new_value = int(entry.value) + amount
                entry.value = new_value

            return new_value

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value atomically."""
        return await self.increment(key, -amount)

    # Additional utility methods

    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key.

        Returns:
            Remaining seconds, None if no expiration, -1 if not found
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return -1
            if entry.expires_at is None:
                return None
            remaining = entry.expires_at - time.time()
            return max(0, int(remaining))

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on an existing key.

        Args:
            key: Cache key
            ttl: New TTL in seconds

        Returns:
            True if key exists and TTL was set
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return False
            entry.expires_at = time.time() + ttl
            return True

    async def persist(self, key: str) -> bool:
        """Remove expiration from a key.

        Args:
            key: Cache key

        Returns:
            True if key exists and expiration was removed
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return False
            entry.expires_at = None
            return True

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            expired_count = sum(
                1 for entry in self._cache.values()
                if entry.is_expired()
            )
            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "default_ttl": self._default_ttl,
                "expired_pending_cleanup": expired_count,
            }
