"""Redis cache implementation.

Provides a Redis-backed cache with TTL support.
Suitable for distributed deployments and production environments.
"""

import asyncio
import logging
from typing import Any, Optional

from javis.cache.base import CacheInterface, CacheSerializer

logger = logging.getLogger(__name__)


class RedisCache(CacheInterface):
    """Redis-backed cache implementation.

    Features:
    - Distributed caching across multiple instances
    - Automatic expiration with TTL
    - Pattern-based key lookup
    - Atomic operations (increment/decrement)
    - Connection pooling

    Example:
        cache = RedisCache(url="redis://localhost:6379")
        await cache.connect()
        await cache.set("user:123", {"name": "John"}, ttl=300)
        user = await cache.get("user:123")
        await cache.close()
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        db: int = 0,
        default_ttl: Optional[int] = None,
        key_prefix: str = "javis:",
        max_connections: int = 10,
        decode_responses: bool = True,
    ):
        """Initialize Redis cache.

        Args:
            url: Redis connection URL
            db: Redis database number
            default_ttl: Default TTL for entries (None = no expiration)
            key_prefix: Prefix for all keys (namespace isolation)
            max_connections: Maximum connections in pool
            decode_responses: Whether to decode bytes to strings
        """
        self._url = url
        self._db = db
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._decode_responses = decode_responses
        self._client = None
        self._serializer = CacheSerializer()

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self._key_prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Remove prefix from key."""
        if key.startswith(self._key_prefix):
            return key[len(self._key_prefix):]
        return key

    async def connect(self) -> None:
        """Establish Redis connection.

        Raises:
            ImportError: If redis package is not installed
            ConnectionError: If unable to connect to Redis
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "Redis package not installed. Install with: pip install redis"
            )

        try:
            self._client = redis.from_url(
                self._url,
                db=self._db,
                max_connections=self._max_connections,
                decode_responses=self._decode_responses,
            )
            # Test connection
            await self._client.ping()
            logger.info(f"Connected to Redis at {self._url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if self._client is None:
            raise RuntimeError(
                "Redis client not connected. Call connect() first."
            )

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        self._ensure_connected()

        try:
            data = await self._client.get(self._make_key(key))
            if data is None:
                return None
            return self._serializer.deserialize(data)
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache."""
        self._ensure_connected()

        try:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            data = self._serializer.serialize(value)

            if effective_ttl is not None:
                await self._client.setex(
                    self._make_key(key),
                    effective_ttl,
                    data,
                )
            else:
                await self._client.set(self._make_key(key), data)

            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        self._ensure_connected()

        try:
            result = await self._client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        self._ensure_connected()

        try:
            result = await self._client.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    async def clear(self) -> int:
        """Clear all keys with our prefix from the cache."""
        self._ensure_connected()

        try:
            pattern = f"{self._key_prefix}*"
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            return 0

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching a pattern."""
        self._ensure_connected()

        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(self._strip_prefix(key))
            return keys
        except Exception as e:
            logger.error(f"Redis KEYS error for pattern {pattern}: {e}")
            return []

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value atomically."""
        self._ensure_connected()

        try:
            result = await self._client.incrby(self._make_key(key), amount)
            return result
        except Exception as e:
            logger.error(f"Redis INCREMENT error for key {key}: {e}")
            raise

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value atomically."""
        return await self.increment(key, -amount)

    # Additional Redis-specific methods

    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key.

        Returns:
            Remaining seconds, None if no expiration, -2 if not found
        """
        self._ensure_connected()

        try:
            result = await self._client.ttl(self._make_key(key))
            if result == -2:
                return -2  # Key does not exist
            if result == -1:
                return None  # No expiration
            return result
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -2

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on an existing key.

        Args:
            key: Cache key
            ttl: New TTL in seconds

        Returns:
            True if key exists and TTL was set
        """
        self._ensure_connected()

        try:
            result = await self._client.expire(self._make_key(key), ttl)
            return result
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False

    async def persist(self, key: str) -> bool:
        """Remove expiration from a key.

        Args:
            key: Cache key

        Returns:
            True if key exists and expiration was removed
        """
        self._ensure_connected()

        try:
            result = await self._client.persist(self._make_key(key))
            return result
        except Exception as e:
            logger.error(f"Redis PERSIST error for key {key}: {e}")
            return False

    async def set_many(
        self,
        mapping: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set multiple values at once using pipeline."""
        self._ensure_connected()

        try:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            pipe = self._client.pipeline()

            for key, value in mapping.items():
                data = self._serializer.serialize(value)
                prefixed_key = self._make_key(key)

                if effective_ttl is not None:
                    pipe.setex(prefixed_key, effective_ttl, data)
                else:
                    pipe.set(prefixed_key, data)

            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis SET_MANY error: {e}")
            return False

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values at once using mget."""
        self._ensure_connected()

        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(prefixed_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._serializer.deserialize(value)

            return result
        except Exception as e:
            logger.error(f"Redis GET_MANY error: {e}")
            return {}

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys at once."""
        self._ensure_connected()

        try:
            if not keys:
                return 0
            prefixed_keys = [self._make_key(k) for k in keys]
            return await self._client.delete(*prefixed_keys)
        except Exception as e:
            logger.error(f"Redis DELETE_MANY error: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Check Redis connection health."""
        try:
            if self._client is None:
                return {
                    "status": "disconnected",
                    "connected": False,
                }

            # Ping Redis
            start = asyncio.get_event_loop().time()
            await self._client.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000

            # Get info
            info = await self._client.info("server")

            return {
                "status": "healthy",
                "connected": True,
                "latency_ms": round(latency, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "url": self._url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "redis",
            "url": self._url,
            "db": self._db,
            "key_prefix": self._key_prefix,
            "default_ttl": self._default_ttl,
            "connected": self._client is not None,
        }
