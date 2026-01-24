"""Abstract cache interface for JAVIS.

Provides a unified caching API that can be backed by different
storage implementations (in-memory, Redis, etc.).
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheInterface(ABC):
    """Abstract base class for cache implementations.

    All cache implementations should inherit from this class
    and implement its abstract methods.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (None = no expiration)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all keys from the cache.

        Returns:
            Number of keys cleared
        """
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard)

        Returns:
            List of matching keys
        """
        pass

    # Convenience methods with default implementations

    async def get_or_set(
        self,
        key: str,
        default_factory: Any,
        ttl: Optional[int] = None,
    ) -> Any:
        """Get a value, setting it if not found.

        Args:
            key: Cache key
            default_factory: Callable to create default value
            ttl: Time-to-live for new values

        Returns:
            Cached or newly created value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Generate new value
        if callable(default_factory):
            value = default_factory()
        else:
            value = default_factory

        await self.set(key, value, ttl)
        return value

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment
        """
        value = await self.get(key)
        new_value = (int(value) if value is not None else 0) + amount
        await self.set(key, new_value)
        return new_value

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value.

        Args:
            key: Cache key
            amount: Amount to decrement by

        Returns:
            New value after decrement
        """
        return await self.increment(key, -amount)

    async def set_many(
        self,
        mapping: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set multiple values at once.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live for all values

        Returns:
            True if all successful
        """
        success = True
        for key, value in mapping.items():
            if not await self.set(key, value, ttl):
                success = False
        return success

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values at once.

        Args:
            keys: List of keys to get

        Returns:
            Dictionary of key-value pairs (missing keys omitted)
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys at once.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys deleted
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count


class CacheSerializer:
    """Handles serialization/deserialization of cache values."""

    @staticmethod
    def serialize(value: Any) -> str:
        """Serialize a value to string."""
        return json.dumps(value, default=str)

    @staticmethod
    def deserialize(data: str) -> Any:
        """Deserialize a string to value."""
        return json.loads(data)
