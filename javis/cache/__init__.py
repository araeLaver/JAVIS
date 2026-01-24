"""Cache module for JAVIS.

Provides a unified caching API with multiple backend support:
- InMemoryCache: Thread-safe in-memory cache for development/single instance
- RedisCache: Distributed cache for production deployments

Example:
    from javis.cache import get_cache, CacheInterface

    cache = get_cache()
    await cache.set("key", {"data": "value"}, ttl=300)
    value = await cache.get("key")
"""

import logging
from typing import Optional

from javis.cache.base import CacheInterface, CacheSerializer
from javis.cache.memory import InMemoryCache, CacheEntry

logger = logging.getLogger(__name__)

# Global cache instance
_cache: Optional[CacheInterface] = None


def get_cache(
    backend: Optional[str] = None,
    **kwargs,
) -> CacheInterface:
    """Get or create the global cache instance.

    Args:
        backend: Cache backend to use ("memory" or "redis")
                If None, auto-detect from config/environment
        **kwargs: Additional arguments passed to cache constructor

    Returns:
        Cache instance

    Example:
        # Auto-detect backend
        cache = get_cache()

        # Force memory backend
        cache = get_cache("memory", max_size=1000)

        # Force Redis backend
        cache = get_cache("redis", url="redis://localhost:6379")
    """
    global _cache

    # Return existing cache if available and no specific backend/kwargs requested
    if _cache is not None and not kwargs:
        # If backend matches or no backend specified, return existing
        if backend is None:
            return _cache
        # Check if existing cache matches requested backend
        existing_type = "redis" if hasattr(_cache, "_client") else "memory"
        if existing_type == backend:
            return _cache

    cache = create_cache(backend, **kwargs)
    _cache = cache

    return cache


def create_cache(
    backend: Optional[str] = None,
    **kwargs,
) -> CacheInterface:
    """Create a new cache instance.

    Args:
        backend: Cache backend to use ("memory" or "redis")
                If None, auto-detect from config/environment
        **kwargs: Additional arguments passed to cache constructor

    Returns:
        New cache instance
    """
    if backend is None:
        backend = _detect_backend()

    if backend == "redis":
        return _create_redis_cache(**kwargs)
    else:
        return _create_memory_cache(**kwargs)


def _detect_backend() -> str:
    """Auto-detect which cache backend to use.

    Priority:
    1. JAVIS_CACHE_BACKEND environment variable
    2. Config setting
    3. Default to "memory"
    """
    import os

    # Check environment variable
    env_backend = os.getenv("JAVIS_CACHE_BACKEND", "").lower()
    if env_backend in ("redis", "memory"):
        return env_backend

    # Check config
    try:
        from javis.utils.config import get_config
        config = get_config()

        # Check if Redis URL is configured
        redis_url = getattr(config, "redis_url", None)
        if redis_url:
            return "redis"

        # Check cache config
        cache_config = getattr(config, "cache", None)
        if cache_config:
            backend_setting = getattr(cache_config, "backend", None)
            if backend_setting in ("redis", "memory"):
                return backend_setting

    except Exception:
        pass

    return "memory"


def _create_memory_cache(**kwargs) -> InMemoryCache:
    """Create in-memory cache instance."""
    # Apply defaults from config if available
    try:
        from javis.utils.config import get_config
        config = get_config()
        cache_config = getattr(config, "cache", None)

        if cache_config:
            kwargs.setdefault("default_ttl", getattr(cache_config, "default_ttl", None))
            kwargs.setdefault("max_size", getattr(cache_config, "max_size", None))

    except Exception:
        pass

    logger.info("Creating in-memory cache")
    return InMemoryCache(**kwargs)


def _create_redis_cache(**kwargs) -> "RedisCache":
    """Create Redis cache instance."""
    from javis.cache.redis import RedisCache

    # Apply defaults from config if available
    try:
        from javis.utils.config import get_config
        import os

        config = get_config()

        # Get Redis URL from environment or config
        redis_url = os.getenv("REDIS_URL") or os.getenv("JAVIS_REDIS_URL")
        if not redis_url:
            redis_url = getattr(config, "redis_url", None)
        if redis_url:
            kwargs.setdefault("url", redis_url)

        # Get cache-specific config
        cache_config = getattr(config, "cache", None)
        if cache_config:
            kwargs.setdefault("default_ttl", getattr(cache_config, "default_ttl", None))
            kwargs.setdefault("key_prefix", getattr(cache_config, "key_prefix", "javis:"))

    except Exception:
        pass

    logger.info(f"Creating Redis cache with URL: {kwargs.get('url', 'redis://localhost:6379')}")
    return RedisCache(**kwargs)


def reset_cache() -> None:
    """Reset global cache instance."""
    global _cache
    _cache = None
    logger.debug("Cache instance reset")


async def close_cache() -> None:
    """Close and cleanup global cache instance."""
    global _cache

    if _cache is not None:
        # Close Redis connection if applicable
        if hasattr(_cache, "close"):
            await _cache.close()
        _cache = None
        logger.info("Cache closed")


__all__ = [
    "CacheInterface",
    "CacheSerializer",
    "CacheEntry",
    "InMemoryCache",
    "RedisCache",
    "get_cache",
    "create_cache",
    "reset_cache",
    "close_cache",
]


# Lazy import for RedisCache to avoid requiring redis package
def __getattr__(name: str):
    if name == "RedisCache":
        from javis.cache.redis import RedisCache
        return RedisCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
