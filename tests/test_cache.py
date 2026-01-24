"""Tests for cache module."""

import asyncio
import pytest
import threading
import time
from unittest.mock import MagicMock, patch, AsyncMock

from javis.cache.base import CacheInterface, CacheSerializer
from javis.cache.memory import InMemoryCache, CacheEntry
from javis.cache import (
    get_cache,
    create_cache,
    reset_cache,
    close_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired_without_expiration(self):
        """Test entry without expiration is never expired."""
        entry = CacheEntry(value="test", expires_at=None)
        assert entry.is_expired() is False

    def test_not_expired_future_time(self):
        """Test entry with future expiration is not expired."""
        entry = CacheEntry(value="test", expires_at=time.time() + 3600)
        assert entry.is_expired() is False

    def test_expired_past_time(self):
        """Test entry with past expiration is expired."""
        entry = CacheEntry(value="test", expires_at=time.time() - 1)
        assert entry.is_expired() is True


class TestCacheSerializer:
    """Tests for CacheSerializer."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        data = {"name": "test", "count": 42}
        result = CacheSerializer.serialize(data)
        assert isinstance(result, str)
        assert "test" in result
        assert "42" in result

    def test_deserialize_dict(self):
        """Test deserializing a dictionary."""
        data = '{"name": "test", "count": 42}'
        result = CacheSerializer.deserialize(data)
        assert result == {"name": "test", "count": 42}

    def test_serialize_list(self):
        """Test serializing a list."""
        data = [1, 2, 3, "test"]
        result = CacheSerializer.serialize(data)
        deserialized = CacheSerializer.deserialize(result)
        assert deserialized == data

    def test_serialize_datetime_to_string(self):
        """Test datetime is serialized to string."""
        from datetime import datetime
        data = {"time": datetime(2024, 1, 1, 12, 0, 0)}
        result = CacheSerializer.serialize(data)
        assert "2024" in result


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance."""
        return InMemoryCache()

    @pytest.fixture
    def cache_with_ttl(self):
        """Create a cache with default TTL."""
        return InMemoryCache(default_ttl=60)

    @pytest.fixture
    def cache_with_max_size(self):
        """Create a cache with max size."""
        return InMemoryCache(max_size=3)

    # Basic operations

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get."""
        await cache.set("key", "value")
        result = await cache.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_dict(self, cache):
        """Test setting and getting a dictionary."""
        data = {"name": "test", "items": [1, 2, 3]}
        await cache.set("data", data)
        result = await cache.get("data")
        assert result == data

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting a key."""
        await cache.set("key", "value")
        result = await cache.delete("key")
        assert result is True
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Test deleting nonexistent key."""
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test checking key existence."""
        await cache.set("key", "value")
        assert await cache.exists("key") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing all keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        count = await cache.clear()
        assert count == 2
        assert await cache.get("key1") is None

    # TTL operations

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache):
        """Test setting a key with TTL."""
        await cache.set("key", "value", ttl=1)
        assert await cache.get("key") == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_default_ttl(self, cache_with_ttl):
        """Test default TTL is applied."""
        await cache_with_ttl.set("key", "value")
        ttl = await cache_with_ttl.ttl("key")
        assert ttl is not None
        assert ttl > 50  # Should be around 60

    @pytest.mark.asyncio
    async def test_ttl_method(self, cache):
        """Test getting TTL of a key."""
        await cache.set("key", "value", ttl=100)
        ttl = await cache.ttl("key")
        assert ttl is not None
        assert 95 <= ttl <= 100

    @pytest.mark.asyncio
    async def test_ttl_no_expiration(self, cache):
        """Test TTL for key without expiration."""
        await cache.set("key", "value")
        ttl = await cache.ttl("key")
        assert ttl is None

    @pytest.mark.asyncio
    async def test_ttl_nonexistent(self, cache):
        """Test TTL for nonexistent key."""
        ttl = await cache.ttl("nonexistent")
        assert ttl == -1

    @pytest.mark.asyncio
    async def test_expire(self, cache):
        """Test setting expiration on existing key."""
        await cache.set("key", "value")
        result = await cache.expire("key", 100)
        assert result is True
        ttl = await cache.ttl("key")
        assert ttl is not None
        assert 95 <= ttl <= 100

    @pytest.mark.asyncio
    async def test_persist(self, cache):
        """Test removing expiration from key."""
        await cache.set("key", "value", ttl=100)
        result = await cache.persist("key")
        assert result is True
        ttl = await cache.ttl("key")
        assert ttl is None

    # Max size operations

    @pytest.mark.asyncio
    async def test_max_size_eviction(self, cache_with_max_size):
        """Test eviction when max size is reached."""
        await cache_with_max_size.set("key1", "value1")
        await cache_with_max_size.set("key2", "value2")
        await cache_with_max_size.set("key3", "value3")
        await cache_with_max_size.set("key4", "value4")

        # Oldest key should be evicted
        assert await cache_with_max_size.get("key1") is None
        assert await cache_with_max_size.get("key4") == "value4"

    # Keys operation

    @pytest.mark.asyncio
    async def test_keys_all(self, cache):
        """Test getting all keys."""
        await cache.set("user:1", "a")
        await cache.set("user:2", "b")
        await cache.set("session:1", "c")

        keys = await cache.keys()
        assert len(keys) == 3
        assert "user:1" in keys

    @pytest.mark.asyncio
    async def test_keys_pattern(self, cache):
        """Test getting keys with pattern."""
        await cache.set("user:1", "a")
        await cache.set("user:2", "b")
        await cache.set("session:1", "c")

        keys = await cache.keys("user:*")
        assert len(keys) == 2
        assert "user:1" in keys
        assert "session:1" not in keys

    # Increment/decrement

    @pytest.mark.asyncio
    async def test_increment(self, cache):
        """Test incrementing a value."""
        result = await cache.increment("counter")
        assert result == 1

        result = await cache.increment("counter")
        assert result == 2

        result = await cache.increment("counter", 5)
        assert result == 7

    @pytest.mark.asyncio
    async def test_decrement(self, cache):
        """Test decrementing a value."""
        await cache.set("counter", 10)
        result = await cache.decrement("counter")
        assert result == 9

        result = await cache.decrement("counter", 5)
        assert result == 4

    # Batch operations

    @pytest.mark.asyncio
    async def test_set_many(self, cache):
        """Test setting multiple values."""
        data = {"key1": "value1", "key2": "value2"}
        result = await cache.set_many(data)
        assert result is True

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_get_many(self, cache):
        """Test getting multiple values."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        result = await cache.get_many(["key1", "key2", "nonexistent"])
        assert result == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_delete_many(self, cache):
        """Test deleting multiple keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        count = await cache.delete_many(["key1", "key2"])
        assert count == 2
        assert await cache.exists("key1") is False
        assert await cache.exists("key3") is True

    # Get or set

    @pytest.mark.asyncio
    async def test_get_or_set_existing(self, cache):
        """Test get_or_set with existing key."""
        await cache.set("key", "original")
        result = await cache.get_or_set("key", "new")
        assert result == "original"

    @pytest.mark.asyncio
    async def test_get_or_set_new(self, cache):
        """Test get_or_set with new key."""
        result = await cache.get_or_set("key", "value")
        assert result == "value"
        assert await cache.get("key") == "value"

    @pytest.mark.asyncio
    async def test_get_or_set_factory(self, cache):
        """Test get_or_set with factory function."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        result1 = await cache.get_or_set("key", factory)
        assert result1 == "value_1"
        assert call_count == 1

        # Second call should return cached value
        result2 = await cache.get_or_set("key", factory)
        assert result2 == "value_1"
        assert call_count == 1  # Factory not called

    # Properties and stats

    def test_size_property(self):
        """Test size property."""
        cache = InMemoryCache()
        assert cache.size == 0

        asyncio.run(cache.set("key", "value"))
        assert cache.size == 1

    def test_stats(self):
        """Test stats method."""
        cache = InMemoryCache(default_ttl=300, max_size=100)
        stats = cache.stats()

        assert stats["type"] == "memory"
        assert stats["size"] == 0
        assert stats["default_ttl"] == 300
        assert stats["max_size"] == 100

    # Thread safety

    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache = InMemoryCache()
        errors = []
        results = []

        def worker(n):
            try:
                for i in range(10):
                    key = f"key_{n}_{i}"
                    asyncio.run(cache.set(key, f"value_{n}_{i}"))
                    value = asyncio.run(cache.get(key))
                    results.append(value is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    # Cleanup

    @pytest.mark.asyncio
    async def test_expired_cleanup_on_access(self):
        """Test expired entries are cleaned up on access."""
        cache = InMemoryCache(cleanup_interval=1)

        await cache.set("expire_soon", "value", ttl=1)
        await cache.set("keep", "value")

        assert cache.size == 2

        await asyncio.sleep(1.5)

        # Trigger cleanup
        await cache.get("keep")

        # Expired entry should be removed
        assert await cache.get("expire_soon") is None


class TestCacheFactory:
    """Tests for cache factory functions."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def teardown_method(self):
        """Reset cache after each test."""
        reset_cache()

    def test_get_cache_memory(self):
        """Test getting memory cache."""
        cache = get_cache("memory")
        assert isinstance(cache, InMemoryCache)

    def test_get_cache_singleton(self):
        """Test cache singleton behavior."""
        cache1 = get_cache("memory")
        cache2 = get_cache()

        assert cache1 is cache2

    def test_create_cache_memory(self):
        """Test creating memory cache."""
        cache = create_cache("memory")
        assert isinstance(cache, InMemoryCache)

    def test_create_cache_with_options(self):
        """Test creating cache with options."""
        cache = create_cache("memory", default_ttl=300, max_size=100)
        assert isinstance(cache, InMemoryCache)
        assert cache._default_ttl == 300
        assert cache._max_size == 100

    @patch.dict("os.environ", {"JAVIS_CACHE_BACKEND": "memory"})
    def test_detect_backend_from_env(self):
        """Test backend detection from environment."""
        reset_cache()
        cache = get_cache()
        assert isinstance(cache, InMemoryCache)

    def test_reset_cache(self):
        """Test resetting cache."""
        cache1 = get_cache("memory")
        reset_cache()
        cache2 = get_cache("memory")

        assert cache1 is not cache2

    @pytest.mark.asyncio
    async def test_close_cache(self):
        """Test closing cache."""
        get_cache("memory")
        await close_cache()
        # Should be able to get new cache
        cache = get_cache("memory")
        assert cache is not None


class TestRedisCache:
    """Tests for RedisCache (mocked)."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.setex = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=1)
        mock.incrby = AsyncMock(return_value=1)
        mock.ttl = AsyncMock(return_value=100)
        mock.expire = AsyncMock(return_value=True)
        mock.persist = AsyncMock(return_value=True)
        mock.close = AsyncMock()
        mock.info = AsyncMock(return_value={"redis_version": "7.0.0"})

        async def scan_iter(match=None):
            if False:
                yield
        mock.scan_iter = scan_iter

        mock.pipeline = MagicMock()
        mock.pipeline.return_value.execute = AsyncMock(return_value=[])
        mock.mget = AsyncMock(return_value=[])

        return mock

    @pytest.fixture
    def redis_cache(self, mock_redis):
        """Create a Redis cache with mocked client."""
        from javis.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379")
        cache._client = mock_redis
        return cache

    @pytest.mark.asyncio
    async def test_get(self, redis_cache, mock_redis):
        """Test Redis get."""
        mock_redis.get.return_value = '{"name": "test"}'
        result = await redis_cache.get("key")
        assert result == {"name": "test"}
        mock_redis.get.assert_called_with("javis:key")

    @pytest.mark.asyncio
    async def test_set(self, redis_cache, mock_redis):
        """Test Redis set."""
        result = await redis_cache.set("key", {"name": "test"})
        assert result is True
        mock_redis.set.assert_called()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, redis_cache, mock_redis):
        """Test Redis set with TTL."""
        result = await redis_cache.set("key", "value", ttl=300)
        assert result is True
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio
    async def test_delete(self, redis_cache, mock_redis):
        """Test Redis delete."""
        result = await redis_cache.delete("key")
        assert result is True
        mock_redis.delete.assert_called_with("javis:key")

    @pytest.mark.asyncio
    async def test_exists(self, redis_cache, mock_redis):
        """Test Redis exists."""
        result = await redis_cache.exists("key")
        assert result is True
        mock_redis.exists.assert_called_with("javis:key")

    @pytest.mark.asyncio
    async def test_increment(self, redis_cache, mock_redis):
        """Test Redis increment."""
        mock_redis.incrby.return_value = 5
        result = await redis_cache.increment("counter", 5)
        assert result == 5
        mock_redis.incrby.assert_called_with("javis:counter", 5)

    @pytest.mark.asyncio
    async def test_ttl(self, redis_cache, mock_redis):
        """Test Redis TTL."""
        mock_redis.ttl.return_value = 100
        result = await redis_cache.ttl("key")
        assert result == 100

    @pytest.mark.asyncio
    async def test_expire(self, redis_cache, mock_redis):
        """Test Redis expire."""
        result = await redis_cache.expire("key", 300)
        assert result is True
        mock_redis.expire.assert_called_with("javis:key", 300)

    @pytest.mark.asyncio
    async def test_persist(self, redis_cache, mock_redis):
        """Test Redis persist."""
        result = await redis_cache.persist("key")
        assert result is True
        mock_redis.persist.assert_called_with("javis:key")

    @pytest.mark.asyncio
    async def test_health_check(self, redis_cache, mock_redis):
        """Test Redis health check."""
        result = await redis_cache.health_check()
        assert result["status"] == "healthy"
        assert result["connected"] is True

    def test_stats(self, redis_cache):
        """Test Redis stats."""
        stats = redis_cache.stats()
        assert stats["type"] == "redis"
        assert stats["connected"] is True

    @pytest.mark.asyncio
    async def test_not_connected_error(self):
        """Test error when not connected."""
        from javis.cache.redis import RedisCache

        cache = RedisCache()
        with pytest.raises(RuntimeError, match="not connected"):
            await cache.get("key")

    @pytest.mark.asyncio
    async def test_close(self, redis_cache, mock_redis):
        """Test Redis close."""
        await redis_cache.close()
        mock_redis.close.assert_called_once()
        assert redis_cache._client is None


class TestCacheInterface:
    """Tests for CacheInterface contract."""

    @pytest.mark.asyncio
    async def test_memory_implements_interface(self):
        """Test InMemoryCache implements CacheInterface."""
        cache = InMemoryCache()

        # All abstract methods should work
        await cache.set("key", "value")
        assert await cache.get("key") == "value"
        assert await cache.exists("key") is True
        assert await cache.delete("key") is True
        assert await cache.keys() == []
        assert await cache.clear() == 0

    @pytest.mark.asyncio
    async def test_interface_type(self):
        """Test cache is instance of CacheInterface."""
        cache = InMemoryCache()
        assert isinstance(cache, CacheInterface)
