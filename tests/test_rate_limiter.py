"""Tests for rate limiting middleware."""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from javis.utils.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitRule,
    RateLimitMiddleware,
    RATE_LIMIT_PROFILES,
    create_default_rules,
    get_rate_limit_middleware,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests == 100
        assert config.window_seconds == 60
        assert config.burst == 100  # Defaults to requests

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(requests=50, window_seconds=30, burst=10)
        assert config.requests == 50
        assert config.window_seconds == 30
        assert config.burst == 10

    def test_burst_defaults_to_requests(self):
        """Test burst defaults to requests when not specified."""
        config = RateLimitConfig(requests=25, window_seconds=60)
        assert config.burst == 25


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        for _ in range(10):
            allowed, info = await limiter.is_allowed("test-key")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(requests_per_window=5, window_seconds=60, burst=5)

        # Use all tokens
        for _ in range(5):
            allowed, _ = await limiter.is_allowed("test-key")
            assert allowed is True

        # Next request should be blocked
        allowed, info = await limiter.is_allowed("test-key")
        assert allowed is False
        assert info["remaining"] == 0

    @pytest.mark.asyncio
    async def test_rate_limit_info(self):
        """Test rate limit info in response."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        allowed, info = await limiter.is_allowed("test-key")

        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info
        assert "window" in info
        assert info["limit"] == 10
        assert info["window"] == 60

    @pytest.mark.asyncio
    async def test_separate_keys(self):
        """Test that different keys have separate limits."""
        limiter = RateLimiter(requests_per_window=2, window_seconds=60, burst=2)

        # User 1 uses their limit
        for _ in range(2):
            allowed, _ = await limiter.is_allowed("user1")
            assert allowed is True

        # User 1 is blocked
        allowed, _ = await limiter.is_allowed("user1")
        assert allowed is False

        # User 2 still has their limit
        allowed, _ = await limiter.is_allowed("user2")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=1, burst=10)

        # Use all tokens
        for _ in range(10):
            await limiter.is_allowed("test-key")

        # Should be blocked
        allowed, _ = await limiter.is_allowed("test-key")
        assert allowed is False

        # Wait for refill
        await asyncio.sleep(0.2)  # Wait for partial refill

        # Should have some tokens now
        allowed, _ = await limiter.is_allowed("test-key")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_state(self):
        """Test getting rate limit state."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60, burst=15)

        state = await limiter.get_state("test-key")

        assert "tokens" in state
        assert "limit" in state
        assert "window" in state
        assert "burst" in state
        assert state["limit"] == 10
        assert state["burst"] == 15

    @pytest.mark.asyncio
    async def test_reset_key(self):
        """Test resetting rate limit for a key."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        # Create state for key
        await limiter.is_allowed("test-key")

        # Reset
        limiter.reset("test-key")

        # Verify key is removed
        assert "test-key" not in limiter._states

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all rate limits."""
        limiter = RateLimiter(requests_per_window=10, window_seconds=60)

        # Create states for multiple keys
        await limiter.is_allowed("key1")
        await limiter.is_allowed("key2")

        # Reset all
        limiter.reset_all()

        # Verify all keys removed
        assert len(limiter._states) == 0


class TestRateLimitRule:
    """Tests for RateLimitRule class."""

    def test_prefix_match(self):
        """Test prefix pattern matching."""
        rule = RateLimitRule("/api/", RateLimitConfig())

        assert rule.matches("/api/users", "GET") is True
        assert rule.matches("/api/chat", "POST") is True
        assert rule.matches("/health", "GET") is False

    def test_exact_match(self):
        """Test exact pattern matching with $ suffix."""
        rule = RateLimitRule("/api/chat$", RateLimitConfig())

        assert rule.matches("/api/chat", "GET") is True
        assert rule.matches("/api/chat/", "GET") is True
        assert rule.matches("/api/chat/history", "GET") is False

    def test_method_filter(self):
        """Test method filtering."""
        rule = RateLimitRule("/api/", RateLimitConfig(), methods=["POST", "PUT"])

        assert rule.matches("/api/users", "POST") is True
        assert rule.matches("/api/users", "PUT") is True
        assert rule.matches("/api/users", "GET") is False
        assert rule.matches("/api/users", "DELETE") is False

    def test_case_insensitive_method(self):
        """Test that method matching is case insensitive."""
        rule = RateLimitRule("/api/", RateLimitConfig(), methods=["post"])

        assert rule.matches("/api/users", "POST") is True
        assert rule.matches("/api/users", "post") is True


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create a FastAPI app with rate limiting."""
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        app.add_middleware(
            RateLimitMiddleware,
            default_limit=RateLimitConfig(requests=5, window_seconds=60, burst=5),
            rules=[],
            enabled=True,
        )

        return app

    def test_allows_requests_within_limit(self, app_with_middleware):
        """Test that requests within limit are allowed."""
        client = TestClient(app_with_middleware)

        for _ in range(5):
            response = client.get("/api/test")
            assert response.status_code == 200

    def test_blocks_requests_over_limit(self, app_with_middleware):
        """Test that requests over limit return 429."""
        client = TestClient(app_with_middleware)

        # Use all tokens
        for _ in range(5):
            client.get("/api/test")

        # Next request should be blocked
        response = client.get("/api/test")
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    def test_rate_limit_headers(self, app_with_middleware):
        """Test that rate limit headers are included."""
        client = TestClient(app_with_middleware)

        response = client.get("/api/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert "X-RateLimit-Window" in response.headers

    def test_retry_after_header_on_429(self, app_with_middleware):
        """Test that Retry-After header is included on 429."""
        client = TestClient(app_with_middleware)

        # Exhaust limit
        for _ in range(5):
            client.get("/api/test")

        response = client.get("/api/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_health_endpoints_bypass(self, app_with_middleware):
        """Test that health endpoints bypass rate limiting."""
        client = TestClient(app_with_middleware)

        # Make many health requests
        for _ in range(20):
            response = client.get("/health")
            assert response.status_code == 200

    def test_disabled_middleware(self):
        """Test that disabled middleware passes all requests."""
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(
            RateLimitMiddleware,
            default_limit=RateLimitConfig(requests=1, window_seconds=60, burst=1),
            enabled=False,  # Disabled
        )

        client = TestClient(app)

        # Should allow unlimited requests
        for _ in range(10):
            response = client.get("/api/test")
            assert response.status_code == 200

    def test_whitelist(self):
        """Test that whitelisted IPs bypass rate limiting."""
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(
            RateLimitMiddleware,
            default_limit=RateLimitConfig(requests=1, window_seconds=60, burst=1),
            whitelist=["testclient"],  # TestClient uses "testclient" as host
            enabled=True,
        )

        client = TestClient(app)

        # Should allow unlimited requests due to whitelist
        for _ in range(10):
            response = client.get("/api/test")
            assert response.status_code == 200

    def test_x_forwarded_for_header(self):
        """Test that X-Forwarded-For header is used for IP detection."""
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(
            RateLimitMiddleware,
            default_limit=RateLimitConfig(requests=2, window_seconds=60, burst=2),
            enabled=True,
        )

        client = TestClient(app)

        # First IP uses its limit
        for _ in range(2):
            response = client.get(
                "/api/test", headers={"X-Forwarded-For": "1.2.3.4"}
            )
            assert response.status_code == 200

        # First IP is blocked
        response = client.get("/api/test", headers={"X-Forwarded-For": "1.2.3.4"})
        assert response.status_code == 429

        # Second IP still works
        response = client.get("/api/test", headers={"X-Forwarded-For": "5.6.7.8"})
        assert response.status_code == 200


class TestRateLimitProfiles:
    """Tests for pre-configured rate limit profiles."""

    def test_profiles_exist(self):
        """Test that expected profiles exist."""
        expected_profiles = ["strict", "standard", "relaxed", "chat", "upload", "auth"]

        for profile in expected_profiles:
            assert profile in RATE_LIMIT_PROFILES
            assert isinstance(RATE_LIMIT_PROFILES[profile], RateLimitConfig)

    def test_strict_profile(self):
        """Test strict profile configuration."""
        config = RATE_LIMIT_PROFILES["strict"]
        assert config.requests == 30
        assert config.burst == 5

    def test_standard_profile(self):
        """Test standard profile configuration."""
        config = RATE_LIMIT_PROFILES["standard"]
        assert config.requests == 100
        assert config.burst == 20

    def test_auth_profile(self):
        """Test auth profile has longer window for brute force protection."""
        config = RATE_LIMIT_PROFILES["auth"]
        assert config.window_seconds == 300  # 5 minutes


class TestDefaultRules:
    """Tests for default rate limit rules."""

    def test_default_rules_created(self):
        """Test that default rules are created."""
        rules = create_default_rules()
        assert len(rules) > 0
        assert all(isinstance(r, RateLimitRule) for r in rules)

    def test_auth_rule_exists(self):
        """Test that auth endpoints have strict limits."""
        rules = create_default_rules()
        auth_rule = next((r for r in rules if "/auth" in r.pattern), None)
        assert auth_rule is not None
        assert auth_rule.config.window_seconds == 300  # Auth profile

    def test_chat_rule_exists(self):
        """Test that chat endpoint has specific limits."""
        rules = create_default_rules()
        chat_rule = next((r for r in rules if "chat" in r.pattern), None)
        assert chat_rule is not None

    def test_upload_rule_exists(self):
        """Test that upload endpoint has strict limits."""
        rules = create_default_rules()
        upload_rule = next((r for r in rules if "upload" in r.pattern), None)
        assert upload_rule is not None


class TestGetRateLimitMiddleware:
    """Tests for get_rate_limit_middleware factory function."""

    def test_creates_middleware(self):
        """Test that factory creates middleware correctly."""
        app = FastAPI()
        middleware = get_rate_limit_middleware(app, enabled=True)

        assert isinstance(middleware, RateLimitMiddleware)
        assert middleware.enabled is True

    def test_with_custom_rules(self):
        """Test factory with custom rules."""
        app = FastAPI()
        custom_rules = [
            RateLimitRule("/custom/", RateLimitConfig(requests=50, window_seconds=30))
        ]
        middleware = get_rate_limit_middleware(app, custom_rules=custom_rules)

        assert len(middleware.rules) == 1
        assert middleware.rules[0].pattern == "/custom/"

    def test_with_whitelist(self):
        """Test factory with whitelist."""
        app = FastAPI()
        middleware = get_rate_limit_middleware(
            app, whitelist=["127.0.0.1", "192.168.1.1"]
        )

        assert "127.0.0.1" in middleware.whitelist
        assert "192.168.1.1" in middleware.whitelist
