"""Rate limiting middleware for JAVIS API.

Provides configurable rate limiting with support for:
- Per-IP and per-user rate limiting
- Different limits for different endpoint patterns
- Sliding window algorithm
- Standard rate limit headers
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""

    requests: int = 100  # Number of requests allowed
    window_seconds: int = 60  # Time window in seconds
    burst: Optional[int] = None  # Optional burst limit (allows short bursts)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests


@dataclass
class RateLimitState:
    """Tracks rate limit state for a single key."""

    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)


class RateLimiter:
    """Token bucket rate limiter with sliding window.

    Uses token bucket algorithm for smooth rate limiting with burst support.
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        burst: Optional[int] = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_window: Number of requests allowed per window
            window_seconds: Time window in seconds
            burst: Maximum burst size (defaults to requests_per_window)
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst = burst or requests_per_window
        self.refill_rate = requests_per_window / window_seconds

        # State storage: key -> RateLimitState
        self._states: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=self.burst, last_update=time.time())
        )
        self._lock = asyncio.Lock()

    def _refill_tokens(self, state: RateLimitState, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - state.last_update
        state.tokens = min(self.burst, state.tokens + elapsed * self.refill_rate)
        state.last_update = now

    async def is_allowed(self, key: str) -> tuple[bool, dict[str, Any]]:
        """Check if a request is allowed for the given key.

        Args:
            key: Unique identifier (IP, user ID, etc.)

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._lock:
            now = time.time()
            state = self._states[key]

            # Refill tokens
            self._refill_tokens(state, now)

            # Calculate info for headers
            remaining = int(state.tokens)
            reset_seconds = int(
                (self.burst - state.tokens) / self.refill_rate
            ) if state.tokens < self.burst else 0

            info = {
                "limit": self.requests_per_window,
                "remaining": max(0, remaining - 1) if remaining > 0 else 0,
                "reset": reset_seconds,
                "window": self.window_seconds,
            }

            # Check if allowed
            if state.tokens >= 1:
                state.tokens -= 1
                return True, info

            return False, info

    async def get_state(self, key: str) -> dict[str, Any]:
        """Get current rate limit state for a key."""
        async with self._lock:
            now = time.time()
            state = self._states[key]
            self._refill_tokens(state, now)

            return {
                "tokens": state.tokens,
                "limit": self.requests_per_window,
                "window": self.window_seconds,
                "burst": self.burst,
            }

    def reset(self, key: str) -> None:
        """Reset rate limit state for a key."""
        if key in self._states:
            del self._states[key]

    def reset_all(self) -> None:
        """Reset all rate limit states."""
        self._states.clear()


class RateLimitRule:
    """A rate limit rule with path pattern matching."""

    def __init__(
        self,
        pattern: str,
        config: RateLimitConfig,
        methods: Optional[list[str]] = None,
    ):
        """Initialize rate limit rule.

        Args:
            pattern: Path pattern (prefix match or exact match with $)
            config: Rate limit configuration
            methods: HTTP methods to match (None = all methods)
        """
        self.pattern = pattern
        self.config = config
        self.methods = [m.upper() for m in methods] if methods else None
        self.exact_match = pattern.endswith("$")
        if self.exact_match:
            self.pattern = pattern[:-1]

        self.limiter = RateLimiter(
            requests_per_window=config.requests,
            window_seconds=config.window_seconds,
            burst=config.burst,
        )

    def matches(self, path: str, method: str) -> bool:
        """Check if this rule matches the request."""
        # Check method
        if self.methods and method.upper() not in self.methods:
            return False

        # Check path
        if self.exact_match:
            return path == self.pattern or path == self.pattern + "/"
        return path.startswith(self.pattern)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting.

    Features:
    - Per-IP rate limiting by default
    - Per-user rate limiting when authenticated
    - Configurable rules for different endpoints
    - Standard rate limit headers
    - Whitelist support
    """

    def __init__(
        self,
        app: ASGIApp,
        default_limit: Optional[RateLimitConfig] = None,
        rules: Optional[list[RateLimitRule]] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        whitelist: Optional[list[str]] = None,
        enabled: bool = True,
    ):
        """Initialize rate limit middleware.

        Args:
            app: ASGI application
            default_limit: Default rate limit configuration
            rules: List of rate limit rules (checked in order)
            key_func: Function to extract rate limit key from request
            whitelist: List of IPs or keys to whitelist
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.default_config = default_limit or RateLimitConfig()
        self.default_limiter = RateLimiter(
            requests_per_window=self.default_config.requests,
            window_seconds=self.default_config.window_seconds,
            burst=self.default_config.burst,
        )
        self.rules = rules or []
        self.key_func = key_func or self._default_key_func
        self.whitelist = set(whitelist or [])
        self.enabled = enabled

    @staticmethod
    def _default_key_func(request: Request) -> str:
        """Default key function using client IP."""
        # Try to get real IP from forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _find_rule(self, path: str, method: str) -> Optional[RateLimitRule]:
        """Find the first matching rule for the request."""
        for rule in self.rules:
            if rule.matches(path, method):
                return rule
        return None

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip health check endpoints
        if request.url.path in ["/health", "/api/health/live", "/api/health/ready"]:
            return await call_next(request)

        # Get rate limit key
        key = self.key_func(request)

        # Check whitelist
        if key in self.whitelist:
            return await call_next(request)

        # Find matching rule or use default
        rule = self._find_rule(request.url.path, request.method)
        limiter = rule.limiter if rule else self.default_limiter

        # Check rate limit
        is_allowed, info = await limiter.is_allowed(key)

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(info["limit"]),
            "X-RateLimit-Remaining": str(info["remaining"]),
            "X-RateLimit-Reset": str(info["reset"]),
            "X-RateLimit-Window": str(info["window"]),
        }

        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {key} on {request.url.path}",
                extra={
                    "client_ip": key,
                    "path": request.url.path,
                    "method": request.method,
                    "limit": info["limit"],
                },
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                        "details": {
                            "retry_after": info["reset"],
                            "limit": info["limit"],
                            "window": info["window"],
                        },
                    }
                },
                headers={**headers, "Retry-After": str(info["reset"])},
            )

        # Process request
        response = await call_next(request)

        # Add headers to successful response
        for header, value in headers.items():
            response.headers[header] = value

        return response


# Pre-configured rate limit profiles
RATE_LIMIT_PROFILES = {
    "strict": RateLimitConfig(requests=30, window_seconds=60, burst=5),
    "standard": RateLimitConfig(requests=100, window_seconds=60, burst=20),
    "relaxed": RateLimitConfig(requests=300, window_seconds=60, burst=50),
    "chat": RateLimitConfig(requests=20, window_seconds=60, burst=5),
    "upload": RateLimitConfig(requests=10, window_seconds=60, burst=3),
    "auth": RateLimitConfig(requests=10, window_seconds=300, burst=5),
}


def create_default_rules() -> list[RateLimitRule]:
    """Create default rate limit rules for JAVIS API."""
    return [
        # Auth endpoints - strict limits to prevent brute force
        RateLimitRule("/api/auth/", RATE_LIMIT_PROFILES["auth"]),
        # Chat endpoints - moderate limits
        RateLimitRule("/api/chat", RATE_LIMIT_PROFILES["chat"], methods=["POST"]),
        # Upload endpoints - strict limits
        RateLimitRule("/api/upload", RATE_LIMIT_PROFILES["upload"], methods=["POST"]),
        # Training endpoints - strict limits
        RateLimitRule("/api/training/", RATE_LIMIT_PROFILES["strict"]),
        # Health endpoints are skipped in middleware
        # All other API endpoints - standard limits
        RateLimitRule("/api/", RATE_LIMIT_PROFILES["standard"]),
    ]


def get_rate_limit_middleware(
    app: ASGIApp,
    enabled: bool = True,
    custom_rules: Optional[list[RateLimitRule]] = None,
    whitelist: Optional[list[str]] = None,
) -> RateLimitMiddleware:
    """Create rate limit middleware with default configuration.

    Args:
        app: ASGI application
        enabled: Whether rate limiting is enabled
        custom_rules: Custom rules (uses defaults if None)
        whitelist: IPs to whitelist

    Returns:
        Configured RateLimitMiddleware
    """
    rules = custom_rules if custom_rules is not None else create_default_rules()

    return RateLimitMiddleware(
        app=app,
        default_limit=RATE_LIMIT_PROFILES["standard"],
        rules=rules,
        whitelist=whitelist,
        enabled=enabled,
    )
