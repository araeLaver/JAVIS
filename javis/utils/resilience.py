"""Service resilience utilities.

Provides Circuit Breaker and Retry patterns for handling
external service failures gracefully.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ==================== Circuit Breaker ====================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.successes += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)
        self.total_requests += 1

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(timezone.utc)
        self.total_requests += 1
        self.total_failures += 1

    def reset(self) -> None:
        """Reset current window statistics."""
        self.failures = 0
        self.successes = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max calls allowed in half-open state
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker("groq-api")

        @breaker
        async def call_groq_api():
            ...

        # Or use as context manager
        async with breaker:
            await call_groq_api()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN

    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._opened_at is not None:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.config.timeout:
                    self._transition_to_half_open()
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        if self._state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' opened after "
                f"{self._stats.consecutive_failures} consecutive failures"
            )
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            self._half_open_calls = 0

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        if self._state != CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' entering half-open state")
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._stats.reset()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        if self._state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._half_open_calls = 0
            self._stats.reset()

    def _handle_success(self) -> None:
        """Handle a successful call."""
        self._stats.record_success()

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()

    def _handle_failure(self, exc: Exception) -> None:
        """Handle a failed call."""
        # Check if exception should be excluded
        if isinstance(exc, self.config.excluded_exceptions):
            return

        self._stats.record_failure()

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to_open()

    async def __aenter__(self):
        """Async context manager entry."""
        async with self._lock:
            if not self._should_allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name,
                    retry_after=self._get_retry_after(),
                )
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        async with self._lock:
            if exc_type is None:
                self._handle_success()
            else:
                self._handle_failure(exc_val)
        return False

    def _get_retry_after(self) -> Optional[float]:
        """Get seconds until circuit might close."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = time.time() - self._opened_at
            remaining = self.config.timeout - elapsed
            return max(0, remaining)
        return None

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for protecting a function with circuit breaker."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                # For sync functions, use run_until_complete pattern
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._execute_sync(func, *args, **kwargs))

            return sync_wrapper

    async def _execute_sync(self, func: Callable, *args, **kwargs):
        """Execute sync function with circuit breaker."""
        async with self:
            return func(*args, **kwargs)

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "stats": {
                "total_requests": self._stats.total_requests,
                "total_failures": self._stats.total_failures,
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "last_failure": (
                    self._stats.last_failure_time.isoformat()
                    if self._stats.last_failure_time
                    else None
                ),
                "last_success": (
                    self._stats.last_success_time.isoformat()
                    if self._stats.last_success_time
                    else None
                ),
            },
            "retry_after": self._get_retry_after(),
        }

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self._transition_to_closed()
        self._stats = CircuitStats()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        message: str,
        circuit_name: str,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.retry_after = retry_after


# ==================== Circuit Breaker Registry ====================


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _breakers: dict[str, CircuitBreaker] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._breakers = {}
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return CircuitBreakerRegistry().get_or_create(name, config)


# ==================== Retry Utilities ====================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add randomness to delays
    retryable_exceptions: tuple = (Exception,)  # Exceptions to retry
    non_retryable_exceptions: tuple = ()  # Exceptions to not retry


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff."""
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add up to 25% jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    non_retryable_exceptions: tuple = (),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Decorator for retrying failed operations with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Multiplier for exponential backoff
        jitter: Add randomness to delays
        retryable_exceptions: Tuple of exceptions to retry
        non_retryable_exceptions: Tuple of exceptions to not retry
        on_retry: Optional callback(exception, attempt) called on each retry

    Example:
        @retry(max_attempts=3, retryable_exceptions=(ConnectionError,))
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exception = None

                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)

                    except config.non_retryable_exceptions as e:
                        # Don't retry these
                        raise

                    except config.retryable_exceptions as e:
                        last_exception = e

                        if attempt == config.max_attempts:
                            logger.warning(
                                f"Function {func.__name__} failed after "
                                f"{config.max_attempts} attempts: {e}"
                            )
                            raise

                        delay = calculate_delay(attempt, config)

                        logger.debug(
                            f"Retry {attempt}/{config.max_attempts} for "
                            f"{func.__name__} after {delay:.2f}s: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        await asyncio.sleep(delay)

                raise last_exception

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_exception = None

                for attempt in range(1, config.max_attempts + 1):
                    try:
                        return func(*args, **kwargs)

                    except config.non_retryable_exceptions as e:
                        raise

                    except config.retryable_exceptions as e:
                        last_exception = e

                        if attempt == config.max_attempts:
                            logger.warning(
                                f"Function {func.__name__} failed after "
                                f"{config.max_attempts} attempts: {e}"
                            )
                            raise

                        delay = calculate_delay(attempt, config)

                        logger.debug(
                            f"Retry {attempt}/{config.max_attempts} for "
                            f"{func.__name__} after {delay:.2f}s: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        time.sleep(delay)

                raise last_exception

            return sync_wrapper

    return decorator


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        on_retry: Optional callback on retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function

    Example:
        result = await retry_async(
            fetch_data,
            url="https://api.example.com",
            config=RetryConfig(max_attempts=5),
        )
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)

        except config.non_retryable_exceptions as e:
            raise

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts:
                raise

            delay = calculate_delay(attempt, config)

            logger.debug(
                f"Retry {attempt}/{config.max_attempts}: {e}, "
                f"waiting {delay:.2f}s"
            )

            if on_retry:
                on_retry(e, attempt)

            await asyncio.sleep(delay)

    raise last_exception


# ==================== Combined Resilience Decorator ====================


def resilient(
    circuit_breaker: Optional[Union[str, CircuitBreaker]] = None,
    retry_config: Optional[RetryConfig] = None,
    fallback: Optional[Callable[..., T]] = None,
):
    """Combined resilience decorator with circuit breaker, retry, and fallback.

    Args:
        circuit_breaker: Circuit breaker name or instance
        retry_config: Retry configuration
        fallback: Fallback function to call on failure

    Example:
        @resilient(
            circuit_breaker="groq-api",
            retry_config=RetryConfig(max_attempts=3),
            fallback=lambda *args, **kwargs: {"error": "Service unavailable"}
        )
        async def call_groq_api(prompt: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get or create circuit breaker
        cb = None
        if circuit_breaker:
            if isinstance(circuit_breaker, str):
                cb = get_circuit_breaker(circuit_breaker)
            else:
                cb = circuit_breaker

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                # Apply circuit breaker if configured
                if cb:
                    async with cb:
                        # Apply retry if configured
                        if retry_config:
                            return await retry_async(
                                func, *args, config=retry_config, **kwargs
                            )
                        return await func(*args, **kwargs)
                else:
                    if retry_config:
                        return await retry_async(
                            func, *args, config=retry_config, **kwargs
                        )
                    return await func(*args, **kwargs)

            except Exception as e:
                if fallback:
                    logger.warning(
                        f"Function {func.__name__} failed, using fallback: {e}"
                    )
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            # For sync functions, wrap in async
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                try:
                    if cb or retry_config:
                        loop = asyncio.get_event_loop()
                        return loop.run_until_complete(
                            async_wrapper(*args, **kwargs)
                        )
                    return func(*args, **kwargs)
                except Exception as e:
                    if fallback:
                        logger.warning(
                            f"Function {func.__name__} failed, using fallback: {e}"
                        )
                        return fallback(*args, **kwargs)
                    raise

            return sync_wrapper

    return decorator


# ==================== Pre-configured Resilience Profiles ====================


# Profiles for different service types
RESILIENCE_PROFILES = {
    "external_api": RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError, IOError),
    ),
    "database": RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
    ),
    "model_inference": RetryConfig(
        max_attempts=2,
        base_delay=2.0,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
    ),
    "integration": RetryConfig(
        max_attempts=3,
        base_delay=1.5,
        max_delay=45.0,
        retryable_exceptions=(ConnectionError, TimeoutError, IOError),
    ),
}

CIRCUIT_BREAKER_PROFILES = {
    "external_api": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=30.0,
    ),
    "database": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=1,
        timeout=15.0,
    ),
    "model_inference": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=60.0,
    ),
    "integration": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=45.0,
    ),
}
