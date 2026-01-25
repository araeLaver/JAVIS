"""Tests for resilience utilities (Circuit Breaker, Retry patterns)."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from javis.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CircuitStats,
    RetryConfig,
    calculate_delay,
    get_circuit_breaker,
    resilient,
    retry,
    retry_async,
    RESILIENCE_PROFILES,
    CIRCUIT_BREAKER_PROFILES,
)


class TestCircuitStats:
    """Tests for CircuitStats dataclass."""

    def test_initial_state(self):
        """Test initial stats values."""
        stats = CircuitStats()
        assert stats.failures == 0
        assert stats.successes == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.total_requests == 0

    def test_record_success(self):
        """Test recording a successful call."""
        stats = CircuitStats()
        stats.record_success()

        assert stats.successes == 1
        assert stats.consecutive_successes == 1
        assert stats.consecutive_failures == 0
        assert stats.total_requests == 1
        assert stats.last_success_time is not None

    def test_record_failure(self):
        """Test recording a failed call."""
        stats = CircuitStats()
        stats.record_failure()

        assert stats.failures == 1
        assert stats.consecutive_failures == 1
        assert stats.consecutive_successes == 0
        assert stats.total_requests == 1
        assert stats.total_failures == 1
        assert stats.last_failure_time is not None

    def test_consecutive_tracking(self):
        """Test consecutive success/failure tracking."""
        stats = CircuitStats()

        # Record 3 successes
        for _ in range(3):
            stats.record_success()
        assert stats.consecutive_successes == 3

        # One failure resets consecutive successes
        stats.record_failure()
        assert stats.consecutive_successes == 0
        assert stats.consecutive_failures == 1

        # More failures
        stats.record_failure()
        assert stats.consecutive_failures == 2

        # Success resets consecutive failures
        stats.record_success()
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 1

    def test_reset(self):
        """Test resetting stats."""
        stats = CircuitStats()
        stats.record_success()
        stats.record_failure()

        stats.reset()

        assert stats.failures == 0
        assert stats.successes == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        # Note: total_requests and total_failures are not reset


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 30.0
        assert config.half_open_max_calls == 3

    def test_custom_values(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout=60.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout == 60.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.5,  # Short timeout for testing
        )
        return CircuitBreaker("test-circuit", config)

    def test_initial_state(self, breaker):
        """Test initial circuit state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self, breaker):
        """Test successful calls keep circuit closed."""
        async with breaker:
            pass  # Successful operation

        assert breaker.is_closed
        assert breaker.stats.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_failure_increments_count(self, breaker):
        """Test failures increment counter."""
        try:
            async with breaker:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert breaker.is_closed
        assert breaker.stats.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold."""
        for i in range(3):  # failure_threshold = 3
            try:
                async with breaker:
                    raise ValueError(f"Error {i}")
            except ValueError:
                pass

        assert breaker.is_open
        assert breaker.stats.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_requests(self, breaker):
        """Test open circuit blocks requests."""
        # Force open
        for _ in range(3):
            try:
                async with breaker:
                    raise ValueError("Error")
            except ValueError:
                pass

        assert breaker.is_open

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.circuit_name == "test-circuit"
        assert exc_info.value.retry_after is not None

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for _ in range(3):
            try:
                async with breaker:
                    raise ValueError("Error")
            except ValueError:
                pass

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(0.6)

        # Next request should be allowed (half-open)
        async with breaker:
            pass

        assert breaker.is_half_open or breaker.is_closed

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self, breaker):
        """Test successful calls in half-open close the circuit."""
        # Open the circuit
        for _ in range(3):
            try:
                async with breaker:
                    raise ValueError("Error")
            except ValueError:
                pass

        await asyncio.sleep(0.6)

        # Successful calls in half-open
        for _ in range(2):  # success_threshold = 2
            async with breaker:
                pass

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        """Test failure in half-open reopens circuit."""
        # Open the circuit
        for _ in range(3):
            try:
                async with breaker:
                    raise ValueError("Error")
            except ValueError:
                pass

        await asyncio.sleep(0.6)

        # Failure in half-open
        try:
            async with breaker:
                raise ValueError("Error")
        except ValueError:
            pass

        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker("decorator-test", CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.5,
        ))

        call_count = 0

        @breaker
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        # Fail twice to open circuit
        for _ in range(2):
            try:
                await failing_function()
            except ValueError:
                pass

        assert call_count == 2
        assert breaker.is_open

        # Third call should be blocked
        with pytest.raises(CircuitOpenError):
            await failing_function()

        assert call_count == 2  # Function not called

    def test_get_status(self, breaker):
        """Test getting circuit status."""
        status = breaker.get_status()

        assert status["name"] == "test-circuit"
        assert status["state"] == "closed"
        assert "stats" in status
        assert status["stats"]["total_requests"] == 0

    def test_manual_reset(self, breaker):
        """Test manually resetting circuit."""
        # Can't directly set state, so this tests reset from any state
        breaker.reset()
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test that excluded exceptions don't trip circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(KeyError,),
        )
        breaker = CircuitBreaker("exclude-test", config)

        # KeyError should not count as failure
        for _ in range(5):
            try:
                async with breaker:
                    raise KeyError("Excluded")
            except KeyError:
                pass

        assert breaker.is_closed
        assert breaker.stats.consecutive_failures == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        CircuitBreakerRegistry._instance = None
        CircuitBreakerRegistry._breakers = {}

    def test_singleton(self):
        """Test registry is singleton."""
        reg1 = CircuitBreakerRegistry()
        reg2 = CircuitBreakerRegistry()
        assert reg1 is reg2

    def test_get_or_create(self):
        """Test getting or creating breakers."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("test")
        breaker2 = registry.get_or_create("test")

        assert breaker1 is breaker2
        assert breaker1.name == "test"

    def test_get_nonexistent(self):
        """Test getting nonexistent breaker."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all_status(self):
        """Test getting all circuit statuses."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("circuit1")
        registry.get_or_create("circuit2")

        status = registry.get_all_status()

        assert "circuit1" in status
        assert "circuit2" in status

    def test_reset_all(self):
        """Test resetting all circuits."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("test")

        # Manually modify state (for testing)
        breaker._state = CircuitState.OPEN

        registry.reset_all()

        assert breaker.is_closed


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker helper."""

    def setup_method(self):
        """Reset registry before each test."""
        CircuitBreakerRegistry._instance = None
        CircuitBreakerRegistry._breakers = {}

    def test_creates_new_breaker(self):
        """Test creating new breaker."""
        breaker = get_circuit_breaker("new-breaker")
        assert breaker.name == "new-breaker"

    def test_returns_existing_breaker(self):
        """Test returning existing breaker."""
        breaker1 = get_circuit_breaker("existing")
        breaker2 = get_circuit_breaker("existing")
        assert breaker1 is breaker2


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False,
        )

        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)
        delay3 = calculate_delay(3, config)

        assert delay1 == 1.0  # 1 * 2^0
        assert delay2 == 2.0  # 1 * 2^1
        assert delay3 == 4.0  # 1 * 2^2

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=10.0,
            exponential_base=2.0,
            max_delay=30.0,
            jitter=False,
        )

        delay = calculate_delay(10, config)
        assert delay == 30.0

    def test_jitter_adds_variance(self):
        """Test jitter adds variance to delay."""
        config = RetryConfig(
            base_delay=10.0,
            jitter=True,
        )

        delays = [calculate_delay(1, config) for _ in range(10)]
        # With jitter, not all delays should be exactly the same
        unique_delays = set(round(d, 3) for d in delays)
        assert len(unique_delays) > 1


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry(max_attempts=3)
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Retry me")
            return "success"

        result = await failing_then_success()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """Test exception raised after max attempts."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exceptions aren't retried."""
        call_count = 0

        @retry(
            max_attempts=3,
            retryable_exceptions=(ConnectionError,),
            non_retryable_exceptions=(ValueError,),
        )
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Don't retry me")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self):
        """Test only specified exceptions are retried."""
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        async def raises_wrong_exception():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            await raises_wrong_exception()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retries = []

        def on_retry(exc, attempt):
            retries.append((str(exc), attempt))

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        async def failing():
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            await failing()

        assert len(retries) == 2  # Called on attempts 1 and 2
        assert retries[0][1] == 1
        assert retries[1][1] == 2


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_basic_usage(self):
        """Test basic retry_async usage."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_async(flaky_func, config=config)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_arguments(self):
        """Test retry_async with function arguments."""
        async def add(a, b):
            return a + b

        result = await retry_async(add, 1, 2)
        assert result == 3


class TestResilientDecorator:
    """Tests for combined resilient decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        CircuitBreakerRegistry._instance = None
        CircuitBreakerRegistry._breakers = {}

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self):
        """Test resilient with circuit breaker."""
        @resilient(circuit_breaker="test-cb")
        async def protected_func():
            return "success"

        result = await protected_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_retry(self):
        """Test resilient with retry."""
        call_count = 0
        config = RetryConfig(max_attempts=3, base_delay=0.01)

        @resilient(retry_config=config)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_fallback(self):
        """Test resilient with fallback."""
        @resilient(fallback=lambda: "fallback")
        async def failing_func():
            raise ValueError("Fail")

        result = await failing_func()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Test resilient with async fallback."""
        async def async_fallback():
            return "async fallback"

        @resilient(fallback=async_fallback)
        async def failing_func():
            raise ValueError("Fail")

        result = await failing_func()
        assert result == "async fallback"

    @pytest.mark.asyncio
    async def test_combined(self):
        """Test resilient with all options."""
        call_count = 0
        config = RetryConfig(max_attempts=2, base_delay=0.01)

        @resilient(
            circuit_breaker="combined-test",
            retry_config=config,
            fallback=lambda: "fallback",
        )
        async def complex_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        # First call: retries then falls back
        result = await complex_func()
        assert result == "fallback"
        assert call_count == 2


class TestResilienceProfiles:
    """Tests for pre-configured resilience profiles."""

    def test_profiles_exist(self):
        """Test all expected profiles exist."""
        expected_profiles = ["external_api", "database", "model_inference", "integration"]

        for profile in expected_profiles:
            assert profile in RESILIENCE_PROFILES
            assert profile in CIRCUIT_BREAKER_PROFILES

    def test_retry_profiles_valid(self):
        """Test retry profiles have valid values."""
        for name, config in RESILIENCE_PROFILES.items():
            assert isinstance(config, RetryConfig)
            assert config.max_attempts > 0
            assert config.base_delay > 0
            assert config.max_delay >= config.base_delay

    def test_circuit_breaker_profiles_valid(self):
        """Test circuit breaker profiles have valid values."""
        for name, config in CIRCUIT_BREAKER_PROFILES.items():
            assert isinstance(config, CircuitBreakerConfig)
            assert config.failure_threshold > 0
            assert config.success_threshold > 0
            assert config.timeout > 0


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_properties(self):
        """Test error has correct properties."""
        error = CircuitOpenError(
            "Circuit is open",
            circuit_name="test-circuit",
            retry_after=30.0,
        )

        assert str(error) == "Circuit is open"
        assert error.circuit_name == "test-circuit"
        assert error.retry_after == 30.0

    def test_error_without_retry_after(self):
        """Test error without retry_after."""
        error = CircuitOpenError(
            "Circuit is open",
            circuit_name="test",
        )

        assert error.retry_after is None


class TestSyncRetryDecorator:
    """Tests for sync retry decorator paths."""

    def test_sync_retry_success(self):
        """Test sync retry decorator success path."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def sync_success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = sync_success()
        assert result == "success"
        assert call_count == 1

    def test_sync_retry_with_retries(self):
        """Test sync retry decorator with retries."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def sync_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry")
            return "success"

        result = sync_flaky()
        assert result == "success"
        assert call_count == 2

    def test_sync_retry_max_attempts_exceeded(self):
        """Test sync retry raises after max attempts."""
        call_count = 0

        @retry(max_attempts=2, base_delay=0.01)
        def sync_always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            sync_always_fails()

        assert call_count == 2

    def test_sync_retry_non_retryable(self):
        """Test sync retry with non-retryable exception."""
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            non_retryable_exceptions=(ValueError,),
        )
        def sync_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable")

        with pytest.raises(ValueError):
            sync_value_error()

        assert call_count == 1

    def test_sync_retry_on_retry_callback(self):
        """Test sync retry on_retry callback."""
        retries = []

        def on_retry(exc, attempt):
            retries.append((str(exc), attempt))

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        def sync_failing():
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            sync_failing()

        assert len(retries) == 2
        assert retries[0][1] == 1
        assert retries[1][1] == 2


class TestRetryAsyncAdvanced:
    """Advanced tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_retry_async_non_retryable_exception(self):
        """Test retry_async with non-retryable exception."""
        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable")

        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            non_retryable_exceptions=(ValueError,),
        )

        with pytest.raises(ValueError):
            await retry_async(raises_value_error, config=config)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_with_on_retry_callback(self):
        """Test retry_async with on_retry callback."""
        call_count = 0
        retries = []

        def on_retry(exc, attempt):
            retries.append(attempt)

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Retry")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        result = await retry_async(flaky_func, config=config, on_retry=on_retry)

        assert result == "success"
        assert len(retries) == 2  # Called on attempts 1 and 2

    @pytest.mark.asyncio
    async def test_retry_async_exhausts_retries(self):
        """Test retry_async raises last exception when exhausted."""
        async def always_fails():
            raise ConnectionError("Always fails")

        config = RetryConfig(max_attempts=2, base_delay=0.01)

        with pytest.raises(ConnectionError):
            await retry_async(always_fails, config=config)


class TestResilientDecoratorAdvanced:
    """Advanced tests for resilient decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        CircuitBreakerRegistry._instance = None
        CircuitBreakerRegistry._breakers = {}

    @pytest.mark.asyncio
    async def test_resilient_with_circuit_breaker_instance(self):
        """Test resilient with circuit breaker instance (not name)."""
        cb = CircuitBreaker("test-instance-cb")

        @resilient(circuit_breaker=cb)
        async def protected_func():
            return "success"

        result = await protected_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_resilient_without_cb_or_retry(self):
        """Test resilient with only fallback."""
        @resilient(fallback=lambda: "fallback")
        async def simple_fail():
            raise ValueError("Fail")

        result = await simple_fail()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_resilient_no_fallback_raises(self):
        """Test resilient without fallback raises exception."""
        @resilient()
        async def raises_error():
            raise ValueError("No fallback")

        with pytest.raises(ValueError):
            await raises_error()

    def test_resilient_sync_function_success(self):
        """Test resilient decorator on sync function."""
        @resilient()
        def sync_success():
            return "sync success"

        result = sync_success()
        assert result == "sync success"

    def test_resilient_sync_with_fallback(self):
        """Test resilient sync function with fallback."""
        @resilient(fallback=lambda: "sync fallback")
        def sync_fail():
            raise ValueError("Fail")

        result = sync_fail()
        assert result == "sync fallback"


class TestCircuitBreakerTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.fixture
    def breaker(self):
        """Create a fast circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.1,
        )
        return CircuitBreaker("transition-test", config)

    @pytest.mark.asyncio
    async def test_closed_to_open_to_half_open(self, breaker):
        """Test full state transition cycle."""
        # Start closed
        assert breaker.is_closed

        # Cause failures to open the circuit
        for _ in range(2):
            try:
                async with breaker:
                    raise ConnectionError("Fail")
            except ConnectionError:
                pass

        # Should be open
        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open on next request
        try:
            async with breaker:
                raise ConnectionError("Still failing")
        except ConnectionError:
            pass

        # Should be back to open after failure in half-open
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_half_open_to_closed(self, breaker):
        """Test half-open to closed transition."""
        # Open the circuit
        for _ in range(2):
            try:
                async with breaker:
                    raise ConnectionError("Fail")
            except ConnectionError:
                pass

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Succeed in half-open state
        async with breaker:
            pass

        # Should be closed now
        assert breaker.is_closed
