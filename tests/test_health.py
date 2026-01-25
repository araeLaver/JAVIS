"""Tests for health check utilities."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from javis.utils.health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    check_groq_api,
    check_storage_system,
    get_health_checker,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Test basic ComponentHealth creation."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.message is None
        assert health.latency_ms is None

    def test_with_all_fields(self):
        """Test ComponentHealth with all fields."""
        now = datetime.now(timezone.utc)
        health = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Partial failure",
            latency_ms=15.5,
            details={"error_count": 3},
            last_check=now,
        )
        assert health.message == "Partial failure"
        assert health.latency_ms == 15.5
        assert health.details["error_count"] == 3
        assert health.last_check == now

    def test_to_dict(self):
        """Test ComponentHealth to_dict conversion."""
        now = datetime.now(timezone.utc)
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=5.123,
            details={"key": "value"},
            last_check=now,
        )
        result = health.to_dict()

        assert result["status"] == "healthy"
        assert result["message"] == "OK"
        assert result["latency_ms"] == 5.12  # Rounded
        assert result["details"] == {"key": "value"}
        assert "last_check" in result

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.UNHEALTHY,
        )
        result = health.to_dict()

        assert result["status"] == "unhealthy"
        assert "message" not in result
        assert "latency_ms" not in result


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_basic_creation(self):
        """Test SystemHealth creation."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            uptime_seconds=3600.5,
            components={},
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.5

    def test_to_dict(self):
        """Test SystemHealth to_dict conversion."""
        component = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            uptime_seconds=100.0,
            components={"test": component},
        )
        result = health.to_dict()

        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"
        assert result["uptime_seconds"] == 100.0
        assert "timestamp" in result
        assert "test" in result["components"]


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_initialization(self):
        """Test HealthChecker initialization."""
        checker = HealthChecker(version="2.0.0")
        assert checker.version == "2.0.0"
        assert checker.is_alive()
        assert checker.uptime_seconds >= 0

    def test_register_check(self):
        """Test registering health checks."""
        checker = HealthChecker()

        def my_check():
            return ComponentHealth(
                name="my_component",
                status=HealthStatus.HEALTHY,
            )

        checker.register_check("my_component", my_check)
        assert "my_component" in checker._checks

    def test_unregister_check(self):
        """Test unregistering health checks."""
        checker = HealthChecker()

        def my_check():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        checker.register_check("test", my_check)
        checker.unregister_check("test")
        assert "test" not in checker._checks

    @pytest.mark.asyncio
    async def test_check_all_empty(self):
        """Test check_all with no registered checks."""
        checker = HealthChecker()
        health = await checker.check_all()

        assert health.status == HealthStatus.UNKNOWN
        assert len(health.components) == 0

    @pytest.mark.asyncio
    async def test_check_all_with_sync_check(self):
        """Test check_all with synchronous checks."""
        checker = HealthChecker()

        def healthy_check():
            return ComponentHealth(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        checker.register_check("healthy", healthy_check)
        health = await checker.check_all()

        assert health.status == HealthStatus.HEALTHY
        assert "healthy" in health.components
        assert health.components["healthy"].status == HealthStatus.HEALTHY
        assert health.components["healthy"].latency_ms is not None

    @pytest.mark.asyncio
    async def test_check_all_with_async_check(self):
        """Test check_all with async checks."""
        checker = HealthChecker()

        async def async_check():
            await asyncio.sleep(0.01)
            return ComponentHealth(
                name="async",
                status=HealthStatus.HEALTHY,
            )

        checker.register_check("async", async_check, is_async=True)
        health = await checker.check_all()

        assert "async" in health.components
        assert health.components["async"].latency_ms >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_check_all_mixed_statuses(self):
        """Test overall status calculation with mixed component statuses."""
        checker = HealthChecker()

        checker.register_check(
            "healthy",
            lambda: ComponentHealth(name="healthy", status=HealthStatus.HEALTHY),
        )
        checker.register_check(
            "degraded",
            lambda: ComponentHealth(name="degraded", status=HealthStatus.DEGRADED),
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_with_unhealthy(self):
        """Test overall status with unhealthy component."""
        checker = HealthChecker()

        checker.register_check(
            "healthy",
            lambda: ComponentHealth(name="healthy", status=HealthStatus.HEALTHY),
        )
        checker.register_check(
            "unhealthy",
            lambda: ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_handles_exception(self):
        """Test that check_all handles exceptions in checks."""
        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed")

        checker.register_check("failing", failing_check)
        health = await checker.check_all()

        assert "failing" in health.components
        assert health.components["failing"].status == HealthStatus.UNHEALTHY
        assert "Check failed" in health.components["failing"].message

    @pytest.mark.asyncio
    async def test_is_ready(self):
        """Test is_ready method."""
        checker = HealthChecker()
        checker.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )

        is_ready, component_status = await checker.is_ready()

        assert is_ready is True
        assert component_status["test"] is True

    @pytest.mark.asyncio
    async def test_is_ready_with_degraded(self):
        """Test is_ready with degraded component."""
        checker = HealthChecker()
        checker.register_check(
            "degraded",
            lambda: ComponentHealth(name="degraded", status=HealthStatus.DEGRADED),
        )

        is_ready, component_status = await checker.is_ready()

        # Degraded is still considered ready
        assert is_ready is True
        assert component_status["degraded"] is True

    @pytest.mark.asyncio
    async def test_is_ready_all_unhealthy(self):
        """Test is_ready when all components are unhealthy."""
        checker = HealthChecker()
        checker.register_check(
            "unhealthy",
            lambda: ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )

        is_ready, component_status = await checker.is_ready()

        assert is_ready is False
        assert component_status["unhealthy"] is False

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test health check result caching."""
        checker = HealthChecker()
        checker._cache_ttl_seconds = 60  # 60 second cache

        call_count = 0

        def counting_check():
            nonlocal call_count
            call_count += 1
            return ComponentHealth(name="counted", status=HealthStatus.HEALTHY)

        checker.register_check("counted", counting_check)

        # First call
        await checker.check_all(use_cache=True)
        assert call_count == 1

        # Second call should use cache
        await checker.check_all(use_cache=True)
        assert call_count == 1

        # Call without cache
        await checker.check_all(use_cache=False)
        assert call_count == 2


class TestDefaultChecks:
    """Tests for default health check functions."""

    def test_check_groq_api_configured(self):
        """Test Groq API check when configured."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            result = check_groq_api()

        assert result.status == HealthStatus.HEALTHY
        assert "configured" in result.message.lower()

    def test_check_groq_api_not_configured(self):
        """Test Groq API check when not configured."""
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=True):
            result = check_groq_api()

        assert result.status == HealthStatus.UNHEALTHY

    def test_check_storage_system(self):
        """Test storage system check."""
        with patch("javis.storage.get_storage_factory") as mock_factory:
            mock_factory.return_value.mode = "local"
            result = check_storage_system()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["mode"] == "local"


class TestGetHealthChecker:
    """Tests for global health checker instance."""

    def test_get_health_checker_singleton(self):
        """Test that get_health_checker returns same instance."""
        # Reset the global instance
        import javis.utils.health as health_module
        health_module._health_checker = None

        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

        # Cleanup
        health_module._health_checker = None


class TestCheckMemorySystem:
    """Tests for check_memory_system function."""

    def test_memory_disabled(self):
        """Test when memory is disabled."""
        from javis.utils.health import check_memory_system

        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False

        result = check_memory_system(mock_config)

        assert result.status == HealthStatus.HEALTHY
        assert result.details["enabled"] is False

    def test_memory_enabled_healthy(self):
        """Test when memory is enabled and working."""
        from javis.utils.health import check_memory_system

        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = True

        with patch("javis.memory.get_memory_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.get_memory_count.return_value = 100
            mock_get.return_value = mock_manager

            result = check_memory_system(mock_config)

            assert result.status == HealthStatus.HEALTHY
            assert result.details["memory_count"] == 100

    def test_memory_enabled_error(self):
        """Test when memory is enabled but has error."""
        from javis.utils.health import check_memory_system

        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = True

        with patch("javis.memory.get_memory_manager") as mock_get:
            mock_get.side_effect = RuntimeError("Memory error")

            result = check_memory_system(mock_config)

            assert result.status == HealthStatus.DEGRADED
            assert "error" in result.details


class TestCheckRAGSystem:
    """Tests for check_rag_system function."""

    def test_rag_disabled(self):
        """Test when RAG is disabled."""
        from javis.utils.health import check_rag_system

        mock_config = MagicMock()
        mock_config.rag.enabled = False

        result = check_rag_system(mock_config)

        assert result.status == HealthStatus.HEALTHY
        assert result.details["enabled"] is False

    def test_rag_enabled_healthy(self):
        """Test when RAG is enabled and working."""
        from javis.utils.health import check_rag_system

        mock_config = MagicMock()
        mock_config.rag.enabled = True

        with patch("javis.rag.get_rag_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.get_stats.return_value = {
                "total_documents": 50,
                "total_chunks": 200,
            }
            mock_get.return_value = mock_manager

            result = check_rag_system(mock_config)

            assert result.status == HealthStatus.HEALTHY
            assert result.details["documents"] == 50
            assert result.details["chunks"] == 200

    def test_rag_enabled_not_initialized(self):
        """Test when RAG is enabled but not initialized."""
        from javis.utils.health import check_rag_system

        mock_config = MagicMock()
        mock_config.rag.enabled = True

        with patch("javis.rag.get_rag_manager") as mock_get:
            mock_get.return_value = None

            result = check_rag_system(mock_config)

            assert result.status == HealthStatus.DEGRADED

    def test_rag_enabled_error(self):
        """Test when RAG is enabled but has error."""
        from javis.utils.health import check_rag_system

        mock_config = MagicMock()
        mock_config.rag.enabled = True

        with patch("javis.rag.get_rag_manager") as mock_get:
            mock_get.side_effect = RuntimeError("RAG error")

            result = check_rag_system(mock_config)

            assert result.status == HealthStatus.DEGRADED


class TestCheckToolsSystem:
    """Tests for check_tools_system function."""

    def test_tools_disabled(self):
        """Test when tools are disabled."""
        from javis.utils.health import check_tools_system

        mock_config = MagicMock()
        mock_config.tools.enabled = False

        result = check_tools_system(mock_config)

        assert result.status == HealthStatus.HEALTHY
        assert result.details["enabled"] is False

    def test_tools_enabled_healthy(self):
        """Test when tools are enabled and working."""
        from javis.utils.health import check_tools_system

        mock_config = MagicMock()
        mock_config.tools.enabled = True

        with patch("javis.tools.get_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.list_tools.return_value = ["tool1", "tool2", "tool3"]
            mock_get.return_value = mock_registry

            result = check_tools_system(mock_config)

            assert result.status == HealthStatus.HEALTHY
            assert result.details["tool_count"] == 3

    def test_tools_enabled_error(self):
        """Test when tools are enabled but has error."""
        from javis.utils.health import check_tools_system

        mock_config = MagicMock()
        mock_config.tools.enabled = True

        with patch("javis.tools.get_registry") as mock_get:
            mock_get.side_effect = RuntimeError("Tools error")

            result = check_tools_system(mock_config)

            assert result.status == HealthStatus.DEGRADED


class TestCheckDatabaseConnection:
    """Tests for check_database_connection function."""

    @pytest.mark.asyncio
    async def test_local_storage(self):
        """Test when using local storage."""
        from javis.utils.health import check_database_connection

        with patch.dict("os.environ", {"JAVIS_STORAGE_MODE": "local"}):
            result = await check_database_connection()

            assert result.status == HealthStatus.HEALTHY
            assert result.details["mode"] == "local"

    @pytest.mark.asyncio
    async def test_cloud_storage_connected(self):
        """Test when cloud storage is connected."""
        from javis.utils.health import check_database_connection

        with patch.dict("os.environ", {"JAVIS_STORAGE_MODE": "cloud"}):
            with patch("javis.storage.cloud.supabase_client.get_supabase_client") as mock_get:
                mock_client = MagicMock()
                mock_get.return_value = mock_client

                result = await check_database_connection()

                assert result.status == HealthStatus.HEALTHY
                assert result.details["provider"] == "supabase"

    @pytest.mark.asyncio
    async def test_cloud_storage_not_initialized(self):
        """Test when cloud storage client is not initialized."""
        from javis.utils.health import check_database_connection

        with patch.dict("os.environ", {"JAVIS_STORAGE_MODE": "cloud"}):
            with patch("javis.storage.cloud.supabase_client.get_supabase_client") as mock_get:
                mock_get.return_value = None

                result = await check_database_connection()

                assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_cloud_storage_error(self):
        """Test when cloud storage connection fails."""
        from javis.utils.health import check_database_connection

        with patch.dict("os.environ", {"JAVIS_STORAGE_MODE": "cloud"}):
            with patch("javis.storage.cloud.supabase_client.get_supabase_client") as mock_get:
                mock_get.side_effect = RuntimeError("Connection failed")

                result = await check_database_connection()

                assert result.status == HealthStatus.UNHEALTHY


class TestCheckCircuitBreakers:
    """Tests for check_circuit_breakers function."""

    def test_no_circuit_breakers(self):
        """Test when no circuit breakers are registered."""
        from javis.utils.health import check_circuit_breakers

        with patch("javis.utils.resilience.CircuitBreakerRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get_all_status.return_value = {}
            mock_registry_class.return_value = mock_registry

            result = check_circuit_breakers()

            assert result.status == HealthStatus.HEALTHY
            assert result.details["count"] == 0

    def test_all_circuits_closed(self):
        """Test when all circuit breakers are closed."""
        from javis.utils.health import check_circuit_breakers

        with patch("javis.utils.resilience.CircuitBreakerRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get_all_status.return_value = {
                "api": {"state": "closed", "failures": 0},
                "db": {"state": "closed", "failures": 0},
            }
            mock_registry_class.return_value = mock_registry

            result = check_circuit_breakers()

            assert result.status == HealthStatus.HEALTHY
            assert result.details["states"]["closed"] == 2

    def test_some_circuits_open(self):
        """Test when some circuit breakers are open."""
        from javis.utils.health import check_circuit_breakers

        with patch("javis.utils.resilience.CircuitBreakerRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get_all_status.return_value = {
                "api": {"state": "open", "failures": 5},
                "db": {"state": "closed", "failures": 0},
            }
            mock_registry_class.return_value = mock_registry

            result = check_circuit_breakers()

            assert result.status == HealthStatus.DEGRADED
            assert "open" in result.message

    def test_some_circuits_half_open(self):
        """Test when some circuit breakers are half-open."""
        from javis.utils.health import check_circuit_breakers

        with patch("javis.utils.resilience.CircuitBreakerRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get_all_status.return_value = {
                "api": {"state": "half_open", "failures": 3},
                "db": {"state": "closed", "failures": 0},
            }
            mock_registry_class.return_value = mock_registry

            result = check_circuit_breakers()

            assert result.status == HealthStatus.DEGRADED
            assert "recovering" in result.message

    def test_circuit_breakers_error(self):
        """Test when circuit breaker check fails."""
        from javis.utils.health import check_circuit_breakers

        with patch("javis.utils.resilience.CircuitBreakerRegistry") as mock_registry_class:
            mock_registry_class.side_effect = RuntimeError("Registry error")

            result = check_circuit_breakers()

            assert result.status == HealthStatus.UNKNOWN


class TestInitializeHealthChecks:
    """Tests for initialize_health_checks function."""

    def test_registers_all_checks(self):
        """Test that all checks are registered."""
        from javis.utils.health import initialize_health_checks

        # Reset global checker
        import javis.utils.health as health_module
        health_module._health_checker = None

        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        checker = initialize_health_checks(mock_config)

        # Check that all expected checks are registered
        all_checks = list(checker._checks.keys()) + list(checker._async_checks.keys())
        assert "groq" in all_checks
        assert "storage" in all_checks
        assert "memory" in all_checks
        assert "rag" in all_checks
        assert "tools" in all_checks
        assert "database" in all_checks
        assert "circuit_breakers" in all_checks

        # Cleanup
        health_module._health_checker = None


class TestHealthCheckerAsyncExceptionHandling:
    """Tests for async exception handling in HealthChecker."""

    @pytest.mark.asyncio
    async def test_async_check_exception_in_gather(self):
        """Test that exceptions from async.gather are handled."""
        checker = HealthChecker()

        async def failing_async_check():
            raise RuntimeError("Async check failed")

        checker.register_check("failing_async", failing_async_check, is_async=True)

        health = await checker.check_all()

        assert "failing_async" in health.components
        assert health.components["failing_async"].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_async_cache_hit(self):
        """Test that async checks use cache properly."""
        checker = HealthChecker()
        checker._cache_ttl_seconds = 60

        call_count = 0

        async def counting_async_check():
            nonlocal call_count
            call_count += 1
            return ComponentHealth(name="counted", status=HealthStatus.HEALTHY)

        checker.register_check("counted", counting_async_check, is_async=True)

        # First call
        await checker.check_all(use_cache=True)
        assert call_count == 1

        # Second call should use cache
        await checker.check_all(use_cache=True)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_calculate_overall_with_all_unknown(self):
        """Test overall status when all are unknown."""
        checker = HealthChecker()

        checker.register_check(
            "unknown",
            lambda: ComponentHealth(name="unknown", status=HealthStatus.UNKNOWN),
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.UNKNOWN
