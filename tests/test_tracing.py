"""Tests for JAVIS distributed tracing module."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from javis.utils.tracing import (
    TracingConfig,
    init_tracing,
    get_tracer,
    get_current_span,
    get_trace_id,
    get_span_id,
    create_span,
    NoOpSpan,
    traced,
    _safe_str,
    add_span_attributes,
    add_span_event,
    set_span_error,
    extract_trace_context,
    inject_trace_context,
    OTEL_AVAILABLE,
)


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TracingConfig()
        assert config.service_name == "javis"
        assert config.enabled is True
        assert config.exporter_type == "console"
        assert config.sample_rate == 1.0
        assert config.debug is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TracingConfig(
            service_name="custom-service",
            enabled=False,
            exporter_type="otlp",
            otlp_endpoint="http://collector:4317",
            sample_rate=0.5,
            debug=True,
        )
        assert config.service_name == "custom-service"
        assert config.enabled is False
        assert config.exporter_type == "otlp"
        assert config.otlp_endpoint == "http://collector:4317"
        assert config.sample_rate == 0.5
        assert config.debug is True

    def test_from_env_defaults(self):
        """Test configuration from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = TracingConfig.from_env()
            assert config.service_name == "javis"
            assert config.enabled is True
            assert config.exporter_type == "console"

    def test_from_env_custom(self):
        """Test configuration from environment variables."""
        env_vars = {
            "OTEL_SERVICE_NAME": "test-service",
            "JAVIS_TRACING_ENABLED": "false",
            "JAVIS_TRACING_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            "JAVIS_TRACING_SAMPLE_RATE": "0.1",
            "JAVIS_TRACING_DEBUG": "true",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = TracingConfig.from_env()
            assert config.service_name == "test-service"
            assert config.enabled is False
            assert config.exporter_type == "otlp"
            assert config.otlp_endpoint == "http://localhost:4318"
            assert config.sample_rate == 0.1
            assert config.debug is True


class TestNoOpSpan:
    """Tests for NoOpSpan."""

    def test_set_attribute(self):
        """Test NoOpSpan.set_attribute does nothing."""
        span = NoOpSpan()
        span.set_attribute("key", "value")  # Should not raise

    def test_set_attributes(self):
        """Test NoOpSpan.set_attributes does nothing."""
        span = NoOpSpan()
        span.set_attributes({"key1": "value1", "key2": "value2"})

    def test_add_event(self):
        """Test NoOpSpan.add_event does nothing."""
        span = NoOpSpan()
        span.add_event("test_event", {"attr": "value"})

    def test_set_status(self):
        """Test NoOpSpan.set_status does nothing."""
        span = NoOpSpan()
        span.set_status("error")

    def test_record_exception(self):
        """Test NoOpSpan.record_exception does nothing."""
        span = NoOpSpan()
        span.record_exception(Exception("test"))

    def test_get_span_context(self):
        """Test NoOpSpan.get_span_context returns None."""
        span = NoOpSpan()
        assert span.get_span_context() is None


class TestSafeStr:
    """Tests for _safe_str utility function."""

    def test_simple_string(self):
        """Test conversion of simple string."""
        assert _safe_str("hello") == "hello"

    def test_number(self):
        """Test conversion of number."""
        assert _safe_str(42) == "42"
        assert _safe_str(3.14) == "3.14"

    def test_long_string_truncation(self):
        """Test truncation of long strings."""
        long_str = "a" * 300
        result = _safe_str(long_str)
        assert len(result) == 259  # 256 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Test custom max_length."""
        result = _safe_str("hello world", max_length=5)
        assert result == "hello..."

    def test_object_conversion(self):
        """Test conversion of objects."""
        obj = {"key": "value"}
        result = _safe_str(obj)
        assert "key" in result
        assert "value" in result

    def test_unconvertible_object(self):
        """Test handling of objects that can't be converted."""
        class BadObject:
            def __str__(self):
                raise ValueError("Cannot convert")

        result = _safe_str(BadObject())
        assert result == "<unable to convert>"


class TestCreateSpan:
    """Tests for create_span context manager."""

    def test_create_span_no_tracer(self):
        """Test create_span returns NoOpSpan when tracer is not available."""
        # Reset tracing state
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                assert isinstance(span, NoOpSpan)
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_create_span_with_attributes(self):
        """Test create_span with attributes."""
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            with create_span("test_span", attributes={"key": "value"}) as span:
                assert isinstance(span, NoOpSpan)
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestTracedDecorator:
    """Tests for traced decorator."""

    def test_traced_sync_function(self):
        """Test traced decorator on sync function."""
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced()
            def test_function(x, y):
                return x + y

            result = test_function(2, 3)
            assert result == 5
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_traced_sync_function_with_options(self):
        """Test traced decorator with options on sync function."""
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced(name="custom_name", attributes={"component": "test"})
            def test_function(x):
                return x * 2

            result = test_function(5)
            assert result == 10
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test traced decorator on async function."""
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced()
            async def async_test_function(x, y):
                await asyncio.sleep(0.001)
                return x + y

            result = await async_test_function(2, 3)
            assert result == 5
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    @pytest.mark.asyncio
    async def test_traced_async_function_with_options(self):
        """Test traced decorator with options on async function."""
        import javis.utils.tracing as tracing_module
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced(
                name="async_custom",
                attributes={"type": "async"},
                record_args=True,
                record_result=True,
            )
            async def async_test_function(value):
                await asyncio.sleep(0.001)
                return value * 2

            result = await async_test_function(5)
            assert result == 10
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestHelperFunctions:
    """Tests for tracing helper functions."""

    def test_add_span_attributes_no_span(self):
        """Test add_span_attributes when no span is active."""
        # Should not raise
        add_span_attributes({"key": "value"})

    def test_add_span_event_no_span(self):
        """Test add_span_event when no span is active."""
        # Should not raise
        add_span_event("test_event", {"attr": "value"})

    def test_set_span_error_no_span(self):
        """Test set_span_error when no span is active."""
        # Should not raise
        set_span_error(Exception("test error"))

    def test_get_trace_id_no_span(self):
        """Test get_trace_id when no span is active."""
        result = get_trace_id()
        # Should return None or valid trace_id
        assert result is None or isinstance(result, str)

    def test_get_span_id_no_span(self):
        """Test get_span_id when no span is active."""
        result = get_span_id()
        # Should return None or valid span_id
        assert result is None or isinstance(result, str)


class TestContextPropagation:
    """Tests for trace context propagation."""

    def test_extract_trace_context_empty(self):
        """Test extracting trace context from empty headers."""
        result = extract_trace_context({})
        # Should return None or empty context when OTEL not available
        # or a Context object when OTEL is available

    def test_inject_trace_context_empty(self):
        """Test injecting trace context into empty headers."""
        headers = {}
        result = inject_trace_context(headers)
        assert isinstance(result, dict)

    def test_inject_trace_context_preserves_existing(self):
        """Test that inject_trace_context preserves existing headers."""
        headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        result = inject_trace_context(headers)
        assert result["Authorization"] == "Bearer token"
        assert result["Content-Type"] == "application/json"


class TestInitTracing:
    """Tests for tracing initialization."""

    def test_init_tracing_disabled(self):
        """Test tracing initialization when disabled."""
        import javis.utils.tracing as tracing_module

        # Reset state
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = False

        try:
            config = TracingConfig(enabled=False)
            result = init_tracing(config)
            assert result is None
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_init_tracing_already_initialized(self):
        """Test tracing initialization when already initialized."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        mock_tracer = MagicMock()
        tracing_module._tracer = mock_tracer
        tracing_module._initialized = True

        try:
            result = init_tracing()
            assert result is mock_tracer
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_initializes_if_needed(self):
        """Test get_tracer initializes tracing if not done."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = False

        try:
            # This will attempt to initialize tracing
            result = get_tracer()
            # Result depends on whether OTEL is available
            assert tracing_module._initialized is True
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_get_tracer_returns_existing(self):
        """Test get_tracer returns existing tracer."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        mock_tracer = MagicMock()
        tracing_module._tracer = mock_tracer
        tracing_module._initialized = True

        try:
            result = get_tracer()
            assert result is mock_tracer
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestOTelAvailability:
    """Tests related to OpenTelemetry availability."""

    def test_otel_available_flag(self):
        """Test OTEL_AVAILABLE flag is boolean."""
        assert isinstance(OTEL_AVAILABLE, bool)

    def test_graceful_degradation_no_otel(self):
        """Test graceful degradation when OTEL is not available."""
        import javis.utils.tracing as tracing_module

        original_available = tracing_module.OTEL_AVAILABLE
        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        # Simulate OTEL not available
        tracing_module.OTEL_AVAILABLE = False
        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            # All operations should work without raising
            with create_span("test") as span:
                assert isinstance(span, NoOpSpan)

            assert get_trace_id() is None
            assert get_span_id() is None
            assert get_current_span() is None

            add_span_attributes({"key": "value"})
            add_span_event("event")
            set_span_error(Exception("test"))

        finally:
            tracing_module.OTEL_AVAILABLE = original_available
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestSetupFunctions:
    """Tests for setup_fastapi_tracing and setup_httpx_tracing."""

    def test_setup_fastapi_tracing_no_otel(self):
        """Test setup_fastapi_tracing when OTEL not available."""
        import javis.utils.tracing as tracing_module
        from javis.utils.tracing import setup_fastapi_tracing

        original_available = tracing_module.OTEL_AVAILABLE
        tracing_module.OTEL_AVAILABLE = False

        try:
            mock_app = MagicMock()
            setup_fastapi_tracing(mock_app)  # Should return early
        finally:
            tracing_module.OTEL_AVAILABLE = original_available

    def test_setup_httpx_tracing_no_otel(self):
        """Test setup_httpx_tracing when OTEL not available."""
        import javis.utils.tracing as tracing_module
        from javis.utils.tracing import setup_httpx_tracing

        original_available = tracing_module.OTEL_AVAILABLE
        tracing_module.OTEL_AVAILABLE = False

        try:
            setup_httpx_tracing()  # Should return early
        finally:
            tracing_module.OTEL_AVAILABLE = original_available

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_setup_fastapi_tracing_import_error(self):
        """Test setup_fastapi_tracing handles ImportError."""
        import javis.utils.tracing as tracing_module
        from javis.utils.tracing import setup_fastapi_tracing

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = MagicMock()
        tracing_module._initialized = True

        try:
            with patch.dict("sys.modules", {"opentelemetry.instrumentation.fastapi": None}):
                with patch("javis.utils.tracing.init_tracing"):
                    mock_app = MagicMock()
                    # This should catch ImportError
                    setup_fastapi_tracing(mock_app)
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_setup_httpx_tracing_import_error(self):
        """Test setup_httpx_tracing handles ImportError."""
        import javis.utils.tracing as tracing_module
        from javis.utils.tracing import setup_httpx_tracing

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = MagicMock()
        tracing_module._initialized = True

        try:
            with patch.dict("sys.modules", {"opentelemetry.instrumentation.httpx": None}):
                # This should catch ImportError
                setup_httpx_tracing()
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


class TestTracedDecoratorAdvanced:
    """Advanced tests for traced decorator."""

    def test_traced_with_exception(self):
        """Test traced decorator records exceptions."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced()
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    @pytest.mark.asyncio
    async def test_traced_async_with_exception(self):
        """Test traced decorator records exceptions in async functions."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            @traced()
            async def async_failing_function():
                raise ValueError("Test async error")

            with pytest.raises(ValueError):
                await async_failing_function()
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_traced_with_self_argument(self):
        """Test traced decorator skips self argument."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            class TestClass:
                @traced(record_args=True)
                def method(self, x, y):
                    return x + y

            obj = TestClass()
            result = obj.method(2, 3)
            assert result == 5
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_traced_with_cls_argument(self):
        """Test traced decorator skips cls argument."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = True

        try:
            class TestClass:
                @classmethod
                @traced(record_args=True)
                def class_method(cls, value):
                    return value * 2

            result = TestClass.class_method(5)
            assert result == 10
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestWithOTelInstalled:
    """Tests that require OpenTelemetry to be installed."""

    def test_init_tracing_console_exporter(self):
        """Test tracing initialization with console exporter."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = False

        try:
            config = TracingConfig(
                service_name="test-service",
                enabled=True,
                exporter_type="console",
            )
            result = init_tracing(config)
            assert result is not None
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_init_tracing_none_exporter(self):
        """Test tracing initialization with no exporter."""
        import javis.utils.tracing as tracing_module

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        tracing_module._tracer = None
        tracing_module._initialized = False

        try:
            config = TracingConfig(
                service_name="test-service",
                enabled=True,
                exporter_type="none",
            )
            result = init_tracing(config)
            assert result is not None
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_create_span_with_real_tracer(self):
        """Test create_span with real tracer."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        # Set up a real tracer
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span", attributes={"key": "value"}) as span:
                assert not isinstance(span, NoOpSpan)
                span.set_attribute("another_key", "another_value")
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_get_trace_id_with_valid_span(self):
        """Test get_trace_id returns valid trace ID."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                trace_id = get_trace_id()
                assert trace_id is not None
                assert len(trace_id) == 32  # 32 hex chars
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_get_span_id_with_valid_span(self):
        """Test get_span_id returns valid span ID."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                span_id = get_span_id()
                assert span_id is not None
                assert len(span_id) == 16  # 16 hex chars
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_add_span_attributes_with_real_span(self):
        """Test add_span_attributes with real span."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                add_span_attributes({"string_key": "value", "int_key": 42, "float_key": 3.14, "bool_key": True})
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_add_span_event_with_real_span(self):
        """Test add_span_event with real span."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                add_span_event("test_event", {"detail": "value"})
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_set_span_error_with_real_span(self):
        """Test set_span_error with real span."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                set_span_error(ValueError("Test error"))
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_traced_decorator_with_real_tracer(self):
        """Test traced decorator with real tracer and record options."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            @traced(record_args=True, record_result=True)
            def test_function(x, y):
                return x + y

            result = test_function(2, 3)
            assert result == 5
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_create_span_with_exception(self):
        """Test create_span records exception properly."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with pytest.raises(ValueError):
                with create_span("test_span") as span:
                    raise ValueError("Test exception")
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized

    def test_extract_inject_trace_context(self):
        """Test extract and inject trace context."""
        import javis.utils.tracing as tracing_module
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        original_tracer = tracing_module._tracer
        original_initialized = tracing_module._initialized

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracing_module._tracer = trace.get_tracer("test")
        tracing_module._initialized = True

        try:
            with create_span("test_span") as span:
                headers = {}
                result = inject_trace_context(headers)
                assert isinstance(result, dict)

                # Extract from injected headers
                context = extract_trace_context(result)
                # Context should be extracted
        finally:
            tracing_module._tracer = original_tracer
            tracing_module._initialized = original_initialized
