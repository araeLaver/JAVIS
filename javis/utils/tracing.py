"""Distributed tracing with OpenTelemetry for JAVIS.

Provides request tracing, span creation, and context propagation
for debugging and performance analysis across services.
"""

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic decorator
T = TypeVar("T")

# Check if OpenTelemetry is available
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, Span, Tracer
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap, get_global_textmap
    from opentelemetry.context import Context

    OTEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not installed. Tracing will be disabled.")
    # Define stubs for when OTel is not available
    trace = None
    TracerProvider = None
    Span = None
    Tracer = None
    StatusCode = None
    Status = None
    Context = None


# Global tracer instance
_tracer: Optional["Tracer"] = None
_initialized: bool = False


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "javis",
        enabled: bool = True,
        exporter_type: str = "console",  # "console", "otlp", "none"
        otlp_endpoint: Optional[str] = None,
        sample_rate: float = 1.0,
        debug: bool = False,
    ):
        """Initialize tracing configuration.

        Args:
            service_name: Name of the service for tracing
            enabled: Whether tracing is enabled
            exporter_type: Type of span exporter ("console", "otlp", "none")
            otlp_endpoint: OTLP collector endpoint (for "otlp" exporter)
            sample_rate: Sampling rate (0.0 to 1.0)
            debug: Enable debug mode with more verbose output
        """
        self.service_name = service_name
        self.enabled = enabled
        self.exporter_type = exporter_type
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.sample_rate = sample_rate
        self.debug = debug

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "javis"),
            enabled=os.getenv("JAVIS_TRACING_ENABLED", "true").lower() in ("true", "1", "yes"),
            exporter_type=os.getenv("JAVIS_TRACING_EXPORTER", "console"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            sample_rate=float(os.getenv("JAVIS_TRACING_SAMPLE_RATE", "1.0")),
            debug=os.getenv("JAVIS_TRACING_DEBUG", "false").lower() in ("true", "1", "yes"),
        )


def init_tracing(config: Optional[TracingConfig] = None) -> Optional["Tracer"]:
    """Initialize OpenTelemetry tracing.

    Args:
        config: Tracing configuration (uses env vars if not provided)

    Returns:
        Tracer instance or None if tracing is disabled/unavailable
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer

    if not OTEL_AVAILABLE:
        logger.info("OpenTelemetry not available, tracing disabled")
        _initialized = True
        return None

    config = config or TracingConfig.from_env()

    if not config.enabled:
        logger.info("Tracing disabled by configuration")
        _initialized = True
        return None

    try:
        # Create resource with service info
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            "service.version": "0.1.0",
            "deployment.environment": os.getenv("JAVIS_ENV", "development"),
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter based on type
        if config.exporter_type == "console":
            # Console exporter for development
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            logger.info("Tracing initialized with console exporter")

        elif config.exporter_type == "otlp":
            # OTLP exporter for production
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                logger.info(f"Tracing initialized with OTLP exporter: {config.otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available, falling back to console")
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)

        elif config.exporter_type == "none":
            logger.info("Tracing initialized with no exporter (spans will be discarded)")

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Set up context propagation
        set_global_textmap(TraceContextTextMapPropagator())

        # Get tracer instance
        _tracer = trace.get_tracer(config.service_name, "0.1.0")
        _initialized = True

        logger.info(f"Tracing initialized for service: {config.service_name}")
        return _tracer

    except Exception as e:
        logger.exception(f"Failed to initialize tracing: {e}")
        _initialized = True
        return None


def get_tracer() -> Optional["Tracer"]:
    """Get the global tracer instance.

    Returns:
        Tracer instance or None if not initialized/available
    """
    global _tracer, _initialized

    if not _initialized:
        init_tracing()

    return _tracer


def get_current_span() -> Optional["Span"]:
    """Get the current active span.

    Returns:
        Current span or None if not available
    """
    if not OTEL_AVAILABLE:
        return None

    return trace.get_current_span()


def get_trace_id() -> Optional[str]:
    """Get the current trace ID as a hex string.

    Returns:
        Trace ID string or None if not available
    """
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_span_id() -> Optional[str]:
    """Get the current span ID as a hex string.

    Returns:
        Span ID string or None if not available
    """
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None


@contextmanager
def create_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    kind: Optional[Any] = None,
):
    """Create a new span as a context manager.

    Args:
        name: Name of the span
        attributes: Optional attributes to attach to the span
        kind: Span kind (CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER)

    Yields:
        The created span or a no-op context if tracing is disabled

    Example:
        with create_span("process_request", {"user_id": "123"}) as span:
            # Do work
            span.set_attribute("result", "success")
    """
    tracer = get_tracer()

    if tracer is None:
        # Return a no-op context
        yield NoOpSpan()
        return

    if kind is None:
        kind = trace.SpanKind.INTERNAL

    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op set_attribute."""
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """No-op set_attributes."""
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """No-op add_event."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op set_status."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op record_exception."""
        pass

    def get_span_context(self) -> None:
        """No-op get_span_context."""
        return None


def traced(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
):
    """Decorator to trace function execution.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Static attributes to attach
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value as attribute

    Example:
        @traced(attributes={"component": "chat_engine"})
        async def process_message(message: str):
            return "response"

        @traced(record_args=True, record_result=True)
        def calculate(x: int, y: int):
            return x + y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                span_attrs = dict(attributes or {})
                span_attrs["function.name"] = func.__name__
                span_attrs["function.module"] = func.__module__

                if record_args:
                    # Record positional args (skip self/cls)
                    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    start_idx = 1 if arg_names and arg_names[0] in ("self", "cls") else 0
                    for i, arg in enumerate(args[start_idx:], start=start_idx):
                        if i < len(arg_names):
                            span_attrs[f"arg.{arg_names[i]}"] = _safe_str(arg)
                    for k, v in kwargs.items():
                        span_attrs[f"arg.{k}"] = _safe_str(v)

                with create_span(span_name, span_attrs) as span:
                    result = await func(*args, **kwargs)
                    if record_result and not isinstance(span, NoOpSpan):
                        span.set_attribute("result", _safe_str(result))
                    return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                span_attrs = dict(attributes or {})
                span_attrs["function.name"] = func.__name__
                span_attrs["function.module"] = func.__module__

                if record_args:
                    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    start_idx = 1 if arg_names and arg_names[0] in ("self", "cls") else 0
                    for i, arg in enumerate(args[start_idx:], start=start_idx):
                        if i < len(arg_names):
                            span_attrs[f"arg.{arg_names[i]}"] = _safe_str(arg)
                    for k, v in kwargs.items():
                        span_attrs[f"arg.{k}"] = _safe_str(v)

                with create_span(span_name, span_attrs) as span:
                    result = func(*args, **kwargs)
                    if record_result and not isinstance(span, NoOpSpan):
                        span.set_attribute("result", _safe_str(result))
                    return result

            return sync_wrapper
    return decorator


def _safe_str(value: Any, max_length: int = 256) -> str:
    """Convert value to string safely, truncating if too long.

    Args:
        value: Value to convert
        max_length: Maximum string length

    Returns:
        String representation
    """
    try:
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
    except Exception:
        return "<unable to convert>"


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span.

    Args:
        attributes: Dictionary of attributes to add
    """
    span = get_current_span()
    if span and not isinstance(span, NoOpSpan):
        for key, value in attributes.items():
            span.set_attribute(key, _safe_str(value) if not isinstance(value, (int, float, bool)) else value)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    span = get_current_span()
    if span and not isinstance(span, NoOpSpan):
        span.add_event(name, attributes)


def set_span_error(error: Exception) -> None:
    """Mark the current span as error.

    Args:
        error: The exception that occurred
    """
    if not OTEL_AVAILABLE:
        return

    span = get_current_span()
    if span:
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)


def extract_trace_context(headers: dict[str, str]) -> Optional["Context"]:
    """Extract trace context from HTTP headers.

    Args:
        headers: HTTP headers dictionary

    Returns:
        Context object or None
    """
    if not OTEL_AVAILABLE:
        return None

    propagator = get_global_textmap()
    return propagator.extract(carrier=headers)


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject trace context into HTTP headers.

    Args:
        headers: HTTP headers dictionary to inject into

    Returns:
        Headers with trace context
    """
    if not OTEL_AVAILABLE:
        return headers

    propagator = get_global_textmap()
    propagator.inject(carrier=headers)
    return headers


# FastAPI middleware integration
def setup_fastapi_tracing(app) -> None:
    """Set up OpenTelemetry instrumentation for FastAPI.

    Args:
        app: FastAPI application instance
    """
    if not OTEL_AVAILABLE:
        logger.info("OpenTelemetry not available, skipping FastAPI instrumentation")
        return

    # Initialize tracing if not already done
    init_tracing()

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,metrics,docs,redoc,openapi.json",
        )
        logger.info("FastAPI instrumentation enabled")

    except ImportError:
        logger.warning("FastAPI instrumentation not available")
    except Exception as e:
        logger.exception(f"Failed to instrument FastAPI: {e}")


def setup_httpx_tracing() -> None:
    """Set up OpenTelemetry instrumentation for httpx client."""
    if not OTEL_AVAILABLE:
        return

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("httpx instrumentation enabled")

    except ImportError:
        logger.warning("httpx instrumentation not available")
    except Exception as e:
        logger.exception(f"Failed to instrument httpx: {e}")


# Import asyncio for decorator checks
import asyncio
