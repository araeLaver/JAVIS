"""Prometheus metrics collection for JAVIS.

Provides metrics collection, tracking, and export for monitoring.
Supports request tracking, business metrics, and system health.
"""

import functools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""

    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class HistogramBucket:
    """Histogram bucket with upper bound."""

    le: float  # Less than or equal
    count: int = 0


class Metric:
    """Base metric class."""

    def __init__(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Optional[list[str]] = None,
    ):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.label_names = labels or []
        self._lock = Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
        """Validate label names match definition."""
        provided = set(labels.keys())
        expected = set(self.label_names)
        if provided != expected:
            raise ValueError(
                f"Label mismatch for {self.name}: "
                f"expected {expected}, got {provided}"
            )


class Counter(Metric):
    """A counter metric that can only increase."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ):
        super().__init__(name, description, MetricType.COUNTER, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, value: float = 1, labels: Optional[dict[str, str]] = None) -> None:
        """Increment counter."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get current counter value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values[key]

    def collect(self) -> list[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Gauge(Metric):
    """A gauge metric that can go up and down."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ):
        super().__init__(name, description, MetricType.GAUGE, labels)
        self._values: dict[tuple, float] = defaultdict(float)

    def set(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set gauge value."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1, labels: Optional[dict[str, str]] = None) -> None:
        """Increment gauge."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1, labels: Optional[dict[str, str]] = None) -> None:
        """Decrement gauge."""
        self.inc(-value, labels)

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get current gauge value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values[key]

    def collect(self) -> list[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Histogram(Metric):
    """A histogram metric for tracking distributions."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
        buckets: Optional[tuple[float, ...]] = None,
    ):
        super().__init__(name, description, MetricType.HISTOGRAM, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple, dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record an observation."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = tuple(sorted(labels.items()))
        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

    def get_stats(self, labels: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """Get histogram statistics."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            total = self._totals[key]
            return {
                "count": total,
                "sum": self._sums[key],
                "avg": self._sums[key] / total if total > 0 else 0,
                "buckets": dict(self._counts[key]),
            }

    def collect(self) -> list[MetricValue]:
        """Collect all metric values in Prometheus format."""
        result = []
        with self._lock:
            for key, buckets in self._counts.items():
                labels = dict(key)
                # Bucket values
                for le, count in buckets.items():
                    bucket_labels = {**labels, "le": str(le)}
                    result.append(MetricValue(value=count, labels=bucket_labels))
                # Sum
                result.append(MetricValue(
                    value=self._sums[key],
                    labels={**labels, "__type__": "sum"}
                ))
                # Count
                result.append(MetricValue(
                    value=self._totals[key],
                    labels={**labels, "__type__": "count"}
                ))
        return result


class Summary(Metric):
    """A summary metric for tracking distributions with quantiles."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
        max_age_seconds: int = 600,
        max_size: int = 1000,
    ):
        super().__init__(name, description, MetricType.SUMMARY, labels)
        self._values: dict[tuple, list[float]] = defaultdict(list)
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._max_size = max_size

    def observe(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record an observation."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = tuple(sorted(labels.items()))
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            self._values[key].append(value)
            # Trim if too many values
            if len(self._values[key]) > self._max_size:
                self._values[key] = self._values[key][-self._max_size:]

    def get_quantile(
        self,
        quantile: float,
        labels: Optional[dict[str, str]] = None
    ) -> float:
        """Get a specific quantile value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            values = sorted(self._values[key])
            if not values:
                return 0
            index = int(len(values) * quantile)
            return values[min(index, len(values) - 1)]

    def collect(self) -> list[MetricValue]:
        """Collect all metric values."""
        result = []
        quantiles = [0.5, 0.9, 0.95, 0.99]
        with self._lock:
            for key in self._values.keys():
                labels = dict(key)
                for q in quantiles:
                    q_labels = {**labels, "quantile": str(q)}
                    result.append(MetricValue(
                        value=self.get_quantile(q, labels),
                        labels=q_labels
                    ))
                result.append(MetricValue(
                    value=self._sums[key],
                    labels={**labels, "__type__": "sum"}
                ))
                result.append(MetricValue(
                    value=self._counts[key],
                    labels={**labels, "__type__": "count"}
                ))
        return result


class MetricsRegistry:
    """Registry for managing all metrics."""

    _instance: Optional["MetricsRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics: dict[str, Metric] = {}
            cls._instance._lock = Lock()
        return cls._instance

    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]
            self._metrics[metric.name] = metric
            return metric

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def unregister(self, name: str) -> None:
        """Unregister a metric."""
        with self._lock:
            self._metrics.pop(name, None)

    def collect_all(self) -> dict[str, list[MetricValue]]:
        """Collect all metrics."""
        with self._lock:
            return {
                name: metric.collect()
                for name, metric in self._metrics.items()
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, metric in self._metrics.items():
                # Add HELP and TYPE
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric.metric_type.value}")

                # Add metric values
                for mv in metric.collect():
                    label_str = ""
                    if mv.labels:
                        # Filter out internal labels
                        display_labels = {
                            k: v for k, v in mv.labels.items()
                            if not k.startswith("__")
                        }
                        if display_labels:
                            label_parts = [
                                f'{k}="{v}"' for k, v in display_labels.items()
                            ]
                            label_str = "{" + ",".join(label_parts) + "}"

                    # Determine metric name suffix
                    suffix = ""
                    if "__type__" in mv.labels:
                        suffix = "_" + mv.labels["__type__"]
                    elif "le" in mv.labels:
                        suffix = "_bucket"

                    lines.append(f"{name}{suffix}{label_str} {mv.value}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._metrics.clear()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return MetricsRegistry()


# ==================== Pre-defined Metrics ====================


# Request metrics
REQUEST_COUNT = Counter(
    "javis_http_requests_total",
    "Total HTTP requests",
    labels=["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "javis_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labels=["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

REQUEST_IN_PROGRESS = Gauge(
    "javis_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    labels=["method", "endpoint"],
)

# Chat metrics
CHAT_REQUESTS = Counter(
    "javis_chat_requests_total",
    "Total chat requests",
    labels=["model", "status"],
)

CHAT_LATENCY = Histogram(
    "javis_chat_duration_seconds",
    "Chat request duration in seconds",
    labels=["model", "mode"],  # mode: with_tools, simple
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

CHAT_TOKENS = Counter(
    "javis_chat_tokens_total",
    "Total tokens used in chat",
    labels=["type"],  # prompt, completion
)

# Tool metrics
TOOL_EXECUTIONS = Counter(
    "javis_tool_executions_total",
    "Total tool executions",
    labels=["tool", "status"],  # status: success, error
)

TOOL_LATENCY = Histogram(
    "javis_tool_duration_seconds",
    "Tool execution duration in seconds",
    labels=["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, float("inf")),
)

# Model metrics
MODEL_REQUESTS = Counter(
    "javis_model_requests_total",
    "Total model API requests",
    labels=["model", "status"],
)

MODEL_LATENCY = Histogram(
    "javis_model_duration_seconds",
    "Model API request duration in seconds",
    labels=["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "javis_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    labels=["name"],
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "javis_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    labels=["name"],
)

# Memory/RAG metrics
MEMORY_OPERATIONS = Counter(
    "javis_memory_operations_total",
    "Total memory operations",
    labels=["operation"],  # store, retrieve, search
)

RAG_OPERATIONS = Counter(
    "javis_rag_operations_total",
    "Total RAG operations",
    labels=["operation"],  # index, search
)

# Error metrics
ERRORS = Counter(
    "javis_errors_total",
    "Total errors",
    labels=["type", "component"],
)

# System metrics
ACTIVE_SESSIONS = Gauge(
    "javis_active_sessions",
    "Number of active sessions",
)

CACHE_HITS = Counter(
    "javis_cache_hits_total",
    "Total cache hits",
    labels=["cache_type"],
)

CACHE_MISSES = Counter(
    "javis_cache_misses_total",
    "Total cache misses",
    labels=["cache_type"],
)


# Register all default metrics
def _register_default_metrics():
    """Register all default metrics with the global registry."""
    registry = get_registry()
    default_metrics = [
        REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS,
        CHAT_REQUESTS, CHAT_LATENCY, CHAT_TOKENS,
        TOOL_EXECUTIONS, TOOL_LATENCY,
        MODEL_REQUESTS, MODEL_LATENCY,
        CIRCUIT_BREAKER_STATE, CIRCUIT_BREAKER_FAILURES,
        MEMORY_OPERATIONS, RAG_OPERATIONS,
        ERRORS, ACTIVE_SESSIONS,
        CACHE_HITS, CACHE_MISSES,
    ]
    for metric in default_metrics:
        registry.register(metric)


_register_default_metrics()


# ==================== Decorators ====================


def track_time(
    histogram: Histogram,
    labels: Optional[dict[str, str]] = None,
):
    """Decorator to track function execution time.

    Args:
        histogram: Histogram metric to record to
        labels: Labels to apply

    Example:
        @track_time(TOOL_LATENCY, {"tool_name": "web_search"})
        async def web_search():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    histogram.observe(time.time() - start, labels)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    histogram.observe(time.time() - start, labels)
            return sync_wrapper
    return decorator


def count_calls(
    counter: Counter,
    labels: Optional[dict[str, str]] = None,
):
    """Decorator to count function calls.

    Args:
        counter: Counter metric to increment
        labels: Labels to apply

    Example:
        @count_calls(TOOL_EXECUTIONS, {"tool_name": "calculator"})
        def calculate():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                counter.inc(labels=labels)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                counter.inc(labels=labels)
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator


# Import asyncio for decorator checks
import asyncio


# Alias for common metric names
ERRORS_TOTAL = ERRORS  # Alias for backwards compatibility
