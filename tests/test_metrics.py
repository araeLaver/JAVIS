"""Tests for JAVIS metrics module."""

import pytest
import time
from unittest.mock import MagicMock, patch
import asyncio

from javis.utils.metrics import (
    MetricType,
    MetricValue,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricsRegistry,
    get_registry,
    track_time,
    count_calls,
    # Pre-defined metrics
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUEST_IN_PROGRESS,
    CHAT_REQUESTS,
    CHAT_LATENCY,
    TOOL_EXECUTIONS,
    MODEL_LATENCY,
    ERRORS_TOTAL,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_init(self):
        """Test counter initialization."""
        counter = Counter("test_counter", "A test counter")
        assert counter.name == "test_counter"
        assert counter.description == "A test counter"
        assert counter.metric_type == MetricType.COUNTER

    def test_counter_inc_no_labels(self):
        """Test counter increment without labels."""
        counter = Counter("test_counter", "A test counter")
        assert counter.get() == 0
        counter.inc()
        assert counter.get() == 1
        counter.inc(5)
        assert counter.get() == 6

    def test_counter_inc_with_labels(self):
        """Test counter increment with labels."""
        counter = Counter(
            "test_counter",
            "A test counter",
            labels=["method", "status"],
        )
        counter.inc(labels={"method": "GET", "status": "200"})
        counter.inc(labels={"method": "GET", "status": "200"})
        counter.inc(labels={"method": "POST", "status": "201"})

        assert counter.get(labels={"method": "GET", "status": "200"}) == 2
        assert counter.get(labels={"method": "POST", "status": "201"}) == 1
        assert counter.get(labels={"method": "DELETE", "status": "404"}) == 0

    def test_counter_label_validation(self):
        """Test counter label validation."""
        counter = Counter(
            "test_counter",
            "A test counter",
            labels=["method", "status"],
        )
        with pytest.raises(ValueError, match="Label mismatch"):
            counter.inc(labels={"method": "GET"})  # Missing status

    def test_counter_collect(self):
        """Test counter collection."""
        counter = Counter(
            "test_counter",
            "A test counter",
            labels=["type"],
        )
        counter.inc(labels={"type": "a"})
        counter.inc(2, labels={"type": "b"})

        values = counter.collect()
        assert len(values) == 2
        assert all(isinstance(v, MetricValue) for v in values)


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_init(self):
        """Test gauge initialization."""
        gauge = Gauge("test_gauge", "A test gauge")
        assert gauge.name == "test_gauge"
        assert gauge.metric_type == MetricType.GAUGE

    def test_gauge_set(self):
        """Test gauge set."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10)
        assert gauge.get() == 10
        gauge.set(5)
        assert gauge.get() == 5

    def test_gauge_inc_dec(self):
        """Test gauge increment and decrement."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.inc()
        assert gauge.get() == 1
        gauge.inc(5)
        assert gauge.get() == 6
        gauge.dec(2)
        assert gauge.get() == 4
        gauge.dec()
        assert gauge.get() == 3

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge(
            "test_gauge",
            "A test gauge",
            labels=["endpoint"],
        )
        gauge.set(5, labels={"endpoint": "/api/chat"})
        gauge.set(3, labels={"endpoint": "/api/health"})

        assert gauge.get(labels={"endpoint": "/api/chat"}) == 5
        assert gauge.get(labels={"endpoint": "/api/health"}) == 3

    def test_gauge_collect(self):
        """Test gauge collection."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(42)
        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 42


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_init(self):
        """Test histogram initialization."""
        histogram = Histogram("test_histogram", "A test histogram")
        assert histogram.name == "test_histogram"
        assert histogram.metric_type == MetricType.HISTOGRAM
        assert histogram.buckets == Histogram.DEFAULT_BUCKETS

    def test_histogram_custom_buckets(self):
        """Test histogram with custom buckets."""
        buckets = (0.1, 0.5, 1.0, float("inf"))
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            buckets=buckets,
        )
        assert histogram.buckets == buckets

    def test_histogram_observe(self):
        """Test histogram observation."""
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(2.0)

        stats = histogram.get_stats()
        assert stats["count"] == 4
        assert stats["sum"] == pytest.approx(3.15)
        assert stats["avg"] == pytest.approx(0.7875)

    def test_histogram_buckets_count(self):
        """Test histogram bucket counting."""
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        histogram.observe(0.05)  # <= 0.1, 0.5, 1.0, inf
        histogram.observe(0.3)   # <= 0.5, 1.0, inf
        histogram.observe(0.8)   # <= 1.0, inf
        histogram.observe(2.0)   # <= inf

        stats = histogram.get_stats()
        buckets = stats["buckets"]
        assert buckets[0.1] == 1
        assert buckets[0.5] == 2
        assert buckets[1.0] == 3
        assert buckets[float("inf")] == 4

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            labels=["method"],
        )
        histogram.observe(0.1, labels={"method": "GET"})
        histogram.observe(0.2, labels={"method": "GET"})
        histogram.observe(0.5, labels={"method": "POST"})

        stats_get = histogram.get_stats(labels={"method": "GET"})
        stats_post = histogram.get_stats(labels={"method": "POST"})

        assert stats_get["count"] == 2
        assert stats_post["count"] == 1


class TestSummary:
    """Tests for Summary metric."""

    def test_summary_init(self):
        """Test summary initialization."""
        summary = Summary("test_summary", "A test summary")
        assert summary.name == "test_summary"
        assert summary.metric_type == MetricType.SUMMARY

    def test_summary_observe(self):
        """Test summary observation."""
        summary = Summary("test_summary", "A test summary")
        for i in range(100):
            summary.observe(i)

        p50 = summary.get_quantile(0.5)
        p99 = summary.get_quantile(0.99)

        assert 45 <= p50 <= 55  # Should be around 50
        assert p99 >= 95  # Should be high

    def test_summary_max_size(self):
        """Test summary max size limit."""
        summary = Summary("test_summary", "A test summary", max_size=10)
        for i in range(100):
            summary.observe(i)

        # Should only keep last 10 values
        p50 = summary.get_quantile(0.5)
        assert p50 >= 90  # Last 10 values are 90-99


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_registry_singleton(self):
        """Test registry is singleton."""
        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()
        assert registry1 is registry2

    def test_get_registry(self):
        """Test get_registry returns singleton."""
        registry = get_registry()
        assert isinstance(registry, MetricsRegistry)
        assert registry is get_registry()

    def test_registry_register(self):
        """Test metric registration."""
        registry = get_registry()
        counter = Counter("unique_test_counter", "Test counter")
        registry.register(counter)

        retrieved = registry.get("unique_test_counter")
        assert retrieved is counter

        # Cleanup
        registry.unregister("unique_test_counter")

    def test_registry_duplicate_register(self):
        """Test registering duplicate metric returns existing."""
        registry = get_registry()
        counter1 = Counter("dup_test_counter", "Test counter")
        counter2 = Counter("dup_test_counter", "Different description")

        registered1 = registry.register(counter1)
        registered2 = registry.register(counter2)

        assert registered1 is registered2
        assert registered1 is counter1

        # Cleanup
        registry.unregister("dup_test_counter")

    def test_registry_export_prometheus(self):
        """Test Prometheus format export."""
        registry = get_registry()

        # Create a fresh counter for testing
        counter = Counter("prometheus_test_counter", "Test counter", labels=["type"])
        registry.register(counter)
        counter.inc(labels={"type": "test"})
        counter.inc(5, labels={"type": "prod"})

        output = registry.export_prometheus()

        assert "# HELP prometheus_test_counter Test counter" in output
        assert "# TYPE prometheus_test_counter counter" in output
        assert 'prometheus_test_counter{type="test"} 1' in output
        assert 'prometheus_test_counter{type="prod"} 5' in output

        # Cleanup
        registry.unregister("prometheus_test_counter")


class TestTrackTimeDecorator:
    """Tests for track_time decorator."""

    def test_track_time_sync(self):
        """Test track_time with sync function."""
        histogram = Histogram("decorator_test_sync", "Test histogram")
        get_registry().register(histogram)

        @track_time(histogram)
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

        stats = histogram.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] >= 0.01

        get_registry().unregister("decorator_test_sync")

    @pytest.mark.asyncio
    async def test_track_time_async(self):
        """Test track_time with async function."""
        histogram = Histogram("decorator_test_async", "Test histogram")
        get_registry().register(histogram)

        @track_time(histogram)
        async def async_slow_function():
            await asyncio.sleep(0.01)
            return "done"

        result = await async_slow_function()
        assert result == "done"

        stats = histogram.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] >= 0.01

        get_registry().unregister("decorator_test_async")


class TestCountCallsDecorator:
    """Tests for count_calls decorator."""

    def test_count_calls_sync(self):
        """Test count_calls with sync function."""
        counter = Counter("call_count_test_sync", "Test counter")
        get_registry().register(counter)

        @count_calls(counter)
        def counted_function():
            return "done"

        counted_function()
        counted_function()
        counted_function()

        assert counter.get() == 3

        get_registry().unregister("call_count_test_sync")

    @pytest.mark.asyncio
    async def test_count_calls_async(self):
        """Test count_calls with async function."""
        counter = Counter("call_count_test_async", "Test counter")
        get_registry().register(counter)

        @count_calls(counter)
        async def async_counted_function():
            return "done"

        await async_counted_function()
        await async_counted_function()

        assert counter.get() == 2

        get_registry().unregister("call_count_test_async")


class TestPreDefinedMetrics:
    """Tests for pre-defined metrics."""

    def test_request_count_exists(self):
        """Test REQUEST_COUNT metric exists."""
        assert REQUEST_COUNT is not None
        assert REQUEST_COUNT.name == "javis_http_requests_total"
        assert "method" in REQUEST_COUNT.label_names
        assert "endpoint" in REQUEST_COUNT.label_names
        assert "status" in REQUEST_COUNT.label_names

    def test_request_latency_exists(self):
        """Test REQUEST_LATENCY metric exists."""
        assert REQUEST_LATENCY is not None
        assert REQUEST_LATENCY.name == "javis_http_request_duration_seconds"
        assert REQUEST_LATENCY.metric_type == MetricType.HISTOGRAM

    def test_request_in_progress_exists(self):
        """Test REQUEST_IN_PROGRESS metric exists."""
        assert REQUEST_IN_PROGRESS is not None
        assert REQUEST_IN_PROGRESS.name == "javis_http_requests_in_progress"
        assert REQUEST_IN_PROGRESS.metric_type == MetricType.GAUGE

    def test_chat_requests_exists(self):
        """Test CHAT_REQUESTS metric exists."""
        assert CHAT_REQUESTS is not None
        assert CHAT_REQUESTS.name == "javis_chat_requests_total"
        assert "model" in CHAT_REQUESTS.label_names
        assert "status" in CHAT_REQUESTS.label_names

    def test_chat_latency_exists(self):
        """Test CHAT_LATENCY metric exists."""
        assert CHAT_LATENCY is not None
        assert CHAT_LATENCY.name == "javis_chat_duration_seconds"
        assert "model" in CHAT_LATENCY.label_names
        assert "mode" in CHAT_LATENCY.label_names

    def test_tool_executions_exists(self):
        """Test TOOL_EXECUTIONS metric exists."""
        assert TOOL_EXECUTIONS is not None
        assert TOOL_EXECUTIONS.name == "javis_tool_executions_total"
        assert "tool" in TOOL_EXECUTIONS.label_names
        assert "status" in TOOL_EXECUTIONS.label_names

    def test_model_latency_exists(self):
        """Test MODEL_LATENCY metric exists."""
        assert MODEL_LATENCY is not None
        assert MODEL_LATENCY.name == "javis_model_duration_seconds"
        assert "model" in MODEL_LATENCY.label_names

    def test_errors_total_exists(self):
        """Test ERRORS_TOTAL metric exists."""
        assert ERRORS_TOTAL is not None
        assert ERRORS_TOTAL.name == "javis_errors_total"
        assert "type" in ERRORS_TOTAL.label_names
        assert "component" in ERRORS_TOTAL.label_names

    def test_metrics_registered_in_registry(self):
        """Test that pre-defined metrics are registered."""
        registry = get_registry()
        assert registry.get("javis_http_requests_total") is not None
        assert registry.get("javis_chat_requests_total") is not None
        assert registry.get("javis_tool_executions_total") is not None


class TestMetricValueTimestamp:
    """Tests for MetricValue timestamp."""

    def test_metric_value_auto_timestamp(self):
        """Test MetricValue auto-generates timestamp."""
        value = MetricValue(value=1.0)
        assert value.timestamp is not None

    def test_metric_value_custom_timestamp(self):
        """Test MetricValue with custom timestamp."""
        from datetime import datetime, timezone
        custom_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        value = MetricValue(value=1.0, timestamp=custom_time)
        assert value.timestamp == custom_time


class TestThreadSafety:
    """Tests for thread safety of metrics."""

    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        import threading

        counter = Counter("thread_safety_test", "Test counter")
        get_registry().register(counter)

        def increment():
            for _ in range(1000):
                counter.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.get() == 10000

        get_registry().unregister("thread_safety_test")

    def test_gauge_thread_safety(self):
        """Test gauge is thread-safe."""
        import threading

        gauge = Gauge("gauge_thread_test", "Test gauge")
        get_registry().register(gauge)

        def modify():
            for _ in range(100):
                gauge.inc()
                gauge.dec()

        threads = [threading.Thread(target=modify) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should end up at 0 since inc/dec are balanced
        assert gauge.get() == 0

        get_registry().unregister("gauge_thread_test")
