"""Tests for javis.training.metrics module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from javis.training.metrics import (
    InferenceMetric,
    HourlyMetrics,
    DailySummary,
    PerformanceMetrics,
    MetricsCollector,
    get_metrics_collector,
)


class TestInferenceMetric:
    """Tests for InferenceMetric model."""

    def test_create_basic_metric(self):
        """Test creating a basic metric."""
        metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id="test-session",
            model_version="v1",
            latency_ms=100.5,
        )
        assert metric.session_id == "test-session"
        assert metric.model_version == "v1"
        assert metric.latency_ms == 100.5
        assert metric.token_count == 0
        assert metric.feedback is None
        assert metric.error is False

    def test_create_full_metric(self):
        """Test creating a metric with all fields."""
        metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id="session-123",
            model_version="v2",
            latency_ms=250.0,
            token_count=100,
            feedback="good",
            error=False,
        )
        assert metric.token_count == 100
        assert metric.feedback == "good"

    def test_metric_with_error(self):
        """Test metric with error flag."""
        metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id="err-session",
            model_version="v1",
            latency_ms=50.0,
            error=True,
        )
        assert metric.error is True


class TestHourlyMetrics:
    """Tests for HourlyMetrics model."""

    def test_create_hourly_metrics(self):
        """Test creating hourly metrics."""
        metrics = HourlyMetrics(
            hour=14,
            requests=100,
            avg_latency_ms=150.5,
            p95_latency_ms=300.0,
            errors=5,
            positive_feedback=20,
            negative_feedback=3,
        )
        assert metrics.hour == 14
        assert metrics.requests == 100
        assert metrics.avg_latency_ms == 150.5
        assert metrics.p95_latency_ms == 300.0
        assert metrics.errors == 5
        assert metrics.positive_feedback == 20
        assert metrics.negative_feedback == 3


class TestDailySummary:
    """Tests for DailySummary model."""

    def test_create_daily_summary(self):
        """Test creating a daily summary."""
        summary = DailySummary(
            date="2024-01-15",
            total_requests=500,
            avg_latency_ms=175.0,
            p95_latency_ms=350.0,
            error_rate=0.02,
            positive_feedback_rate=0.8,
            negative_feedback_rate=0.05,
        )
        assert summary.date == "2024-01-15"
        assert summary.total_requests == 500
        assert summary.error_rate == 0.02


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics model."""

    def test_create_performance_metrics(self):
        """Test creating performance metrics."""
        now = datetime.now()
        start = now - timedelta(days=7)

        metrics = PerformanceMetrics(
            version="v1",
            period_start=start,
            period_end=now,
            total_requests=1000,
            avg_latency_ms=200.0,
            p95_latency_ms=500.0,
            positive_feedback_rate=0.75,
            negative_feedback_rate=0.08,
            error_rate=0.03,
        )
        assert metrics.version == "v1"
        assert metrics.total_requests == 1000
        assert metrics.positive_feedback_rate == 0.75


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def temp_metrics_dir(self):
        """Create a temporary directory for metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def collector(self, temp_metrics_dir):
        """Create a MetricsCollector with temp directory."""
        with patch("javis.training.metrics.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                training=MagicMock(
                    metrics=MagicMock(enabled=True, retention_days=30)
                )
            )
            return MetricsCollector(metrics_dir=temp_metrics_dir)

    def test_init_creates_directory(self, temp_metrics_dir):
        """Test that init creates the metrics directory."""
        sub_dir = temp_metrics_dir / "new_subdir"
        with patch("javis.training.metrics.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                training=MagicMock(metrics=MagicMock(enabled=True))
            )
            collector = MetricsCollector(metrics_dir=sub_dir)
            assert sub_dir.exists()

    def test_get_version_dir(self, collector, temp_metrics_dir):
        """Test getting version directory."""
        version_dir = collector._get_version_dir("v1")
        assert version_dir == temp_metrics_dir / "v1"
        assert version_dir.exists()

    def test_get_daily_file(self, collector, temp_metrics_dir):
        """Test getting daily file path."""
        file_path = collector._get_daily_file("v1", "2024-01-15")
        assert file_path == temp_metrics_dir / "v1" / "2024-01-15.json"

    def test_record_inference_disabled(self, temp_metrics_dir):
        """Test that recording is skipped when disabled."""
        with patch("javis.training.metrics.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                training=MagicMock(metrics=MagicMock(enabled=False))
            )
            collector = MetricsCollector(metrics_dir=temp_metrics_dir)

            metric = InferenceMetric(
                timestamp=datetime.now(),
                session_id="test",
                model_version="v1",
                latency_ms=100.0,
            )
            collector.record_inference(metric)
            assert len(collector._buffer) == 0

    def test_record_inference_adds_to_buffer(self, collector):
        """Test that recording adds metric to buffer."""
        metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id="test",
            model_version="v1",
            latency_ms=100.0,
        )
        collector.record_inference(metric)
        assert len(collector._buffer) == 1
        assert collector._buffer[0] == metric

    def test_record_convenience_method(self, collector):
        """Test the convenience record method."""
        collector.record(
            session_id="sess-1",
            model_version="v1",
            latency_ms=150.0,
            token_count=50,
            feedback="good",
            error=False,
        )
        assert len(collector._buffer) == 1
        assert collector._buffer[0].session_id == "sess-1"
        assert collector._buffer[0].feedback == "good"

    def test_record_with_invalid_feedback(self, collector):
        """Test that invalid feedback is ignored."""
        collector.record(
            session_id="sess-1",
            model_version="v1",
            latency_ms=100.0,
            feedback="invalid",
        )
        assert collector._buffer[0].feedback is None

    def test_flush_buffer_empty(self, collector):
        """Test flushing empty buffer does nothing."""
        collector._flush_buffer()  # Should not raise

    def test_flush_buffer_writes_file(self, collector, temp_metrics_dir):
        """Test that flush writes metrics to file."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        metric = InferenceMetric(
            timestamp=now,
            session_id="test",
            model_version="v1",
            latency_ms=100.0,
        )
        collector._buffer.append(metric)
        collector._flush_buffer()

        file_path = temp_metrics_dir / "v1" / f"{date_str}.json"
        assert file_path.exists()

        with open(file_path, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["session_id"] == "test"

    def test_flush_buffer_appends_to_existing(self, collector, temp_metrics_dir):
        """Test that flush appends to existing file."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Create existing file
        file_path = version_dir / f"{date_str}.json"
        existing_data = [{"session_id": "existing", "timestamp": now.isoformat(),
                         "model_version": "v1", "latency_ms": 50.0}]
        with open(file_path, "w") as f:
            json.dump(existing_data, f)

        # Add new metric and flush
        metric = InferenceMetric(
            timestamp=now,
            session_id="new",
            model_version="v1",
            latency_ms=100.0,
        )
        collector._buffer.append(metric)
        collector._flush_buffer()

        with open(file_path, "r") as f:
            data = json.load(f)
        assert len(data) == 2

    def test_append_to_file_corrupted_file(self, collector, temp_metrics_dir):
        """Test appending when existing file is corrupted."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Create corrupted file
        file_path = version_dir / f"{date_str}.json"
        with open(file_path, "w") as f:
            f.write("invalid json")

        metric = InferenceMetric(
            timestamp=now,
            session_id="new",
            model_version="v1",
            latency_ms=100.0,
        )
        collector._append_to_file("v1", date_str, [metric])

        with open(file_path, "r") as f:
            data = json.load(f)
        assert len(data) == 1

    def test_buffer_flush_on_large_buffer(self, collector, temp_metrics_dir):
        """Test that buffer is flushed when it reaches 100."""
        now = datetime.now()
        for i in range(100):
            metric = InferenceMetric(
                timestamp=now,
                session_id=f"test-{i}",
                model_version="v1",
                latency_ms=100.0,
            )
            collector.record_inference(metric)

        # Buffer should be flushed
        assert len(collector._buffer) == 0

    def test_buffer_flush_on_new_day(self, collector, temp_metrics_dir):
        """Test that buffer is flushed on date change."""
        collector._buffer_date = "2024-01-01"  # Set old date
        collector._buffer.append(
            InferenceMetric(
                timestamp=datetime(2024, 1, 1, 12, 0),
                session_id="old",
                model_version="v1",
                latency_ms=100.0,
            )
        )

        # Record new metric (should trigger flush)
        new_metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id="new",
            model_version="v1",
            latency_ms=100.0,
        )
        collector.record_inference(new_metric)

        # Old metric should have been flushed, new one in buffer
        assert len(collector._buffer) == 1
        assert collector._buffer[0].session_id == "new"

    def test_load_daily_metrics_no_file(self, collector):
        """Test loading when file doesn't exist."""
        metrics = collector._load_daily_metrics("nonexistent", "2024-01-15")
        assert metrics == []

    def test_load_daily_metrics_success(self, collector, temp_metrics_dir):
        """Test loading metrics from file."""
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        now = datetime.now()
        data = [
            {
                "timestamp": now.isoformat(),
                "session_id": "test",
                "model_version": "v1",
                "latency_ms": 100.0,
                "token_count": 0,
                "feedback": None,
                "error": False,
            }
        ]

        file_path = version_dir / "2024-01-15.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        metrics = collector._load_daily_metrics("v1", "2024-01-15")
        assert len(metrics) == 1
        assert metrics[0].session_id == "test"

    def test_load_daily_metrics_corrupted(self, collector, temp_metrics_dir):
        """Test loading corrupted file returns empty list."""
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        file_path = version_dir / "2024-01-15.json"
        with open(file_path, "w") as f:
            f.write("not valid json")

        metrics = collector._load_daily_metrics("v1", "2024-01-15")
        assert metrics == []

    def test_calculate_p95_empty(self, collector):
        """Test p95 calculation with empty list."""
        assert collector._calculate_p95([]) == 0

    def test_calculate_p95_single(self, collector):
        """Test p95 calculation with single value."""
        assert collector._calculate_p95([100.0]) == 100.0

    def test_calculate_p95_multiple(self, collector):
        """Test p95 calculation with multiple values."""
        values = list(range(1, 101))  # 1 to 100
        p95 = collector._calculate_p95(values)
        # idx = int(100 * 0.95) = 95, sorted_values[95] = 96 (0-indexed)
        assert p95 == 96

    def test_get_metrics_no_data(self, collector):
        """Test getting metrics with no data."""
        now = datetime.now()
        start = now - timedelta(days=1)

        metrics = collector.get_metrics("v1", start, now)
        assert metrics.version == "v1"
        assert metrics.total_requests == 0
        assert metrics.avg_latency_ms == 0
        assert metrics.positive_feedback_rate == 0

    def test_get_metrics_with_data(self, collector, temp_metrics_dir):
        """Test getting aggregated metrics."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        data = [
            {
                "timestamp": now.isoformat(),
                "session_id": "s1",
                "model_version": "v1",
                "latency_ms": 100.0,
                "token_count": 10,
                "feedback": "good",
                "error": False,
            },
            {
                "timestamp": now.isoformat(),
                "session_id": "s2",
                "model_version": "v1",
                "latency_ms": 200.0,
                "token_count": 20,
                "feedback": "bad",
                "error": False,
            },
            {
                "timestamp": now.isoformat(),
                "session_id": "s3",
                "model_version": "v1",
                "latency_ms": 150.0,
                "token_count": 15,
                "feedback": None,
                "error": True,
            },
        ]

        file_path = version_dir / f"{date_str}.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        start = now - timedelta(hours=1)
        metrics = collector.get_metrics("v1", start, now + timedelta(hours=1))

        assert metrics.total_requests == 3
        assert metrics.avg_latency_ms == 150.0
        assert metrics.positive_feedback_rate == pytest.approx(1/3)
        assert metrics.negative_feedback_rate == pytest.approx(1/3)
        assert metrics.error_rate == pytest.approx(1/3)

    def test_get_metrics_default_period(self, collector):
        """Test getting metrics with default period."""
        metrics = collector.get_metrics("v1")
        assert metrics.version == "v1"
        # Default is 7 days
        assert (metrics.period_end - metrics.period_start).days == 7

    def test_get_daily_summary_no_data(self, collector):
        """Test daily summary with no data."""
        summary = collector.get_daily_summary("v1", "2024-01-15")
        assert summary is None

    def test_get_daily_summary_with_data(self, collector, temp_metrics_dir):
        """Test daily summary with data."""
        now = datetime.now()
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        data = [
            {
                "timestamp": now.isoformat(),
                "session_id": "s1",
                "model_version": "v1",
                "latency_ms": 100.0,
                "token_count": 10,
                "feedback": "good",
                "error": False,
            },
            {
                "timestamp": now.isoformat(),
                "session_id": "s2",
                "model_version": "v1",
                "latency_ms": 200.0,
                "token_count": 20,
                "feedback": None,
                "error": True,
            },
        ]

        file_path = version_dir / "2024-01-15.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        summary = collector.get_daily_summary("v1", "2024-01-15")
        assert summary is not None
        assert summary.date == "2024-01-15"
        assert summary.total_requests == 2
        assert summary.avg_latency_ms == 150.0
        assert summary.error_rate == 0.5
        assert summary.positive_feedback_rate == 0.5

    def test_compare_versions(self, collector, temp_metrics_dir):
        """Test comparing two versions."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Create v1 data
        v1_dir = temp_metrics_dir / "v1"
        v1_dir.mkdir(parents=True)
        with open(v1_dir / f"{date_str}.json", "w") as f:
            json.dump([{
                "timestamp": now.isoformat(),
                "session_id": "s1",
                "model_version": "v1",
                "latency_ms": 100.0,
                "token_count": 10,
                "feedback": "good",
                "error": False,
            }], f)

        # Create v2 data
        v2_dir = temp_metrics_dir / "v2"
        v2_dir.mkdir(parents=True)
        with open(v2_dir / f"{date_str}.json", "w") as f:
            json.dump([{
                "timestamp": now.isoformat(),
                "session_id": "s2",
                "model_version": "v2",
                "latency_ms": 80.0,
                "token_count": 12,
                "feedback": "good",
                "error": False,
            }], f)

        comparison = collector.compare_versions("v1", "v2", period_days=1)

        assert comparison["version_a"]["version"] == "v1"
        assert comparison["version_b"]["version"] == "v2"
        assert comparison["period_days"] == 1
        assert "latency_change_pct" in comparison["comparison"]

    def test_compare_versions_safe_diff_zero(self, collector, temp_metrics_dir):
        """Test version comparison with zero values."""
        comparison = collector.compare_versions("empty1", "empty2", period_days=1)
        assert comparison["version_a"]["requests"] == 0
        assert comparison["version_b"]["requests"] == 0

    def test_cleanup_old_metrics(self, collector, temp_metrics_dir):
        """Test cleaning up old metrics files."""
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        # Create old file (more than 30 days old)
        old_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        old_file = version_dir / f"{old_date}.json"
        with open(old_file, "w") as f:
            json.dump([], f)

        # Create recent file
        recent_date = datetime.now().strftime("%Y-%m-%d")
        recent_file = version_dir / f"{recent_date}.json"
        with open(recent_file, "w") as f:
            json.dump([], f)

        removed = collector.cleanup_old_metrics(days=30)

        assert removed == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_cleanup_old_metrics_invalid_filename(self, collector, temp_metrics_dir):
        """Test cleanup with invalid filename."""
        version_dir = temp_metrics_dir / "v1"
        version_dir.mkdir(parents=True)

        # Create file with invalid date format
        invalid_file = version_dir / "invalid-date.json"
        with open(invalid_file, "w") as f:
            json.dump([], f)

        removed = collector.cleanup_old_metrics(days=30)
        assert removed == 0  # Invalid file not removed
        assert invalid_file.exists()

    def test_cleanup_skips_non_directories(self, collector, temp_metrics_dir):
        """Test cleanup skips files in root."""
        # Create a file in metrics root
        root_file = temp_metrics_dir / "some_file.txt"
        with open(root_file, "w") as f:
            f.write("test")

        removed = collector.cleanup_old_metrics(days=30)
        assert removed == 0

    def test_get_dashboard_data_empty(self, collector):
        """Test getting dashboard data with no data."""
        data = collector.get_dashboard_data()

        assert data["period"] == "last_7_days"
        assert data["versions"] == []
        assert data["total_requests"] == 0
        assert "generated_at" in data

    def test_get_dashboard_data_with_versions(self, collector, temp_metrics_dir):
        """Test getting dashboard data with version data."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Create v1 data
        v1_dir = temp_metrics_dir / "v1"
        v1_dir.mkdir(parents=True)
        with open(v1_dir / f"{date_str}.json", "w") as f:
            json.dump([{
                "timestamp": now.isoformat(),
                "session_id": "s1",
                "model_version": "v1",
                "latency_ms": 100.0,
                "token_count": 10,
                "feedback": "good",
                "error": False,
            }], f)

        data = collector.get_dashboard_data()

        assert len(data["versions"]) == 1
        assert data["versions"][0]["version"] == "v1"
        assert data["versions"][0]["requests"] == 1
        assert data["total_requests"] == 1


class TestGetMetricsCollector:
    """Tests for get_metrics_collector singleton."""

    def test_returns_collector(self):
        """Test that it returns a MetricsCollector."""
        import javis.training.metrics as metrics_module

        # Reset singleton
        metrics_module._collector = None

        with patch("javis.training.metrics.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                training=MagicMock(metrics=MagicMock(enabled=True, retention_days=30))
            )
            collector = get_metrics_collector()
            assert isinstance(collector, MetricsCollector)

    def test_returns_same_instance(self):
        """Test that it returns the same instance."""
        import javis.training.metrics as metrics_module

        # Reset singleton
        metrics_module._collector = None

        with patch("javis.training.metrics.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                training=MagicMock(metrics=MagicMock(enabled=True, retention_days=30))
            )
            collector1 = get_metrics_collector()
            collector2 = get_metrics_collector()
            assert collector1 is collector2
