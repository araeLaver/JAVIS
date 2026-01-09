"""Performance metrics collection system."""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from javis.utils.config import get_config

logger = logging.getLogger(__name__)


class InferenceMetric(BaseModel):
    """Single inference metric record."""

    timestamp: datetime
    session_id: str
    model_version: str
    latency_ms: float
    token_count: int = 0
    feedback: Optional[Literal["good", "bad"]] = None
    error: bool = False


class HourlyMetrics(BaseModel):
    """Hourly aggregated metrics."""

    hour: int
    requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    errors: int
    positive_feedback: int
    negative_feedback: int


class DailySummary(BaseModel):
    """Daily summary metrics."""

    date: str
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    positive_feedback_rate: float
    negative_feedback_rate: float


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics for a version."""

    version: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    positive_feedback_rate: float
    negative_feedback_rate: float
    error_rate: float


class MetricsCollector:
    """Collects and aggregates model performance metrics."""

    def __init__(self, metrics_dir: Optional[Path] = None):
        config = get_config()
        self.config = config.training.metrics

        if metrics_dir is None:
            metrics_dir = Path(__file__).parent.parent.parent / "data" / "metrics"

        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for current day
        self._buffer: list[InferenceMetric] = []
        self._buffer_date: Optional[str] = None

    def _get_version_dir(self, version: str) -> Path:
        """Get directory for version metrics."""
        version_dir = self.metrics_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    def _get_daily_file(self, version: str, date: str) -> Path:
        """Get file path for daily metrics."""
        return self._get_version_dir(version) / f"{date}.json"

    def _flush_buffer(self) -> None:
        """Flush in-memory buffer to disk."""
        if not self._buffer:
            return

        # Group by version and date
        grouped: dict[tuple[str, str], list[InferenceMetric]] = {}
        for metric in self._buffer:
            date_str = metric.timestamp.strftime("%Y-%m-%d")
            key = (metric.model_version, date_str)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(metric)

        # Save each group
        for (version, date_str), metrics in grouped.items():
            self._append_to_file(version, date_str, metrics)

        self._buffer.clear()

    def _append_to_file(
        self, version: str, date_str: str, metrics: list[InferenceMetric]
    ) -> None:
        """Append metrics to daily file."""
        file_path = self._get_daily_file(version, date_str)

        # Load existing data
        existing: list[dict] = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        # Append new metrics
        for m in metrics:
            data = m.model_dump()
            data["timestamp"] = data["timestamp"].isoformat()
            existing.append(data)

        # Save
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)

    def record_inference(self, metric: InferenceMetric) -> None:
        """Record an inference metric.

        Args:
            metric: InferenceMetric to record
        """
        if not self.config.enabled:
            return

        # Check if we need to flush buffer (new day)
        today = datetime.now().strftime("%Y-%m-%d")
        if self._buffer_date != today:
            self._flush_buffer()
            self._buffer_date = today

        self._buffer.append(metric)

        # Flush if buffer is large
        if len(self._buffer) >= 100:
            self._flush_buffer()

    def record(
        self,
        session_id: str,
        model_version: str,
        latency_ms: float,
        token_count: int = 0,
        feedback: Optional[str] = None,
        error: bool = False,
    ) -> None:
        """Convenience method to record a metric.

        Args:
            session_id: Session identifier
            model_version: Model version used
            latency_ms: Response latency in milliseconds
            token_count: Number of tokens generated
            feedback: Optional feedback ("good" or "bad")
            error: Whether an error occurred
        """
        metric = InferenceMetric(
            timestamp=datetime.now(),
            session_id=session_id,
            model_version=model_version,
            latency_ms=latency_ms,
            token_count=token_count,
            feedback=feedback if feedback in ("good", "bad") else None,
            error=error,
        )
        self.record_inference(metric)

    def _load_daily_metrics(self, version: str, date_str: str) -> list[InferenceMetric]:
        """Load metrics for a specific day."""
        file_path = self._get_daily_file(version, date_str)

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = []
            for item in data:
                if isinstance(item.get("timestamp"), str):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                metrics.append(InferenceMetric(**item))
            return metrics

        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Failed to load metrics from {file_path}: {e}")
            return []

    def _calculate_p95(self, values: list[float]) -> float:
        """Calculate 95th percentile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def get_metrics(
        self,
        version: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """Get aggregated metrics for a version.

        Args:
            version: Model version
            start: Start of period (default: 7 days ago)
            end: End of period (default: now)

        Returns:
            PerformanceMetrics aggregation
        """
        # Flush buffer first
        self._flush_buffer()

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=7)

        # Collect all metrics in range
        all_metrics: list[InferenceMetric] = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            metrics = self._load_daily_metrics(version, date_str)

            # Filter by time range
            for m in metrics:
                if start <= m.timestamp <= end:
                    all_metrics.append(m)

            current += timedelta(days=1)

        if not all_metrics:
            return PerformanceMetrics(
                version=version,
                period_start=start,
                period_end=end,
                total_requests=0,
                avg_latency_ms=0,
                p95_latency_ms=0,
                positive_feedback_rate=0,
                negative_feedback_rate=0,
                error_rate=0,
            )

        # Calculate metrics
        latencies = [m.latency_ms for m in all_metrics]
        errors = sum(1 for m in all_metrics if m.error)
        positive = sum(1 for m in all_metrics if m.feedback == "good")
        negative = sum(1 for m in all_metrics if m.feedback == "bad")
        total = len(all_metrics)

        return PerformanceMetrics(
            version=version,
            period_start=start,
            period_end=end,
            total_requests=total,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=self._calculate_p95(latencies),
            positive_feedback_rate=positive / total if total > 0 else 0,
            negative_feedback_rate=negative / total if total > 0 else 0,
            error_rate=errors / total if total > 0 else 0,
        )

    def get_daily_summary(self, version: str, date_str: str) -> Optional[DailySummary]:
        """Get summary for a specific day.

        Args:
            version: Model version
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DailySummary or None if no data
        """
        metrics = self._load_daily_metrics(version, date_str)

        if not metrics:
            return None

        latencies = [m.latency_ms for m in metrics]
        errors = sum(1 for m in metrics if m.error)
        positive = sum(1 for m in metrics if m.feedback == "good")
        negative = sum(1 for m in metrics if m.feedback == "bad")
        total = len(metrics)

        return DailySummary(
            date=date_str,
            total_requests=total,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=self._calculate_p95(latencies),
            error_rate=errors / total if total > 0 else 0,
            positive_feedback_rate=positive / total if total > 0 else 0,
            negative_feedback_rate=negative / total if total > 0 else 0,
        )

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
        period_days: int = 7,
    ) -> dict:
        """Compare metrics between two versions.

        Args:
            version_a: First version
            version_b: Second version
            period_days: Number of days to compare

        Returns:
            Comparison dictionary
        """
        end = datetime.now()
        start = end - timedelta(days=period_days)

        metrics_a = self.get_metrics(version_a, start, end)
        metrics_b = self.get_metrics(version_b, start, end)

        def safe_diff(a: float, b: float) -> float:
            if a == 0:
                return 0 if b == 0 else float("inf")
            return (b - a) / a * 100

        return {
            "period_days": period_days,
            "version_a": {
                "version": version_a,
                "requests": metrics_a.total_requests,
                "avg_latency_ms": round(metrics_a.avg_latency_ms, 2),
                "p95_latency_ms": round(metrics_a.p95_latency_ms, 2),
                "positive_feedback_rate": round(metrics_a.positive_feedback_rate, 4),
                "error_rate": round(metrics_a.error_rate, 4),
            },
            "version_b": {
                "version": version_b,
                "requests": metrics_b.total_requests,
                "avg_latency_ms": round(metrics_b.avg_latency_ms, 2),
                "p95_latency_ms": round(metrics_b.p95_latency_ms, 2),
                "positive_feedback_rate": round(metrics_b.positive_feedback_rate, 4),
                "error_rate": round(metrics_b.error_rate, 4),
            },
            "comparison": {
                "latency_change_pct": round(
                    safe_diff(metrics_a.avg_latency_ms, metrics_b.avg_latency_ms), 2
                ),
                "feedback_change_pct": round(
                    safe_diff(
                        metrics_a.positive_feedback_rate,
                        metrics_b.positive_feedback_rate,
                    ),
                    2,
                ),
                "error_change_pct": round(
                    safe_diff(metrics_a.error_rate, metrics_b.error_rate), 2
                ),
            },
        }

    def cleanup_old_metrics(self, days: Optional[int] = None) -> int:
        """Remove metrics older than retention period.

        Args:
            days: Days to retain (default from config)

        Returns:
            Number of files removed
        """
        if days is None:
            days = self.config.retention_days

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for version_dir in self.metrics_dir.iterdir():
            if not version_dir.is_dir():
                continue

            for file_path in version_dir.glob("*.json"):
                try:
                    # Parse date from filename
                    date_str = file_path.stem
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")

                    if file_date < cutoff:
                        file_path.unlink()
                        removed += 1
                        logger.debug(f"Removed old metrics file: {file_path}")

                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing {file_path}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old metrics files")

        return removed

    def get_dashboard_data(self) -> dict:
        """Get data for metrics dashboard.

        Returns:
            Dashboard data dictionary
        """
        self._flush_buffer()

        # Get all versions
        versions = []
        for version_dir in self.metrics_dir.iterdir():
            if version_dir.is_dir():
                versions.append(version_dir.name)

        # Get recent metrics for each version
        end = datetime.now()
        start = end - timedelta(days=7)

        version_data = []
        for version in versions:
            metrics = self.get_metrics(version, start, end)
            version_data.append({
                "version": version,
                "requests": metrics.total_requests,
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "positive_rate": round(metrics.positive_feedback_rate * 100, 1),
                "error_rate": round(metrics.error_rate * 100, 1),
            })

        return {
            "period": "last_7_days",
            "versions": version_data,
            "total_requests": sum(v["requests"] for v in version_data),
            "generated_at": datetime.now().isoformat(),
        }


# Singleton instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global MetricsCollector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
