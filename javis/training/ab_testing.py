"""A/B testing system for comparing model versions."""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from javis.utils.config import get_config

logger = logging.getLogger(__name__)

TestStatus = Literal["active", "paused", "completed", "cancelled"]


class ABTestConfig(BaseModel):
    """A/B test configuration."""

    test_id: str
    version_a: str  # Control version
    version_b: str  # Treatment version
    traffic_split: float  # 0.0-1.0, percentage for version B
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TestStatus = "active"
    success_metric: str = "feedback_rate"  # feedback_rate, latency, error_rate
    description: Optional[str] = None


class VersionMetrics(BaseModel):
    """Metrics for a version in an A/B test."""

    version: str
    requests: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    total_latency_ms: float = 0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests > 0 else 0

    @property
    def feedback_rate(self) -> float:
        total_feedback = self.positive_feedback + self.negative_feedback
        return self.positive_feedback / total_feedback if total_feedback > 0 else 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests > 0 else 0


class ABTestResult(BaseModel):
    """Results from an A/B test."""

    test_id: str
    version_a_metrics: VersionMetrics
    version_b_metrics: VersionMetrics
    winner: Optional[str] = None
    statistical_significance: float = 0
    sample_size_a: int = 0
    sample_size_b: int = 0
    recommendation: str = ""


class ABTestManager:
    """Manages A/B tests between model versions."""

    def __init__(self, tests_dir: Optional[Path] = None):
        config = get_config()
        self.config = config.training.ab_testing

        if tests_dir is None:
            tests_dir = Path(__file__).parent.parent.parent / "data" / "ab_tests"

        self.tests_dir = Path(tests_dir)
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        # Session assignments cache
        self._assignments: dict[str, dict[str, str]] = {}  # test_id -> session_id -> version

    def _generate_test_id(self, version_a: str, version_b: str) -> str:
        """Generate a unique test ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"test_{version_a}_{version_b}_{timestamp}"

    def _get_test_file(self, test_id: str) -> Path:
        """Get file path for a test."""
        return self.tests_dir / f"{test_id}.json"

    def _load_test(self, test_id: str) -> Optional[dict]:
        """Load test data from file."""
        file_path = self._get_test_file(test_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load test {test_id}: {e}")
            return None

    def _save_test(self, test_id: str, data: dict) -> None:
        """Save test data to file."""
        file_path = self._get_test_file(test_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_test(
        self,
        version_a: str,
        version_b: str,
        traffic_split: Optional[float] = None,
        description: Optional[str] = None,
    ) -> str:
        """Create a new A/B test.

        Args:
            version_a: Control version
            version_b: Treatment version
            traffic_split: Traffic percentage for version B (default from config)
            description: Optional test description

        Returns:
            Test ID
        """
        if traffic_split is None:
            traffic_split = self.config.default_traffic_split

        test_id = self._generate_test_id(version_a, version_b)

        config = ABTestConfig(
            test_id=test_id,
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            description=description,
        )

        data = {
            "config": config.model_dump(),
            "metrics": {
                "version_a": VersionMetrics(version=version_a).model_dump(),
                "version_b": VersionMetrics(version=version_b).model_dump(),
            },
            "assignments": {},
        }

        # Convert datetime to string for JSON
        data["config"]["start_time"] = data["config"]["start_time"].isoformat()
        if data["config"]["end_time"]:
            data["config"]["end_time"] = data["config"]["end_time"].isoformat()

        self._save_test(test_id, data)
        logger.info(f"Created A/B test: {test_id}")

        return test_id

    def get_test(self, test_id: str) -> Optional[ABTestConfig]:
        """Get test configuration.

        Args:
            test_id: Test ID

        Returns:
            ABTestConfig or None
        """
        data = self._load_test(test_id)
        if not data:
            return None

        config_data = data["config"]
        if isinstance(config_data.get("start_time"), str):
            config_data["start_time"] = datetime.fromisoformat(config_data["start_time"])
        if config_data.get("end_time") and isinstance(config_data["end_time"], str):
            config_data["end_time"] = datetime.fromisoformat(config_data["end_time"])

        return ABTestConfig(**config_data)

    def get_active_test(self) -> Optional[ABTestConfig]:
        """Get the currently active A/B test.

        Returns:
            Active ABTestConfig or None
        """
        for file_path in self.tests_dir.glob("*.json"):
            test_id = file_path.stem
            test = self.get_test(test_id)
            if test and test.status == "active":
                return test
        return None

    def list_tests(self, status: Optional[TestStatus] = None) -> list[ABTestConfig]:
        """List all A/B tests.

        Args:
            status: Filter by status

        Returns:
            List of ABTestConfig objects
        """
        tests = []
        for file_path in self.tests_dir.glob("*.json"):
            test = self.get_test(file_path.stem)
            if test:
                if status is None or test.status == status:
                    tests.append(test)

        # Sort by start time, newest first
        tests.sort(key=lambda t: t.start_time, reverse=True)
        return tests

    def assign_version(self, test_id: str, session_id: str) -> str:
        """Assign a version to a session for an A/B test.

        Uses consistent hashing to ensure the same session always gets the same version.

        Args:
            test_id: Test ID
            session_id: Session ID

        Returns:
            Assigned version string
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        config = data["config"]
        if config["status"] != "active":
            # Return control version if test is not active
            return config["version_a"]

        # Check existing assignment
        assignments = data.get("assignments", {})
        if session_id in assignments:
            return assignments[session_id]

        # Consistent hash-based assignment
        hash_input = f"{test_id}:{session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000

        if normalized < config["traffic_split"]:
            assigned = config["version_b"]
        else:
            assigned = config["version_a"]

        # Save assignment
        assignments[session_id] = assigned
        data["assignments"] = assignments
        self._save_test(test_id, data)

        return assigned

    def record_metric(
        self,
        test_id: str,
        session_id: str,
        latency_ms: float = 0,
        feedback: Optional[str] = None,
        error: bool = False,
    ) -> None:
        """Record a metric for an A/B test.

        Args:
            test_id: Test ID
            session_id: Session ID
            latency_ms: Response latency
            feedback: User feedback ("good" or "bad")
            error: Whether an error occurred
        """
        data = self._load_test(test_id)
        if not data:
            return

        # Determine which version was used
        version = data.get("assignments", {}).get(session_id)
        if not version:
            return

        # Update metrics
        metrics_key = "version_a" if version == data["config"]["version_a"] else "version_b"
        metrics = data["metrics"][metrics_key]

        metrics["requests"] += 1
        metrics["total_latency_ms"] += latency_ms

        if feedback == "good":
            metrics["positive_feedback"] += 1
        elif feedback == "bad":
            metrics["negative_feedback"] += 1

        if error:
            metrics["errors"] += 1

        self._save_test(test_id, data)

    def _calculate_statistical_significance(
        self,
        p1: float,
        n1: int,
        p2: float,
        n2: int,
    ) -> float:
        """Calculate statistical significance using two-proportion z-test.

        Args:
            p1: Proportion for version A
            n1: Sample size for version A
            p2: Proportion for version B
            n2: Sample size for version B

        Returns:
            Confidence level (0-1)
        """
        if n1 == 0 or n2 == 0:
            return 0

        # Pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

        if p_pooled == 0 or p_pooled == 1:
            return 0

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se == 0:
            return 0

        # Z-score
        z = abs(p1 - p2) / se

        # Convert to confidence level (simplified approximation)
        # Using normal distribution CDF approximation
        confidence = 1 - math.exp(-0.5 * z * z)

        return min(confidence, 0.9999)

    def calculate_results(self, test_id: str) -> ABTestResult:
        """Calculate results for an A/B test.

        Args:
            test_id: Test ID

        Returns:
            ABTestResult with analysis
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        metrics_a = VersionMetrics(**data["metrics"]["version_a"])
        metrics_b = VersionMetrics(**data["metrics"]["version_b"])

        # Determine winner based on success metric
        success_metric = data["config"].get("success_metric", "feedback_rate")

        if success_metric == "feedback_rate":
            a_value = metrics_a.feedback_rate
            b_value = metrics_b.feedback_rate
            higher_is_better = True
        elif success_metric == "latency":
            a_value = metrics_a.avg_latency_ms
            b_value = metrics_b.avg_latency_ms
            higher_is_better = False
        else:  # error_rate
            a_value = metrics_a.error_rate
            b_value = metrics_b.error_rate
            higher_is_better = False

        # Calculate statistical significance
        significance = self._calculate_statistical_significance(
            metrics_a.feedback_rate,
            metrics_a.requests,
            metrics_b.feedback_rate,
            metrics_b.requests,
        )

        # Determine winner
        winner = None
        recommendation = "Insufficient data for conclusion"

        min_sample = self.config.min_sample_size
        if metrics_a.requests >= min_sample and metrics_b.requests >= min_sample:
            if significance >= self.config.significance_threshold:
                if higher_is_better:
                    winner = metrics_a.version if a_value > b_value else metrics_b.version
                else:
                    winner = metrics_a.version if a_value < b_value else metrics_b.version

                winner_name = "A" if winner == metrics_a.version else "B"
                recommendation = f"Version {winner_name} ({winner}) is the winner with {significance*100:.1f}% confidence"
            else:
                recommendation = f"No statistically significant difference ({significance*100:.1f}% confidence)"

        return ABTestResult(
            test_id=test_id,
            version_a_metrics=metrics_a,
            version_b_metrics=metrics_b,
            winner=winner,
            statistical_significance=significance,
            sample_size_a=metrics_a.requests,
            sample_size_b=metrics_b.requests,
            recommendation=recommendation,
        )

    def complete_test(self, test_id: str, winner: Optional[str] = None) -> None:
        """Complete an A/B test.

        Args:
            test_id: Test ID
            winner: Manually specified winner (optional)
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        data["config"]["status"] = "completed"
        data["config"]["end_time"] = datetime.now().isoformat()

        if winner:
            data["declared_winner"] = winner

        self._save_test(test_id, data)
        logger.info(f"Completed A/B test: {test_id}")

    def pause_test(self, test_id: str) -> None:
        """Pause an A/B test.

        Args:
            test_id: Test ID
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        data["config"]["status"] = "paused"
        self._save_test(test_id, data)
        logger.info(f"Paused A/B test: {test_id}")

    def resume_test(self, test_id: str) -> None:
        """Resume a paused A/B test.

        Args:
            test_id: Test ID
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        if data["config"]["status"] != "paused":
            raise ValueError(f"Test {test_id} is not paused")

        data["config"]["status"] = "active"
        self._save_test(test_id, data)
        logger.info(f"Resumed A/B test: {test_id}")

    def cancel_test(self, test_id: str) -> None:
        """Cancel an A/B test.

        Args:
            test_id: Test ID
        """
        data = self._load_test(test_id)
        if not data:
            raise ValueError(f"Test not found: {test_id}")

        data["config"]["status"] = "cancelled"
        data["config"]["end_time"] = datetime.now().isoformat()
        self._save_test(test_id, data)
        logger.info(f"Cancelled A/B test: {test_id}")

    def delete_test(self, test_id: str) -> bool:
        """Delete an A/B test.

        Args:
            test_id: Test ID

        Returns:
            True if deleted
        """
        file_path = self._get_test_file(test_id)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted A/B test: {test_id}")
            return True
        return False


# Singleton instance
_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """Get the global ABTestManager instance."""
    global _manager
    if _manager is None:
        _manager = ABTestManager()
    return _manager
