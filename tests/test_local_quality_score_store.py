"""Tests for javis.storage.local.quality_score_store module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from javis.storage.local.quality_score_store import LocalQualityScoreStore
from javis.storage.base import QualityScoreRecord


class TestLocalQualityScoreStore:
    """Tests for LocalQualityScoreStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create quality score store with temp directory."""
        return LocalQualityScoreStore(scores_dir=temp_dir)

    @pytest.fixture
    def sample_score(self):
        """Create sample quality score record."""
        return QualityScoreRecord(
            conversation_id="conv-001",
            overall_score=4.5,
            length_score=4.0,
            coherence_score=5.0,
            feedback_score=4.5,
            recency_score=4.0,
            computed_at=datetime.now(),
        )

    def test_init_creates_directory(self, temp_dir):
        """Test initialization creates scores directory."""
        scores_dir = temp_dir / "scores"
        assert not scores_dir.exists()

        store = LocalQualityScoreStore(scores_dir=scores_dir)

        assert scores_dir.exists()

    def test_init_default_directory(self):
        """Test initialization with default directory."""
        store = LocalQualityScoreStore()
        assert store.scores_dir.exists()

    def test_init_loads_existing_cache(self, temp_dir):
        """Test initialization loads existing cache file."""
        cache_file = temp_dir / "scores_cache.json"
        cache_data = {
            "conv-001": {
                "conversation_id": "conv-001",
                "overall_score": 4.0,
                "length_score": 4.0,
                "coherence_score": 4.0,
                "feedback_score": 4.0,
                "recency_score": 4.0,
                "computed_at": "2024-01-01T10:00:00",
            }
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        store = LocalQualityScoreStore(scores_dir=temp_dir)

        assert "conv-001" in store._cache

    def test_init_handles_corrupted_cache(self, temp_dir):
        """Test initialization handles corrupted cache file."""
        cache_file = temp_dir / "scores_cache.json"
        cache_file.write_text("not valid json {{{")

        store = LocalQualityScoreStore(scores_dir=temp_dir)

        assert store._cache == {}

    def test_save_score(self, store, sample_score):
        """Test saving a quality score."""
        result = store.save(sample_score)

        assert result == "conv-001"
        assert "conv-001" in store._cache

    def test_get_existing_score(self, store, sample_score):
        """Test getting an existing score."""
        store.save(sample_score)

        result = store.get("conv-001")

        assert result is not None
        assert result.conversation_id == "conv-001"
        assert result.overall_score == 4.5

    def test_get_nonexistent_score(self, store):
        """Test getting non-existent score returns None."""
        result = store.get("nonexistent")
        assert result is None

    def test_list_all_empty(self, store):
        """Test listing all scores when empty."""
        result = store.list_all()
        assert result == []

    def test_list_all_multiple(self, store):
        """Test listing multiple scores."""
        for i in range(3):
            score = QualityScoreRecord(
                conversation_id=f"conv-{i}",
                overall_score=3.0 + i * 0.5,
                length_score=4.0,
                coherence_score=4.0,
                feedback_score=4.0,
                recency_score=4.0,
                computed_at=datetime.now(),
            )
            store.save(score)

        result = store.list_all()

        assert len(result) == 3
        # Should be sorted by overall_score descending
        assert result[0].overall_score == 4.0

    def test_list_all_with_min_score(self, store):
        """Test listing scores with minimum score filter."""
        for i in range(5):
            score = QualityScoreRecord(
                conversation_id=f"conv-{i}",
                overall_score=float(i + 1),
                length_score=4.0,
                coherence_score=4.0,
                feedback_score=4.0,
                recency_score=4.0,
                computed_at=datetime.now(),
            )
            store.save(score)

        result = store.list_all(min_score=3.0)

        assert len(result) == 3  # scores 3.0, 4.0, 5.0

    def test_list_all_with_limit(self, store):
        """Test listing scores with limit."""
        for i in range(10):
            score = QualityScoreRecord(
                conversation_id=f"conv-{i}",
                overall_score=float(i),
                length_score=4.0,
                coherence_score=4.0,
                feedback_score=4.0,
                recency_score=4.0,
                computed_at=datetime.now(),
            )
            store.save(score)

        result = store.list_all(limit=5)

        assert len(result) == 5

    def test_get_high_quality_ids(self, store):
        """Test getting high quality conversation IDs."""
        scores = [
            (1.0, "low-1"),
            (2.5, "low-2"),
            (3.5, "high-1"),
            (4.5, "high-2"),
            (5.0, "high-3"),
        ]
        for overall, conv_id in scores:
            score = QualityScoreRecord(
                conversation_id=conv_id,
                overall_score=overall,
                length_score=4.0,
                coherence_score=4.0,
                feedback_score=4.0,
                recency_score=4.0,
                computed_at=datetime.now(),
            )
            store.save(score)

        result = store.get_high_quality_ids(min_score=3.5)

        assert len(result) == 3
        assert "high-1" in result
        assert "low-1" not in result

    def test_delete_existing(self, store, sample_score):
        """Test deleting existing score."""
        store.save(sample_score)
        assert store.get("conv-001") is not None

        result = store.delete("conv-001")

        assert result is True
        assert store.get("conv-001") is None

    def test_delete_nonexistent(self, store):
        """Test deleting non-existent score returns False."""
        result = store.delete("nonexistent")
        assert result is False

    def test_clear(self, store, sample_score):
        """Test clearing all scores."""
        store.save(sample_score)
        assert len(store._cache) > 0

        store.clear()

        assert len(store._cache) == 0

    def test_export_report_default_path(self, store, sample_score, temp_dir):
        """Test exporting report with default path."""
        store.save(sample_score)

        result = store.export_report()

        assert result.exists()
        assert "quality_report" in result.name

        with open(result) as f:
            report = json.load(f)
        assert "statistics" in report
        assert "scores" in report

    def test_export_report_custom_path(self, store, sample_score, temp_dir):
        """Test exporting report with custom path."""
        store.save(sample_score)
        output_path = temp_dir / "custom_report.json"

        result = store.export_report(output_path=output_path)

        assert result == output_path
        assert output_path.exists()

    def test_export_report_statistics(self, store):
        """Test export report includes correct statistics."""
        for i in range(4):
            score = QualityScoreRecord(
                conversation_id=f"conv-{i}",
                overall_score=float(i + 2),  # 2.0, 3.0, 4.0, 5.0
                length_score=4.0,
                coherence_score=4.0,
                feedback_score=4.0,
                recency_score=4.0,
                computed_at=datetime.now(),
            )
            store.save(score)

        result_path = store.export_report()

        with open(result_path) as f:
            report = json.load(f)

        stats = report["statistics"]
        assert stats["total_conversations"] == 4
        assert stats["average_score"] == 3.5  # (2+3+4+5)/4
        assert stats["high_quality_count"] == 2  # 4.0 and 5.0 >= 3.5

    def test_export_report_empty(self, store, temp_dir):
        """Test exporting report when no scores."""
        result_path = store.export_report()

        with open(result_path) as f:
            report = json.load(f)

        stats = report["statistics"]
        assert stats["total_conversations"] == 0
        assert stats["average_score"] == 0.0
        assert stats["high_quality_rate"] == 0
