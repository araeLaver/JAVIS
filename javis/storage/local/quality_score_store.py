"""Local file-based quality score storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.storage.base import QualityScoreRecord, QualityScoreStoreInterface

logger = logging.getLogger(__name__)


class LocalQualityScoreStore(QualityScoreStoreInterface):
    """Local file-based implementation of quality score storage."""

    def __init__(self, scores_dir: Optional[Path] = None):
        """Initialize local quality score store.

        Args:
            scores_dir: Directory for storing quality scores.
                        Defaults to data/quality_scores.
        """
        if scores_dir is None:
            scores_dir = (
                Path(__file__).parent.parent.parent.parent / "data" / "quality_scores"
            )
        self.scores_dir = Path(scores_dir)
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        # Cache file for all scores
        self._cache_file = self.scores_dir / "scores_cache.json"
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load scores cache from file."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save scores cache to file."""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")

    def save(self, score: QualityScoreRecord) -> str:
        """Save a quality score."""
        score_data = score.model_dump()
        score_data["computed_at"] = score_data["computed_at"].isoformat()

        self._cache[score.conversation_id] = score_data
        self._save_cache()

        logger.debug(f"Saved quality score for: {score.conversation_id}")
        return score.conversation_id

    def get(self, conversation_id: str) -> Optional[QualityScoreRecord]:
        """Get quality score for a conversation."""
        if conversation_id not in self._cache:
            return None

        data = self._cache[conversation_id].copy()
        if isinstance(data.get("computed_at"), str):
            data["computed_at"] = datetime.fromisoformat(data["computed_at"])

        return QualityScoreRecord(**data)

    def list_all(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[QualityScoreRecord]:
        """List quality scores with optional filters."""
        scores = []

        for conv_id, data in self._cache.items():
            score_data = data.copy()
            if isinstance(score_data.get("computed_at"), str):
                score_data["computed_at"] = datetime.fromisoformat(
                    score_data["computed_at"]
                )

            score = QualityScoreRecord(**score_data)

            # Apply min_score filter
            if min_score is not None and score.overall_score < min_score:
                continue

            scores.append(score)

        # Sort by overall_score descending
        scores.sort(key=lambda s: s.overall_score, reverse=True)

        if limit:
            return scores[:limit]
        return scores

    def get_high_quality_ids(self, min_score: float = 3.5) -> list[str]:
        """Get conversation IDs with high quality scores."""
        return [
            score.conversation_id
            for score in self.list_all(min_score=min_score)
        ]

    def delete(self, conversation_id: str) -> bool:
        """Delete a quality score."""
        if conversation_id in self._cache:
            del self._cache[conversation_id]
            self._save_cache()
            return True
        return False

    def clear(self) -> None:
        """Clear all cached scores."""
        self._cache = {}
        self._save_cache()

    def export_report(self, output_path: Optional[Path] = None) -> Path:
        """Export quality scores report.

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = (
                self.scores_dir
                / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        output_path = Path(output_path)

        scores = self.list_all()

        # Calculate statistics
        if scores:
            total = len(scores)
            avg_score = sum(s.overall_score for s in scores) / total
            high_quality = len([s for s in scores if s.overall_score >= 3.5])
        else:
            total = 0
            avg_score = 0.0
            high_quality = 0

        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": {
                "total_conversations": total,
                "average_score": round(avg_score, 2),
                "high_quality_count": high_quality,
                "high_quality_rate": round(high_quality / total, 2) if total > 0 else 0,
            },
            "scores": [
                {
                    "conversation_id": s.conversation_id,
                    "overall_score": s.overall_score,
                    "length_score": s.length_score,
                    "coherence_score": s.coherence_score,
                    "feedback_score": s.feedback_score,
                    "recency_score": s.recency_score,
                    "computed_at": s.computed_at.isoformat(),
                }
                for s in scores
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported quality report to {output_path}")
        return output_path
