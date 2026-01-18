"""Cloud-based quality score storage using Supabase."""

import logging
from datetime import datetime
from typing import Optional

from javis.storage.base import QualityScoreRecord, QualityScoreStoreInterface
from javis.storage.cloud.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Table name
QUALITY_SCORES_TABLE = "quality_scores"


class CloudQualityScoreStore(QualityScoreStoreInterface):
    """Cloud-based implementation of quality score storage using Supabase."""

    def __init__(self):
        """Initialize cloud quality score store."""
        self.client = get_supabase_client()

    def save(self, score: QualityScoreRecord) -> str:
        """Save a quality score."""
        data = {
            "conversation_id": score.conversation_id,
            "overall_score": score.overall_score,
            "length_score": score.length_score,
            "coherence_score": score.coherence_score,
            "feedback_score": score.feedback_score,
            "recency_score": score.recency_score,
            "computed_at": score.computed_at.isoformat(),
        }

        self.client.upsert(QUALITY_SCORES_TABLE, data, on_conflict="conversation_id")
        logger.debug(f"Saved quality score for: {score.conversation_id}")
        return score.conversation_id

    def get(self, conversation_id: str) -> Optional[QualityScoreRecord]:
        """Get quality score for a conversation."""
        data = self.client.select_one(
            QUALITY_SCORES_TABLE,
            filters={"conversation_id": conversation_id},
        )

        if not data:
            return None

        return self._record_from_data(data)

    def _record_from_data(self, data: dict) -> QualityScoreRecord:
        """Convert database record to QualityScoreRecord."""
        return QualityScoreRecord(
            conversation_id=data["conversation_id"],
            overall_score=data["overall_score"],
            length_score=data["length_score"],
            coherence_score=data["coherence_score"],
            feedback_score=data["feedback_score"],
            recency_score=data["recency_score"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
        )

    def list_all(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[QualityScoreRecord]:
        """List quality scores with optional filters."""
        filters = {}

        if min_score is not None:
            filters["overall_score"] = {"op": "gte", "val": min_score}

        data = self.client.select(
            QUALITY_SCORES_TABLE,
            filters=filters if filters else None,
            order="overall_score",
            desc=True,
            limit=limit,
        )

        return [self._record_from_data(d) for d in data]

    def get_high_quality_ids(self, min_score: float = 3.5) -> list[str]:
        """Get conversation IDs with high quality scores."""
        scores = self.list_all(min_score=min_score)
        return [s.conversation_id for s in scores]

    def delete(self, conversation_id: str) -> bool:
        """Delete a quality score."""
        result = self.client.delete(
            QUALITY_SCORES_TABLE,
            {"conversation_id": conversation_id},
        )

        if result:
            logger.debug(f"Deleted quality score for: {conversation_id}")
            return True
        return False

    def clear(self) -> None:
        """Clear all quality scores.

        Note: This is a destructive operation. Use with caution.
        """
        # Get all scores and delete them
        all_scores = self.list_all()
        for score in all_scores:
            self.delete(score.conversation_id)
        logger.info(f"Cleared {len(all_scores)} quality scores")

    def get_statistics(self) -> dict:
        """Get quality score statistics."""
        all_scores = self.list_all()

        if not all_scores:
            return {
                "total": 0,
                "average_score": 0.0,
                "high_quality_count": 0,
                "high_quality_rate": 0.0,
            }

        total = len(all_scores)
        avg_score = sum(s.overall_score for s in all_scores) / total
        high_quality = len([s for s in all_scores if s.overall_score >= 3.5])

        return {
            "total": total,
            "average_score": round(avg_score, 2),
            "high_quality_count": high_quality,
            "high_quality_rate": round(high_quality / total, 2),
        }
