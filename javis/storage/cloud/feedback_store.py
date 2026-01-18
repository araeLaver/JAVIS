"""Cloud-based feedback storage using Supabase."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from javis.storage.base import FeedbackRecord, FeedbackStoreInterface, FeedbackType
from javis.storage.cloud.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Table name
FEEDBACK_TABLE = "feedback"


class CloudFeedbackStore(FeedbackStoreInterface):
    """Cloud-based implementation of feedback storage using Supabase."""

    def __init__(self):
        """Initialize cloud feedback store."""
        self.client = get_supabase_client()

    def _generate_id(self) -> str:
        """Generate a unique feedback ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"fb_{timestamp}_{unique}"

    def save(self, feedback: FeedbackRecord) -> str:
        """Save a feedback record."""
        if not feedback.id:
            feedback = feedback.model_copy(update={"id": self._generate_id()})

        data = {
            "id": feedback.id,
            "session_id": feedback.session_id,
            "conversation_id": feedback.conversation_id,
            "timestamp": feedback.timestamp.isoformat(),
            "feedback_type": feedback.feedback_type,
            "quality_score": feedback.quality_score,
            "prompt": feedback.prompt,
            "response": feedback.response,
            "corrected_response": feedback.corrected_response,
            "metadata": feedback.metadata,
        }

        self.client.upsert(FEEDBACK_TABLE, data, on_conflict="id")
        logger.info(f"Stored {feedback.feedback_type} feedback: {feedback.id}")
        return feedback.id

    def get(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Get a feedback record by ID."""
        data = self.client.select_one(
            FEEDBACK_TABLE,
            filters={"id": feedback_id},
        )

        if not data:
            return None

        return self._record_from_data(data)

    def _record_from_data(self, data: dict) -> FeedbackRecord:
        """Convert database record to FeedbackRecord."""
        return FeedbackRecord(
            id=data["id"],
            session_id=data["session_id"],
            conversation_id=data.get("conversation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            feedback_type=data["feedback_type"],
            quality_score=data.get("quality_score"),
            prompt=data["prompt"],
            response=data["response"],
            corrected_response=data.get("corrected_response"),
            metadata=data.get("metadata", {}),
        )

    def list_by_type(
        self,
        feedback_type: FeedbackType,
        limit: Optional[int] = None,
    ) -> list[FeedbackRecord]:
        """List feedback by type."""
        data = self.client.select(
            FEEDBACK_TABLE,
            filters={"feedback_type": feedback_type},
            order="timestamp",
            desc=True,
            limit=limit,
        )

        return [self._record_from_data(d) for d in data]

    def list_all(self, limit: Optional[int] = None) -> list[FeedbackRecord]:
        """List all feedback records."""
        data = self.client.select(
            FEEDBACK_TABLE,
            order="timestamp",
            desc=True,
            limit=limit,
        )

        return [self._record_from_data(d) for d in data]

    def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics."""
        good_count = self.client.count(FEEDBACK_TABLE, {"feedback_type": "good"})
        bad_count = self.client.count(FEEDBACK_TABLE, {"feedback_type": "bad"})
        corrected_count = self.client.count(
            FEEDBACK_TABLE, {"feedback_type": "corrected"}
        )

        total = good_count + bad_count + corrected_count

        return {
            "total": total,
            "good": good_count,
            "bad": bad_count,
            "corrected": corrected_count,
            "positive_rate": good_count / total if total > 0 else 0,
            "correction_rate": (
                corrected_count / (bad_count + corrected_count)
                if (bad_count + corrected_count) > 0
                else 0
            ),
        }

    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback record."""
        result = self.client.delete(
            FEEDBACK_TABLE,
            {"id": feedback_id},
        )

        if result:
            logger.info(f"Deleted feedback: {feedback_id}")
            return True
        return False

    def export_for_training(
        self,
        output_path: Path,
        include_corrected: bool = True,
    ) -> Path:
        """Export feedback as JSONL for training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        training_data = []

        # Add good responses
        for fb in self.list_by_type("good"):
            training_data.append({
                "messages": [
                    {"role": "user", "content": fb.prompt},
                    {"role": "assistant", "content": fb.response},
                ]
            })

        # Add corrected responses
        if include_corrected:
            for fb in self.list_by_type("corrected"):
                if fb.corrected_response:
                    training_data.append({
                        "messages": [
                            {"role": "user", "content": fb.prompt},
                            {"role": "assistant", "content": fb.corrected_response},
                        ]
                    })

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(training_data)} training examples to {output_path}")
        return output_path
