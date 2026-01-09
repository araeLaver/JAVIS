"""Enhanced feedback storage system."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

FeedbackType = Literal["good", "bad", "corrected"]


class EnhancedFeedback(BaseModel):
    """Enhanced feedback record."""

    id: str
    session_id: str
    conversation_id: Optional[str] = None
    timestamp: datetime
    feedback_type: FeedbackType
    quality_score: Optional[float] = None  # 1-5 scale
    prompt: str
    response: str
    corrected_response: Optional[str] = None
    metadata: dict = {}


class FeedbackStore:
    """Stores and manages feedback data with separate directories."""

    def __init__(self, feedback_dir: Optional[Path] = None):
        if feedback_dir is None:
            feedback_dir = Path(__file__).parent.parent.parent / "data" / "feedback"

        self.feedback_dir = Path(feedback_dir)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create feedback directories if they don't exist."""
        for subdir in ["good", "bad", "corrected"]:
            (self.feedback_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _get_month_dir(self, feedback_type: FeedbackType, date: datetime) -> Path:
        """Get the month directory for a feedback type."""
        month_str = date.strftime("%Y-%m")
        month_dir = self.feedback_dir / feedback_type / month_str
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir

    def _generate_id(self) -> str:
        """Generate a unique feedback ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"fb_{timestamp}_{unique}"

    def store_feedback(self, feedback: EnhancedFeedback) -> str:
        """Store a feedback record.

        Args:
            feedback: EnhancedFeedback object to store

        Returns:
            Feedback ID
        """
        if not feedback.id:
            feedback.id = self._generate_id()

        month_dir = self._get_month_dir(feedback.feedback_type, feedback.timestamp)
        file_path = month_dir / f"{feedback.id}.json"

        data = feedback.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Stored {feedback.feedback_type} feedback: {feedback.id}")
        return feedback.id

    def store_good_response(
        self,
        session_id: str,
        prompt: str,
        response: str,
        quality_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a good response.

        Args:
            session_id: Session identifier
            prompt: User prompt
            response: Assistant response
            quality_score: Optional quality score (1-5)
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        feedback = EnhancedFeedback(
            id=self._generate_id(),
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type="good",
            quality_score=quality_score,
            prompt=prompt,
            response=response,
            metadata=metadata or {},
        )
        return self.store_feedback(feedback)

    def store_bad_response(
        self,
        session_id: str,
        prompt: str,
        response: str,
        corrected_response: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a bad response, optionally with correction.

        Args:
            session_id: Session identifier
            prompt: User prompt
            response: Original assistant response
            corrected_response: Optional corrected response
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        feedback_type: FeedbackType = "corrected" if corrected_response else "bad"

        feedback = EnhancedFeedback(
            id=self._generate_id(),
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=feedback_type,
            quality_score=1.0 if not corrected_response else None,
            prompt=prompt,
            response=response,
            corrected_response=corrected_response,
            metadata=metadata or {},
        )
        return self.store_feedback(feedback)

    def _load_feedback_from_dir(
        self, feedback_type: FeedbackType, limit: Optional[int] = None
    ) -> list[EnhancedFeedback]:
        """Load feedback records from a directory.

        Args:
            feedback_type: Type of feedback to load
            limit: Maximum number of records to return

        Returns:
            List of EnhancedFeedback objects
        """
        feedbacks = []
        type_dir = self.feedback_dir / feedback_type

        if not type_dir.exists():
            return feedbacks

        # Get all JSON files, sorted by modification time (newest first)
        files = sorted(type_dir.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

        if limit:
            files = files[:limit]

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert timestamp string to datetime
                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                feedbacks.append(EnhancedFeedback(**data))

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        return feedbacks

    def get_good_responses(self, limit: Optional[int] = None) -> list[EnhancedFeedback]:
        """Get responses with positive feedback.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of good feedback records
        """
        return self._load_feedback_from_dir("good", limit)

    def get_bad_responses(self, limit: Optional[int] = None) -> list[EnhancedFeedback]:
        """Get responses with negative feedback.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of bad feedback records
        """
        return self._load_feedback_from_dir("bad", limit)

    def get_corrected_responses(self, limit: Optional[int] = None) -> list[EnhancedFeedback]:
        """Get corrected responses.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of corrected feedback records
        """
        return self._load_feedback_from_dir("corrected", limit)

    def get_all_feedback(self, limit: Optional[int] = None) -> list[EnhancedFeedback]:
        """Get all feedback records.

        Args:
            limit: Maximum total records to return

        Returns:
            List of all feedback records
        """
        all_feedback = []
        for feedback_type in ["good", "bad", "corrected"]:
            all_feedback.extend(self._load_feedback_from_dir(feedback_type))

        # Sort by timestamp, newest first
        all_feedback.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            return all_feedback[:limit]
        return all_feedback

    def get_statistics(self) -> dict:
        """Get feedback statistics.

        Returns:
            Dictionary with feedback statistics
        """
        good = self.get_good_responses()
        bad = self.get_bad_responses()
        corrected = self.get_corrected_responses()

        total = len(good) + len(bad) + len(corrected)

        return {
            "total": total,
            "good": len(good),
            "bad": len(bad),
            "corrected": len(corrected),
            "positive_rate": len(good) / total if total > 0 else 0,
            "correction_rate": len(corrected) / (len(bad) + len(corrected)) if (len(bad) + len(corrected)) > 0 else 0,
        }

    def export_training_data(
        self,
        output_path: Optional[Path] = None,
        include_corrected: bool = True,
    ) -> Path:
        """Export feedback data as training JSONL.

        Args:
            output_path: Output file path
            include_corrected: Include corrected responses as training data

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_dir = (
                Path(__file__).parent.parent.parent / "data" / "training" / "exported"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        training_data = []

        # Add good responses
        for fb in self.get_good_responses():
            training_data.append({
                "messages": [
                    {"role": "user", "content": fb.prompt},
                    {"role": "assistant", "content": fb.response},
                ]
            })

        # Add corrected responses (using the corrected version)
        if include_corrected:
            for fb in self.get_corrected_responses():
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

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback record.

        Args:
            feedback_id: Feedback ID to delete

        Returns:
            True if deleted, False if not found
        """
        for feedback_type in ["good", "bad", "corrected"]:
            type_dir = self.feedback_dir / feedback_type
            for file_path in type_dir.rglob(f"{feedback_id}.json"):
                file_path.unlink()
                logger.info(f"Deleted feedback: {feedback_id}")
                return True

        return False


# Singleton instance
_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get the global FeedbackStore instance."""
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
