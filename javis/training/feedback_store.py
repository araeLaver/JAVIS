"""Enhanced feedback storage system."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from javis.storage import get_feedback_store as get_storage_feedback_store
from javis.storage.base import FeedbackRecord, FeedbackStoreInterface

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

    def to_storage_record(self) -> FeedbackRecord:
        """Convert to storage record format."""
        return FeedbackRecord(
            id=self.id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            timestamp=self.timestamp,
            feedback_type=self.feedback_type,  # type: ignore
            quality_score=self.quality_score,
            prompt=self.prompt,
            response=self.response,
            corrected_response=self.corrected_response,
            metadata=self.metadata,
        )

    @classmethod
    def from_storage_record(cls, record: FeedbackRecord) -> "EnhancedFeedback":
        """Create from storage record."""
        return cls(
            id=record.id,
            session_id=record.session_id,
            conversation_id=record.conversation_id,
            timestamp=record.timestamp,
            feedback_type=record.feedback_type,
            quality_score=record.quality_score,
            prompt=record.prompt,
            response=record.response,
            corrected_response=record.corrected_response,
            metadata=record.metadata,
        )


class FeedbackStore:
    """Stores and manages feedback data.

    This class wraps the storage interface to provide backward-compatible
    API while using the storage factory (local or cloud).
    """

    def __init__(
        self,
        feedback_dir: Optional[Path] = None,
        store: Optional[FeedbackStoreInterface] = None,
    ):
        """Initialize feedback store.

        Args:
            feedback_dir: Legacy parameter for backward compatibility.
            store: Storage interface to use. If None, uses storage factory.
        """
        self._store = store

        # Keep feedback_dir for backward compatibility
        if feedback_dir is None:
            feedback_dir = Path(__file__).parent.parent.parent / "data" / "feedback"
        self.feedback_dir = Path(feedback_dir)
        self._ensure_directories()

    @property
    def store(self) -> FeedbackStoreInterface:
        """Get the feedback store (lazy initialization)."""
        if self._store is None:
            self._store = get_storage_feedback_store()
        return self._store

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

        # Use storage backend
        result = self.store.save(feedback.to_storage_record())
        logger.info(f"Stored {feedback.feedback_type} feedback: {feedback.id}")
        return result

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
        """Load feedback records by type.

        Args:
            feedback_type: Type of feedback to load
            limit: Maximum number of records to return

        Returns:
            List of EnhancedFeedback objects
        """
        # Use storage backend
        records = self.store.list_by_type(feedback_type, limit=limit)  # type: ignore
        return [EnhancedFeedback.from_storage_record(r) for r in records]

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
        # Use storage backend
        records = self.store.list_all(limit=limit)
        return [EnhancedFeedback.from_storage_record(r) for r in records]

    def get_statistics(self) -> dict:
        """Get feedback statistics.

        Returns:
            Dictionary with feedback statistics
        """
        # Use storage backend
        return self.store.get_statistics()

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

        # Use storage backend
        return self.store.export_for_training(
            output_path=Path(output_path),
            include_corrected=include_corrected,
        )

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback record.

        Args:
            feedback_id: Feedback ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Use storage backend
        result = self.store.delete(feedback_id)
        if result:
            logger.info(f"Deleted feedback: {feedback_id}")
        return result


# Singleton instance
_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get the global FeedbackStore instance."""
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
