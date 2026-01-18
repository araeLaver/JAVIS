"""Abstract base interfaces for storage backends."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel


# ============================================================================
# Data Models (shared across all backends)
# ============================================================================

class ConversationTurn(BaseModel):
    """Single conversation turn."""
    role: str
    content: str
    timestamp: str


class ConversationRecord(BaseModel):
    """Full conversation record."""
    session_id: str
    started_at: str
    turns: list[ConversationTurn] = []
    feedback: Optional[str] = None  # good, bad, None
    tags: list[str] = []


FeedbackType = Literal["good", "bad", "corrected"]


class FeedbackRecord(BaseModel):
    """Feedback record."""
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


CorrectionType = Literal["factual", "style", "format", "completeness", "other"]


class CorrectionRecord(BaseModel):
    """Response correction record."""
    id: str
    session_id: str
    timestamp: datetime
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: CorrectionType
    notes: Optional[str] = None
    metadata: dict = {}


class QualityScoreRecord(BaseModel):
    """Quality score for a conversation."""
    conversation_id: str
    overall_score: float
    length_score: float
    coherence_score: float
    feedback_score: float
    recency_score: float
    computed_at: datetime


class ModelVersionRecord(BaseModel):
    """Model version metadata."""
    version: str
    created_at: datetime
    base_model: str
    dataset_size: int
    status: str  # active, inactive, failed
    training_config: dict = {}
    adapter_path: Optional[str] = None  # Local path or HF Hub path


# ============================================================================
# Storage Interfaces
# ============================================================================

class ConversationStoreInterface(ABC):
    """Abstract interface for conversation storage."""

    @abstractmethod
    def save(self, conversation: ConversationRecord) -> str:
        """Save a conversation.

        Args:
            conversation: ConversationRecord to save

        Returns:
            Identifier (path or ID) of the saved conversation
        """
        pass

    @abstractmethod
    def get(self, session_id: str) -> Optional[ConversationRecord]:
        """Get a conversation by session ID.

        Args:
            session_id: Session identifier

        Returns:
            ConversationRecord or None
        """
        pass

    @abstractmethod
    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        feedback_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ConversationRecord]:
        """List all conversations with optional filters.

        Args:
            start: Start date filter
            end: End date filter
            feedback_filter: Filter by feedback type
            limit: Maximum number to return

        Returns:
            List of ConversationRecord objects
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total conversations.

        Returns:
            Total number of conversations
        """
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a conversation.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def export_for_training(
        self,
        output_path: Path,
        feedback_filter: Optional[str] = None,
        min_turns: int = 2,
    ) -> Path:
        """Export conversations as JSONL for training.

        Args:
            output_path: Output file path
            feedback_filter: Filter by feedback type
            min_turns: Minimum number of turns required

        Returns:
            Path to exported file
        """
        pass


class FeedbackStoreInterface(ABC):
    """Abstract interface for feedback storage."""

    @abstractmethod
    def save(self, feedback: FeedbackRecord) -> str:
        """Save a feedback record.

        Args:
            feedback: FeedbackRecord to save

        Returns:
            Feedback ID
        """
        pass

    @abstractmethod
    def get(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Get a feedback record by ID.

        Args:
            feedback_id: Feedback identifier

        Returns:
            FeedbackRecord or None
        """
        pass

    @abstractmethod
    def list_by_type(
        self,
        feedback_type: FeedbackType,
        limit: Optional[int] = None,
    ) -> list[FeedbackRecord]:
        """List feedback by type.

        Args:
            feedback_type: Type of feedback
            limit: Maximum number to return

        Returns:
            List of FeedbackRecord objects
        """
        pass

    @abstractmethod
    def list_all(self, limit: Optional[int] = None) -> list[FeedbackRecord]:
        """List all feedback records.

        Args:
            limit: Maximum number to return

        Returns:
            List of FeedbackRecord objects
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Dictionary with statistics
        """
        pass

    @abstractmethod
    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback record.

        Args:
            feedback_id: Feedback identifier

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def export_for_training(
        self,
        output_path: Path,
        include_corrected: bool = True,
    ) -> Path:
        """Export feedback as JSONL for training.

        Args:
            output_path: Output file path
            include_corrected: Include corrected responses

        Returns:
            Path to exported file
        """
        pass


class CorrectionStoreInterface(ABC):
    """Abstract interface for correction storage."""

    @abstractmethod
    def save(self, correction: CorrectionRecord) -> str:
        """Save a correction record.

        Args:
            correction: CorrectionRecord to save

        Returns:
            Correction ID
        """
        pass

    @abstractmethod
    def get(self, correction_id: str) -> Optional[CorrectionRecord]:
        """Get a correction by ID.

        Args:
            correction_id: Correction identifier

        Returns:
            CorrectionRecord or None
        """
        pass

    @abstractmethod
    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        correction_type: Optional[CorrectionType] = None,
        limit: Optional[int] = None,
    ) -> list[CorrectionRecord]:
        """List corrections with optional filters.

        Args:
            start: Start date filter
            end: End date filter
            correction_type: Type filter
            limit: Maximum number to return

        Returns:
            List of CorrectionRecord objects
        """
        pass

    @abstractmethod
    def update(
        self,
        correction_id: str,
        corrected_response: Optional[str] = None,
        correction_type: Optional[CorrectionType] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update an existing correction.

        Args:
            correction_id: Correction identifier
            corrected_response: New corrected response
            correction_type: New correction type
            notes: New notes

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    def delete(self, correction_id: str) -> bool:
        """Delete a correction.

        Args:
            correction_id: Correction identifier

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get correction statistics.

        Returns:
            Dictionary with statistics
        """
        pass

    @abstractmethod
    def export_for_training(self, output_path: Path) -> Path:
        """Export corrections as JSONL for training.

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        pass


class QualityScoreStoreInterface(ABC):
    """Abstract interface for quality score storage."""

    @abstractmethod
    def save(self, score: QualityScoreRecord) -> str:
        """Save a quality score.

        Args:
            score: QualityScoreRecord to save

        Returns:
            Conversation ID
        """
        pass

    @abstractmethod
    def get(self, conversation_id: str) -> Optional[QualityScoreRecord]:
        """Get quality score for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            QualityScoreRecord or None
        """
        pass

    @abstractmethod
    def list_all(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[QualityScoreRecord]:
        """List quality scores with optional filters.

        Args:
            min_score: Minimum overall score filter
            limit: Maximum number to return

        Returns:
            List of QualityScoreRecord objects
        """
        pass

    @abstractmethod
    def get_high_quality_ids(self, min_score: float = 3.5) -> list[str]:
        """Get conversation IDs with high quality scores.

        Args:
            min_score: Minimum score threshold

        Returns:
            List of conversation IDs
        """
        pass


class ModelStoreInterface(ABC):
    """Abstract interface for model version storage."""

    @abstractmethod
    def save_version(
        self,
        version: str,
        adapter_path: Path,
        metadata: ModelVersionRecord,
    ) -> str:
        """Save a new model version.

        Args:
            version: Version identifier
            adapter_path: Local path to adapter files
            metadata: Version metadata

        Returns:
            Version identifier or remote path
        """
        pass

    @abstractmethod
    def get_version(self, version: str) -> Optional[ModelVersionRecord]:
        """Get version metadata.

        Args:
            version: Version identifier

        Returns:
            ModelVersionRecord or None
        """
        pass

    @abstractmethod
    def list_versions(self) -> list[ModelVersionRecord]:
        """List all versions.

        Returns:
            List of ModelVersionRecord objects, sorted by date descending
        """
        pass

    @abstractmethod
    def get_active_version(self) -> Optional[str]:
        """Get the currently active version.

        Returns:
            Active version identifier or None
        """
        pass

    @abstractmethod
    def set_active_version(self, version: str) -> bool:
        """Set the active version.

        Args:
            version: Version identifier

        Returns:
            True if set successfully
        """
        pass

    @abstractmethod
    def get_adapter_path(self, version: str) -> Optional[Path]:
        """Get the path to adapter files for a version.

        For local storage, returns the local path.
        For cloud storage, downloads and returns a cached local path.

        Args:
            version: Version identifier

        Returns:
            Path to adapter files or None
        """
        pass

    @abstractmethod
    def update_status(self, version: str, status: str) -> bool:
        """Update version status.

        Args:
            version: Version identifier
            status: New status (active, inactive, failed)

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    def delete_version(self, version: str) -> bool:
        """Delete a version.

        Args:
            version: Version identifier

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def cleanup_old_versions(self, keep: int = 5) -> list[str]:
        """Remove old versions, keeping the most recent ones.

        Args:
            keep: Number of versions to keep

        Returns:
            List of deleted version identifiers
        """
        pass
