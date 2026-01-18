"""Response correction system for improving training data."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from javis.storage import get_correction_store as get_storage_correction_store
from javis.storage.base import CorrectionRecord, CorrectionStoreInterface

logger = logging.getLogger(__name__)

CorrectionType = Literal["factual", "style", "format", "completeness", "other"]


class ResponseCorrection(BaseModel):
    """User correction for a response."""

    id: str
    session_id: str
    timestamp: datetime
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: CorrectionType
    notes: Optional[str] = None
    metadata: dict = {}

    def to_storage_record(self) -> CorrectionRecord:
        """Convert to storage record format."""
        return CorrectionRecord(
            id=self.id,
            session_id=self.session_id,
            timestamp=self.timestamp,
            original_prompt=self.original_prompt,
            original_response=self.original_response,
            corrected_response=self.corrected_response,
            correction_type=self.correction_type,  # type: ignore
            notes=self.notes,
            metadata=self.metadata,
        )

    @classmethod
    def from_storage_record(cls, record: CorrectionRecord) -> "ResponseCorrection":
        """Create from storage record."""
        return cls(
            id=record.id,
            session_id=record.session_id,
            timestamp=record.timestamp,
            original_prompt=record.original_prompt,
            original_response=record.original_response,
            corrected_response=record.corrected_response,
            correction_type=record.correction_type,
            notes=record.notes,
            metadata=record.metadata,
        )


class CorrectionManager:
    """Manages response corrections for training data improvement.

    This class wraps the storage interface to provide backward-compatible
    API while using the storage factory (local or cloud).
    """

    def __init__(
        self,
        corrections_dir: Optional[Path] = None,
        store: Optional[CorrectionStoreInterface] = None,
    ):
        """Initialize correction manager.

        Args:
            corrections_dir: Legacy parameter for backward compatibility.
            store: Storage interface to use. If None, uses storage factory.
        """
        self._store = store

        # Keep corrections_dir for backward compatibility
        if corrections_dir is None:
            corrections_dir = Path(__file__).parent.parent.parent / "data" / "corrections"
        self.corrections_dir = Path(corrections_dir)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

    @property
    def store(self) -> CorrectionStoreInterface:
        """Get the correction store (lazy initialization)."""
        if self._store is None:
            self._store = get_storage_correction_store()
        return self._store

    def _generate_id(self) -> str:
        """Generate a unique correction ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"corr_{timestamp}_{unique}"

    def _get_month_dir(self, date: datetime) -> Path:
        """Get the month directory for a correction."""
        month_str = date.strftime("%Y-%m")
        month_dir = self.corrections_dir / month_str
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir

    def add_correction(
        self,
        session_id: str,
        original_prompt: str,
        original_response: str,
        corrected_response: str,
        correction_type: CorrectionType = "other",
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a new response correction.

        Args:
            session_id: Session identifier
            original_prompt: Original user prompt
            original_response: Original assistant response
            corrected_response: Corrected response
            correction_type: Type of correction
            notes: Additional notes
            metadata: Additional metadata

        Returns:
            Correction ID
        """
        correction_id = self._generate_id()
        timestamp = datetime.now()

        correction = ResponseCorrection(
            id=correction_id,
            session_id=session_id,
            timestamp=timestamp,
            original_prompt=original_prompt,
            original_response=original_response,
            corrected_response=corrected_response,
            correction_type=correction_type,
            notes=notes,
            metadata=metadata or {},
        )

        # Use storage backend
        result = self.store.save(correction.to_storage_record())
        logger.info(f"Added correction: {correction_id}")
        return result

    def get_correction(self, correction_id: str) -> Optional[ResponseCorrection]:
        """Get a correction by ID.

        Args:
            correction_id: Correction ID

        Returns:
            ResponseCorrection or None
        """
        # Use storage backend
        record = self.store.get(correction_id)
        if record:
            return ResponseCorrection.from_storage_record(record)
        return None

    def get_corrections(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        correction_type: Optional[CorrectionType] = None,
        limit: Optional[int] = None,
    ) -> list[ResponseCorrection]:
        """Get corrections with optional filters.

        Args:
            start: Start date filter
            end: End date filter
            correction_type: Type filter
            limit: Maximum number to return

        Returns:
            List of ResponseCorrection objects
        """
        # Use storage backend
        records = self.store.list_all(
            start=start,
            end=end,
            correction_type=correction_type,  # type: ignore
            limit=limit,
        )
        return [ResponseCorrection.from_storage_record(r) for r in records]

    def update_correction(
        self,
        correction_id: str,
        corrected_response: Optional[str] = None,
        correction_type: Optional[CorrectionType] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update an existing correction.

        Args:
            correction_id: Correction ID
            corrected_response: New corrected response
            correction_type: New correction type
            notes: New notes

        Returns:
            True if updated
        """
        # Use storage backend
        result = self.store.update(
            correction_id=correction_id,
            corrected_response=corrected_response,
            correction_type=correction_type,  # type: ignore
            notes=notes,
        )
        if result:
            logger.info(f"Updated correction: {correction_id}")
        return result

    def delete_correction(self, correction_id: str) -> bool:
        """Delete a correction.

        Args:
            correction_id: Correction ID

        Returns:
            True if deleted
        """
        # Use storage backend
        result = self.store.delete(correction_id)
        if result:
            logger.info(f"Deleted correction: {correction_id}")
        return result

    def export_for_training(self, output_path: Optional[Path] = None) -> Path:
        """Export corrections as training JSONL.

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_dir = (
                Path(__file__).parent.parent.parent / "data" / "training" / "exported"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Use storage backend
        return self.store.export_for_training(Path(output_path))

    def get_statistics(self) -> dict:
        """Get correction statistics.

        Returns:
            Dictionary with statistics
        """
        # Use storage backend
        return self.store.get_statistics()


# Singleton instance
_manager: Optional[CorrectionManager] = None


def get_correction_manager() -> CorrectionManager:
    """Get the global CorrectionManager instance."""
    global _manager
    if _manager is None:
        _manager = CorrectionManager()
    return _manager
