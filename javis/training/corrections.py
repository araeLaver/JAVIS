"""Response correction system for improving training data."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

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


class CorrectionManager:
    """Manages response corrections for training data improvement."""

    def __init__(self, corrections_dir: Optional[Path] = None):
        if corrections_dir is None:
            corrections_dir = Path(__file__).parent.parent.parent / "data" / "corrections"

        self.corrections_dir = Path(corrections_dir)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

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

        month_dir = self._get_month_dir(timestamp)
        file_path = month_dir / f"{correction_id}.json"

        data = correction.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Added correction: {correction_id}")
        return correction_id

    def get_correction(self, correction_id: str) -> Optional[ResponseCorrection]:
        """Get a correction by ID.

        Args:
            correction_id: Correction ID

        Returns:
            ResponseCorrection or None
        """
        for file_path in self.corrections_dir.rglob(f"{correction_id}.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                return ResponseCorrection(**data)

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load correction {correction_id}: {e}")
                return None

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
        corrections = []

        # Get all JSON files, sorted by modification time
        files = sorted(
            self.corrections_dir.rglob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                correction = ResponseCorrection(**data)

                # Apply filters
                if start and correction.timestamp < start:
                    continue
                if end and correction.timestamp > end:
                    continue
                if correction_type and correction.correction_type != correction_type:
                    continue

                corrections.append(correction)

                if limit and len(corrections) >= limit:
                    break

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        return corrections

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
        for file_path in self.corrections_dir.rglob(f"{correction_id}.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if corrected_response is not None:
                    data["corrected_response"] = corrected_response
                if correction_type is not None:
                    data["correction_type"] = correction_type
                if notes is not None:
                    data["notes"] = notes

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                logger.info(f"Updated correction: {correction_id}")
                return True

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to update correction {correction_id}: {e}")
                return False

        return False

    def delete_correction(self, correction_id: str) -> bool:
        """Delete a correction.

        Args:
            correction_id: Correction ID

        Returns:
            True if deleted
        """
        for file_path in self.corrections_dir.rglob(f"{correction_id}.json"):
            file_path.unlink()
            logger.info(f"Deleted correction: {correction_id}")
            return True

        return False

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

        corrections = self.get_corrections()
        training_data = []

        for corr in corrections:
            training_data.append({
                "messages": [
                    {"role": "user", "content": corr.original_prompt},
                    {"role": "assistant", "content": corr.corrected_response},
                ]
            })

        with open(output_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(training_data)} corrections to {output_path}")
        return output_path

    def get_statistics(self) -> dict:
        """Get correction statistics.

        Returns:
            Dictionary with statistics
        """
        corrections = self.get_corrections()

        by_type: dict[str, int] = {}
        for corr in corrections:
            by_type[corr.correction_type] = by_type.get(corr.correction_type, 0) + 1

        return {
            "total": len(corrections),
            "by_type": by_type,
            "this_month": sum(
                1
                for c in corrections
                if c.timestamp.month == datetime.now().month
                and c.timestamp.year == datetime.now().year
            ),
        }


# Singleton instance
_manager: Optional[CorrectionManager] = None


def get_correction_manager() -> CorrectionManager:
    """Get the global CorrectionManager instance."""
    global _manager
    if _manager is None:
        _manager = CorrectionManager()
    return _manager
