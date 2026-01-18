"""Local file-based correction storage."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from javis.storage.base import (
    CorrectionRecord,
    CorrectionStoreInterface,
    CorrectionType,
)

logger = logging.getLogger(__name__)


class LocalCorrectionStore(CorrectionStoreInterface):
    """Local file-based implementation of correction storage."""

    def __init__(self, corrections_dir: Optional[Path] = None):
        """Initialize local correction store.

        Args:
            corrections_dir: Directory for storing corrections.
                             Defaults to data/corrections.
        """
        if corrections_dir is None:
            corrections_dir = (
                Path(__file__).parent.parent.parent.parent / "data" / "corrections"
            )
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

    def save(self, correction: CorrectionRecord) -> str:
        """Save a correction record."""
        if not correction.id:
            correction = correction.model_copy(update={"id": self._generate_id()})

        month_dir = self._get_month_dir(correction.timestamp)
        file_path = month_dir / f"{correction.id}.json"

        data = correction.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Added correction: {correction.id}")
        return correction.id

    def get(self, correction_id: str) -> Optional[CorrectionRecord]:
        """Get a correction by ID."""
        for file_path in self.corrections_dir.rglob(f"{correction_id}.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                return CorrectionRecord(**data)

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load correction {correction_id}: {e}")
                return None

        return None

    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        correction_type: Optional[CorrectionType] = None,
        limit: Optional[int] = None,
    ) -> list[CorrectionRecord]:
        """List corrections with optional filters."""
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

                correction = CorrectionRecord(**data)

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

    def update(
        self,
        correction_id: str,
        corrected_response: Optional[str] = None,
        correction_type: Optional[CorrectionType] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update an existing correction."""
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

    def delete(self, correction_id: str) -> bool:
        """Delete a correction."""
        for file_path in self.corrections_dir.rglob(f"{correction_id}.json"):
            file_path.unlink()
            logger.info(f"Deleted correction: {correction_id}")
            return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get correction statistics."""
        corrections = self.list_all()

        by_type: dict[str, int] = {}
        for corr in corrections:
            by_type[corr.correction_type] = by_type.get(corr.correction_type, 0) + 1

        this_month = sum(
            1
            for c in corrections
            if c.timestamp.month == datetime.now().month
            and c.timestamp.year == datetime.now().year
        )

        return {
            "total": len(corrections),
            "by_type": by_type,
            "this_month": this_month,
        }

    def export_for_training(self, output_path: Path) -> Path:
        """Export corrections as JSONL for training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        corrections = self.list_all()
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
