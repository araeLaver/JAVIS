"""Cloud-based correction storage using Supabase."""

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
from javis.storage.cloud.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Table name
CORRECTIONS_TABLE = "corrections"


class CloudCorrectionStore(CorrectionStoreInterface):
    """Cloud-based implementation of correction storage using Supabase."""

    def __init__(self):
        """Initialize cloud correction store."""
        self.client = get_supabase_client()

    def _generate_id(self) -> str:
        """Generate a unique correction ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"corr_{timestamp}_{unique}"

    def save(self, correction: CorrectionRecord) -> str:
        """Save a correction record."""
        if not correction.id:
            correction = correction.model_copy(update={"id": self._generate_id()})

        data = {
            "id": correction.id,
            "session_id": correction.session_id,
            "timestamp": correction.timestamp.isoformat(),
            "original_prompt": correction.original_prompt,
            "original_response": correction.original_response,
            "corrected_response": correction.corrected_response,
            "correction_type": correction.correction_type,
            "notes": correction.notes,
            "metadata": correction.metadata,
        }

        self.client.upsert(CORRECTIONS_TABLE, data, on_conflict="id")
        logger.info(f"Added correction: {correction.id}")
        return correction.id

    def get(self, correction_id: str) -> Optional[CorrectionRecord]:
        """Get a correction by ID."""
        data = self.client.select_one(
            CORRECTIONS_TABLE,
            filters={"id": correction_id},
        )

        if not data:
            return None

        return self._record_from_data(data)

    def _record_from_data(self, data: dict) -> CorrectionRecord:
        """Convert database record to CorrectionRecord."""
        return CorrectionRecord(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            original_prompt=data["original_prompt"],
            original_response=data["original_response"],
            corrected_response=data["corrected_response"],
            correction_type=data["correction_type"],
            notes=data.get("notes"),
            metadata=data.get("metadata", {}),
        )

    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        correction_type: Optional[CorrectionType] = None,
        limit: Optional[int] = None,
    ) -> list[CorrectionRecord]:
        """List corrections with optional filters."""
        filters = {}

        if correction_type:
            filters["correction_type"] = correction_type

        data = self.client.select(
            CORRECTIONS_TABLE,
            filters=filters if filters else None,
            order="timestamp",
            desc=True,
            limit=limit,
        )

        corrections = []
        for d in data:
            record = self._record_from_data(d)

            # Apply date filters
            if start and record.timestamp < start:
                continue
            if end and record.timestamp > end:
                continue

            corrections.append(record)

        return corrections

    def update(
        self,
        correction_id: str,
        corrected_response: Optional[str] = None,
        correction_type: Optional[CorrectionType] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update an existing correction."""
        update_data = {}

        if corrected_response is not None:
            update_data["corrected_response"] = corrected_response
        if correction_type is not None:
            update_data["correction_type"] = correction_type
        if notes is not None:
            update_data["notes"] = notes

        if not update_data:
            return False

        result = self.client.update(
            CORRECTIONS_TABLE,
            update_data,
            {"id": correction_id},
        )

        if result:
            logger.info(f"Updated correction: {correction_id}")
            return True
        return False

    def delete(self, correction_id: str) -> bool:
        """Delete a correction."""
        result = self.client.delete(
            CORRECTIONS_TABLE,
            {"id": correction_id},
        )

        if result:
            logger.info(f"Deleted correction: {correction_id}")
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get correction statistics."""
        all_corrections = self.list_all()

        by_type: dict[str, int] = {}
        for corr in all_corrections:
            by_type[corr.correction_type] = by_type.get(corr.correction_type, 0) + 1

        this_month = sum(
            1
            for c in all_corrections
            if c.timestamp.month == datetime.now().month
            and c.timestamp.year == datetime.now().year
        )

        return {
            "total": len(all_corrections),
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
