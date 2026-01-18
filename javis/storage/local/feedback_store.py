"""Local file-based feedback storage."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from javis.storage.base import FeedbackRecord, FeedbackStoreInterface, FeedbackType

logger = logging.getLogger(__name__)


class LocalFeedbackStore(FeedbackStoreInterface):
    """Local file-based implementation of feedback storage."""

    def __init__(self, feedback_dir: Optional[Path] = None):
        """Initialize local feedback store.

        Args:
            feedback_dir: Directory for storing feedback.
                          Defaults to data/feedback.
        """
        if feedback_dir is None:
            feedback_dir = Path(__file__).parent.parent.parent.parent / "data" / "feedback"
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

    def save(self, feedback: FeedbackRecord) -> str:
        """Save a feedback record."""
        if not feedback.id:
            feedback = feedback.model_copy(update={"id": self._generate_id()})

        month_dir = self._get_month_dir(feedback.feedback_type, feedback.timestamp)
        file_path = month_dir / f"{feedback.id}.json"

        data = feedback.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Stored {feedback.feedback_type} feedback: {feedback.id}")
        return feedback.id

    def get(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Get a feedback record by ID."""
        for feedback_type in ["good", "bad", "corrected"]:
            type_dir = self.feedback_dir / feedback_type
            for file_path in type_dir.rglob(f"{feedback_id}.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data.get("timestamp"), str):
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                    return FeedbackRecord(**data)
                except (json.JSONDecodeError, IOError, ValueError) as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    return None

        return None

    def _load_from_dir(
        self,
        feedback_type: FeedbackType,
        limit: Optional[int] = None,
    ) -> list[FeedbackRecord]:
        """Load feedback records from a directory."""
        feedbacks = []
        type_dir = self.feedback_dir / feedback_type

        if not type_dir.exists():
            return feedbacks

        # Get all JSON files, sorted by modification time (newest first)
        files = sorted(
            type_dir.rglob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if limit:
            files = files[:limit]

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                feedbacks.append(FeedbackRecord(**data))

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        return feedbacks

    def list_by_type(
        self,
        feedback_type: FeedbackType,
        limit: Optional[int] = None,
    ) -> list[FeedbackRecord]:
        """List feedback by type."""
        return self._load_from_dir(feedback_type, limit)

    def list_all(self, limit: Optional[int] = None) -> list[FeedbackRecord]:
        """List all feedback records."""
        all_feedback = []
        for feedback_type in ["good", "bad", "corrected"]:
            all_feedback.extend(self._load_from_dir(feedback_type))  # type: ignore

        # Sort by timestamp, newest first
        all_feedback.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            return all_feedback[:limit]
        return all_feedback

    def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics."""
        good = self.list_by_type("good")
        bad = self.list_by_type("bad")
        corrected = self.list_by_type("corrected")

        total = len(good) + len(bad) + len(corrected)

        return {
            "total": total,
            "good": len(good),
            "bad": len(bad),
            "corrected": len(corrected),
            "positive_rate": len(good) / total if total > 0 else 0,
            "correction_rate": (
                len(corrected) / (len(bad) + len(corrected))
                if (len(bad) + len(corrected)) > 0
                else 0
            ),
        }

    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback record."""
        for feedback_type in ["good", "bad", "corrected"]:
            type_dir = self.feedback_dir / feedback_type
            for file_path in type_dir.rglob(f"{feedback_id}.json"):
                file_path.unlink()
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

        # Add corrected responses (using the corrected version)
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
