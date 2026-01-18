"""Local file-based conversation storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.storage.base import (
    ConversationRecord,
    ConversationStoreInterface,
    ConversationTurn,
)

logger = logging.getLogger(__name__)


class LocalConversationStore(ConversationStoreInterface):
    """Local file-based implementation of conversation storage."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize local conversation store.

        Args:
            data_dir: Directory for storing conversations.
                      Defaults to data/conversations.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "conversations"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_month_dir(self, date: datetime) -> Path:
        """Get the month directory for a date."""
        month_str = date.strftime("%Y-%m")
        month_dir = self.data_dir / month_str
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir

    def save(self, conversation: ConversationRecord) -> str:
        """Save a conversation."""
        timestamp = datetime.now()
        month_dir = self._get_month_dir(timestamp)

        filename = f"{conversation.session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = month_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved conversation: {filepath}")
        return str(filepath)

    def get(self, session_id: str) -> Optional[ConversationRecord]:
        """Get a conversation by session ID."""
        # Search for the most recent file matching the session_id
        matching_files = list(self.data_dir.rglob(f"{session_id}_*.json"))
        if not matching_files:
            return None

        # Get the most recent one
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ConversationRecord(**data)
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Failed to load {latest_file}: {e}")
            return None

    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        feedback_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ConversationRecord]:
        """List all conversations with optional filters."""
        conversations = []

        # Get all JSON files sorted by modification time (newest first)
        files = sorted(
            self.data_dir.rglob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                conv = ConversationRecord(**data)

                # Apply date filters
                conv_date = datetime.fromisoformat(conv.started_at)
                if start and conv_date < start:
                    continue
                if end and conv_date > end:
                    continue

                # Apply feedback filter
                if feedback_filter and conv.feedback != feedback_filter:
                    continue

                conversations.append(conv)

                if limit and len(conversations) >= limit:
                    break

            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue

        return conversations

    def count(self) -> int:
        """Count total conversations."""
        return len(list(self.data_dir.rglob("*.json")))

    def delete(self, session_id: str) -> bool:
        """Delete a conversation."""
        matching_files = list(self.data_dir.rglob(f"{session_id}_*.json"))
        if not matching_files:
            return False

        for filepath in matching_files:
            filepath.unlink()
            logger.info(f"Deleted conversation: {filepath}")

        return True

    def export_for_training(
        self,
        output_path: Path,
        feedback_filter: Optional[str] = None,
        min_turns: int = 2,
    ) -> Path:
        """Export conversations as JSONL for training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conversations = self.list_all(feedback_filter=feedback_filter)
        training_data = []

        for conv in conversations:
            # Filter by minimum turns
            if len(conv.turns) < min_turns:
                continue

            # Convert to training format
            messages = []
            for turn in conv.turns:
                messages.append({
                    "role": turn.role,
                    "content": turn.content,
                })

            training_data.append({"messages": messages})

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(training_data)} conversations to {output_path}")
        return output_path
