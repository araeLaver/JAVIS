"""Conversation logger for fine-tuning data collection."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.storage import get_conversation_store
from javis.storage.base import (
    ConversationRecord,
    ConversationStoreInterface,
    ConversationTurn as StorageConversationTurn,
)


class ConversationTurn(BaseModel):
    """Single conversation turn."""
    role: str
    content: str
    timestamp: str


class ConversationLog(BaseModel):
    """Full conversation log."""
    session_id: str
    started_at: str
    turns: list[ConversationTurn] = []
    feedback: Optional[str] = None  # good, bad, None
    tags: list[str] = []

    def to_storage_record(self) -> ConversationRecord:
        """Convert to storage record format."""
        return ConversationRecord(
            session_id=self.session_id,
            started_at=self.started_at,
            turns=[
                StorageConversationTurn(
                    role=t.role,
                    content=t.content,
                    timestamp=t.timestamp,
                )
                for t in self.turns
            ],
            feedback=self.feedback,
            tags=self.tags,
        )

    @classmethod
    def from_storage_record(cls, record: ConversationRecord) -> "ConversationLog":
        """Create from storage record."""
        return cls(
            session_id=record.session_id,
            started_at=record.started_at,
            turns=[
                ConversationTurn(
                    role=t.role,
                    content=t.content,
                    timestamp=t.timestamp,
                )
                for t in record.turns
            ],
            feedback=record.feedback,
            tags=record.tags,
        )


class ConversationLogger:
    """Logger for collecting fine-tuning data.

    Uses the storage factory to save conversations to either local files
    or cloud storage (Supabase) depending on configuration.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        store: Optional[ConversationStoreInterface] = None,
    ):
        """Initialize conversation logger.

        Args:
            data_dir: Legacy parameter for backward compatibility.
            store: Storage interface to use. If None, uses storage factory.
        """
        self._store = store

        # Keep data_dir for backward compatibility
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "conversations"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 현재 활성 대화들
        self.active_conversations: dict[str, ConversationLog] = {}

    @property
    def store(self) -> ConversationStoreInterface:
        """Get the conversation store (lazy initialization)."""
        if self._store is None:
            self._store = get_conversation_store()
        return self._store

    def start_conversation(self, session_id: str) -> ConversationLog:
        """Start a new conversation."""
        conv = ConversationLog(
            session_id=session_id,
            started_at=datetime.now().isoformat()
        )
        self.active_conversations[session_id] = conv
        return conv

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a turn to the conversation."""
        if session_id not in self.active_conversations:
            self.start_conversation(session_id)

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self.active_conversations[session_id].turns.append(turn)

    def add_feedback(self, session_id: str, feedback: str):
        """Add feedback (good/bad) to a conversation."""
        if session_id in self.active_conversations:
            self.active_conversations[session_id].feedback = feedback

    def add_tags(self, session_id: str, tags: list[str]):
        """Add tags to a conversation."""
        if session_id in self.active_conversations:
            self.active_conversations[session_id].tags.extend(tags)

    def save_conversation(self, session_id: str):
        """Save conversation using the storage backend."""
        if session_id not in self.active_conversations:
            return

        conv = self.active_conversations[session_id]

        # Use storage backend
        result = self.store.save(conv.to_storage_record())

        return result

    def end_conversation(self, session_id: str):
        """End and save a conversation."""
        filepath = self.save_conversation(session_id)
        if session_id in self.active_conversations:
            del self.active_conversations[session_id]
        return filepath

    def export_for_training(
        self,
        output_path: Optional[Path] = None,
        feedback_filter: Optional[str] = None,
    ) -> Path:
        """Export conversations in training format (JSONL).

        Uses the storage backend to export conversations.

        Args:
            output_path: Output file path (default: auto-generated)
            feedback_filter: Filter by feedback type (good/bad/None)

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_dir = self.data_dir.parent / "training" / "exported"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
        output_path = Path(output_path)

        # Use storage backend to export
        return self.store.export_for_training(
            output_path=output_path,
            feedback_filter=feedback_filter,
            min_turns=2,
        )


# 전역 로거 인스턴스
_logger: Optional[ConversationLogger] = None


def get_logger() -> ConversationLogger:
    """Get the global conversation logger."""
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger
