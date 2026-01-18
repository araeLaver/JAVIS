"""Cloud-based conversation storage using Supabase."""

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
from javis.storage.cloud.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Table names
CONVERSATIONS_TABLE = "conversations"
CONVERSATION_TURNS_TABLE = "conversation_turns"


class CloudConversationStore(ConversationStoreInterface):
    """Cloud-based implementation of conversation storage using Supabase."""

    def __init__(self):
        """Initialize cloud conversation store."""
        self.client = get_supabase_client()

    def save(self, conversation: ConversationRecord) -> str:
        """Save a conversation."""
        # Insert or update main conversation record
        conv_data = {
            "session_id": conversation.session_id,
            "started_at": conversation.started_at,
            "feedback": conversation.feedback,
            "tags": conversation.tags,
            "updated_at": datetime.now().isoformat(),
        }

        # Check if conversation exists
        existing = self.client.select_one(
            CONVERSATIONS_TABLE,
            filters={"session_id": conversation.session_id},
        )

        if existing:
            # Update existing
            self.client.update(
                CONVERSATIONS_TABLE,
                conv_data,
                {"session_id": conversation.session_id},
            )
            conv_id = existing["id"]
        else:
            # Insert new
            result = self.client.insert(CONVERSATIONS_TABLE, conv_data)
            conv_id = result["id"]

        # Delete existing turns and insert new ones
        self.client.delete(
            CONVERSATION_TURNS_TABLE,
            {"conversation_id": conv_id},
        )

        # Insert turns
        for idx, turn in enumerate(conversation.turns):
            turn_data = {
                "conversation_id": conv_id,
                "turn_index": idx,
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
            }
            self.client.insert(CONVERSATION_TURNS_TABLE, turn_data)

        logger.info(f"Saved conversation: {conversation.session_id}")
        return conversation.session_id

    def get(self, session_id: str) -> Optional[ConversationRecord]:
        """Get a conversation by session ID."""
        # Get conversation
        conv = self.client.select_one(
            CONVERSATIONS_TABLE,
            filters={"session_id": session_id},
        )

        if not conv:
            return None

        # Get turns
        turns_data = self.client.select(
            CONVERSATION_TURNS_TABLE,
            filters={"conversation_id": conv["id"]},
            order="turn_index",
            desc=False,
        )

        turns = [
            ConversationTurn(
                role=t["role"],
                content=t["content"],
                timestamp=t["timestamp"],
            )
            for t in turns_data
        ]

        return ConversationRecord(
            session_id=conv["session_id"],
            started_at=conv["started_at"],
            turns=turns,
            feedback=conv.get("feedback"),
            tags=conv.get("tags", []),
        )

    def list_all(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        feedback_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ConversationRecord]:
        """List all conversations with optional filters."""
        filters = {}

        if feedback_filter:
            filters["feedback"] = feedback_filter

        # Get conversations
        convs = self.client.select(
            CONVERSATIONS_TABLE,
            filters=filters if filters else None,
            order="started_at",
            desc=True,
            limit=limit,
        )

        conversations = []
        for conv in convs:
            # Apply date filters
            started_at = datetime.fromisoformat(conv["started_at"])
            if start and started_at < start:
                continue
            if end and started_at > end:
                continue

            # Get turns
            turns_data = self.client.select(
                CONVERSATION_TURNS_TABLE,
                filters={"conversation_id": conv["id"]},
                order="turn_index",
                desc=False,
            )

            turns = [
                ConversationTurn(
                    role=t["role"],
                    content=t["content"],
                    timestamp=t["timestamp"],
                )
                for t in turns_data
            ]

            conversations.append(
                ConversationRecord(
                    session_id=conv["session_id"],
                    started_at=conv["started_at"],
                    turns=turns,
                    feedback=conv.get("feedback"),
                    tags=conv.get("tags", []),
                )
            )

        return conversations

    def count(self) -> int:
        """Count total conversations."""
        return self.client.count(CONVERSATIONS_TABLE)

    def delete(self, session_id: str) -> bool:
        """Delete a conversation."""
        # Get conversation ID first
        conv = self.client.select_one(
            CONVERSATIONS_TABLE,
            filters={"session_id": session_id},
        )

        if not conv:
            return False

        # Delete turns first (foreign key constraint)
        self.client.delete(
            CONVERSATION_TURNS_TABLE,
            {"conversation_id": conv["id"]},
        )

        # Delete conversation
        self.client.delete(
            CONVERSATIONS_TABLE,
            {"session_id": session_id},
        )

        logger.info(f"Deleted conversation: {session_id}")
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
