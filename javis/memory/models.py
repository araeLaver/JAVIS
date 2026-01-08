"""Memory data models for JAVIS."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry stored in vector DB."""

    id: str = Field(description="Unique identifier (UUID)")
    session_id: str = Field(description="Source conversation session ID")
    summary: str = Field(description="Summarized conversation content")
    topics: list[str] = Field(default_factory=list, description="Extracted topics/tags")
    created_at: datetime = Field(default_factory=datetime.now)
    turn_count: int = Field(default=0, description="Number of turns in original conversation")
    metadata: dict = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """Result from memory search."""

    entry: MemoryEntry
    score: float = Field(description="Similarity score (0-1)")
    distance: float = Field(default=0.0, description="Vector distance")


class ConversationSummary(BaseModel):
    """Output from conversation summarizer."""

    summary: str = Field(description="Summarized conversation")
    topics: list[str] = Field(default_factory=list, description="Extracted topics")
    key_entities: list[str] = Field(default_factory=list, description="Key entities mentioned")
    sentiment: Optional[str] = Field(default=None, description="Overall sentiment")
