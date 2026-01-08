"""RAG data models for JAVIS."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentSource(str, Enum):
    """Source type of a document."""
    LOCAL_FILE = "local_file"
    CODEBASE = "codebase"
    WEB_PAGE = "web_page"


class Document(BaseModel):
    """A document loaded into the RAG system."""

    id: str = Field(description="Unique identifier (UUID)")
    source: DocumentSource = Field(description="Document source type")
    source_path: str = Field(description="Original path or URL")
    title: Optional[str] = Field(default=None, description="Document title")
    content: str = Field(description="Full document content")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of a document for embedding."""

    id: str = Field(description="Unique identifier (UUID)")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Position in document (0-indexed)")
    start_char: int = Field(default=0, description="Start character position in original")
    end_char: int = Field(default=0, description="End character position in original")
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from RAG retrieval."""

    chunk: DocumentChunk
    document: Optional[Document] = None
    score: float = Field(description="Similarity score (0-1)")
    distance: float = Field(default=0.0, description="Vector distance")


class DocumentStats(BaseModel):
    """Statistics about the document store."""

    total_documents: int = 0
    total_chunks: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
