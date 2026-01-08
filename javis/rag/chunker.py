"""Text chunking utilities for JAVIS RAG system."""

import logging
import re
import uuid
from typing import Optional

from javis.rag.models import DocumentChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            separator: Primary separator to try splitting on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict] = None
    ) -> list[DocumentChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            document_id: Parent document ID
            metadata: Optional metadata to include in chunks

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []

        # Normalize whitespace
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk
                end = len(text)
            else:
                # Try to find a good break point
                end = self._find_break_point(text, start, end)

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start with overlap
            if end >= len(text):
                break

            start = end - self.chunk_overlap
            if start < 0:
                start = 0

            # Prevent infinite loop
            if start >= end:
                start = end

        logger.debug(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """
        Find a good break point (paragraph, sentence, or word boundary).

        Args:
            text: Full text
            start: Start position
            end: Target end position

        Returns:
            Best break point position
        """
        # Look for paragraph break (double newline)
        paragraph_break = text.rfind('\n\n', start, end)
        if paragraph_break > start + self.chunk_size // 2:
            return paragraph_break + 2

        # Look for single newline
        line_break = text.rfind('\n', start, end)
        if line_break > start + self.chunk_size // 2:
            return line_break + 1

        # Look for sentence end
        for pattern in ['. ', '! ', '? ', '。', '！', '？']:
            sentence_end = text.rfind(pattern, start, end)
            if sentence_end > start + self.chunk_size // 2:
                return sentence_end + len(pattern)

        # Look for word boundary (space)
        space = text.rfind(' ', start, end)
        if space > start + self.chunk_size // 2:
            return space + 1

        # No good break point found, use original end
        return end


class CodeChunker(TextChunker):
    """Specialized chunker for source code."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        super().__init__(chunk_size, chunk_overlap, separator="\n\n")

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find break point respecting code structure."""
        # Look for function/class definitions
        for pattern in ['\ndef ', '\nclass ', '\nasync def ']:
            match = text.rfind(pattern, start, end)
            if match > start + self.chunk_size // 3:
                return match + 1

        # Look for blank lines between code blocks
        blank_line = text.rfind('\n\n', start, end)
        if blank_line > start + self.chunk_size // 2:
            return blank_line + 2

        # Fall back to regular line break
        return super()._find_break_point(text, start, end)


def get_chunker(
    source_type: str = "text",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> TextChunker:
    """
    Get appropriate chunker for source type.

    Args:
        source_type: Type of content (text, code)
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        TextChunker or CodeChunker instance
    """
    if source_type == "code":
        return CodeChunker(chunk_size, chunk_overlap)
    return TextChunker(chunk_size, chunk_overlap)
