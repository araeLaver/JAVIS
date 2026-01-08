"""Base loader interface for JAVIS RAG system."""

from abc import ABC, abstractmethod
from typing import Optional

from javis.rag.models import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    async def load(self, source: str, **kwargs) -> Document:
        """
        Load a single document from source.

        Args:
            source: Source path, URL, or identifier
            **kwargs: Additional loader-specific options

        Returns:
            Loaded Document object
        """
        pass

    @abstractmethod
    async def can_load(self, source: str) -> bool:
        """
        Check if this loader can handle the given source.

        Args:
            source: Source to check

        Returns:
            True if this loader can handle the source
        """
        pass

    def get_title(self, source: str, content: str) -> Optional[str]:
        """
        Extract or generate a title for the document.

        Args:
            source: Source path or URL
            content: Document content

        Returns:
            Document title or None
        """
        return None
