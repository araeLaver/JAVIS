"""Document loaders for JAVIS RAG system."""

from javis.rag.loaders.base import BaseLoader
from javis.rag.loaders.file_loader import FileLoader
from javis.rag.loaders.code_loader import CodeLoader
from javis.rag.loaders.web_loader import WebLoader
from javis.rag.models import DocumentSource

__all__ = [
    "BaseLoader",
    "FileLoader",
    "CodeLoader",
    "WebLoader",
    "get_loader",
    "LoaderRegistry",
]


class LoaderRegistry:
    """Registry for document loaders."""

    _loaders: dict[DocumentSource, BaseLoader] = {}
    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize default loaders."""
        if cls._initialized:
            return

        cls._loaders = {
            DocumentSource.LOCAL_FILE: FileLoader(),
            DocumentSource.CODEBASE: CodeLoader(),
            DocumentSource.WEB_PAGE: WebLoader(),
        }
        cls._initialized = True

    @classmethod
    def get(cls, source_type: DocumentSource) -> BaseLoader:
        """Get loader for source type."""
        cls.initialize()
        loader = cls._loaders.get(source_type)
        if not loader:
            raise ValueError(f"No loader for source type: {source_type}")
        return loader

    @classmethod
    def register(cls, source_type: DocumentSource, loader: BaseLoader) -> None:
        """Register a custom loader."""
        cls.initialize()
        cls._loaders[source_type] = loader

    @classmethod
    async def auto_detect(cls, source: str) -> BaseLoader:
        """
        Auto-detect and return appropriate loader.

        Args:
            source: Source path or URL

        Returns:
            Appropriate loader instance
        """
        cls.initialize()

        # Check each loader
        for source_type, loader in cls._loaders.items():
            if await loader.can_load(source):
                return loader

        raise ValueError(f"No loader can handle source: {source}")


def get_loader(source_type: DocumentSource) -> BaseLoader:
    """Get loader for source type."""
    return LoaderRegistry.get(source_type)
