"""JAVIS RAG System - Retrieval-Augmented Generation for document-based Q&A."""

from javis.rag.models import (
    DocumentSource,
    Document,
    DocumentChunk,
    RetrievalResult,
    DocumentStats,
)
from javis.rag.chunker import TextChunker, CodeChunker, get_chunker
from javis.rag.store import DocumentStore
from javis.rag.manager import RAGManager, get_rag_manager
from javis.rag.loaders import (
    BaseLoader,
    FileLoader,
    CodeLoader,
    WebLoader,
    LoaderRegistry,
    get_loader,
)

__all__ = [
    # Models
    "DocumentSource",
    "Document",
    "DocumentChunk",
    "RetrievalResult",
    "DocumentStats",
    # Chunking
    "TextChunker",
    "CodeChunker",
    "get_chunker",
    # Store
    "DocumentStore",
    # Loaders
    "BaseLoader",
    "FileLoader",
    "CodeLoader",
    "WebLoader",
    "LoaderRegistry",
    "get_loader",
    # Manager
    "RAGManager",
    "get_rag_manager",
    # Initialization
    "initialize_rag",
]


_initialized = False


def initialize_rag(
    db_path: str = "./data/vectors/documents",
    collection_name: str = "javis_documents",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> RAGManager:
    """
    Initialize the RAG system.

    Args:
        db_path: Path for ChromaDB storage
        collection_name: Name of the collection
        embedding_model: Model for embeddings

    Returns:
        Initialized RAGManager instance
    """
    global _initialized

    manager = RAGManager.get_instance(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

    _initialized = True
    return manager


def is_initialized() -> bool:
    """Check if RAG system is initialized."""
    return _initialized


def shutdown_rag() -> None:
    """Shutdown the RAG system."""
    global _initialized

    if _initialized:
        manager = get_rag_manager()
        manager.shutdown()
        RAGManager.reset_instance()
        _initialized = False
