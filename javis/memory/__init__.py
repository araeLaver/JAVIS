"""JAVIS Memory System - Long-term conversation memory with vector storage."""

from javis.memory.models import MemoryEntry, MemorySearchResult, ConversationSummary
from javis.memory.embeddings import EmbeddingService, get_embedding_service
from javis.memory.store import VectorStore
from javis.memory.summarizer import ConversationSummarizer
from javis.memory.manager import MemoryManager, get_memory_manager

__all__ = [
    # Models
    "MemoryEntry",
    "MemorySearchResult",
    "ConversationSummary",
    # Services
    "EmbeddingService",
    "get_embedding_service",
    "VectorStore",
    "ConversationSummarizer",
    # Manager
    "MemoryManager",
    "get_memory_manager",
    # Initialization
    "initialize_memory",
]


_initialized = False


def initialize_memory(
    db_path: str = "./data/vectors/memory",
    collection_name: str = "javis_memory",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> MemoryManager:
    """
    Initialize the memory system.

    Args:
        db_path: Path for ChromaDB storage
        collection_name: Name of the vector collection
        embedding_model: Model for generating embeddings

    Returns:
        Initialized MemoryManager instance
    """
    global _initialized

    manager = MemoryManager.get_instance(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

    _initialized = True
    return manager


def is_initialized() -> bool:
    """Check if memory system is initialized."""
    return _initialized


def shutdown_memory() -> None:
    """Shutdown the memory system."""
    global _initialized

    if _initialized:
        manager = get_memory_manager()
        manager.shutdown()
        MemoryManager.reset_instance()
        EmbeddingService.reset_instance()
        _initialized = False
