"""Memory manager for JAVIS long-term memory system."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from javis.memory.models import MemoryEntry, MemorySearchResult, ConversationSummary
from javis.memory.embeddings import EmbeddingService, get_embedding_service
from javis.memory.store import VectorStore
from javis.memory.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class MemoryManager:
    """Long-term memory manager (singleton)."""

    _instance: Optional["MemoryManager"] = None

    def __init__(
        self,
        db_path: str = "./data/vectors/memory",
        collection_name: str = "javis_memory",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize memory manager.

        Args:
            db_path: Path for ChromaDB storage
            collection_name: Name of the vector collection
            embedding_model: Model for generating embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize components
        self.embeddings = get_embedding_service(embedding_model)
        self.store = VectorStore(db_path, collection_name)
        self.summarizer = ConversationSummarizer()

        logger.info(f"MemoryManager initialized. DB: {db_path}")

    @classmethod
    def get_instance(
        cls,
        db_path: str = "./data/vectors/memory",
        collection_name: str = "javis_memory",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> "MemoryManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(db_path, collection_name, embedding_model)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    async def store_conversation(
        self,
        session_id: str,
        turns: list[dict],
        metadata: Optional[dict] = None
    ) -> Optional[MemoryEntry]:
        """
        Store a conversation in long-term memory.

        Args:
            session_id: Session identifier
            turns: List of conversation turns
            metadata: Optional additional metadata

        Returns:
            Created MemoryEntry or None if failed
        """
        if not turns or len(turns) < 2:
            logger.debug(f"Skipping short conversation: {session_id}")
            return None

        try:
            # Summarize conversation
            summary = await self.summarizer.summarize(turns, session_id)

            # Create memory entry
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                session_id=session_id,
                summary=summary.summary,
                topics=summary.topics,
                created_at=datetime.now(),
                turn_count=len(turns),
                metadata={
                    **(metadata or {}),
                    "key_entities": summary.key_entities,
                    "sentiment": summary.sentiment
                }
            )

            # Generate embedding
            embedding = self.embeddings.embed_single(summary.summary)

            # Store in vector DB
            self.store.add(
                ids=[entry.id],
                embeddings=[embedding],
                documents=[summary.summary],
                metadatas=[{
                    "session_id": entry.session_id,
                    "topics": ",".join(entry.topics),
                    "turn_count": entry.turn_count,
                    "created_at": entry.created_at.isoformat()
                }]
            )

            logger.info(f"Stored memory: {entry.id} for session {session_id}")
            return entry

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return None

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3
    ) -> list[MemorySearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)

        Returns:
            List of MemorySearchResult
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_single(query)

            # Search vector store
            results = self.store.search(query_embedding, n_results=limit)

            # Convert to MemorySearchResult
            search_results = []
            for i, doc_id in enumerate(results["ids"]):
                distance = results["distances"][i]
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distance

                if score < min_score:
                    continue

                metadata = results["metadatas"][i]
                topics = metadata.get("topics", "").split(",") if metadata.get("topics") else []

                entry = MemoryEntry(
                    id=doc_id,
                    session_id=metadata.get("session_id", ""),
                    summary=results["documents"][i],
                    topics=topics,
                    turn_count=metadata.get("turn_count", 0),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
                )

                search_results.append(MemorySearchResult(
                    entry=entry,
                    score=score,
                    distance=distance
                ))

            logger.debug(f"Found {len(search_results)} memories for query: {query[:50]}...")
            return search_results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def get_relevant_context(
        self,
        query: str,
        limit: int = 3,
        max_chars: int = 1000
    ) -> str:
        """
        Get formatted context for prompt injection.

        Args:
            query: User query
            limit: Maximum memories to include
            max_chars: Maximum characters in context

        Returns:
            Formatted context string
        """
        results = await self.search_memories(query, limit=limit)

        if not results:
            return ""

        context_parts = ["### 관련 과거 대화:"]
        total_chars = 0

        for i, result in enumerate(results, 1):
            entry = result.entry
            line = f"[{i}] ({entry.created_at.strftime('%Y-%m-%d')}): {entry.summary}"

            if total_chars + len(line) > max_chars:
                break

            context_parts.append(line)
            total_chars += len(line)

        return "\n".join(context_parts)

    def format_for_prompt(self, results: list[MemorySearchResult]) -> str:
        """
        Format search results for prompt injection.

        Args:
            results: List of MemorySearchResult

        Returns:
            Formatted string
        """
        if not results:
            return ""

        lines = ["### 관련 과거 대화:"]
        for i, result in enumerate(results, 1):
            entry = result.entry
            date_str = entry.created_at.strftime("%Y-%m-%d")
            topics_str = ", ".join(entry.topics[:3]) if entry.topics else ""

            line = f"[Memory {i}] {date_str}"
            if topics_str:
                line += f" ({topics_str})"
            line += f": {entry.summary}"

            lines.append(line)

        return "\n".join(lines)

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            self.store.delete([memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    async def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "count": self.store.count(),
            "db_path": self.db_path,
            "collection": self.collection_name
        }

    def shutdown(self) -> None:
        """Cleanup resources."""
        logger.info("MemoryManager shutting down")
        # ChromaDB handles persistence automatically


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return MemoryManager.get_instance()
