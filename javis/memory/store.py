"""ChromaDB vector store for JAVIS memory system."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for vector storage."""

    def __init__(
        self,
        path: str,
        collection_name: str = "javis_memory",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize vector store.

        Args:
            path: Path to persist ChromaDB data
            collection_name: Name of the collection
            embedding_function: Optional custom embedding function
        """
        self.path = Path(path)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = embedding_function

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                logger.info(f"Initializing ChromaDB at: {self.path}")
                self._client = chromadb.PersistentClient(
                    path=str(self.path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except ImportError:
                logger.error("chromadb not installed. Run: pip install chromadb")
                raise
        return self._client

    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Collection '{self.collection_name}' ready. Count: {self._collection.count()}")
        return self._collection

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None
    ) -> None:
        """
        Add vectors to the store.

        Args:
            ids: Unique identifiers
            embeddings: Vector embeddings
            documents: Original text documents
            metadatas: Optional metadata for each document
        """
        collection = self._get_collection()

        if metadatas is None:
            metadatas = [{}] * len(ids)

        # ChromaDB requires metadata values to be str, int, float, or bool
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                elif isinstance(v, list):
                    cleaned[k] = ",".join(str(x) for x in v)
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas
        )
        logger.debug(f"Added {len(ids)} vectors to collection")

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ) -> dict:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        collection = self._get_collection()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)

        # Flatten results (query returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def get(self, ids: list[str]) -> dict:
        """
        Get vectors by IDs.

        Args:
            ids: List of IDs to retrieve

        Returns:
            Dict with ids, documents, metadatas
        """
        collection = self._get_collection()
        return collection.get(
            ids=ids,
            include=["documents", "metadatas"]
        )

    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors by IDs.

        Args:
            ids: List of IDs to delete
        """
        collection = self._get_collection()
        collection.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} vectors from collection")

    def count(self) -> int:
        """Get total count of vectors in collection."""
        collection = self._get_collection()
        return collection.count()

    def reset(self) -> None:
        """Reset (delete) the entire collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Collection '{self.collection_name}' reset")
        except Exception as e:
            logger.warning(f"Failed to reset collection: {e}")

    def list_all(self, limit: int = 100, offset: int = 0) -> dict:
        """
        List all vectors in collection.

        Args:
            limit: Maximum number to return
            offset: Offset for pagination

        Returns:
            Dict with ids, documents, metadatas
        """
        collection = self._get_collection()
        return collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )
