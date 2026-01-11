"""ChromaDB vector store for JAVIS RAG system."""

import json
import logging
from pathlib import Path
from typing import Optional

from javis.rag.models import Document, DocumentChunk, DocumentSource

logger = logging.getLogger(__name__)


class DocumentStore:
    """ChromaDB wrapper for document and chunk storage."""

    def __init__(
        self,
        path: str,
        collection_name: str = "javis_documents"
    ):
        """
        Initialize document store.

        Args:
            path: Path to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.path = Path(path)
        self.collection_name = collection_name
        self._client = None
        self._chunks_collection = None
        self._docs_collection = None

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                logger.info(f"Initializing ChromaDB for RAG at: {self.path}")
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

    def _get_chunks_collection(self):
        """Get or create chunks collection."""
        if self._chunks_collection is None:
            client = self._get_client()
            self._chunks_collection = client.get_or_create_collection(
                name=f"{self.collection_name}_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Chunks collection ready. Count: {self._chunks_collection.count()}")
        return self._chunks_collection

    def _get_docs_collection(self):
        """Get or create documents metadata collection."""
        if self._docs_collection is None:
            client = self._get_client()
            self._docs_collection = client.get_or_create_collection(
                name=f"{self.collection_name}_docs"
            )
            logger.info(f"Docs collection ready. Count: {self._docs_collection.count()}")
        return self._docs_collection

    def add_document(self, document: Document) -> None:
        """
        Store document metadata.

        Args:
            document: Document to store
        """
        docs = self._get_docs_collection()

        docs.add(
            ids=[document.id],
            documents=[document.content[:1000]],  # Store preview
            metadatas=[{
                "source": document.source.value,
                "source_path": document.source_path,
                "title": document.title or "",
                "created_at": document.created_at.isoformat(),
                "metadata_json": json.dumps(document.metadata)
            }]
        )
        logger.debug(f"Stored document: {document.id}")

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]]
    ) -> None:
        """
        Store document chunks with embeddings.

        Args:
            chunks: List of chunks
            embeddings: Corresponding embeddings
        """
        if not chunks:
            return

        collection = self._get_chunks_collection()

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.debug(f"Added {len(chunks)} chunks")

    def search_chunks(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            n_results: Number of results
            document_id: Optional filter by document

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        collection = self._get_chunks_collection()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if document_id:
            kwargs["where"] = {"document_id": document_id}

        results = collection.query(**kwargs)

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def get_document(self, document_id: str) -> Optional[dict]:
        """Get document by ID."""
        docs = self._get_docs_collection()
        result = docs.get(ids=[document_id], include=["documents", "metadatas"])

        if not result["ids"]:
            return None

        meta = result["metadatas"][0]
        return {
            "id": document_id,
            "source": meta.get("source"),
            "source_path": meta.get("source_path"),
            "title": meta.get("title"),
            "created_at": meta.get("created_at"),
            "metadata": json.loads(meta.get("metadata_json", "{}"))
        }

    def get_documents_batch(self, document_ids: list[str]) -> dict[str, dict]:
        """
        Get multiple documents by IDs in a single query.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dict mapping document_id to document data
        """
        if not document_ids:
            return {}

        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(document_ids))

        docs = self._get_docs_collection()
        result = docs.get(ids=unique_ids, include=["documents", "metadatas"])

        documents = {}
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            documents[doc_id] = {
                "id": doc_id,
                "source": meta.get("source"),
                "source_path": meta.get("source_path"),
                "title": meta.get("title"),
                "created_at": meta.get("created_at"),
                "metadata": json.loads(meta.get("metadata_json", "{}"))
            }

        return documents

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document and its chunks.

        Args:
            document_id: Document ID to delete
        """
        # Delete chunks
        chunks = self._get_chunks_collection()
        try:
            chunks.delete(where={"document_id": document_id})
        except Exception as e:
            logger.warning(f"Failed to delete chunks: {e}")

        # Delete document
        docs = self._get_docs_collection()
        try:
            docs.delete(ids=[document_id])
        except Exception as e:
            logger.warning(f"Failed to delete document: {e}")

        logger.info(f"Deleted document: {document_id}")

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List all documents."""
        docs = self._get_docs_collection()
        result = docs.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )

        documents = []
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            documents.append({
                "id": doc_id,
                "source": meta.get("source"),
                "source_path": meta.get("source_path"),
                "title": meta.get("title"),
                "created_at": meta.get("created_at"),
            })

        return documents

    def count_documents(self) -> int:
        """Get total document count."""
        return self._get_docs_collection().count()

    def count_chunks(self) -> int:
        """Get total chunk count."""
        return self._get_chunks_collection().count()

    def reset(self) -> None:
        """Reset all collections."""
        client = self._get_client()

        for name in [f"{self.collection_name}_chunks", f"{self.collection_name}_docs"]:
            try:
                client.delete_collection(name)
            except Exception as e:
                logger.warning(f"Failed to delete collection {name}: {e}")

        self._chunks_collection = None
        self._docs_collection = None
        logger.info("RAG collections reset")
