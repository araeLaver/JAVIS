"""RAG manager for JAVIS retrieval-augmented generation system."""

import logging
from typing import Optional

from javis.memory.embeddings import EmbeddingService, get_embedding_service
from javis.rag.models import (
    Document, DocumentChunk, DocumentSource,
    RetrievalResult, DocumentStats
)
from javis.rag.store import DocumentStore
from javis.rag.chunker import TextChunker, CodeChunker
from javis.rag.loaders import LoaderRegistry, FileLoader, CodeLoader, WebLoader

logger = logging.getLogger(__name__)


class RAGManager:
    """RAG system manager (singleton)."""

    _instance: Optional["RAGManager"] = None

    def __init__(
        self,
        db_path: str = "./data/vectors/documents",
        collection_name: str = "javis_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG manager.

        Args:
            db_path: Path for ChromaDB storage
            collection_name: Name of the collection
            embedding_model: Model for embeddings
            chunk_size: Default chunk size
            chunk_overlap: Default chunk overlap
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.embeddings = get_embedding_service(embedding_model)
        self.store = DocumentStore(db_path, collection_name)
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.code_chunker = CodeChunker(chunk_size * 2, chunk_overlap * 2)

        # Initialize loaders
        LoaderRegistry.initialize()

        logger.info(f"RAGManager initialized. DB: {db_path}")

    @classmethod
    def get_instance(
        cls,
        db_path: str = "./data/vectors/documents",
        collection_name: str = "javis_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> "RAGManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(db_path, collection_name, embedding_model)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    async def add_document(
        self,
        source_type: str,
        source_path: str,
        **kwargs
    ) -> Optional[Document]:
        """
        Add a document to the RAG system.

        Args:
            source_type: Type of source (file, web, code)
            source_path: Path or URL to the source
            **kwargs: Additional loader options

        Returns:
            Created Document or None if failed
        """
        try:
            # Map source type to enum
            type_map = {
                "file": DocumentSource.LOCAL_FILE,
                "local_file": DocumentSource.LOCAL_FILE,
                "web": DocumentSource.WEB_PAGE,
                "web_page": DocumentSource.WEB_PAGE,
                "url": DocumentSource.WEB_PAGE,
                "code": DocumentSource.CODEBASE,
                "codebase": DocumentSource.CODEBASE,
            }
            doc_source = type_map.get(source_type.lower())
            if not doc_source:
                raise ValueError(f"Unknown source type: {source_type}")

            # Load document
            loader = LoaderRegistry.get(doc_source)
            document = await loader.load(source_path, **kwargs)

            # Choose chunker based on source
            if doc_source == DocumentSource.CODEBASE:
                chunker = self.code_chunker
            else:
                chunker = self.text_chunker

            # Chunk document
            chunks = chunker.chunk(
                document.content,
                document.id,
                {"source": doc_source.value, "title": document.title}
            )

            if not chunks:
                logger.warning(f"No chunks created for: {source_path}")
                return None

            # Generate embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embeddings.embed(chunk_texts)

            # Store document and chunks
            self.store.add_document(document)
            self.store.add_chunks(chunks, embeddings)

            logger.info(f"Added document: {document.title} ({len(chunks)} chunks)")
            return document

        except Exception as e:
            logger.error(f"Failed to add document {source_path}: {e}")
            return None

    async def add_codebase(
        self,
        directory: str,
        extensions: Optional[set[str]] = None,
        max_files: int = 100
    ) -> list[Document]:
        """
        Add all code files from a directory.

        Args:
            directory: Directory path
            extensions: File extensions to include
            max_files: Maximum files to process

        Returns:
            List of added Documents
        """
        loader = CodeLoader()
        documents = await loader.load_directory(directory, extensions, max_files)

        added = []
        for doc in documents:
            # Chunk each document
            chunks = self.code_chunker.chunk(
                doc.content,
                doc.id,
                {"source": doc.source.value, "title": doc.title}
            )

            if chunks:
                embeddings = self.embeddings.embed([c.content for c in chunks])
                self.store.add_document(doc)
                self.store.add_chunks(chunks, embeddings)
                added.append(doc)

        logger.info(f"Added {len(added)} code files from {directory}")
        return added

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        document_id: Optional[str] = None
    ) -> list[RetrievalResult]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score
            document_id: Optional filter by document

        Returns:
            List of RetrievalResult
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_single(query)

            # Search chunks
            results = self.store.search_chunks(
                query_embedding,
                n_results=top_k,
                document_id=document_id
            )

            # Convert to RetrievalResult
            retrieval_results = []
            for i, chunk_id in enumerate(results["ids"]):
                distance = results["distances"][i]
                score = 1 - distance  # cosine distance to similarity

                if score < min_score:
                    continue

                metadata = results["metadatas"][i]

                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=results["documents"][i],
                    chunk_index=metadata.get("chunk_index", 0),
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", 0),
                )

                # Get document metadata
                doc_data = self.store.get_document(chunk.document_id)
                document = None
                if doc_data:
                    document = Document(
                        id=doc_data["id"],
                        source=DocumentSource(doc_data["source"]),
                        source_path=doc_data["source_path"],
                        title=doc_data.get("title"),
                        content="",  # Don't load full content
                        metadata=doc_data.get("metadata", {})
                    )

                retrieval_results.append(RetrievalResult(
                    chunk=chunk,
                    document=document,
                    score=score,
                    distance=distance
                ))

            logger.debug(f"Found {len(retrieval_results)} results for: {query[:50]}...")
            return retrieval_results

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    async def get_context_for_query(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 2000
    ) -> str:
        """
        Get formatted context for prompt injection.

        Args:
            query: User query
            top_k: Number of results
            max_chars: Maximum characters

        Returns:
            Formatted context string
        """
        results = await self.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = ["### 관련 문서:"]
        total_chars = 0

        for i, result in enumerate(results, 1):
            source = ""
            if result.document:
                source = result.document.title or result.document.source_path

            header = f"\n[{i}] {source} (관련도: {result.score:.2f})"
            content = result.chunk.content

            # Truncate if needed
            available = max_chars - total_chars - len(header)
            if available <= 0:
                break

            if len(content) > available:
                content = content[:available] + "..."

            context_parts.append(f"{header}\n{content}")
            total_chars += len(header) + len(content)

        return "\n".join(context_parts)

    def format_for_prompt(self, results: list[RetrievalResult]) -> str:
        """
        Format search results for prompt injection.

        Args:
            results: List of RetrievalResult

        Returns:
            Formatted string
        """
        if not results:
            return ""

        lines = ["### 관련 문서:"]
        for i, result in enumerate(results, 1):
            source = ""
            if result.document:
                source = f"[{result.document.title or result.document.source_path}]"

            lines.append(f"\n**[{i}]** {source} (점수: {result.score:.2f})")
            lines.append(result.chunk.content)

        return "\n".join(lines)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            self.store.delete_document(document_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """List all documents."""
        return self.store.list_documents(limit, offset)

    async def get_stats(self) -> DocumentStats:
        """Get RAG system statistics."""
        docs = self.store.list_documents(limit=1000)

        by_source: dict[str, int] = {}
        for doc in docs:
            source = doc.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        return DocumentStats(
            total_documents=self.store.count_documents(),
            total_chunks=self.store.count_chunks(),
            by_source=by_source
        )

    def shutdown(self) -> None:
        """Cleanup resources."""
        logger.info("RAGManager shutting down")


def get_rag_manager() -> RAGManager:
    """Get the global RAG manager instance."""
    return RAGManager.get_instance()
