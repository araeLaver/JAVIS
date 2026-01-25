"""Unit tests for JAVIS RAG system."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from javis.rag.models import (
    Document,
    DocumentChunk,
    DocumentSource,
    RetrievalResult,
    DocumentStats
)
from javis.rag.manager import RAGManager


class TestRAGModels:
    """Tests for RAG data models."""

    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(
            id="doc-123",
            source=DocumentSource.LOCAL_FILE,
            source_path="/path/to/file.txt",
            title="Test Document",
            content="This is the content"
        )

        assert doc.id == "doc-123"
        assert doc.source == DocumentSource.LOCAL_FILE
        assert doc.title == "Test Document"

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-123",
            content="Chunk content",
            chunk_index=0,
            start_char=0,
            end_char=13
        )

        assert chunk.document_id == "doc-123"
        assert chunk.chunk_index == 0

    def test_retrieval_result(self):
        """Test creating a RetrievalResult."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-123",
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12
        )

        result = RetrievalResult(
            chunk=chunk,
            document=None,
            score=0.85,
            distance=0.15
        )

        assert result.score == 0.85
        assert result.chunk.content == "Test content"

    def test_document_stats(self):
        """Test DocumentStats model."""
        stats = DocumentStats(
            total_documents=10,
            total_chunks=150,
            by_source={"local_file": 5, "web_page": 5}
        )

        assert stats.total_documents == 10
        assert stats.by_source["local_file"] == 5

    def test_document_source_values(self):
        """Test DocumentSource enum values."""
        assert DocumentSource.LOCAL_FILE.value == "local_file"
        assert DocumentSource.WEB_PAGE.value == "web_page"
        assert DocumentSource.CODEBASE.value == "codebase"


class TestRAGManager:
    """Tests for RAGManager."""

    @pytest.fixture(autouse=True)
    def reset_instance(self):
        """Reset singleton before each test."""
        RAGManager.reset_instance()
        yield
        RAGManager.reset_instance()

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding service."""
        mock = MagicMock()
        mock.embed_single.return_value = [0.1] * 384
        mock.embed.return_value = [[0.1] * 384]
        return mock

    @pytest.fixture
    def mock_store(self):
        """Create mock document store."""
        mock = MagicMock()
        mock.count_documents.return_value = 0
        mock.count_chunks.return_value = 0
        mock.add_document = MagicMock()
        mock.add_chunks = MagicMock()
        mock.delete_document = MagicMock()
        mock.list_documents.return_value = []
        mock.search_chunks.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }
        mock.get_documents_batch.return_value = {}
        return mock

    @pytest.fixture
    def manager(self, mock_embeddings, mock_store):
        """Create RAG manager with mocked dependencies."""
        with patch("javis.rag.manager.get_embedding_service", return_value=mock_embeddings), \
             patch("javis.rag.manager.DocumentStore", return_value=mock_store), \
             patch("javis.rag.manager.LoaderRegistry"):
            manager = RAGManager(
                db_path="./test_vectors",
                collection_name="test_docs"
            )
            manager.embeddings = mock_embeddings
            manager.store = mock_store
            return manager

    def test_singleton_pattern(self):
        """Test that RAGManager is a singleton."""
        with patch("javis.rag.manager.get_embedding_service"), \
             patch("javis.rag.manager.DocumentStore"), \
             patch("javis.rag.manager.LoaderRegistry"):
            instance1 = RAGManager.get_instance()
            instance2 = RAGManager.get_instance()
            assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_add_document_file(self, manager, mock_store):
        """Test adding a file document."""
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=Document(
            id="doc-123",
            source=DocumentSource.LOCAL_FILE,
            source_path="/test/file.txt",
            title="Test File",
            content="File content here"
        ))

        with patch("javis.rag.manager.LoaderRegistry.get", return_value=mock_loader):
            result = await manager.add_document(
                source_type="file",
                source_path="/test/file.txt"
            )

        assert result is not None
        assert result.title == "Test File"
        mock_store.add_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_invalid_source_type(self, manager):
        """Test adding document with invalid source type returns None."""
        # RAG manager catches the ValueError internally and returns None
        result = await manager.add_document(
            source_type="invalid_type",
            source_path="/test/path"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_add_document_web(self, manager, mock_store):
        """Test adding a web document."""
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=Document(
            id="doc-web",
            source=DocumentSource.WEB_PAGE,
            source_path="https://example.com",
            title="Example Page",
            content="Web page content"
        ))

        with patch("javis.rag.manager.LoaderRegistry.get", return_value=mock_loader):
            result = await manager.add_document(
                source_type="web",
                source_path="https://example.com"
            )

        assert result is not None
        assert result.source == DocumentSource.WEB_PAGE

    @pytest.mark.asyncio
    async def test_add_document_empty_chunks(self, manager, mock_store):
        """Test adding document that produces no chunks."""
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=Document(
            id="doc-empty",
            source=DocumentSource.LOCAL_FILE,
            source_path="/test/empty.txt",
            title="Empty",
            content=""  # Empty content
        ))

        # Mock text_chunker to return empty list
        manager.text_chunker.chunk = MagicMock(return_value=[])

        with patch("javis.rag.manager.LoaderRegistry.get", return_value=mock_loader):
            result = await manager.add_document(
                source_type="file",
                source_path="/test/empty.txt"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_search_success(self, manager, mock_store, mock_embeddings):
        """Test successful RAG search."""
        mock_store.search_chunks.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "distances": [0.1, 0.2],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [
                {"document_id": "doc-1", "chunk_index": 0, "start_char": 0, "end_char": 10},
                {"document_id": "doc-1", "chunk_index": 1, "start_char": 10, "end_char": 20}
            ]
        }
        mock_store.get_documents_batch.return_value = {
            "doc-1": {
                "id": "doc-1",
                "source": "local_file",
                "source_path": "/test/file.txt",
                "title": "Test Doc",
                "metadata": {}
            }
        }

        results = await manager.search("test query", top_k=5)

        assert len(results) == 2
        assert results[0].score == 0.9  # 1 - 0.1
        assert results[0].chunk.content == "Content 1"
        mock_embeddings.embed_single.assert_called_with("test query")

    @pytest.mark.asyncio
    async def test_search_filters_low_scores(self, manager, mock_store):
        """Test that low score results are filtered."""
        mock_store.search_chunks.return_value = {
            "ids": ["chunk-1"],
            "distances": [0.8],  # score = 0.2, below min_score
            "documents": ["Low relevance"],
            "metadatas": [{"document_id": "doc-1", "chunk_index": 0, "start_char": 0, "end_char": 10}]
        }

        results = await manager.search("query", min_score=0.5)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_empty_results(self, manager, mock_store):
        """Test search with no results."""
        mock_store.search_chunks.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }

        results = await manager.search("nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_context_for_query(self, manager, mock_store):
        """Test getting formatted context."""
        mock_store.search_chunks.return_value = {
            "ids": ["chunk-1"],
            "distances": [0.1],
            "documents": ["Relevant content here"],
            "metadatas": [{"document_id": "doc-1", "chunk_index": 0, "start_char": 0, "end_char": 20}]
        }
        mock_store.get_documents_batch.return_value = {
            "doc-1": {
                "id": "doc-1",
                "source": "local_file",
                "source_path": "/test/file.txt",
                "title": "Test Doc"
            }
        }

        context = await manager.get_context_for_query("question", top_k=3)

        assert "관련 문서" in context
        assert "Relevant content here" in context

    @pytest.mark.asyncio
    async def test_get_context_for_query_empty(self, manager, mock_store):
        """Test getting context when no results."""
        mock_store.search_chunks.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }

        context = await manager.get_context_for_query("query")

        assert context == ""

    def test_format_for_prompt(self, manager):
        """Test formatting results for prompt."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            content="Chunk content here",
            chunk_index=0,
            start_char=0,
            end_char=18
        )
        doc = Document(
            id="doc-1",
            source=DocumentSource.LOCAL_FILE,
            source_path="/path/to/file.txt",
            title="My Document",
            content=""
        )
        results = [RetrievalResult(chunk=chunk, document=doc, score=0.85, distance=0.15)]

        formatted = manager.format_for_prompt(results)

        assert "관련 문서" in formatted
        assert "My Document" in formatted
        assert "점수: 0.85" in formatted
        assert "Chunk content here" in formatted

    def test_format_for_prompt_empty(self, manager):
        """Test formatting empty results."""
        formatted = manager.format_for_prompt([])
        assert formatted == ""

    @pytest.mark.asyncio
    async def test_delete_document(self, manager, mock_store):
        """Test deleting a document."""
        result = await manager.delete_document("doc-123")

        assert result is True
        mock_store.delete_document.assert_called_once_with("doc-123")

    @pytest.mark.asyncio
    async def test_delete_document_failure(self, manager, mock_store):
        """Test delete failure handling."""
        mock_store.delete_document.side_effect = Exception("Delete failed")

        result = await manager.delete_document("doc-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_documents(self, manager, mock_store):
        """Test listing documents."""
        mock_store.list_documents.return_value = [
            {"id": "doc-1", "title": "Doc 1"},
            {"id": "doc-2", "title": "Doc 2"}
        ]

        docs = await manager.list_documents(limit=10, offset=0)

        assert len(docs) == 2
        mock_store.list_documents.assert_called_with(10, 0)

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, mock_store):
        """Test getting RAG statistics."""
        mock_store.count_documents.return_value = 25
        mock_store.count_chunks.return_value = 500
        mock_store.list_documents.return_value = [
            {"source": "local_file"},
            {"source": "local_file"},
            {"source": "web_page"}
        ]

        stats = await manager.get_stats()

        assert stats.total_documents == 25
        assert stats.total_chunks == 500
        assert stats.by_source["local_file"] == 2
        assert stats.by_source["web_page"] == 1

    def test_shutdown(self, manager):
        """Test shutdown doesn't raise errors."""
        manager.shutdown()  # Should not raise


class TestRAGManagerSourceTypeMapping:
    """Tests for source type mapping in RAGManager."""

    @pytest.fixture(autouse=True)
    def reset_instance(self):
        RAGManager.reset_instance()
        yield
        RAGManager.reset_instance()

    @pytest.fixture
    def manager(self):
        with patch("javis.rag.manager.get_embedding_service"), \
             patch("javis.rag.manager.DocumentStore"), \
             patch("javis.rag.manager.LoaderRegistry"):
            return RAGManager()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source_type,expected_source", [
        ("file", DocumentSource.LOCAL_FILE),
        ("local_file", DocumentSource.LOCAL_FILE),
        ("web", DocumentSource.WEB_PAGE),
        ("web_page", DocumentSource.WEB_PAGE),
        ("url", DocumentSource.WEB_PAGE),
        ("code", DocumentSource.CODEBASE),
        ("codebase", DocumentSource.CODEBASE),
    ])
    async def test_source_type_mapping(self, manager, source_type, expected_source):
        """Test all valid source type mappings."""
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=Document(
            id="doc-1",
            source=expected_source,
            source_path="/test",
            title="Test",
            content="Content"
        ))
        manager.text_chunker.chunk = MagicMock(return_value=[])  # Force no chunks

        with patch("javis.rag.manager.LoaderRegistry.get", return_value=mock_loader):
            # This will return None due to no chunks, but we're testing the mapping
            await manager.add_document(source_type=source_type, source_path="/test")

        mock_loader.load.assert_called_once()


class TestRAGManagerErrorHandling:
    """Tests for RAG error handling."""

    @pytest.fixture(autouse=True)
    def reset_instance(self):
        RAGManager.reset_instance()
        yield
        RAGManager.reset_instance()

    @pytest.mark.asyncio
    async def test_add_document_loader_error(self):
        """Test error handling when loader fails."""
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(side_effect=Exception("Load failed"))

        with patch("javis.rag.manager.get_embedding_service"), \
             patch("javis.rag.manager.DocumentStore"), \
             patch("javis.rag.manager.LoaderRegistry") as mock_registry:
            mock_registry.get.return_value = mock_loader

            manager = RAGManager()
            result = await manager.add_document(
                source_type="file",
                source_path="/nonexistent"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test error handling during search."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_single.side_effect = Exception("Embedding failed")

        with patch("javis.rag.manager.get_embedding_service", return_value=mock_embeddings), \
             patch("javis.rag.manager.DocumentStore"), \
             patch("javis.rag.manager.LoaderRegistry"):
            manager = RAGManager()
            manager.embeddings = mock_embeddings

            results = await manager.search("query")

            assert results == []
