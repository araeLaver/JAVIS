"""Unit tests for JAVIS memory system."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from javis.memory.models import MemoryEntry, MemorySearchResult, ConversationSummary
from javis.memory.manager import MemoryManager


class TestMemoryEntry:
    """Tests for MemoryEntry model."""

    def test_create_memory_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test-123",
            session_id="session-456",
            summary="Test conversation summary",
            topics=["coding", "python"],
            turn_count=5,
            created_at=datetime.now()
        )

        assert entry.id == "test-123"
        assert entry.session_id == "session-456"
        assert "coding" in entry.topics
        assert entry.turn_count == 5

    def test_memory_entry_with_metadata(self):
        """Test memory entry with metadata."""
        entry = MemoryEntry(
            id="test-123",
            session_id="session-456",
            summary="Test",
            topics=[],
            turn_count=2,
            created_at=datetime.now(),
            metadata={"key_entities": ["Python", "API"], "sentiment": "positive"}
        )

        assert entry.metadata["sentiment"] == "positive"
        assert "Python" in entry.metadata["key_entities"]


class TestMemorySearchResult:
    """Tests for MemorySearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        entry = MemoryEntry(
            id="test-123",
            session_id="session-456",
            summary="Test",
            topics=[],
            turn_count=2,
            created_at=datetime.now()
        )

        result = MemorySearchResult(
            entry=entry,
            score=0.85,
            distance=0.15
        )

        assert result.entry.id == "test-123"
        assert result.score == 0.85
        assert result.distance == 0.15


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture(autouse=True)
    def reset_instance(self):
        """Reset singleton instance before each test."""
        MemoryManager.reset_instance()
        yield
        MemoryManager.reset_instance()

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding service."""
        mock = MagicMock()
        mock.embed_single.return_value = [0.1] * 384
        mock.embed.return_value = [[0.1] * 384]
        return mock

    @pytest.fixture
    def mock_store(self):
        """Create mock vector store."""
        mock = MagicMock()
        mock.count.return_value = 0
        mock.add = MagicMock()
        mock.delete = MagicMock()
        mock.search.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }
        return mock

    @pytest.fixture
    def mock_summarizer(self):
        """Create mock conversation summarizer."""
        mock = MagicMock()
        mock.summarize = AsyncMock(return_value=ConversationSummary(
            summary="Test summary of conversation",
            topics=["topic1", "topic2"],
            key_entities=["entity1"],
            sentiment="neutral"
        ))
        return mock

    @pytest.fixture
    def manager(self, mock_embeddings, mock_store, mock_summarizer):
        """Create memory manager with mocked dependencies."""
        with patch("javis.memory.manager.get_embedding_service", return_value=mock_embeddings), \
             patch("javis.memory.manager.VectorStore", return_value=mock_store), \
             patch("javis.memory.manager.ConversationSummarizer", return_value=mock_summarizer):
            manager = MemoryManager(
                db_path="./test_db",
                collection_name="test_memory"
            )
            manager.embeddings = mock_embeddings
            manager.store = mock_store
            manager.summarizer = mock_summarizer
            return manager

    def test_singleton_pattern(self, manager):
        """Test that MemoryManager is a singleton."""
        with patch("javis.memory.manager.get_embedding_service"), \
             patch("javis.memory.manager.VectorStore"), \
             patch("javis.memory.manager.ConversationSummarizer"):
            instance1 = MemoryManager.get_instance()
            instance2 = MemoryManager.get_instance()
            assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_store_conversation_success(self, manager, mock_store, mock_summarizer):
        """Test storing a conversation."""
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = await manager.store_conversation(
            session_id="test-session",
            turns=turns,
            metadata={"source": "test"}
        )

        assert result is not None
        assert result.session_id == "test-session"
        assert result.summary == "Test summary of conversation"
        mock_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_conversation_short_conversation(self, manager):
        """Test that short conversations are skipped."""
        turns = [{"role": "user", "content": "Hi"}]  # Only 1 turn

        result = await manager.store_conversation(
            session_id="test-session",
            turns=turns
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_store_conversation_empty(self, manager):
        """Test that empty conversations are skipped."""
        result = await manager.store_conversation(
            session_id="test-session",
            turns=[]
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_search_memories_success(self, manager, mock_store, mock_embeddings):
        """Test searching memories."""
        mock_store.search.return_value = {
            "ids": ["mem-1", "mem-2"],
            "distances": [0.1, 0.3],
            "documents": ["Summary 1", "Summary 2"],
            "metadatas": [
                {
                    "session_id": "sess-1",
                    "topics": "coding,python",
                    "turn_count": 5,
                    "created_at": datetime.now().isoformat()
                },
                {
                    "session_id": "sess-2",
                    "topics": "design",
                    "turn_count": 3,
                    "created_at": datetime.now().isoformat()
                }
            ]
        }

        results = await manager.search_memories("How to code?", limit=5)

        assert len(results) == 2
        assert results[0].entry.id == "mem-1"
        assert results[0].score == 0.9  # 1 - 0.1
        mock_embeddings.embed_single.assert_called_with("How to code?")

    @pytest.mark.asyncio
    async def test_search_memories_filters_low_scores(self, manager, mock_store):
        """Test that low score results are filtered."""
        mock_store.search.return_value = {
            "ids": ["mem-1"],
            "distances": [0.9],  # score = 0.1, below default min_score=0.3
            "documents": ["Low relevance"],
            "metadatas": [{
                "session_id": "sess-1",
                "topics": "",
                "turn_count": 2,
                "created_at": datetime.now().isoformat()
            }]
        }

        results = await manager.search_memories("query", min_score=0.3)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_memories_empty_results(self, manager, mock_store):
        """Test search with no results."""
        mock_store.search.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }

        results = await manager.search_memories("nonexistent query")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, manager, mock_store):
        """Test getting formatted context."""
        mock_store.search.return_value = {
            "ids": ["mem-1"],
            "distances": [0.2],
            "documents": ["Previous discussion about Python"],
            "metadatas": [{
                "session_id": "sess-1",
                "topics": "python",
                "turn_count": 4,
                "created_at": datetime.now().isoformat()
            }]
        }

        context = await manager.get_relevant_context("Python question")

        assert "관련 과거 대화" in context
        assert "Previous discussion about Python" in context

    @pytest.mark.asyncio
    async def test_get_relevant_context_empty(self, manager, mock_store):
        """Test getting context when no memories exist."""
        mock_store.search.return_value = {
            "ids": [],
            "distances": [],
            "documents": [],
            "metadatas": []
        }

        context = await manager.get_relevant_context("query")

        assert context == ""

    def test_format_for_prompt(self, manager):
        """Test formatting search results for prompt."""
        entry = MemoryEntry(
            id="mem-1",
            session_id="sess-1",
            summary="Discussion about AI",
            topics=["AI", "ML", "deep learning"],
            turn_count=5,
            created_at=datetime(2024, 1, 15)
        )
        results = [MemorySearchResult(entry=entry, score=0.85, distance=0.15)]

        formatted = manager.format_for_prompt(results)

        assert "관련 과거 대화" in formatted
        assert "2024-01-15" in formatted
        assert "AI, ML, deep learning" in formatted
        assert "Discussion about AI" in formatted

    def test_format_for_prompt_empty(self, manager):
        """Test formatting empty results."""
        formatted = manager.format_for_prompt([])
        assert formatted == ""

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, manager, mock_store):
        """Test deleting a memory."""
        result = await manager.delete_memory("mem-123")

        assert result is True
        mock_store.delete.assert_called_once_with(["mem-123"])

    @pytest.mark.asyncio
    async def test_delete_memory_failure(self, manager, mock_store):
        """Test delete failure handling."""
        mock_store.delete.side_effect = Exception("Delete failed")

        result = await manager.delete_memory("mem-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, mock_store):
        """Test getting memory statistics."""
        mock_store.count.return_value = 42

        stats = await manager.get_stats()

        assert stats["count"] == 42
        assert stats["db_path"] == "./test_db"
        assert stats["collection"] == "test_memory"

    def test_shutdown(self, manager):
        """Test shutdown doesn't raise errors."""
        manager.shutdown()  # Should not raise


class TestMemoryManagerEdgeCases:
    """Edge case tests for MemoryManager."""

    @pytest.fixture(autouse=True)
    def reset_instance(self):
        """Reset singleton before each test."""
        MemoryManager.reset_instance()
        yield
        MemoryManager.reset_instance()

    @pytest.mark.asyncio
    async def test_store_conversation_error_handling(self):
        """Test error handling during conversation storage."""
        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(
            side_effect=Exception("Summarization failed")
        )

        with patch("javis.memory.manager.get_embedding_service"), \
             patch("javis.memory.manager.VectorStore"), \
             patch("javis.memory.manager.ConversationSummarizer", return_value=mock_summarizer):
            manager = MemoryManager()
            manager.summarizer = mock_summarizer

            result = await manager.store_conversation(
                session_id="test",
                turns=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"}
                ]
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_search_memories_error_handling(self):
        """Test error handling during memory search."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_single.side_effect = Exception("Embedding failed")

        with patch("javis.memory.manager.get_embedding_service", return_value=mock_embeddings), \
             patch("javis.memory.manager.VectorStore"), \
             patch("javis.memory.manager.ConversationSummarizer"):
            manager = MemoryManager()
            manager.embeddings = mock_embeddings

            results = await manager.search_memories("query")

            assert results == []


class TestConversationSummarizer:
    """Tests for ConversationSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Create summarizer instance."""
        from javis.memory.summarizer import ConversationSummarizer
        return ConversationSummarizer(api_key="test-key")

    @pytest.mark.asyncio
    async def test_summarize_empty_turns(self, summarizer):
        """Test summarizing empty conversation."""
        result = await summarizer.summarize([])

        assert result.summary == "빈 대화"
        assert result.topics == []
        assert result.key_entities == []

    def test_format_conversation_basic(self, summarizer):
        """Test formatting conversation turns."""
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = summarizer._format_conversation(turns)

        assert "사용자: Hello" in result
        assert "AI: Hi there!" in result

    def test_format_conversation_skips_system(self, summarizer):
        """Test that system messages are skipped."""
        turns = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        result = summarizer._format_conversation(turns)

        assert "helpful assistant" not in result
        assert "사용자: Hello" in result

    def test_format_conversation_handles_missing_content(self, summarizer):
        """Test handling turns with missing content."""
        turns = [
            {"role": "user"},  # No content
            {"role": "assistant", "content": "Response"}
        ]

        result = summarizer._format_conversation(turns)

        assert "사용자:" in result
        assert "AI: Response" in result

    def test_parse_summary_valid_json(self, summarizer):
        """Test parsing valid JSON summary."""
        json_str = '{"summary": "Test summary", "topics": ["coding"], "key_entities": ["Python"], "sentiment": "positive"}'

        result = summarizer._parse_summary(json_str)

        assert result.summary == "Test summary"
        assert result.topics == ["coding"]
        assert result.key_entities == ["Python"]
        assert result.sentiment == "positive"

    def test_parse_summary_with_code_blocks(self, summarizer):
        """Test parsing JSON wrapped in code blocks."""
        json_str = '```json\n{"summary": "Test", "topics": [], "key_entities": []}\n```'

        result = summarizer._parse_summary(json_str)

        assert result.summary == "Test"

    def test_parse_summary_invalid_json(self, summarizer):
        """Test fallback when JSON is invalid."""
        json_str = "This is not valid JSON"

        result = summarizer._parse_summary(json_str)

        assert result.summary == "This is not valid JSON"
        assert result.topics == []
        assert result.key_entities == []

    def test_fallback_summary_basic(self, summarizer):
        """Test fallback summary generation."""
        turns = [
            {"role": "user", "content": "How do I code in Python?"},
            {"role": "assistant", "content": "Here's how..."},
            {"role": "user", "content": "What about JavaScript?"}
        ]

        result = summarizer._fallback_summary(turns)

        assert "2개의 질문/요청" in result.summary
        assert "How do I code in Python?" in result.summary

    def test_fallback_summary_single_message(self, summarizer):
        """Test fallback with single user message."""
        turns = [
            {"role": "user", "content": "Hello"}
        ]

        result = summarizer._fallback_summary(turns)

        assert "1개의 질문/요청" in result.summary
        assert "Hello" in result.summary

    def test_fallback_summary_no_user_messages(self, summarizer):
        """Test fallback with no user messages."""
        turns = [
            {"role": "assistant", "content": "Hello!"}
        ]

        result = summarizer._fallback_summary(turns)

        assert "0개의 질문/요청" in result.summary

    @pytest.mark.asyncio
    async def test_summarize_llm_failure_uses_fallback(self, summarizer):
        """Test that LLM failure uses fallback summary."""
        turns = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response"}
        ]

        with patch.object(summarizer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")

            result = await summarizer.summarize(turns)

            assert "1개의 질문/요청" in result.summary

    @pytest.mark.asyncio
    async def test_call_llm_no_api_key(self):
        """Test that missing API key raises error."""
        from javis.memory.summarizer import ConversationSummarizer
        summarizer = ConversationSummarizer(api_key=None)

        # Ensure env var is not set for this test
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=True):
            summarizer.api_key = None
            with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
                await summarizer._call_llm("test conversation")


class TestConversationSummaryModel:
    """Tests for ConversationSummary model."""

    def test_create_summary(self):
        """Test creating a conversation summary."""
        summary = ConversationSummary(
            summary="Test summary",
            topics=["topic1", "topic2"],
            key_entities=["entity1"]
        )

        assert summary.summary == "Test summary"
        assert len(summary.topics) == 2
        assert summary.sentiment is None

    def test_summary_with_sentiment(self):
        """Test summary with sentiment."""
        summary = ConversationSummary(
            summary="Happy conversation",
            topics=[],
            key_entities=[],
            sentiment="positive"
        )

        assert summary.sentiment == "positive"
