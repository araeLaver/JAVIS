"""Tests for javis.data.conversation_logger module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from javis.data.conversation_logger import (
    ConversationTurn,
    ConversationLog,
    ConversationLogger,
    get_logger,
)
from javis.storage.base import ConversationRecord, ConversationTurn as StorageConversationTurn


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            role="user",
            content="Hello",
            timestamp="2024-01-01T10:00:00"
        )
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.timestamp == "2024-01-01T10:00:00"

    def test_create_assistant_turn(self):
        """Test creating an assistant turn."""
        turn = ConversationTurn(
            role="assistant",
            content="Hi there!",
            timestamp="2024-01-01T10:00:01"
        )
        assert turn.role == "assistant"


class TestConversationLog:
    """Tests for ConversationLog model."""

    def test_create_minimal_log(self):
        """Test creating minimal conversation log."""
        log = ConversationLog(
            session_id="test-session",
            started_at="2024-01-01T10:00:00"
        )
        assert log.session_id == "test-session"
        assert log.turns == []
        assert log.feedback is None
        assert log.tags == []

    def test_create_full_log(self):
        """Test creating full conversation log."""
        turns = [
            ConversationTurn(role="user", content="Hi", timestamp="2024-01-01T10:00:00"),
            ConversationTurn(role="assistant", content="Hello!", timestamp="2024-01-01T10:00:01"),
        ]
        log = ConversationLog(
            session_id="session-123",
            started_at="2024-01-01T10:00:00",
            turns=turns,
            feedback="good",
            tags=["support", "greeting"]
        )
        assert len(log.turns) == 2
        assert log.feedback == "good"
        assert "support" in log.tags

    def test_to_storage_record(self):
        """Test conversion to storage record."""
        turns = [
            ConversationTurn(role="user", content="Question", timestamp="2024-01-01T10:00:00"),
            ConversationTurn(role="assistant", content="Answer", timestamp="2024-01-01T10:00:01"),
        ]
        log = ConversationLog(
            session_id="session-456",
            started_at="2024-01-01T10:00:00",
            turns=turns,
            feedback="good",
            tags=["help"]
        )

        record = log.to_storage_record()

        assert isinstance(record, ConversationRecord)
        assert record.session_id == "session-456"
        assert len(record.turns) == 2
        assert record.turns[0].role == "user"
        assert record.turns[0].content == "Question"
        assert record.feedback == "good"
        assert "help" in record.tags

    def test_from_storage_record(self):
        """Test creation from storage record."""
        storage_turns = [
            StorageConversationTurn(role="user", content="Test", timestamp="2024-01-01T10:00:00"),
            StorageConversationTurn(role="assistant", content="Response", timestamp="2024-01-01T10:00:01"),
        ]
        record = ConversationRecord(
            session_id="record-session",
            started_at="2024-01-01T09:59:00",
            turns=storage_turns,
            feedback="bad",
            tags=["bug"]
        )

        log = ConversationLog.from_storage_record(record)

        assert log.session_id == "record-session"
        assert log.started_at == "2024-01-01T09:59:00"
        assert len(log.turns) == 2
        assert log.turns[0].role == "user"
        assert log.turns[0].content == "Test"
        assert log.feedback == "bad"
        assert "bug" in log.tags


class TestConversationLogger:
    """Tests for ConversationLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_store(self):
        """Create mock conversation store."""
        store = MagicMock()
        store.save.return_value = Path("/tmp/test.json")
        store.export_for_training.return_value = Path("/tmp/export.jsonl")
        return store

    def test_init_with_data_dir(self, temp_dir):
        """Test initialization with custom data directory."""
        logger = ConversationLogger(data_dir=temp_dir)
        assert logger.data_dir == temp_dir
        assert logger.data_dir.exists()

    def test_init_default_data_dir(self):
        """Test initialization with default data directory."""
        logger = ConversationLogger()
        assert logger.data_dir.exists()
        assert "conversations" in str(logger.data_dir)

    def test_init_with_store(self, mock_store):
        """Test initialization with custom store."""
        logger = ConversationLogger(store=mock_store)
        assert logger._store is mock_store

    def test_store_property_lazy_init(self, temp_dir):
        """Test lazy initialization of store property."""
        logger = ConversationLogger(data_dir=temp_dir)
        assert logger._store is None

        with patch("javis.data.conversation_logger.get_conversation_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            store = logger.store
            assert store is mock_store
            mock_get_store.assert_called_once()

    def test_store_property_returns_cached(self, mock_store):
        """Test store property returns cached instance."""
        logger = ConversationLogger(store=mock_store)

        # Access store property multiple times
        store1 = logger.store
        store2 = logger.store

        assert store1 is store2
        assert store1 is mock_store

    def test_start_conversation(self, temp_dir):
        """Test starting a new conversation."""
        logger = ConversationLogger(data_dir=temp_dir)
        conv = logger.start_conversation("session-001")

        assert conv.session_id == "session-001"
        assert "session-001" in logger.active_conversations
        assert len(conv.turns) == 0

    def test_add_turn_to_existing_conversation(self, temp_dir):
        """Test adding turn to existing conversation."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("session-002")

        logger.add_turn("session-002", "user", "Hello!")

        conv = logger.active_conversations["session-002"]
        assert len(conv.turns) == 1
        assert conv.turns[0].role == "user"
        assert conv.turns[0].content == "Hello!"

    def test_add_turn_auto_starts_conversation(self, temp_dir):
        """Test adding turn auto-starts conversation if not exists."""
        logger = ConversationLogger(data_dir=temp_dir)

        logger.add_turn("new-session", "user", "Hi")

        assert "new-session" in logger.active_conversations
        assert len(logger.active_conversations["new-session"].turns) == 1

    def test_add_multiple_turns(self, temp_dir):
        """Test adding multiple turns to conversation."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("multi-turn")

        logger.add_turn("multi-turn", "user", "Question 1")
        logger.add_turn("multi-turn", "assistant", "Answer 1")
        logger.add_turn("multi-turn", "user", "Question 2")
        logger.add_turn("multi-turn", "assistant", "Answer 2")

        conv = logger.active_conversations["multi-turn"]
        assert len(conv.turns) == 4
        assert conv.turns[0].role == "user"
        assert conv.turns[1].role == "assistant"

    def test_add_feedback(self, temp_dir):
        """Test adding feedback to conversation."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("feedback-session")

        logger.add_feedback("feedback-session", "good")

        conv = logger.active_conversations["feedback-session"]
        assert conv.feedback == "good"

    def test_add_feedback_bad(self, temp_dir):
        """Test adding negative feedback."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("bad-session")

        logger.add_feedback("bad-session", "bad")

        assert logger.active_conversations["bad-session"].feedback == "bad"

    def test_add_feedback_nonexistent_session(self, temp_dir):
        """Test adding feedback to non-existent session does nothing."""
        logger = ConversationLogger(data_dir=temp_dir)

        logger.add_feedback("nonexistent", "good")

        assert "nonexistent" not in logger.active_conversations

    def test_add_tags(self, temp_dir):
        """Test adding tags to conversation."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("tag-session")

        logger.add_tags("tag-session", ["support", "bug"])

        conv = logger.active_conversations["tag-session"]
        assert "support" in conv.tags
        assert "bug" in conv.tags

    def test_add_tags_multiple_times(self, temp_dir):
        """Test adding tags multiple times extends list."""
        logger = ConversationLogger(data_dir=temp_dir)
        logger.start_conversation("multi-tag")

        logger.add_tags("multi-tag", ["tag1"])
        logger.add_tags("multi-tag", ["tag2", "tag3"])

        conv = logger.active_conversations["multi-tag"]
        assert len(conv.tags) == 3

    def test_add_tags_nonexistent_session(self, temp_dir):
        """Test adding tags to non-existent session does nothing."""
        logger = ConversationLogger(data_dir=temp_dir)

        logger.add_tags("nonexistent", ["tag"])

        assert "nonexistent" not in logger.active_conversations

    def test_save_conversation(self, mock_store, temp_dir):
        """Test saving conversation."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)
        logger.start_conversation("save-session")
        logger.add_turn("save-session", "user", "Test message")

        result = logger.save_conversation("save-session")

        mock_store.save.assert_called_once()
        assert result == Path("/tmp/test.json")

    def test_save_conversation_nonexistent(self, mock_store, temp_dir):
        """Test saving non-existent conversation returns None."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)

        result = logger.save_conversation("nonexistent")

        assert result is None
        mock_store.save.assert_not_called()

    def test_end_conversation(self, mock_store, temp_dir):
        """Test ending conversation saves and removes from active."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)
        logger.start_conversation("end-session")
        logger.add_turn("end-session", "user", "Message")

        result = logger.end_conversation("end-session")

        assert "end-session" not in logger.active_conversations
        mock_store.save.assert_called_once()

    def test_end_conversation_nonexistent(self, mock_store, temp_dir):
        """Test ending non-existent conversation."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)

        result = logger.end_conversation("nonexistent")

        assert result is None

    def test_export_for_training_default_path(self, mock_store, temp_dir):
        """Test export with default output path."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)

        result = logger.export_for_training()

        mock_store.export_for_training.assert_called_once()
        call_kwargs = mock_store.export_for_training.call_args.kwargs
        assert "output_path" in call_kwargs
        assert "training" in str(call_kwargs["output_path"])
        assert call_kwargs["min_turns"] == 2

    def test_export_for_training_custom_path(self, mock_store, temp_dir):
        """Test export with custom output path."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)
        output_path = temp_dir / "custom_export.jsonl"

        result = logger.export_for_training(output_path=output_path)

        call_kwargs = mock_store.export_for_training.call_args.kwargs
        assert call_kwargs["output_path"] == output_path

    def test_export_for_training_with_filter(self, mock_store, temp_dir):
        """Test export with feedback filter."""
        logger = ConversationLogger(data_dir=temp_dir, store=mock_store)

        result = logger.export_for_training(feedback_filter="good")

        call_kwargs = mock_store.export_for_training.call_args.kwargs
        assert call_kwargs["feedback_filter"] == "good"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_instance(self):
        """Test get_logger returns ConversationLogger instance."""
        # Reset global logger
        import javis.data.conversation_logger as module
        module._logger = None

        logger = get_logger()

        assert isinstance(logger, ConversationLogger)

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance (singleton)."""
        # Reset global logger
        import javis.data.conversation_logger as module
        module._logger = None

        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2

    def test_get_logger_creates_on_first_call(self):
        """Test get_logger creates new logger on first call."""
        import javis.data.conversation_logger as module
        module._logger = None

        assert module._logger is None

        logger = get_logger()

        assert module._logger is not None
        assert module._logger is logger
