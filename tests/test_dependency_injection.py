"""Tests for dependency injection components."""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from javis.core.container import (
    ServiceContainer,
    get_container,
    reset_container,
    bootstrap_services,
)
from javis.core.session import (
    Message,
    Session,
    InMemorySessionManager,
    get_session_manager,
    set_session_manager,
    reset_session_manager,
)


class TestServiceContainer:
    """Tests for ServiceContainer class."""

    def test_register_singleton(self):
        """Test registering a singleton service."""
        container = ServiceContainer()
        service = {"name": "test_service"}

        container.register_singleton("test", service)

        assert container.get("test") is service

    def test_register_factory(self):
        """Test registering a factory function."""
        container = ServiceContainer()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"created": call_count}

        container.register_factory("test", factory)

        # First call creates instance
        result1 = container.get("test")
        assert result1["created"] == 1
        assert call_count == 1

        # Second call returns cached instance
        result2 = container.get("test")
        assert result2 is result1
        assert call_count == 1  # Factory not called again

    def test_get_not_found(self):
        """Test getting a non-existent service."""
        container = ServiceContainer()

        with pytest.raises(KeyError) as exc_info:
            container.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_optional(self):
        """Test getting optional service."""
        container = ServiceContainer()
        container.register_singleton("exists", "value")

        assert container.get_optional("exists") == "value"
        assert container.get_optional("missing") is None

    def test_has(self):
        """Test checking if service exists."""
        container = ServiceContainer()
        container.register_singleton("singleton", "value")
        container.register_factory("factory", lambda: "value")

        assert container.has("singleton") is True
        assert container.has("factory") is True
        assert container.has("missing") is False

    def test_remove(self):
        """Test removing a service."""
        container = ServiceContainer()
        container.register_singleton("test", "value")

        container.remove("test")

        assert container.has("test") is False

    def test_reset(self):
        """Test resetting services while keeping factories."""
        container = ServiceContainer()
        container.register_singleton("singleton", "value")
        container.register_factory("factory", lambda: "new_value")

        # Access factory to create instance
        container.get("factory")

        container.reset()

        assert container.has("singleton") is False
        assert container.has("factory") is True  # Factory remains

    def test_reset_all(self):
        """Test fully resetting container."""
        container = ServiceContainer()
        container.register_singleton("singleton", "value")
        container.register_factory("factory", lambda: "value")

        container.reset_all()

        assert container.has("singleton") is False
        assert container.has("factory") is False

    def test_registered_services(self):
        """Test listing registered services."""
        container = ServiceContainer()
        container.register_singleton("a", "value")
        container.register_factory("b", lambda: "value")

        services = container.registered_services

        assert "a" in services
        assert "b" in services
        assert len(services) == 2

    def test_thread_safety(self):
        """Test thread-safe access."""
        container = ServiceContainer()
        results = []
        errors = []

        def register_and_get(name: str):
            try:
                container.register_singleton(name, f"value_{name}")
                result = container.get(name)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_and_get, args=(f"service_{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestGlobalContainer:
    """Tests for global container functions."""

    def setup_method(self):
        """Reset container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset container after each test."""
        reset_container()

    def test_get_container_singleton(self):
        """Test that get_container returns same instance."""
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_reset_container(self):
        """Test resetting global container."""
        container1 = get_container()
        container1.register_singleton("test", "value")

        reset_container()

        container2 = get_container()
        assert container2 is not container1
        assert container2.has("test") is False

    def test_bootstrap_services(self):
        """Test bootstrapping services with mock config."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(mock_config)

        assert container.has("config")
        assert container.get("config") is mock_config
        assert container.has("storage_factory")
        assert container.has("health_checker")


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert isinstance(msg.metadata, dict)

    def test_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role="assistant",
            content="Hi",
            metadata={"tokens": 10}
        )

        assert msg.metadata["tokens"] == 10


class TestSession:
    """Tests for Session dataclass."""

    def test_creation(self):
        """Test session creation."""
        session = Session(id="test-session")

        assert session.id == "test-session"
        assert session.messages == []
        assert session.message_count == 0
        assert session.created_at is not None
        assert session.updated_at is not None

    def test_message_count(self):
        """Test message count property."""
        session = Session(
            id="test",
            messages=[
                Message(role="system", content="Hello"),
                Message(role="user", content="Hi"),
            ]
        )

        assert session.message_count == 2


class TestInMemorySessionManager:
    """Tests for InMemorySessionManager."""

    @pytest.fixture
    def manager(self):
        """Create session manager."""
        return InMemorySessionManager(default_system_prompt="You are a test bot.")

    def test_get_or_create_new_session(self, manager):
        """Test creating a new session."""
        session = manager.get_or_create_session("new-session")

        assert session.id == "new-session"
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"
        assert "test bot" in session.messages[0].content

    def test_get_or_create_existing_session(self, manager):
        """Test getting existing session."""
        session1 = manager.get_or_create_session("existing")
        session1.messages.append(Message(role="user", content="Hello"))

        session2 = manager.get_or_create_session("existing")

        assert session2 is session1
        assert len(session2.messages) == 2

    def test_get_session_not_found(self, manager):
        """Test getting non-existent session."""
        result = manager.get_session("nonexistent")

        assert result is None

    def test_save_session(self, manager):
        """Test saving a session."""
        session = Session(
            id="save-test",
            messages=[Message(role="user", content="Test")]
        )

        manager.save_session(session)

        retrieved = manager.get_session("save-test")
        assert retrieved is not None
        assert retrieved.id == "save-test"

    def test_add_message(self, manager):
        """Test adding a message to session."""
        manager.get_or_create_session("msg-test")

        manager.add_message("msg-test", Message(role="user", content="Hello"))

        session = manager.get_session("msg-test")
        assert len(session.messages) == 2  # system + user

    def test_clear_session(self, manager):
        """Test clearing a session."""
        manager.get_or_create_session("clear-test")

        result = manager.clear_session("clear-test")

        assert result is True
        assert manager.get_session("clear-test") is None

    def test_clear_nonexistent_session(self, manager):
        """Test clearing non-existent session."""
        result = manager.clear_session("nonexistent")

        assert result is False

    def test_list_sessions(self, manager):
        """Test listing all sessions."""
        manager.get_or_create_session("session-1")
        manager.get_or_create_session("session-2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        session_ids = [s.id for s in sessions]
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    def test_cleanup_old_sessions(self, manager):
        """Test cleaning up old sessions."""
        # Create a session and make it "old" by directly modifying internal state
        session = manager.get_or_create_session("old-session")
        # Directly modify the session's updated_at (bypass save_session which updates timestamp)
        with manager._lock:
            manager._sessions["old-session"].updated_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Create a fresh session
        manager.get_or_create_session("new-session")

        # Cleanup sessions older than 1 hour
        cleaned = manager.cleanup_old_sessions(max_age_seconds=3600)

        assert cleaned == 1
        assert manager.get_session("old-session") is None
        assert manager.get_session("new-session") is not None

    def test_get_messages_for_api(self, manager):
        """Test getting messages in API format."""
        session = manager.get_or_create_session("api-test")
        manager.add_message("api-test", Message(role="user", content="Hello"))
        manager.add_message("api-test", Message(role="assistant", content="Hi"))

        messages = manager.get_messages_for_api("api-test")

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are a test bot."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi"}

    def test_get_messages_for_api_empty(self, manager):
        """Test getting messages for non-existent session."""
        messages = manager.get_messages_for_api("nonexistent")

        assert messages == []

    def test_thread_safety(self, manager):
        """Test thread-safe operations."""
        errors = []

        def create_and_modify(session_id: str):
            try:
                manager.get_or_create_session(session_id)
                for i in range(5):
                    manager.add_message(
                        session_id,
                        Message(role="user", content=f"Message {i}")
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_and_modify, args=(f"session_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        sessions = manager.list_sessions()
        assert len(sessions) == 5


class TestGlobalSessionManager:
    """Tests for global session manager functions."""

    def setup_method(self):
        """Reset session manager before each test."""
        reset_session_manager()

    def teardown_method(self):
        """Reset session manager after each test."""
        reset_session_manager()

    def test_get_session_manager_singleton(self):
        """Test that get_session_manager returns same instance."""
        with patch("javis.utils.config.get_config") as mock_config:
            mock_config.return_value.conversation.system_prompt = "Test"

            mgr1 = get_session_manager()
            mgr2 = get_session_manager()

            assert mgr1 is mgr2

    def test_set_session_manager(self):
        """Test setting custom session manager."""
        custom_manager = InMemorySessionManager(default_system_prompt="Custom")

        set_session_manager(custom_manager)

        assert get_session_manager() is custom_manager

    def test_reset_session_manager(self):
        """Test resetting session manager."""
        with patch("javis.utils.config.get_config") as mock_config:
            mock_config.return_value.conversation.system_prompt = "Test"

            mgr1 = get_session_manager()
            reset_session_manager()
            mgr2 = get_session_manager()

            assert mgr2 is not mgr1
