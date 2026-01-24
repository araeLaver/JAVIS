"""Session management for JAVIS.

Provides abstraction for session storage with support for:
- In-memory sessions (default)
- Pluggable storage backends
- Session lifecycle management
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Chat session."""

    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)


class SessionManager(ABC):
    """Abstract base class for session management.

    Implementations should handle:
    - Session creation and retrieval
    - Message persistence
    - Session cleanup
    """

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        pass

    @abstractmethod
    def get_or_create_session(
        self, session_id: str, system_prompt: Optional[str] = None
    ) -> Session:
        """Get or create a session.

        Args:
            session_id: Session identifier
            system_prompt: Optional system prompt for new sessions

        Returns:
            Existing or newly created session
        """
        pass

    @abstractmethod
    def save_session(self, session: Session) -> None:
        """Save a session.

        Args:
            session: Session to save
        """
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session.

        Args:
            session_id: Session identifier
            message: Message to add
        """
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> bool:
        """Clear a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared, False if not found
        """
        pass

    @abstractmethod
    def list_sessions(self) -> list[Session]:
        """List all sessions.

        Returns:
            List of all sessions
        """
        pass

    @abstractmethod
    def cleanup_old_sessions(self, max_age_seconds: int) -> int:
        """Clean up old sessions.

        Args:
            max_age_seconds: Maximum session age in seconds

        Returns:
            Number of sessions cleaned up
        """
        pass


class InMemorySessionManager(SessionManager):
    """In-memory session manager.

    Thread-safe implementation that stores sessions in memory.
    Suitable for development and single-instance deployments.
    """

    def __init__(self, default_system_prompt: Optional[str] = None):
        """Initialize session manager.

        Args:
            default_system_prompt: Default system prompt for new sessions
        """
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._default_system_prompt = default_system_prompt or "You are a helpful assistant."

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create_session(
        self, session_id: str, system_prompt: Optional[str] = None
    ) -> Session:
        """Get or create a session."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            # Create new session with system message
            prompt = system_prompt or self._default_system_prompt
            session = Session(
                id=session_id,
                messages=[Message(role="system", content=prompt)],
            )
            self._sessions[session_id] = session
            logger.debug(f"Created new session: {session_id}")
            return session

    def save_session(self, session: Session) -> None:
        """Save a session."""
        with self._lock:
            session.updated_at = datetime.now(timezone.utc)
            self._sessions[session.id] = session

    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append(message)
                session.updated_at = datetime.now(timezone.utc)

    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Cleared session: {session_id}")
                return True
            return False

    def list_sessions(self) -> list[Session]:
        """List all sessions."""
        with self._lock:
            return list(self._sessions.values())

    def cleanup_old_sessions(self, max_age_seconds: int) -> int:
        """Clean up old sessions."""
        now = datetime.now(timezone.utc)
        cleaned = 0

        with self._lock:
            to_remove = []
            for session_id, session in self._sessions.items():
                age = (now - session.updated_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(session_id)

            for session_id in to_remove:
                del self._sessions[session_id]
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old sessions")
        return cleaned

    def get_messages_for_api(self, session_id: str) -> list[dict[str, str]]:
        """Get messages in API format (for model clients).

        Args:
            session_id: Session identifier

        Returns:
            List of message dicts with 'role' and 'content'
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return [{"role": m.role, "content": m.content} for m in session.messages]


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance.

    Returns:
        The global SessionManager
    """
    global _session_manager
    if _session_manager is None:
        with _session_manager_lock:
            if _session_manager is None:
                from javis.utils.config import get_config

                config = get_config()
                _session_manager = InMemorySessionManager(
                    default_system_prompt=config.conversation.system_prompt
                )
    return _session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager (for testing or custom implementations).

    Args:
        manager: Session manager to use
    """
    global _session_manager
    with _session_manager_lock:
        _session_manager = manager


def reset_session_manager() -> None:
    """Reset the global session manager (for testing)."""
    global _session_manager
    with _session_manager_lock:
        _session_manager = None
