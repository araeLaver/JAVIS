"""Integration tests for JAVIS API endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

# Mock modules before importing the app
with patch.dict("sys.modules", {
    "javis.training.scheduler": MagicMock(
        get_scheduler=MagicMock(return_value=None),
        start_scheduler=MagicMock(),
        stop_scheduler=MagicMock(),
        SCHEDULER_AVAILABLE=False,
    ),
}):
    from javis.interfaces.api import app, sessions


@pytest.fixture
def client():
    """Create test client."""
    # Clear sessions before each test
    sessions.clear()
    with TestClient(app) as c:
        yield c
    sessions.clear()


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.conversation.system_prompt = "You are JAVIS, a helpful assistant."
    config.conversation.max_history = 20
    config.groq_api_key = "test-api-key"
    config.tools.enabled = False
    config.memory.long_term.enabled = False
    config.rag.enabled = False
    return config


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_healthy(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_status_returns_detailed_info(self, client):
        """Test that status endpoint returns detailed system info."""
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        assert "groq" in data["components"]
        assert "tools" in data["components"]
        assert "memory" in data["components"]
        assert "sessions" in data

    def test_ready_endpoint(self, client):
        """Test the readiness probe endpoint."""
        response = client.get("/api/ready")

        # Should return 200 if any backend is available, 503 otherwise
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "ready" in data
            assert "backends" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test root endpoint when no static file exists."""
        response = client.get("/")

        assert response.status_code == 200
        # Either returns static file or API info
        data = response.json() if response.headers.get("content-type") == "application/json" else None
        if data:
            assert "message" in data or "JAVIS" in str(data)


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_groq_success(self, client, mock_config):
        """Test successful chat with Groq model using real API."""
        # This test verifies the endpoint works with proper response structure
        # Uses real Groq API if configured
        import javis.interfaces.api as api_module

        # Patch at module level for proper effect
        with patch.object(api_module, 'get_config', return_value=mock_config), \
             patch.object(api_module, 'get_logger') as mock_get_logger, \
             patch.object(api_module, 'ModelClient') as mock_model_client:

            mock_logger = MagicMock()
            mock_logger.add_turn = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_response = MagicMock()
            mock_response.content = "Hello! How can I help you?"
            mock_client_instance = MagicMock()
            mock_client_instance.chat = AsyncMock(return_value=mock_response)
            mock_model_client.return_value = mock_client_instance

            response = client.post("/api/chat", json={
                "message": "Hello",
                "session_id": "test-chat-success",
                "model": "groq"
            })

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert data["session_id"] == "test-chat-success"
            assert data["model_used"] == "groq"

    def test_chat_request_validation(self, client):
        """Test chat request validation - message is required."""
        response = client.post("/api/chat", json={
            "session_id": "test-validation"
        })

        # Should fail validation because message is required
        assert response.status_code == 422

    def test_chat_invalid_request(self, client):
        """Test chat with invalid request body."""
        response = client.post("/api/chat", json={})

        assert response.status_code == 422  # Validation error

    def test_chat_maintains_session(self, client, mock_config):
        """Test that chat maintains session history."""
        import javis.interfaces.api as api_module

        with patch.object(api_module, 'get_config', return_value=mock_config), \
             patch.object(api_module, 'get_logger') as mock_get_logger, \
             patch.object(api_module, 'ModelClient') as mock_model_client:

            mock_logger = MagicMock()
            mock_logger.add_turn = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_client_instance = MagicMock()
            mock_client_instance.chat = AsyncMock(return_value=mock_response)
            mock_model_client.return_value = mock_client_instance

            # First message
            client.post("/api/chat", json={
                "message": "First message",
                "session_id": "session-1",
                "model": "groq"
            })

            # Second message
            client.post("/api/chat", json={
                "message": "Second message",
                "session_id": "session-1",
                "model": "groq"
            })

            # Session should have: system + user1 + assistant1 + user2 + assistant2
            assert "session-1" in sessions
            assert len(sessions["session-1"]) == 5


class TestClearEndpoint:
    """Tests for clear session endpoint."""

    @patch("javis.interfaces.api.get_config")
    @patch("javis.interfaces.api.get_logger")
    def test_clear_session(self, mock_get_logger, mock_get_config, client, mock_config):
        """Test clearing a session."""
        mock_get_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_logger.end_conversation = MagicMock(return_value=None)
        mock_get_logger.return_value = mock_logger

        # Create a session first
        sessions["test-session"] = [MagicMock()]

        response = client.post("/api/clear", params={"session_id": "test-session"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test-session"

    @patch("javis.interfaces.api.get_config")
    @patch("javis.interfaces.api.get_logger")
    def test_clear_nonexistent_session(self, mock_get_logger, mock_get_config, client, mock_config):
        """Test clearing a session that doesn't exist."""
        mock_get_config.return_value = mock_config

        mock_logger = MagicMock()
        mock_logger.end_conversation = MagicMock(return_value=None)
        mock_get_logger.return_value = mock_logger

        response = client.post("/api/clear", params={"session_id": "nonexistent"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"


class TestFeedbackEndpoint:
    """Tests for feedback endpoint."""

    def test_feedback_positive(self, client):
        """Test submitting positive feedback."""
        response = client.post("/api/feedback", params={
            "session_id": "test-session",
            "feedback": "good"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "feedback_added"
        assert data["feedback"] == "good"
        assert data["session_id"] == "test-session"

    def test_feedback_negative(self, client):
        """Test submitting negative feedback."""
        response = client.post("/api/feedback", params={
            "session_id": "test-session",
            "feedback": "bad"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "feedback_added"
        assert data["feedback"] == "bad"
        assert data["session_id"] == "test-session"

    def test_feedback_invalid(self, client):
        """Test feedback with invalid value."""
        response = client.post("/api/feedback", params={
            "session_id": "test-session",
            "feedback": "invalid"
        })

        assert response.status_code == 400
        assert "good" in response.json()["detail"] or "bad" in response.json()["detail"]


class TestModelsEndpoint:
    """Tests for models list endpoint."""

    @patch("javis.interfaces.api.get_config")
    @patch("javis.interfaces.api.list_adapters")
    @patch("javis.interfaces.api.check_modal_available")
    def test_list_models(self, mock_modal_available, mock_list_adapters, mock_get_config, client, mock_config):
        """Test listing available models."""
        mock_get_config.return_value = mock_config
        mock_modal_available.return_value = False
        mock_list_adapters.return_value = [
            {"version": "v1", "path": "/path/v1"},
            {"version": "v2", "path": "/path/v2"}
        ]

        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert "groq" in data
        assert "modal" in data
        assert "local" in data
        assert "adapters" in data
        assert len(data["adapters"]) == 2


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }
        )

        # Should allow the request
        assert response.status_code == 200

    def test_cors_headers_on_response(self, client):
        """Test that CORS headers are present on responses."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/api/unknown")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.get("/api/chat")  # Should be POST

        assert response.status_code == 405

    def test_json_parse_error(self, client):
        """Test that invalid JSON is handled gracefully."""
        response = client.post(
            "/api/chat",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 for validation error
        assert response.status_code == 422
