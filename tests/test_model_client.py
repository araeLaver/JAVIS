"""Tests for javis.models.client module."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from javis.models.client import (
    Message,
    ChatRequest,
    ChatResponse,
    ModelClient,
)


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test different message roles."""
        system = Message(role="system", content="You are helpful")
        user = Message(role="user", content="Hi")
        assistant = Message(role="assistant", content="Hello!")

        assert system.role == "system"
        assert user.role == "user"
        assert assistant.role == "assistant"


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_create_minimal_request(self):
        """Test creating request with minimal fields."""
        messages = [Message(role="user", content="Hello")]
        request = ChatRequest(messages=messages)

        assert len(request.messages) == 1
        assert request.max_tokens == 2048
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop is None

    def test_create_full_request(self):
        """Test creating request with all fields."""
        messages = [Message(role="user", content="Hello")]
        request = ChatRequest(
            messages=messages,
            max_tokens=1000,
            temperature=0.5,
            top_p=0.8,
            stop=[".", "\n"],
        )

        assert request.max_tokens == 1000
        assert request.temperature == 0.5
        assert request.top_p == 0.8
        assert request.stop == [".", "\n"]


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_create_minimal_response(self):
        """Test creating response with minimal fields."""
        response = ChatResponse(content="Hello!")
        assert response.content == "Hello!"
        assert response.finish_reason is None
        assert response.usage is None

    def test_create_full_response(self):
        """Test creating response with all fields."""
        response = ChatResponse(
            content="Hello!",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        assert response.content == "Hello!"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15


class TestModelClient:
    """Tests for ModelClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.groq_api_key = "test-groq-key"
        config.model.generation.max_tokens = 2048
        config.model.generation.temperature = 0.7
        config.model.generation.top_p = 0.9
        config.model.generation.stop_sequences = None
        return config

    def test_init_success(self, mock_config):
        """Test successful initialization."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()
            assert client.api_key == "test-groq-key"
            assert client.base_url == "https://api.groq.com/openai/v1"
            assert client.model == "llama-3.1-8b-instant"

    def test_init_no_api_key(self):
        """Test initialization without API key raises error."""
        mock_config = MagicMock()
        mock_config.groq_api_key = None

        with patch("javis.models.client.get_config", return_value=mock_config):
            with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
                ModelClient()

    @pytest.mark.asyncio
    async def test_chat_success(self, mock_config):
        """Test successful chat completion."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = AsyncMock()
                mock_http_client.__aenter__.return_value = mock_http_client
                mock_http_client.post.return_value = mock_response
                mock_client_class.return_value = mock_http_client

                messages = [Message(role="user", content="Hi")]
                response = await client.chat(messages)

                assert response.content == "Hello!"
                assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_stop_sequences(self, mock_config):
        """Test chat with stop sequences."""
        mock_config.model.generation.stop_sequences = [".", "\n"]

        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}],
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = AsyncMock()
                mock_http_client.__aenter__.return_value = mock_http_client
                mock_http_client.post.return_value = mock_response
                mock_client_class.return_value = mock_http_client

                messages = [Message(role="user", content="Hi")]
                response = await client.chat(messages)

                # Verify stop was included in payload
                call_args = mock_http_client.post.call_args
                payload = call_args.kwargs["json"]
                assert "stop" in payload
                assert payload["stop"] == [".", "\n"]

    def test_chat_sync(self, mock_config):
        """Test synchronous chat method."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            mock_response = ChatResponse(content="Hello sync!")

            with patch.object(client, "chat", new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = mock_response

                messages = [Message(role="user", content="Hi")]
                response = client.chat_sync(messages)

                assert response.content == "Hello sync!"

    @pytest.mark.asyncio
    async def test_chat_stream(self, mock_config):
        """Test streaming chat completion."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            # Create mock for streaming
            async def mock_aiter_lines():
                yield ""  # Empty line (should be skipped)
                yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
                yield 'data: {"choices": [{"delta": {"content": " World"}}]}'
                yield 'data: {"choices": [{"delta": {}}]}'  # No content
                yield "data: [DONE]"

            mock_stream_response = MagicMock()
            mock_stream_response.aiter_lines = mock_aiter_lines
            mock_stream_response.raise_for_status = MagicMock()

            # Create an async context manager class
            class MockStreamCM:
                async def __aenter__(self):
                    return mock_stream_response

                async def __aexit__(self, *args):
                    pass

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = MagicMock()
                mock_http_client.stream.return_value = MockStreamCM()

                # Create an async context manager for the client itself
                class MockClientCM:
                    async def __aenter__(self):
                        return mock_http_client

                    async def __aexit__(self, *args):
                        pass

                mock_client_class.return_value = MockClientCM()

                messages = [Message(role="user", content="Hi")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_chat_stream_with_stop_sequences(self, mock_config):
        """Test streaming with stop sequences."""
        mock_config.model.generation.stop_sequences = ["END"]

        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            async def mock_aiter_lines():
                yield 'data: {"choices": [{"delta": {"content": "Test"}}]}'
                yield "data: [DONE]"

            mock_stream_response = MagicMock()
            mock_stream_response.aiter_lines = mock_aiter_lines
            mock_stream_response.raise_for_status = MagicMock()

            class MockStreamCM:
                async def __aenter__(self):
                    return mock_stream_response

                async def __aexit__(self, *args):
                    pass

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = MagicMock()
                mock_http_client.stream.return_value = MockStreamCM()

                class MockClientCM:
                    async def __aenter__(self):
                        return mock_http_client

                    async def __aexit__(self, *args):
                        pass

                mock_client_class.return_value = MockClientCM()

                messages = [Message(role="user", content="Hi")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                # Verify stop was included
                call_args = mock_http_client.stream.call_args
                payload = call_args.kwargs["json"]
                assert payload["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_chat_stream_invalid_json(self, mock_config):
        """Test streaming handles invalid JSON gracefully."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            async def mock_aiter_lines():
                yield "data: invalid json"  # Should be skipped
                yield 'data: {"choices": [{"delta": {"content": "Valid"}}]}'
                yield "data: [DONE]"

            mock_stream_response = MagicMock()
            mock_stream_response.aiter_lines = mock_aiter_lines
            mock_stream_response.raise_for_status = MagicMock()

            class MockStreamCM:
                async def __aenter__(self):
                    return mock_stream_response

                async def __aexit__(self, *args):
                    pass

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = MagicMock()
                mock_http_client.stream.return_value = MockStreamCM()

                class MockClientCM:
                    async def __aenter__(self):
                        return mock_http_client

                    async def __aexit__(self, *args):
                        pass

                mock_client_class.return_value = MockClientCM()

                messages = [Message(role="user", content="Hi")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                assert chunks == ["Valid"]

    @pytest.mark.asyncio
    async def test_chat_stream_line_without_data_prefix(self, mock_config):
        """Test streaming skips lines without data prefix."""
        with patch("javis.models.client.get_config", return_value=mock_config):
            client = ModelClient()

            async def mock_aiter_lines():
                yield "not-data-line"  # Should be skipped
                yield ": comment"  # Should be skipped
                yield 'data: {"choices": [{"delta": {"content": "Ok"}}]}'
                yield "data: [DONE]"

            mock_stream_response = MagicMock()
            mock_stream_response.aiter_lines = mock_aiter_lines
            mock_stream_response.raise_for_status = MagicMock()

            class MockStreamCM:
                async def __aenter__(self):
                    return mock_stream_response

                async def __aexit__(self, *args):
                    pass

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_http_client = MagicMock()
                mock_http_client.stream.return_value = MockStreamCM()

                class MockClientCM:
                    async def __aenter__(self):
                        return mock_http_client

                    async def __aexit__(self, *args):
                        pass

                mock_client_class.return_value = MockClientCM()

                messages = [Message(role="user", content="Hi")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                assert chunks == ["Ok"]
