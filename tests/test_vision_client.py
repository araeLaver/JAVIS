"""Tests for javis.models.vision_client module."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from javis.models.vision_client import (
    VisionMessage,
    VisionChatResponse,
    VisionModelClient,
)


class TestVisionMessage:
    """Tests for VisionMessage model."""

    def test_create_basic_message(self):
        """Test creating a basic message."""
        msg = VisionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_create_message_with_images(self):
        """Test creating a message with images."""
        images = [{"url": "https://example.com/image.png", "detail": "auto"}]
        msg = VisionMessage(role="user", content="Describe this", images=images)
        assert msg.images == images
        assert len(msg.images) == 1


class TestVisionChatResponse:
    """Tests for VisionChatResponse model."""

    def test_create_basic_response(self):
        """Test creating a basic response."""
        resp = VisionChatResponse(content="This is an image of a cat.")
        assert resp.content == "This is an image of a cat."
        assert resp.finish_reason is None
        assert resp.usage is None
        assert resp.has_vision is False
        assert resp.vision_tokens is None

    def test_create_full_response(self):
        """Test creating a response with all fields."""
        resp = VisionChatResponse(
            content="A landscape photo",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            has_vision=True,
            vision_tokens=80,
        )
        assert resp.finish_reason == "stop"
        assert resp.usage["prompt_tokens"] == 100
        assert resp.has_vision is True
        assert resp.vision_tokens == 80


class TestVisionModelClient:
    """Tests for VisionModelClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-openai-key"
        config.anthropic_api_key = "test-anthropic-key"
        config.groq_api_key = "test-groq-key"
        config.model.generation.max_tokens = 1000
        config.model.generation.temperature = 0.7
        return config

    def test_init_openai(self, mock_config):
        """Test initializing OpenAI client."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            assert client.provider == "openai"
            assert client.model == "gpt-4o"
            assert client.base_url == "https://api.openai.com/v1"
            assert client.api_key == "test-openai-key"

    def test_init_anthropic(self, mock_config):
        """Test initializing Anthropic client."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")
            assert client.provider == "anthropic"
            assert client.model == "claude-3-5-sonnet-20241022"
            assert client.base_url == "https://api.anthropic.com/v1"
            assert client.api_key == "test-anthropic-key"

    def test_init_groq(self, mock_config):
        """Test initializing Groq client."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="groq")
            assert client.provider == "groq"
            assert client.model == "llama-3.2-90b-vision-preview"
            assert client.api_key == "test-groq-key"

    def test_init_custom_model(self, mock_config):
        """Test initializing with custom model."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai", model="gpt-4o-mini")
            assert client.model == "gpt-4o-mini"

    def test_init_custom_api_key(self, mock_config):
        """Test initializing with custom API key."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai", api_key="custom-key")
            assert client.api_key == "custom-key"

    def test_init_no_api_key_raises(self):
        """Test that missing API key raises error."""
        mock_config = MagicMock()
        mock_config.openai_api_key = None
        mock_config.anthropic_api_key = None
        mock_config.groq_api_key = None

        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                VisionModelClient(provider="openai")

    def test_get_api_key_openai(self, mock_config):
        """Test getting OpenAI API key."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            assert client._get_api_key() == "test-openai-key"

    def test_get_api_key_anthropic(self, mock_config):
        """Test getting Anthropic API key."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")
            assert client._get_api_key() == "test-anthropic-key"

    def test_get_api_key_groq(self, mock_config):
        """Test getting Groq API key."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="groq")
            assert client._get_api_key() == "test-groq-key"

    def test_get_api_key_unknown(self, mock_config):
        """Test getting API key for unknown provider returns None."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            client.provider = "unknown"
            assert client._get_api_key() is None

    def test_get_instance_singleton(self, mock_config):
        """Test singleton instance creation."""
        VisionModelClient._instance = None

        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            instance1 = VisionModelClient.get_instance(provider="openai")
            instance2 = VisionModelClient.get_instance(provider="openai")
            assert instance1 is instance2

    def test_get_instance_different_provider(self, mock_config):
        """Test that different provider creates new instance."""
        VisionModelClient._instance = None

        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            instance1 = VisionModelClient.get_instance(provider="openai")
            instance2 = VisionModelClient.get_instance(provider="anthropic")
            assert instance1 is not instance2

    def test_detect_media_type_png(self, mock_config):
        """Test detecting PNG media type."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
            assert client._detect_media_type(png_header) == "image/png"

    def test_detect_media_type_jpeg(self, mock_config):
        """Test detecting JPEG media type."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            jpeg_header = b"\xff\xd8" + b"\x00" * 10
            assert client._detect_media_type(jpeg_header) == "image/jpeg"

    def test_detect_media_type_gif87a(self, mock_config):
        """Test detecting GIF87a media type."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            gif_header = b"GIF87a" + b"\x00" * 10
            assert client._detect_media_type(gif_header) == "image/gif"

    def test_detect_media_type_gif89a(self, mock_config):
        """Test detecting GIF89a media type."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            gif_header = b"GIF89a" + b"\x00" * 10
            assert client._detect_media_type(gif_header) == "image/gif"

    def test_detect_media_type_webp(self, mock_config):
        """Test detecting WebP media type."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            webp_header = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
            assert client._detect_media_type(webp_header) == "image/webp"

    def test_detect_media_type_unknown(self, mock_config):
        """Test detecting unknown media type defaults to jpeg."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            unknown_header = b"\x00\x00\x00\x00" + b"\x00" * 10
            assert client._detect_media_type(unknown_header) == "image/jpeg"

    def test_get_media_type_from_extension(self, mock_config):
        """Test getting media type from file extension."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            assert client._get_media_type_from_extension(".png") == "image/png"
            assert client._get_media_type_from_extension(".jpg") == "image/jpeg"
            assert client._get_media_type_from_extension(".jpeg") == "image/jpeg"
            assert client._get_media_type_from_extension(".gif") == "image/gif"
            assert client._get_media_type_from_extension(".webp") == "image/webp"
            assert client._get_media_type_from_extension(".PNG") == "image/png"
            assert client._get_media_type_from_extension(".unknown") == "image/jpeg"

    def test_parse_data_url(self, mock_config):
        """Test parsing data URL."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            data_url = "data:image/png;base64,iVBORw0KGgoAAAANS"
            media_type, data = client._parse_data_url(data_url)
            assert media_type == "image/png"
            assert data == "iVBORw0KGgoAAAANS"

    @pytest.mark.asyncio
    async def test_prepare_image_bytes(self, mock_config):
        """Test preparing image from bytes."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            result = await client._prepare_image(png_bytes, "auto")

            assert "url" in result
            assert result["url"].startswith("data:image/png;base64,")
            assert result["detail"] == "auto"

    @pytest.mark.asyncio
    async def test_prepare_image_file_path(self, mock_config):
        """Test preparing image from file path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            f.write(png_content)
            f.flush()

            with patch("javis.models.vision_client.get_config", return_value=mock_config):
                client = VisionModelClient(provider="openai")
                result = await client._prepare_image(Path(f.name), "high")

                assert "url" in result
                assert result["url"].startswith("data:image/png;base64,")
                assert result["detail"] == "high"

    @pytest.mark.asyncio
    async def test_prepare_image_file_not_found(self, mock_config):
        """Test preparing image from nonexistent file."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            with pytest.raises(FileNotFoundError):
                await client._prepare_image(Path("/nonexistent/image.png"), "auto")

    @pytest.mark.asyncio
    async def test_prepare_image_url(self, mock_config):
        """Test preparing image from URL."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")
            result = await client._prepare_image("https://example.com/image.png", "low")

            assert result["url"] == "https://example.com/image.png"
            assert result["detail"] == "low"

    @pytest.mark.asyncio
    async def test_prepare_image_string_path(self, mock_config):
        """Test preparing image from string path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpg_content = b"\xff\xd8" + b"\x00" * 50
            f.write(jpg_content)
            f.flush()

            with patch("javis.models.vision_client.get_config", return_value=mock_config):
                client = VisionModelClient(provider="openai")
                result = await client._prepare_image(f.name, "auto")

                assert "url" in result
                assert result["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_chat_routes_to_anthropic(self, mock_config):
        """Test that chat routes to Anthropic for Anthropic provider."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")

            with patch.object(
                client, "_chat_anthropic", new_callable=AsyncMock
            ) as mock_anthropic:
                mock_anthropic.return_value = VisionChatResponse(content="test")

                messages = [VisionMessage(role="user", content="Hello")]
                await client.chat(messages)

                mock_anthropic.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_chat_routes_to_openai(self, mock_config):
        """Test that chat routes to OpenAI-compatible for OpenAI provider."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")

            with patch.object(
                client, "_chat_openai_compatible", new_callable=AsyncMock
            ) as mock_openai:
                mock_openai.return_value = VisionChatResponse(content="test")

                messages = [VisionMessage(role="user", content="Hello")]
                await client.chat(messages)

                mock_openai.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_analyze_image(self, mock_config):
        """Test analyze_image method."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")

            with patch.object(
                client, "_prepare_image", new_callable=AsyncMock
            ) as mock_prepare:
                mock_prepare.return_value = {"url": "data:image/png;base64,abc", "detail": "auto"}

                with patch.object(
                    client, "chat", new_callable=AsyncMock
                ) as mock_chat:
                    mock_chat.return_value = VisionChatResponse(content="A cat")

                    result = await client.analyze_image(
                        "https://example.com/cat.png",
                        "What is in this image?"
                    )

                    assert result.content == "A cat"
                    mock_prepare.assert_called_once()
                    mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_stream_routes_to_anthropic(self, mock_config):
        """Test that chat_stream routes to Anthropic for Anthropic provider."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")

            async def mock_stream(messages):
                yield "Hello"
                yield " World"

            with patch.object(client, "_stream_anthropic", mock_stream):
                messages = [VisionMessage(role="user", content="Hello")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_chat_stream_routes_to_openai(self, mock_config):
        """Test that chat_stream routes to OpenAI-compatible for OpenAI provider."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")

            async def mock_stream(messages):
                yield "Hello"
                yield " World"

            with patch.object(client, "_stream_openai_compatible", mock_stream):
                messages = [VisionMessage(role="user", content="Hello")]
                chunks = []
                async for chunk in client.chat_stream(messages):
                    chunks.append(chunk)

                assert chunks == ["Hello", " World"]


class TestVisionClientOpenAICompatible:
    """Tests for OpenAI-compatible API methods."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-openai-key"
        config.model.generation.max_tokens = 1000
        config.model.generation.temperature = 0.7
        return config

    @pytest.mark.asyncio
    async def test_chat_openai_compatible_text_only(self, mock_config):
        """Test OpenAI-compatible chat with text only."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                messages = [VisionMessage(role="user", content="Hello")]
                result = await client._chat_openai_compatible(messages)

                assert result.content == "Hello!"
                assert result.finish_reason == "stop"
                assert result.has_vision is False

    @pytest.mark.asyncio
    async def test_chat_openai_compatible_with_images(self, mock_config):
        """Test OpenAI-compatible chat with images."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="openai")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "A cat"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"image_tokens": 80},
                },
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                messages = [
                    VisionMessage(
                        role="user",
                        content="What is this?",
                        images=[{"url": "https://example.com/cat.png"}],
                    )
                ]
                result = await client._chat_openai_compatible(messages)

                assert result.content == "A cat"
                assert result.has_vision is True
                assert result.vision_tokens == 80


class TestVisionClientAnthropic:
    """Tests for Anthropic API methods."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.anthropic_api_key = "test-anthropic-key"
        config.model.generation.max_tokens = 1000
        config.model.generation.temperature = 0.7
        return config

    @pytest.mark.asyncio
    async def test_chat_anthropic_text_only(self, mock_config):
        """Test Anthropic chat with text only."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "Hello!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                messages = [VisionMessage(role="user", content="Hello")]
                result = await client._chat_anthropic(messages)

                assert result.content == "Hello!"
                assert result.finish_reason == "end_turn"
                assert result.has_vision is False

    @pytest.mark.asyncio
    async def test_chat_anthropic_with_data_url_image(self, mock_config):
        """Test Anthropic chat with data URL image."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "A cat"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 5},
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                messages = [
                    VisionMessage(
                        role="user",
                        content="What is this?",
                        images=[{"url": "data:image/png;base64,iVBORw0KGgo"}],
                    )
                ]
                result = await client._chat_anthropic(messages)

                assert result.content == "A cat"
                assert result.has_vision is True

    @pytest.mark.asyncio
    async def test_chat_anthropic_with_http_url_image(self, mock_config):
        """Test Anthropic chat with HTTP URL image."""
        with patch("javis.models.vision_client.get_config", return_value=mock_config):
            client = VisionModelClient(provider="anthropic")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "A dog"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 5},
            }

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                messages = [
                    VisionMessage(
                        role="user",
                        content="What is this?",
                        images=[{"url": "https://example.com/dog.png"}],
                    )
                ]
                result = await client._chat_anthropic(messages)

                assert result.content == "A dog"
                assert result.has_vision is True
