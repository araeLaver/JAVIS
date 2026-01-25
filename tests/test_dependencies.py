"""Tests for FastAPI dependency injection providers."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from javis.interfaces.dependencies import (
    get_config_dependency,
    get_session_manager_dependency,
    get_storage_factory_dependency,
    get_conversation_store_dependency,
    get_feedback_store_dependency,
    get_correction_store_dependency,
    get_memory_manager_dependency,
    require_memory_manager,
    get_rag_manager_dependency,
    require_rag_manager,
    get_tool_registry_dependency,
    require_tool_registry,
    get_health_checker_dependency,
    get_correction_manager_dependency,
    get_feedback_manager_dependency,
    ModelClientProvider,
    get_model_client_provider,
    get_container_dependency,
)
from javis.utils.config import Config


class TestConfigDependency:
    """Tests for config dependency."""

    def test_get_config_dependency_returns_config(self):
        """Test that get_config_dependency returns a Config instance."""
        config = get_config_dependency()
        assert isinstance(config, Config)

    def test_get_config_dependency_returns_same_instance(self):
        """Test that get_config_dependency returns the singleton instance."""
        config1 = get_config_dependency()
        config2 = get_config_dependency()
        assert config1 is config2


class TestSessionDependency:
    """Tests for session manager dependency."""

    def test_get_session_manager_dependency(self):
        """Test that get_session_manager_dependency returns session manager."""
        with patch("javis.core.session.get_session_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            result = get_session_manager_dependency()

            assert result is mock_manager
            mock_get.assert_called_once()


class TestStorageDependencies:
    """Tests for storage dependencies."""

    def test_get_storage_factory_dependency(self):
        """Test that get_storage_factory_dependency returns storage factory."""
        with patch("javis.storage.factory.get_storage_factory") as mock_get:
            mock_factory = MagicMock()
            mock_get.return_value = mock_factory

            result = get_storage_factory_dependency()

            assert result is mock_factory
            mock_get.assert_called_once()

    def test_get_conversation_store_dependency(self):
        """Test that get_conversation_store_dependency returns conversation store."""
        with patch("javis.storage.get_conversation_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = get_conversation_store_dependency()

            assert result is mock_store
            mock_get.assert_called_once()

    def test_get_feedback_store_dependency(self):
        """Test that get_feedback_store_dependency returns feedback store."""
        with patch("javis.storage.get_feedback_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = get_feedback_store_dependency()

            assert result is mock_store
            mock_get.assert_called_once()

    def test_get_correction_store_dependency(self):
        """Test that get_correction_store_dependency returns correction store."""
        with patch("javis.storage.get_correction_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = get_correction_store_dependency()

            assert result is mock_store
            mock_get.assert_called_once()


class TestMemoryDependencies:
    """Tests for memory dependencies."""

    def test_get_memory_manager_dependency_enabled(self):
        """Test memory manager dependency when enabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = True

        with patch("javis.memory.get_memory_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            result = get_memory_manager_dependency(mock_config)

            assert result is mock_manager
            mock_get.assert_called_once()

    def test_get_memory_manager_dependency_disabled(self):
        """Test memory manager dependency when disabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False

        result = get_memory_manager_dependency(mock_config)

        assert result is None

    def test_require_memory_manager_success(self):
        """Test require_memory_manager with valid manager."""
        mock_manager = MagicMock()

        result = require_memory_manager(mock_manager)

        assert result is mock_manager

    def test_require_memory_manager_raises_when_none(self):
        """Test require_memory_manager raises HTTPException when None."""
        with pytest.raises(HTTPException) as exc_info:
            require_memory_manager(None)

        assert exc_info.value.status_code == 400
        assert "Memory system is not enabled" in exc_info.value.detail


class TestRAGDependencies:
    """Tests for RAG dependencies."""

    def test_get_rag_manager_dependency_enabled(self):
        """Test RAG manager dependency when enabled."""
        mock_config = MagicMock()
        mock_config.rag.enabled = True

        with patch("javis.rag.get_rag_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            result = get_rag_manager_dependency(mock_config)

            assert result is mock_manager
            mock_get.assert_called_once()

    def test_get_rag_manager_dependency_disabled(self):
        """Test RAG manager dependency when disabled."""
        mock_config = MagicMock()
        mock_config.rag.enabled = False

        result = get_rag_manager_dependency(mock_config)

        assert result is None

    def test_require_rag_manager_success(self):
        """Test require_rag_manager with valid manager."""
        mock_manager = MagicMock()

        result = require_rag_manager(mock_manager)

        assert result is mock_manager

    def test_require_rag_manager_raises_when_none(self):
        """Test require_rag_manager raises HTTPException when None."""
        with pytest.raises(HTTPException) as exc_info:
            require_rag_manager(None)

        assert exc_info.value.status_code == 400
        assert "RAG system is not enabled" in exc_info.value.detail


class TestToolsDependencies:
    """Tests for tools dependencies."""

    def test_get_tool_registry_dependency_enabled(self):
        """Test tool registry dependency when enabled."""
        mock_config = MagicMock()
        mock_config.tools.enabled = True

        with patch("javis.tools.get_registry") as mock_get:
            mock_registry = MagicMock()
            mock_get.return_value = mock_registry

            result = get_tool_registry_dependency(mock_config)

            assert result is mock_registry
            mock_get.assert_called_once()

    def test_get_tool_registry_dependency_disabled(self):
        """Test tool registry dependency when disabled."""
        mock_config = MagicMock()
        mock_config.tools.enabled = False

        result = get_tool_registry_dependency(mock_config)

        assert result is None

    def test_require_tool_registry_success(self):
        """Test require_tool_registry with valid registry."""
        mock_registry = MagicMock()

        result = require_tool_registry(mock_registry)

        assert result is mock_registry

    def test_require_tool_registry_raises_when_none(self):
        """Test require_tool_registry raises HTTPException when None."""
        with pytest.raises(HTTPException) as exc_info:
            require_tool_registry(None)

        assert exc_info.value.status_code == 400
        assert "Tools system is not enabled" in exc_info.value.detail


class TestHealthDependency:
    """Tests for health checker dependency."""

    def test_get_health_checker_dependency(self):
        """Test that get_health_checker_dependency returns health checker."""
        with patch("javis.utils.health.get_health_checker") as mock_get:
            mock_checker = MagicMock()
            mock_get.return_value = mock_checker

            result = get_health_checker_dependency()

            assert result is mock_checker
            mock_get.assert_called_once()


class TestTrainingDependencies:
    """Tests for training dependencies."""

    def test_get_correction_manager_dependency(self):
        """Test that get_correction_manager_dependency returns correction manager."""
        with patch("javis.training.corrections.get_correction_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            result = get_correction_manager_dependency()

            assert result is mock_manager
            mock_get.assert_called_once()

    def test_get_feedback_manager_dependency(self):
        """Test that get_feedback_manager_dependency returns feedback store."""
        with patch("javis.training.feedback_store.get_feedback_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = get_feedback_manager_dependency()

            assert result is mock_store
            mock_get.assert_called_once()


class TestModelClientProvider:
    """Tests for ModelClientProvider."""

    def setup_method(self):
        """Reset ModelClientProvider before each test."""
        ModelClientProvider.reset()

    def teardown_method(self):
        """Reset ModelClientProvider after each test."""
        ModelClientProvider.reset()

    def test_reset_clears_cached_clients(self):
        """Test that reset clears cached clients."""
        ModelClientProvider._local_client = MagicMock()
        ModelClientProvider._modal_client = MagicMock()

        ModelClientProvider.reset()

        assert ModelClientProvider._local_client is None
        assert ModelClientProvider._modal_client is None

    @patch("javis.models.local_client.LocalModelClient")
    @patch("javis.models.local_client.get_latest_adapter")
    def test_get_local_client_creates_new(self, mock_get_adapter, mock_client_class):
        """Test get_local_client creates new client when none exists."""
        mock_get_adapter.return_value = "/path/to/adapter"
        mock_client = MagicMock()
        mock_client.adapter_path = "/path/to/adapter"
        mock_client_class.return_value = mock_client

        result = ModelClientProvider.get_local_client()

        assert result is mock_client
        mock_client_class.assert_called_once_with(adapter_path="/path/to/adapter")

    @patch("javis.models.local_client.LocalModelClient")
    @patch("javis.models.local_client.get_latest_adapter")
    def test_get_local_client_with_custom_version(self, mock_get_adapter, mock_client_class):
        """Test get_local_client with custom adapter version."""
        mock_client = MagicMock()
        mock_client.adapter_path = "/custom/adapter"
        mock_client_class.return_value = mock_client

        result = ModelClientProvider.get_local_client(adapter_version="/custom/adapter")

        assert result is mock_client
        mock_client_class.assert_called_once_with(adapter_path="/custom/adapter")
        mock_get_adapter.assert_not_called()

    @patch("javis.models.local_client.LocalModelClient")
    @patch("javis.models.local_client.get_latest_adapter")
    def test_get_local_client_returns_cached(self, mock_get_adapter, mock_client_class):
        """Test get_local_client returns cached client."""
        mock_get_adapter.return_value = "/path/to/adapter"
        mock_client = MagicMock()
        mock_client.adapter_path = "/path/to/adapter"
        mock_client_class.return_value = mock_client

        result1 = ModelClientProvider.get_local_client()
        result2 = ModelClientProvider.get_local_client()

        assert result1 is result2
        assert mock_client_class.call_count == 1

    @patch("javis.models.local_client.LocalModelClient")
    @patch("javis.models.local_client.get_latest_adapter")
    def test_get_local_client_recreates_for_different_adapter(self, mock_get_adapter, mock_client_class):
        """Test get_local_client creates new client for different adapter."""
        mock_get_adapter.return_value = "/path/to/adapter"

        mock_client1 = MagicMock()
        mock_client1.adapter_path = "/path/to/adapter"
        mock_client2 = MagicMock()
        mock_client2.adapter_path = "/different/adapter"
        mock_client_class.side_effect = [mock_client1, mock_client2]

        result1 = ModelClientProvider.get_local_client()
        result2 = ModelClientProvider.get_local_client(adapter_version="/different/adapter")

        assert result1 is not result2
        assert mock_client_class.call_count == 2

    @patch("javis.models.modal_client.ModalInferenceClient")
    @patch("javis.models.modal_client.get_latest_adapter_path")
    def test_get_modal_client_creates_new(self, mock_get_adapter, mock_client_class):
        """Test get_modal_client creates new client."""
        mock_get_adapter.return_value = "/path/to/modal/adapter"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = ModelClientProvider.get_modal_client()

        assert result is mock_client
        mock_client_class.assert_called_once_with(adapter_path="/path/to/modal/adapter")

    @patch("javis.models.modal_client.ModalInferenceClient")
    @patch("javis.models.modal_client.get_latest_adapter_path")
    def test_get_modal_client_with_custom_version(self, mock_get_adapter, mock_client_class):
        """Test get_modal_client with custom adapter version."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = ModelClientProvider.get_modal_client(adapter_version="/custom/modal/adapter")

        assert result is mock_client
        mock_client_class.assert_called_once_with(adapter_path="/custom/modal/adapter")
        mock_get_adapter.assert_not_called()

    @patch("javis.models.modal_client.ModalInferenceClient")
    @patch("javis.models.modal_client.get_latest_adapter_path")
    def test_get_modal_client_always_creates_new(self, mock_get_adapter, mock_client_class):
        """Test get_modal_client creates fresh client each time."""
        mock_get_adapter.return_value = "/path/to/modal/adapter"
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_client_class.side_effect = [mock_client1, mock_client2]

        result1 = ModelClientProvider.get_modal_client()
        result2 = ModelClientProvider.get_modal_client()

        # Modal clients are created fresh each time
        assert mock_client_class.call_count == 2


class TestModelClientProviderDependency:
    """Tests for model client provider dependency."""

    def test_get_model_client_provider(self):
        """Test that get_model_client_provider returns ModelClientProvider."""
        result = get_model_client_provider()

        assert result is ModelClientProvider


class TestContainerDependency:
    """Tests for container dependency."""

    def test_get_container_dependency(self):
        """Test that get_container_dependency returns container."""
        with patch("javis.core.container.get_container") as mock_get:
            mock_container = MagicMock()
            mock_get.return_value = mock_container

            result = get_container_dependency()

            assert result is mock_container
            mock_get.assert_called_once()
