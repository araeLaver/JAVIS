"""Tests for javis.storage.factory module."""

import os
from unittest.mock import patch, MagicMock

import pytest

from javis.storage.factory import (
    StorageFactory,
    get_storage_factory,
    reset_storage_factory,
    get_conversation_store,
    get_feedback_store,
    get_correction_store,
    get_model_store,
    get_quality_score_store,
)


class TestStorageFactory:
    """Tests for StorageFactory class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset factory before each test."""
        reset_storage_factory()
        yield
        reset_storage_factory()

    def test_init_default_mode(self):
        """Test initialization with default mode (local)."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            factory = StorageFactory()
            assert factory.mode == "local"

    def test_init_explicit_mode_local(self):
        """Test initialization with explicit local mode."""
        factory = StorageFactory(mode="local")
        assert factory.mode == "local"

    def test_init_explicit_mode_cloud(self):
        """Test initialization with explicit cloud mode."""
        factory = StorageFactory(mode="cloud")
        assert factory.mode == "cloud"

    def test_init_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "cloud"}, clear=False):
            factory = StorageFactory()
            assert factory.mode == "cloud"

    def test_get_conversation_store_local(self):
        """Test getting local conversation store."""
        factory = StorageFactory(mode="local")
        store = factory.get_conversation_store()

        from javis.storage.local.conversation_store import LocalConversationStore
        assert isinstance(store, LocalConversationStore)

    def test_get_conversation_store_cloud(self):
        """Test getting cloud conversation store."""
        factory = StorageFactory(mode="cloud")

        with patch("javis.storage.cloud.conversation_store.CloudConversationStore") as MockStore:
            mock_instance = MagicMock()
            MockStore.return_value = mock_instance

            # Force reimport to use mock
            factory._conversation_store = None
            with patch.dict("sys.modules", {"javis.storage.cloud.conversation_store": MagicMock()}):
                with patch("javis.storage.factory.StorageFactory.get_conversation_store") as mock_get:
                    mock_get.return_value = mock_instance
                    store = factory.get_conversation_store()
                    assert store is mock_instance

    def test_get_conversation_store_cached(self):
        """Test that conversation store is cached."""
        factory = StorageFactory(mode="local")
        store1 = factory.get_conversation_store()
        store2 = factory.get_conversation_store()
        assert store1 is store2

    def test_get_feedback_store_local(self):
        """Test getting local feedback store."""
        factory = StorageFactory(mode="local")
        store = factory.get_feedback_store()

        from javis.storage.local.feedback_store import LocalFeedbackStore
        assert isinstance(store, LocalFeedbackStore)

    def test_get_feedback_store_cached(self):
        """Test that feedback store is cached."""
        factory = StorageFactory(mode="local")
        store1 = factory.get_feedback_store()
        store2 = factory.get_feedback_store()
        assert store1 is store2

    def test_get_correction_store_local(self):
        """Test getting local correction store."""
        factory = StorageFactory(mode="local")
        store = factory.get_correction_store()

        from javis.storage.local.correction_store import LocalCorrectionStore
        assert isinstance(store, LocalCorrectionStore)

    def test_get_correction_store_cached(self):
        """Test that correction store is cached."""
        factory = StorageFactory(mode="local")
        store1 = factory.get_correction_store()
        store2 = factory.get_correction_store()
        assert store1 is store2

    def test_get_model_store_local(self):
        """Test getting local model store."""
        factory = StorageFactory(mode="local")
        store = factory.get_model_store()

        from javis.storage.local.model_store import LocalModelStore
        assert isinstance(store, LocalModelStore)

    def test_get_model_store_cached(self):
        """Test that model store is cached."""
        factory = StorageFactory(mode="local")
        store1 = factory.get_model_store()
        store2 = factory.get_model_store()
        assert store1 is store2

    def test_get_quality_score_store_local(self):
        """Test getting local quality score store."""
        factory = StorageFactory(mode="local")
        store = factory.get_quality_score_store()

        from javis.storage.local.quality_score_store import LocalQualityScoreStore
        assert isinstance(store, LocalQualityScoreStore)

    def test_get_quality_score_store_cached(self):
        """Test that quality score store is cached."""
        factory = StorageFactory(mode="local")
        store1 = factory.get_quality_score_store()
        store2 = factory.get_quality_score_store()
        assert store1 is store2


class TestGlobalFactory:
    """Tests for global factory functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset factory before each test."""
        reset_storage_factory()
        yield
        reset_storage_factory()

    def test_get_storage_factory_creates_new(self):
        """Test that get_storage_factory creates a new factory."""
        factory = get_storage_factory(mode="local")
        assert isinstance(factory, StorageFactory)
        assert factory.mode == "local"

    def test_get_storage_factory_returns_same(self):
        """Test that get_storage_factory returns the same instance."""
        factory1 = get_storage_factory(mode="local")
        factory2 = get_storage_factory()  # Mode ignored for existing factory
        assert factory1 is factory2

    def test_reset_storage_factory(self):
        """Test resetting the global factory."""
        factory1 = get_storage_factory(mode="local")
        reset_storage_factory()
        factory2 = get_storage_factory(mode="local")
        assert factory1 is not factory2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset factory before each test with local mode."""
        reset_storage_factory()
        # Force local mode by setting environment variable
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            yield
        reset_storage_factory()

    def test_get_conversation_store_convenience(self):
        """Test convenience function for conversation store."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            store = get_conversation_store()
            from javis.storage.local.conversation_store import LocalConversationStore
            assert isinstance(store, LocalConversationStore)

    def test_get_feedback_store_convenience(self):
        """Test convenience function for feedback store."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            store = get_feedback_store()
            from javis.storage.local.feedback_store import LocalFeedbackStore
            assert isinstance(store, LocalFeedbackStore)

    def test_get_correction_store_convenience(self):
        """Test convenience function for correction store."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            store = get_correction_store()
            from javis.storage.local.correction_store import LocalCorrectionStore
            assert isinstance(store, LocalCorrectionStore)

    def test_get_model_store_convenience(self):
        """Test convenience function for model store."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            store = get_model_store()
            from javis.storage.local.model_store import LocalModelStore
            assert isinstance(store, LocalModelStore)

    def test_get_quality_score_store_convenience(self):
        """Test convenience function for quality score store."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}, clear=False):
            store = get_quality_score_store()
            from javis.storage.local.quality_score_store import LocalQualityScoreStore
            assert isinstance(store, LocalQualityScoreStore)
