"""Storage factory for creating appropriate storage backends."""

import logging
import os
from typing import Literal, Optional

from javis.storage.base import (
    ConversationStoreInterface,
    CorrectionStoreInterface,
    FeedbackStoreInterface,
    ModelStoreInterface,
    QualityScoreStoreInterface,
)

logger = logging.getLogger(__name__)

StorageMode = Literal["local", "cloud"]


class StorageFactory:
    """Factory for creating storage backends based on configuration."""

    def __init__(self, mode: Optional[StorageMode] = None):
        """Initialize storage factory.

        Args:
            mode: Storage mode (local or cloud).
                  If None, reads from JAVIS_STORAGE_MODE env var,
                  defaults to 'local'.
        """
        if mode is None:
            mode = os.getenv("JAVIS_STORAGE_MODE", "local")  # type: ignore
        self.mode: StorageMode = mode  # type: ignore

        # Cache instances
        self._conversation_store: Optional[ConversationStoreInterface] = None
        self._feedback_store: Optional[FeedbackStoreInterface] = None
        self._correction_store: Optional[CorrectionStoreInterface] = None
        self._model_store: Optional[ModelStoreInterface] = None
        self._quality_score_store: Optional[QualityScoreStoreInterface] = None

        logger.info(f"StorageFactory initialized with mode: {self.mode}")

    def get_conversation_store(self) -> ConversationStoreInterface:
        """Get conversation store instance.

        Returns:
            ConversationStoreInterface implementation
        """
        if self._conversation_store is None:
            if self.mode == "cloud":
                from javis.storage.cloud.conversation_store import (
                    CloudConversationStore,
                )
                self._conversation_store = CloudConversationStore()
            else:
                from javis.storage.local.conversation_store import (
                    LocalConversationStore,
                )
                self._conversation_store = LocalConversationStore()
        return self._conversation_store

    def get_feedback_store(self) -> FeedbackStoreInterface:
        """Get feedback store instance.

        Returns:
            FeedbackStoreInterface implementation
        """
        if self._feedback_store is None:
            if self.mode == "cloud":
                from javis.storage.cloud.feedback_store import CloudFeedbackStore
                self._feedback_store = CloudFeedbackStore()
            else:
                from javis.storage.local.feedback_store import LocalFeedbackStore
                self._feedback_store = LocalFeedbackStore()
        return self._feedback_store

    def get_correction_store(self) -> CorrectionStoreInterface:
        """Get correction store instance.

        Returns:
            CorrectionStoreInterface implementation
        """
        if self._correction_store is None:
            if self.mode == "cloud":
                from javis.storage.cloud.correction_store import CloudCorrectionStore
                self._correction_store = CloudCorrectionStore()
            else:
                from javis.storage.local.correction_store import LocalCorrectionStore
                self._correction_store = LocalCorrectionStore()
        return self._correction_store

    def get_model_store(self) -> ModelStoreInterface:
        """Get model store instance.

        Returns:
            ModelStoreInterface implementation
        """
        if self._model_store is None:
            if self.mode == "cloud":
                from javis.storage.cloud.hf_model_store import HuggingFaceModelStore
                self._model_store = HuggingFaceModelStore()
            else:
                from javis.storage.local.model_store import LocalModelStore
                self._model_store = LocalModelStore()
        return self._model_store

    def get_quality_score_store(self) -> QualityScoreStoreInterface:
        """Get quality score store instance.

        Returns:
            QualityScoreStoreInterface implementation
        """
        if self._quality_score_store is None:
            if self.mode == "cloud":
                from javis.storage.cloud.quality_score_store import (
                    CloudQualityScoreStore,
                )
                self._quality_score_store = CloudQualityScoreStore()
            else:
                from javis.storage.local.quality_score_store import (
                    LocalQualityScoreStore,
                )
                self._quality_score_store = LocalQualityScoreStore()
        return self._quality_score_store


# Global factory instance
_factory: Optional[StorageFactory] = None


def get_storage_factory(mode: Optional[StorageMode] = None) -> StorageFactory:
    """Get or create the global storage factory.

    Args:
        mode: Storage mode. Only used when creating a new factory.

    Returns:
        StorageFactory instance
    """
    global _factory
    if _factory is None:
        _factory = StorageFactory(mode)
    return _factory


def reset_storage_factory() -> None:
    """Reset the global storage factory. Useful for testing."""
    global _factory
    _factory = None


# Convenience functions
def get_conversation_store() -> ConversationStoreInterface:
    """Get the conversation store from the global factory."""
    return get_storage_factory().get_conversation_store()


def get_feedback_store() -> FeedbackStoreInterface:
    """Get the feedback store from the global factory."""
    return get_storage_factory().get_feedback_store()


def get_correction_store() -> CorrectionStoreInterface:
    """Get the correction store from the global factory."""
    return get_storage_factory().get_correction_store()


def get_model_store() -> ModelStoreInterface:
    """Get the model store from the global factory."""
    return get_storage_factory().get_model_store()


def get_quality_score_store() -> QualityScoreStoreInterface:
    """Get the quality score store from the global factory."""
    return get_storage_factory().get_quality_score_store()
