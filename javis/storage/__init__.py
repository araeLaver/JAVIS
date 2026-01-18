"""JAVIS Storage Abstraction Layer.

Provides a unified interface for local and cloud storage backends.
Supports Supabase (PostgreSQL) for data and HuggingFace Hub for model adapters.
"""

from javis.storage.base import (
    ConversationStoreInterface,
    FeedbackStoreInterface,
    CorrectionStoreInterface,
    ModelStoreInterface,
    QualityScoreStoreInterface,
)
from javis.storage.factory import (
    StorageFactory,
    get_storage_factory,
    get_conversation_store,
    get_feedback_store,
    get_correction_store,
    get_model_store,
    get_quality_score_store,
)

__all__ = [
    # Interfaces
    "ConversationStoreInterface",
    "FeedbackStoreInterface",
    "CorrectionStoreInterface",
    "ModelStoreInterface",
    "QualityScoreStoreInterface",
    # Factory
    "StorageFactory",
    "get_storage_factory",
    "get_conversation_store",
    "get_feedback_store",
    "get_correction_store",
    "get_model_store",
    "get_quality_score_store",
]
