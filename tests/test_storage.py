"""Tests for the storage abstraction layer."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from javis.storage.base import (
    ConversationRecord,
    ConversationTurn,
    CorrectionRecord,
    FeedbackRecord,
    ModelVersionRecord,
    QualityScoreRecord,
)
from javis.storage.factory import StorageFactory
from javis.storage.local.conversation_store import LocalConversationStore
from javis.storage.local.correction_store import LocalCorrectionStore
from javis.storage.local.feedback_store import LocalFeedbackStore
from javis.storage.local.model_store import LocalModelStore
from javis.storage.local.quality_score_store import LocalQualityScoreStore


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_conversation():
    """Create a sample conversation record."""
    return ConversationRecord(
        session_id="test-session-123",
        started_at=datetime.now().isoformat(),
        turns=[
            ConversationTurn(
                role="user",
                content="Hello, how are you?",
                timestamp=datetime.now().isoformat(),
            ),
            ConversationTurn(
                role="assistant",
                content="I'm doing well, thank you!",
                timestamp=datetime.now().isoformat(),
            ),
        ],
        feedback="good",
        tags=["greeting", "test"],
    )


@pytest.fixture
def sample_feedback():
    """Create a sample feedback record."""
    return FeedbackRecord(
        id="fb_test_123",
        session_id="test-session-123",
        timestamp=datetime.now(),
        feedback_type="good",
        quality_score=4.5,
        prompt="What is the weather?",
        response="I don't have access to weather data.",
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_correction():
    """Create a sample correction record."""
    return CorrectionRecord(
        id="corr_test_123",
        session_id="test-session-123",
        timestamp=datetime.now(),
        original_prompt="Explain Python",
        original_response="Python is a snake.",
        corrected_response="Python is a programming language.",
        correction_type="factual",
        notes="Fixed incorrect response",
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_quality_score():
    """Create a sample quality score record."""
    return QualityScoreRecord(
        conversation_id="test-session-123",
        overall_score=4.2,
        length_score=4.0,
        coherence_score=4.5,
        feedback_score=5.0,
        recency_score=3.5,
        computed_at=datetime.now(),
    )


@pytest.fixture
def sample_model_version():
    """Create a sample model version record."""
    return ModelVersionRecord(
        version="v20240101_120000",
        created_at=datetime.now(),
        base_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_size=100,
        status="ready",
        training_config={"epochs": 3, "batch_size": 4},
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Local Conversation Store Tests
# ============================================================================


class TestLocalConversationStore:
    """Tests for LocalConversationStore."""

    def test_save_and_get(self, temp_dir, sample_conversation):
        """Test saving and retrieving a conversation."""
        store = LocalConversationStore(data_dir=temp_dir)

        # Save
        result = store.save(sample_conversation)
        assert result is not None

        # Get
        retrieved = store.get(sample_conversation.session_id)
        assert retrieved is not None
        assert retrieved.session_id == sample_conversation.session_id
        assert len(retrieved.turns) == len(sample_conversation.turns)
        assert retrieved.feedback == sample_conversation.feedback

    def test_list_all(self, temp_dir, sample_conversation):
        """Test listing all conversations."""
        store = LocalConversationStore(data_dir=temp_dir)

        # Save multiple conversations
        for i in range(3):
            conv = ConversationRecord(
                session_id=f"session-{i}",
                started_at=datetime.now().isoformat(),
                turns=[
                    ConversationTurn(
                        role="user",
                        content=f"Message {i}",
                        timestamp=datetime.now().isoformat(),
                    )
                ],
            )
            store.save(conv)

        # List all
        all_convs = store.list_all()
        assert len(all_convs) == 3

    def test_count(self, temp_dir, sample_conversation):
        """Test counting conversations."""
        store = LocalConversationStore(data_dir=temp_dir)

        assert store.count() == 0

        store.save(sample_conversation)
        assert store.count() == 1

    def test_delete(self, temp_dir, sample_conversation):
        """Test deleting a conversation."""
        store = LocalConversationStore(data_dir=temp_dir)

        store.save(sample_conversation)
        assert store.get(sample_conversation.session_id) is not None

        result = store.delete(sample_conversation.session_id)
        assert result is True
        assert store.get(sample_conversation.session_id) is None

    def test_export_for_training(self, temp_dir, sample_conversation):
        """Test exporting conversations for training."""
        store = LocalConversationStore(data_dir=temp_dir)
        store.save(sample_conversation)

        output_path = temp_dir / "export.jsonl"
        result = store.export_for_training(output_path)

        assert result.exists()
        with open(result, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert "messages" in data


# ============================================================================
# Local Feedback Store Tests
# ============================================================================


class TestLocalFeedbackStore:
    """Tests for LocalFeedbackStore."""

    def test_save_and_get(self, temp_dir, sample_feedback):
        """Test saving and retrieving feedback."""
        store = LocalFeedbackStore(feedback_dir=temp_dir)

        result = store.save(sample_feedback)
        assert result == sample_feedback.id

        retrieved = store.get(sample_feedback.id)
        assert retrieved is not None
        assert retrieved.id == sample_feedback.id
        assert retrieved.feedback_type == sample_feedback.feedback_type

    def test_list_by_type(self, temp_dir, sample_feedback):
        """Test listing feedback by type."""
        store = LocalFeedbackStore(feedback_dir=temp_dir)
        store.save(sample_feedback)

        good_feedback = store.list_by_type("good")
        assert len(good_feedback) == 1

        bad_feedback = store.list_by_type("bad")
        assert len(bad_feedback) == 0

    def test_get_statistics(self, temp_dir, sample_feedback):
        """Test getting feedback statistics."""
        store = LocalFeedbackStore(feedback_dir=temp_dir)
        store.save(sample_feedback)

        stats = store.get_statistics()
        assert "total" in stats
        assert stats["total"] >= 1


# ============================================================================
# Local Correction Store Tests
# ============================================================================


class TestLocalCorrectionStore:
    """Tests for LocalCorrectionStore."""

    def test_save_and_get(self, temp_dir, sample_correction):
        """Test saving and retrieving a correction."""
        store = LocalCorrectionStore(corrections_dir=temp_dir)

        result = store.save(sample_correction)
        assert result == sample_correction.id

        retrieved = store.get(sample_correction.id)
        assert retrieved is not None
        assert retrieved.id == sample_correction.id
        assert retrieved.correction_type == sample_correction.correction_type

    def test_update(self, temp_dir, sample_correction):
        """Test updating a correction."""
        store = LocalCorrectionStore(corrections_dir=temp_dir)
        store.save(sample_correction)

        result = store.update(
            correction_id=sample_correction.id,
            corrected_response="Updated response",
            notes="Updated notes",
        )
        assert result is True

        retrieved = store.get(sample_correction.id)
        assert retrieved.corrected_response == "Updated response"
        assert retrieved.notes == "Updated notes"

    def test_delete(self, temp_dir, sample_correction):
        """Test deleting a correction."""
        store = LocalCorrectionStore(corrections_dir=temp_dir)
        store.save(sample_correction)

        result = store.delete(sample_correction.id)
        assert result is True
        assert store.get(sample_correction.id) is None


# ============================================================================
# Local Quality Score Store Tests
# ============================================================================


class TestLocalQualityScoreStore:
    """Tests for LocalQualityScoreStore."""

    def test_save_and_get(self, temp_dir, sample_quality_score):
        """Test saving and retrieving a quality score."""
        store = LocalQualityScoreStore(scores_dir=temp_dir)

        result = store.save(sample_quality_score)
        assert result == sample_quality_score.conversation_id

        retrieved = store.get(sample_quality_score.conversation_id)
        assert retrieved is not None
        assert retrieved.overall_score == sample_quality_score.overall_score

    def test_get_high_quality_ids(self, temp_dir, sample_quality_score):
        """Test getting high quality conversation IDs."""
        store = LocalQualityScoreStore(scores_dir=temp_dir)
        store.save(sample_quality_score)

        high_quality = store.get_high_quality_ids(min_score=4.0)
        assert sample_quality_score.conversation_id in high_quality

        very_high = store.get_high_quality_ids(min_score=4.5)
        assert sample_quality_score.conversation_id not in very_high


# ============================================================================
# Local Model Store Tests
# ============================================================================


class TestLocalModelStore:
    """Tests for LocalModelStore."""

    def test_save_and_get_version(self, temp_dir, sample_model_version):
        """Test saving and retrieving a model version."""
        store = LocalModelStore(models_dir=temp_dir)

        # Create a mock adapter directory
        adapter_path = temp_dir / "adapter_source"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        result = store.save_version(
            version=sample_model_version.version,
            adapter_path=adapter_path,
            metadata=sample_model_version,
        )
        assert result is not None

        retrieved = store.get_version(sample_model_version.version)
        assert retrieved is not None
        assert retrieved.version == sample_model_version.version

    def test_list_versions(self, temp_dir):
        """Test listing model versions."""
        store = LocalModelStore(models_dir=temp_dir)

        # Create multiple versions
        for i in range(3):
            version = f"v{i}"
            adapter_path = temp_dir / f"adapter_{i}"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            metadata = ModelVersionRecord(
                version=version,
                created_at=datetime.now(),
                base_model="test-model",
                dataset_size=i * 10,
                status="ready",
            )
            store.save_version(version, adapter_path, metadata)

        versions = store.list_versions()
        assert len(versions) == 3

    def test_set_active_version(self, temp_dir, sample_model_version):
        """Test setting active version."""
        store = LocalModelStore(models_dir=temp_dir)

        adapter_path = temp_dir / "adapter_source"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        store.save_version(
            version=sample_model_version.version,
            adapter_path=adapter_path,
            metadata=sample_model_version,
        )

        result = store.set_active_version(sample_model_version.version)
        assert result is True

        active = store.get_active_version()
        assert active == sample_model_version.version


# ============================================================================
# Storage Factory Tests
# ============================================================================


class TestStorageFactory:
    """Tests for StorageFactory."""

    def test_local_mode(self, temp_dir):
        """Test factory in local mode."""
        with patch.dict(os.environ, {"JAVIS_STORAGE_MODE": "local"}):
            factory = StorageFactory()

            conv_store = factory.get_conversation_store()
            assert isinstance(conv_store, LocalConversationStore)

            feedback_store = factory.get_feedback_store()
            assert isinstance(feedback_store, LocalFeedbackStore)

    def test_default_is_local(self, temp_dir):
        """Test that default mode is local."""
        # Clear the environment variable
        env = os.environ.copy()
        if "JAVIS_STORAGE_MODE" in env:
            del env["JAVIS_STORAGE_MODE"]

        with patch.dict(os.environ, env, clear=True):
            factory = StorageFactory()
            assert factory.mode == "local"

    def test_cloud_mode_without_credentials(self):
        """Test that cloud mode fails gracefully without credentials."""
        with patch.dict(os.environ, {
            "JAVIS_STORAGE_MODE": "cloud",
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": "",
        }):
            factory = StorageFactory()

            # Should raise an error when trying to get cloud store
            with pytest.raises(Exception):
                factory.get_conversation_store()


# ============================================================================
# Integration Tests
# ============================================================================


class TestStorageIntegration:
    """Integration tests for storage layer."""

    def test_conversation_workflow(self, temp_dir, sample_conversation):
        """Test complete conversation workflow."""
        store = LocalConversationStore(data_dir=temp_dir)

        # 1. Save conversation
        store.save(sample_conversation)

        # 2. Retrieve and verify
        retrieved = store.get(sample_conversation.session_id)
        assert retrieved is not None

        # 3. List all
        all_convs = store.list_all()
        assert len(all_convs) == 1

        # 4. Export for training
        export_path = temp_dir / "export.jsonl"
        store.export_for_training(export_path)
        assert export_path.exists()

        # 5. Delete
        store.delete(sample_conversation.session_id)
        assert store.get(sample_conversation.session_id) is None

    def test_feedback_correction_workflow(
        self, temp_dir, sample_feedback, sample_correction
    ):
        """Test feedback and correction workflow."""
        feedback_store = LocalFeedbackStore(feedback_dir=temp_dir / "feedback")
        correction_store = LocalCorrectionStore(corrections_dir=temp_dir / "corrections")

        # Save feedback and correction
        feedback_store.save(sample_feedback)
        correction_store.save(sample_correction)

        # Get statistics
        fb_stats = feedback_store.get_statistics()
        corr_stats = correction_store.get_statistics()

        assert fb_stats["total"] >= 1
        assert corr_stats["total"] >= 1
