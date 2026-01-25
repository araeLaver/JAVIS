"""Unit tests for JAVIS training modules."""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from javis.training.corrections import (
    ResponseCorrection,
    CorrectionManager,
    get_correction_manager,
)
from javis.training.feedback_store import (
    EnhancedFeedback,
    FeedbackStore,
    get_feedback_store,
)
from javis.storage.local.correction_store import LocalCorrectionStore
from javis.storage.local.feedback_store import LocalFeedbackStore


class TestResponseCorrection:
    """Tests for ResponseCorrection model."""

    def test_create_correction(self):
        """Test creating a response correction."""
        correction = ResponseCorrection(
            id="corr_123",
            session_id="sess_456",
            timestamp=datetime.now(),
            original_prompt="How do I code?",
            original_response="Code like this...",
            corrected_response="Actually, code like this...",
            correction_type="factual",
        )

        assert correction.id == "corr_123"
        assert correction.correction_type == "factual"
        assert correction.notes is None

    def test_correction_with_notes(self):
        """Test correction with notes and metadata."""
        correction = ResponseCorrection(
            id="corr_123",
            session_id="sess_456",
            timestamp=datetime.now(),
            original_prompt="What is Python?",
            original_response="A snake",
            corrected_response="A programming language",
            correction_type="factual",
            notes="Completely wrong answer",
            metadata={"severity": "high"},
        )

        assert correction.notes == "Completely wrong answer"
        assert correction.metadata["severity"] == "high"


class TestCorrectionManager:
    """Tests for CorrectionManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create correction manager with temp directory."""
        # Inject LocalCorrectionStore with temp directory to avoid global factory
        local_store = LocalCorrectionStore(corrections_dir=temp_dir)
        return CorrectionManager(corrections_dir=temp_dir, store=local_store)

    def test_init_creates_directory(self, temp_dir):
        """Test that init creates the corrections directory."""
        subdir = temp_dir / "corrections"
        manager = CorrectionManager(corrections_dir=subdir)
        assert subdir.exists()

    def test_generate_id(self, manager):
        """Test ID generation."""
        id1 = manager._generate_id()
        id2 = manager._generate_id()

        assert id1.startswith("corr_")
        assert id1 != id2

    def test_get_month_dir(self, manager):
        """Test month directory creation."""
        date = datetime(2024, 3, 15)
        month_dir = manager._get_month_dir(date)

        assert month_dir.exists()
        assert "2024-03" in str(month_dir)

    def test_add_correction(self, manager, temp_dir):
        """Test adding a correction."""
        correction_id = manager.add_correction(
            session_id="sess_123",
            original_prompt="What is 2+2?",
            original_response="5",
            corrected_response="4",
            correction_type="factual",
            notes="Wrong math",
        )

        assert correction_id.startswith("corr_")

        # Check file was created
        files = list(temp_dir.rglob("*.json"))
        assert len(files) == 1

    def test_get_correction(self, manager):
        """Test retrieving a correction."""
        correction_id = manager.add_correction(
            session_id="sess_123",
            original_prompt="Hello",
            original_response="Hi",
            corrected_response="Hello there!",
            correction_type="style",
        )

        correction = manager.get_correction(correction_id)

        assert correction is not None
        assert correction.id == correction_id
        assert correction.original_prompt == "Hello"
        assert correction.correction_type == "style"

    def test_get_correction_not_found(self, manager):
        """Test getting a non-existent correction."""
        result = manager.get_correction("nonexistent_id")
        assert result is None

    def test_get_corrections_empty(self, manager):
        """Test getting corrections when none exist."""
        corrections = manager.get_corrections()
        assert corrections == []

    def test_get_corrections_with_filters(self, manager):
        """Test getting corrections with filters."""
        # Add a factual correction
        manager.add_correction(
            session_id="sess_1",
            original_prompt="Q1",
            original_response="R1",
            corrected_response="C1",
            correction_type="factual",
        )
        # Add a style correction
        manager.add_correction(
            session_id="sess_2",
            original_prompt="Q2",
            original_response="R2",
            corrected_response="C2",
            correction_type="style",
        )

        # Get only factual corrections
        factual = manager.get_corrections(correction_type="factual")
        assert len(factual) == 1
        assert factual[0].correction_type == "factual"

    def test_get_corrections_with_limit(self, manager):
        """Test getting corrections with limit."""
        for i in range(5):
            manager.add_correction(
                session_id=f"sess_{i}",
                original_prompt=f"Q{i}",
                original_response=f"R{i}",
                corrected_response=f"C{i}",
                correction_type="other",
            )

        corrections = manager.get_corrections(limit=3)
        assert len(corrections) == 3

    def test_update_correction(self, manager):
        """Test updating a correction."""
        correction_id = manager.add_correction(
            session_id="sess_123",
            original_prompt="Test",
            original_response="Old",
            corrected_response="New",
            correction_type="style",
        )

        result = manager.update_correction(
            correction_id,
            corrected_response="Updated response",
            notes="Updated notes",
        )

        assert result is True

        updated = manager.get_correction(correction_id)
        assert updated.corrected_response == "Updated response"
        assert updated.notes == "Updated notes"

    def test_update_correction_not_found(self, manager):
        """Test updating non-existent correction."""
        result = manager.update_correction("nonexistent", corrected_response="test")
        assert result is False

    def test_delete_correction(self, manager):
        """Test deleting a correction."""
        correction_id = manager.add_correction(
            session_id="sess_123",
            original_prompt="Test",
            original_response="Old",
            corrected_response="New",
            correction_type="other",
        )

        result = manager.delete_correction(correction_id)
        assert result is True

        # Verify deletion
        assert manager.get_correction(correction_id) is None

    def test_delete_correction_not_found(self, manager):
        """Test deleting non-existent correction."""
        result = manager.delete_correction("nonexistent")
        assert result is False

    def test_export_for_training(self, manager, temp_dir):
        """Test exporting corrections as training data."""
        manager.add_correction(
            session_id="sess_1",
            original_prompt="Question 1",
            original_response="Wrong answer",
            corrected_response="Right answer",
            correction_type="factual",
        )
        manager.add_correction(
            session_id="sess_2",
            original_prompt="Question 2",
            original_response="Bad response",
            corrected_response="Good response",
            correction_type="style",
        )

        output_path = temp_dir / "training.jsonl"
        result = manager.export_for_training(output_path)

        assert result.exists()

        with open(result) as f:
            lines = f.readlines()

        assert len(lines) == 2

        data = json.loads(lines[0])
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_get_statistics(self, manager):
        """Test getting correction statistics."""
        manager.add_correction(
            session_id="sess_1",
            original_prompt="Q1",
            original_response="R1",
            corrected_response="C1",
            correction_type="factual",
        )
        manager.add_correction(
            session_id="sess_2",
            original_prompt="Q2",
            original_response="R2",
            corrected_response="C2",
            correction_type="factual",
        )
        manager.add_correction(
            session_id="sess_3",
            original_prompt="Q3",
            original_response="R3",
            corrected_response="C3",
            correction_type="style",
        )

        stats = manager.get_statistics()

        assert stats["total"] == 3
        assert stats["by_type"]["factual"] == 2
        assert stats["by_type"]["style"] == 1


class TestEnhancedFeedback:
    """Tests for EnhancedFeedback model."""

    def test_create_feedback(self):
        """Test creating feedback."""
        feedback = EnhancedFeedback(
            id="fb_123",
            session_id="sess_456",
            timestamp=datetime.now(),
            feedback_type="good",
            prompt="How are you?",
            response="I'm doing well!",
        )

        assert feedback.id == "fb_123"
        assert feedback.feedback_type == "good"
        assert feedback.quality_score is None

    def test_feedback_with_correction(self):
        """Test feedback with correction."""
        feedback = EnhancedFeedback(
            id="fb_123",
            session_id="sess_456",
            timestamp=datetime.now(),
            feedback_type="corrected",
            prompt="Question",
            response="Bad answer",
            corrected_response="Good answer",
        )

        assert feedback.corrected_response == "Good answer"


class TestFeedbackStore:
    """Tests for FeedbackStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create feedback store with temp directory."""
        # Inject LocalFeedbackStore with temp directory to avoid global factory
        local_store = LocalFeedbackStore(feedback_dir=temp_dir)
        return FeedbackStore(feedback_dir=temp_dir, store=local_store)

    def test_init_creates_directories(self, temp_dir):
        """Test that init creates subdirectories."""
        store = FeedbackStore(feedback_dir=temp_dir)

        assert (temp_dir / "good").exists()
        assert (temp_dir / "bad").exists()
        assert (temp_dir / "corrected").exists()

    def test_generate_id(self, store):
        """Test ID generation."""
        id1 = store._generate_id()
        id2 = store._generate_id()

        assert id1.startswith("fb_")
        assert id1 != id2

    def test_store_good_response(self, store, temp_dir):
        """Test storing good response."""
        feedback_id = store.store_good_response(
            session_id="sess_123",
            prompt="Hello",
            response="Hi there!",
            quality_score=5.0,
        )

        assert feedback_id.startswith("fb_")

        # Check file was created in good directory
        files = list((temp_dir / "good").rglob("*.json"))
        assert len(files) == 1

    def test_store_bad_response(self, store, temp_dir):
        """Test storing bad response."""
        feedback_id = store.store_bad_response(
            session_id="sess_123",
            prompt="What is Python?",
            response="A snake",
        )

        assert feedback_id.startswith("fb_")

        # Check file was created in bad directory
        files = list((temp_dir / "bad").rglob("*.json"))
        assert len(files) == 1

    def test_store_bad_response_with_correction(self, store, temp_dir):
        """Test storing bad response with correction goes to corrected."""
        feedback_id = store.store_bad_response(
            session_id="sess_123",
            prompt="What is Python?",
            response="A snake",
            corrected_response="A programming language",
        )

        # Check file was created in corrected directory
        files = list((temp_dir / "corrected").rglob("*.json"))
        assert len(files) == 1

    def test_get_good_responses(self, store):
        """Test getting good responses."""
        store.store_good_response("sess_1", "Q1", "A1")
        store.store_good_response("sess_2", "Q2", "A2")

        responses = store.get_good_responses()
        assert len(responses) == 2

    def test_get_bad_responses(self, store):
        """Test getting bad responses."""
        store.store_bad_response("sess_1", "Q1", "A1")

        responses = store.get_bad_responses()
        assert len(responses) == 1

    def test_get_corrected_responses(self, store):
        """Test getting corrected responses."""
        store.store_bad_response("sess_1", "Q1", "A1", "C1")

        responses = store.get_corrected_responses()
        assert len(responses) == 1
        assert responses[0].corrected_response == "C1"

    def test_get_all_feedback(self, store):
        """Test getting all feedback."""
        store.store_good_response("sess_1", "Q1", "A1")
        store.store_bad_response("sess_2", "Q2", "A2")
        store.store_bad_response("sess_3", "Q3", "A3", "C3")

        all_feedback = store.get_all_feedback()
        assert len(all_feedback) == 3

    def test_get_all_feedback_with_limit(self, store):
        """Test getting all feedback with limit."""
        for i in range(5):
            store.store_good_response(f"sess_{i}", f"Q{i}", f"A{i}")

        feedback = store.get_all_feedback(limit=3)
        assert len(feedback) == 3

    def test_get_statistics(self, store):
        """Test getting feedback statistics."""
        store.store_good_response("sess_1", "Q1", "A1")
        store.store_good_response("sess_2", "Q2", "A2")
        store.store_bad_response("sess_3", "Q3", "A3")
        store.store_bad_response("sess_4", "Q4", "A4", "C4")

        stats = store.get_statistics()

        assert stats["total"] == 4
        assert stats["good"] == 2
        assert stats["bad"] == 1
        assert stats["corrected"] == 1
        assert stats["positive_rate"] == 0.5

    def test_export_training_data(self, store, temp_dir):
        """Test exporting training data."""
        store.store_good_response("sess_1", "Q1", "A1")
        store.store_bad_response("sess_2", "Q2", "A2", "C2")

        output_path = temp_dir / "training.jsonl"
        result = store.export_training_data(output_path)

        assert result.exists()

        with open(result) as f:
            lines = f.readlines()

        # Should include good response and corrected response
        assert len(lines) == 2

    def test_export_training_data_exclude_corrected(self, store, temp_dir):
        """Test exporting without corrected responses."""
        store.store_good_response("sess_1", "Q1", "A1")
        store.store_bad_response("sess_2", "Q2", "A2", "C2")

        output_path = temp_dir / "training.jsonl"
        result = store.export_training_data(output_path, include_corrected=False)

        with open(result) as f:
            lines = f.readlines()

        # Should only include good response
        assert len(lines) == 1

    def test_delete_feedback(self, store):
        """Test deleting feedback."""
        feedback_id = store.store_good_response("sess_1", "Q1", "A1")

        result = store.delete_feedback(feedback_id)
        assert result is True

        # Verify deletion
        responses = store.get_good_responses()
        assert len(responses) == 0

    def test_delete_feedback_not_found(self, store):
        """Test deleting non-existent feedback."""
        result = store.delete_feedback("nonexistent")
        assert result is False


class TestSingletonPatterns:
    """Tests for singleton patterns."""

    def test_get_correction_manager_singleton(self):
        """Test that get_correction_manager returns singleton."""
        import javis.training.corrections as corrections_module

        # Reset the singleton
        corrections_module._manager = None

        with patch.object(CorrectionManager, '__init__', return_value=None):
            manager1 = get_correction_manager()
            manager2 = get_correction_manager()
            assert manager1 is manager2

        corrections_module._manager = None

    def test_get_feedback_store_singleton(self):
        """Test that get_feedback_store returns singleton."""
        import javis.training.feedback_store as feedback_module

        # Reset the singleton
        feedback_module._store = None

        with patch.object(FeedbackStore, '__init__', return_value=None):
            store1 = get_feedback_store()
            store2 = get_feedback_store()
            assert store1 is store2

        feedback_module._store = None
