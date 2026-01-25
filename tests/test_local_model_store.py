"""Tests for javis.storage.local.model_store module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from javis.storage.local.model_store import LocalModelStore
from javis.storage.base import ModelVersionRecord


class TestLocalModelStore:
    """Tests for LocalModelStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create model store with temp directory."""
        return LocalModelStore(models_dir=temp_dir)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample model version metadata."""
        return ModelVersionRecord(
            version="v1.0.0",
            base_model="llama-2",
            dataset_size=1000,
            training_config={"epochs": 3, "batch_size": 4},
            status="trained",
            created_at=datetime.now(),
        )

    @pytest.fixture
    def adapter_dir(self, temp_dir):
        """Create sample adapter directory with files."""
        adapter = temp_dir / "adapter_source"
        adapter.mkdir()
        (adapter / "config.json").write_text('{"test": true}')
        (adapter / "adapter_model.bin").write_bytes(b"fake model data")
        return adapter

    @pytest.fixture
    def adapter_file(self, temp_dir):
        """Create sample adapter file."""
        adapter_file = temp_dir / "adapter.bin"
        adapter_file.write_bytes(b"single adapter file")
        return adapter_file

    def test_init_creates_directory(self, temp_dir):
        """Test initialization creates models directory."""
        models_dir = temp_dir / "models"
        assert not models_dir.exists()

        store = LocalModelStore(models_dir=models_dir)

        assert models_dir.exists()

    def test_init_default_directory(self):
        """Test initialization with default directory."""
        store = LocalModelStore()
        assert store.models_dir.exists()

    def test_save_version_with_directory(self, store, adapter_dir, sample_metadata):
        """Test saving version with adapter directory."""
        result = store.save_version("v1.0.0", adapter_dir, sample_metadata)

        assert result == "v1.0.0"
        assert store._get_version_dir("v1.0.0").exists()
        assert store._get_metadata_path("v1.0.0").exists()
        assert store._get_adapter_dir("v1.0.0").exists()
        assert (store._get_adapter_dir("v1.0.0") / "config.json").exists()

    def test_save_version_with_file(self, store, adapter_file, sample_metadata):
        """Test saving version with single adapter file."""
        result = store.save_version("v1.1.0", adapter_file, sample_metadata)

        assert result == "v1.1.0"
        adapter_dir = store._get_adapter_dir("v1.1.0")
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter.bin").exists()

    def test_save_version_overwrites_existing_adapter(self, store, adapter_dir, sample_metadata):
        """Test saving version overwrites existing adapter directory."""
        # Save first version
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        # Create new adapter dir
        new_adapter = adapter_dir.parent / "new_adapter"
        new_adapter.mkdir()
        (new_adapter / "new_file.json").write_text('{"new": true}')

        # Save again with new adapter
        store.save_version("v1.0.0", new_adapter, sample_metadata)

        # Should have new file, not old ones
        adapter_saved = store._get_adapter_dir("v1.0.0")
        assert (adapter_saved / "new_file.json").exists()

    def test_get_version_exists(self, store, adapter_dir, sample_metadata):
        """Test getting existing version."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        result = store.get_version("v1.0.0")

        assert result is not None
        assert result.version == "v1.0.0"
        assert result.base_model == "llama-2"

    def test_get_version_not_exists(self, store):
        """Test getting non-existent version returns None."""
        result = store.get_version("nonexistent")
        assert result is None

    def test_get_version_corrupted_metadata(self, store, temp_dir):
        """Test getting version with corrupted metadata returns None."""
        # Create version dir with corrupted metadata
        version_dir = store._get_version_dir("corrupt")
        version_dir.mkdir(parents=True)
        metadata_path = store._get_metadata_path("corrupt")
        metadata_path.write_text("not valid json {{{")

        result = store.get_version("corrupt")
        assert result is None

    def test_list_versions_empty(self, store):
        """Test listing versions when empty."""
        result = store.list_versions()
        assert result == []

    def test_list_versions_multiple(self, store, adapter_dir, sample_metadata):
        """Test listing multiple versions."""
        # Save multiple versions
        for i in range(3):
            meta = ModelVersionRecord(
                version=f"v1.{i}.0",
                base_model="llama-2",
                dataset_size=1000,
                training_config={},
                status="trained",
                created_at=datetime(2024, 1, i + 1),
            )
            store.save_version(f"v1.{i}.0", adapter_dir, meta)

        result = store.list_versions()

        assert len(result) == 3
        # Should be sorted by created_at descending
        assert result[0].version == "v1.2.0"

    def test_list_versions_skips_hidden_dirs(self, store, adapter_dir, sample_metadata):
        """Test that list_versions skips hidden directories."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        # Create hidden directory
        hidden_dir = store.models_dir / ".hidden"
        hidden_dir.mkdir()

        result = store.list_versions()
        assert len(result) == 1

    def test_list_versions_skips_files(self, store, adapter_dir, sample_metadata):
        """Test that list_versions skips files (non-directories)."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        # Create a file in models dir
        (store.models_dir / "some_file.txt").write_text("test")

        result = store.list_versions()
        assert len(result) == 1

    def test_get_active_version_none(self, store):
        """Test getting active version when none set."""
        result = store.get_active_version()
        assert result is None

    def test_set_and_get_active_version(self, store, adapter_dir, sample_metadata):
        """Test setting and getting active version."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        success = store.set_active_version("v1.0.0")

        assert success is True
        assert store.get_active_version() == "v1.0.0"

    def test_set_active_version_nonexistent(self, store):
        """Test setting active version for non-existent version fails."""
        result = store.set_active_version("nonexistent")
        assert result is False

    def test_set_active_version_deactivates_previous(self, store, adapter_dir, sample_metadata):
        """Test setting new active version deactivates previous."""
        # Save two versions
        store.save_version("v1.0.0", adapter_dir, sample_metadata)
        meta2 = ModelVersionRecord(
            version="v2.0.0",
            base_model="llama-2",
            dataset_size=2000,
            training_config={},
            status="trained",
            created_at=datetime.now(),
        )
        store.save_version("v2.0.0", adapter_dir, meta2)

        # Set first as active
        store.set_active_version("v1.0.0")
        assert store.get_active_version() == "v1.0.0"

        # Set second as active
        store.set_active_version("v2.0.0")

        assert store.get_active_version() == "v2.0.0"
        # v1.0.0 should be inactive
        v1_meta = store.get_version("v1.0.0")
        assert v1_meta.status == "inactive"

    def test_get_adapter_path_exists(self, store, adapter_dir, sample_metadata):
        """Test getting adapter path for existing version."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        result = store.get_adapter_path("v1.0.0")

        assert result is not None
        assert result.exists()

    def test_get_adapter_path_not_exists(self, store):
        """Test getting adapter path for non-existent version returns None."""
        result = store.get_adapter_path("nonexistent")
        assert result is None

    def test_update_status(self, store, adapter_dir, sample_metadata):
        """Test updating version status."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        result = store.update_status("v1.0.0", "deployed")

        assert result is True
        version = store.get_version("v1.0.0")
        assert version.status == "deployed"

    def test_update_status_nonexistent(self, store):
        """Test updating status for non-existent version fails."""
        result = store.update_status("nonexistent", "active")
        assert result is False

    def test_update_status_corrupted_metadata(self, store, temp_dir):
        """Test updating status with corrupted metadata returns False."""
        # Create version dir with corrupted metadata
        version_dir = store._get_version_dir("corrupt")
        version_dir.mkdir(parents=True)
        metadata_path = store._get_metadata_path("corrupt")
        metadata_path.write_text("invalid json")

        result = store.update_status("corrupt", "active")
        assert result is False

    def test_delete_version(self, store, adapter_dir, sample_metadata):
        """Test deleting a version."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)
        assert store._get_version_dir("v1.0.0").exists()

        result = store.delete_version("v1.0.0")

        assert result is True
        assert not store._get_version_dir("v1.0.0").exists()

    def test_delete_version_nonexistent(self, store):
        """Test deleting non-existent version returns False."""
        result = store.delete_version("nonexistent")
        assert result is False

    def test_delete_active_version_fails(self, store, adapter_dir, sample_metadata):
        """Test deleting active version fails."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)
        store.set_active_version("v1.0.0")

        result = store.delete_version("v1.0.0")

        assert result is False
        assert store._get_version_dir("v1.0.0").exists()

    def test_cleanup_old_versions(self, store, adapter_dir):
        """Test cleaning up old versions."""
        # Create 7 versions
        for i in range(7):
            meta = ModelVersionRecord(
                version=f"v1.{i}.0",
                base_model="llama-2",
                dataset_size=1000 + i * 100,
                training_config={},
                status="trained",
                created_at=datetime(2024, 1, i + 1),
            )
            store.save_version(f"v1.{i}.0", adapter_dir, meta)

        # Keep only 5
        deleted = store.cleanup_old_versions(keep=5)

        assert len(deleted) == 2
        assert len(store.list_versions()) == 5
        # Oldest versions should be deleted (v1.0.0 and v1.1.0)
        assert "v1.0.0" in deleted
        assert "v1.1.0" in deleted

    def test_cleanup_old_versions_keeps_active(self, store, adapter_dir):
        """Test cleanup keeps active version even if old."""
        # Create versions
        for i in range(5):
            meta = ModelVersionRecord(
                version=f"v1.{i}.0",
                base_model="llama-2",
                dataset_size=1000 + i * 100,
                training_config={},
                status="trained",
                created_at=datetime(2024, 1, i + 1),
            )
            store.save_version(f"v1.{i}.0", adapter_dir, meta)

        # Set oldest as active
        store.set_active_version("v1.0.0")

        # Keep only 2 (but v1.0.0 is active so should not be deleted)
        deleted = store.cleanup_old_versions(keep=2)

        # v1.3.0 and v1.2.0 should be kept (newest 2)
        # v1.0.0 should be kept (active)
        # v1.1.0 should be deleted
        assert "v1.0.0" not in deleted
        assert store._get_version_dir("v1.0.0").exists()

    def test_cleanup_old_versions_nothing_to_delete(self, store, adapter_dir, sample_metadata):
        """Test cleanup when there are fewer versions than keep."""
        store.save_version("v1.0.0", adapter_dir, sample_metadata)

        deleted = store.cleanup_old_versions(keep=5)

        assert deleted == []
        assert len(store.list_versions()) == 1


class TestModelVersionRecord:
    """Tests for ModelVersionRecord model."""

    def test_create_minimal_record(self):
        """Test creating minimal model version record."""
        record = ModelVersionRecord(
            version="v1.0.0",
            base_model="llama-2",
            dataset_size=1000,
            status="trained",
            created_at=datetime.now(),
        )
        assert record.version == "v1.0.0"
        assert record.adapter_path is None
        assert record.training_config == {}

    def test_create_full_record(self):
        """Test creating full model version record."""
        record = ModelVersionRecord(
            version="v2.0.0",
            base_model="llama-3",
            dataset_size=5000,
            training_config={"epochs": 5, "lr": 1e-4},
            status="deployed",
            created_at=datetime(2024, 1, 15),
            adapter_path="/path/to/adapter",
        )
        assert record.training_config["epochs"] == 5
        assert record.adapter_path == "/path/to/adapter"
        assert record.dataset_size == 5000
