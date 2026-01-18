"""Model version management with rollback capability."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.storage import get_model_store
from javis.storage.base import ModelStoreInterface, ModelVersionRecord


class VersionInfo(BaseModel):
    """Model version information."""

    version: str
    created_at: str
    base_model: str
    dataset_size: int = 0
    status: str = "ready"  # ready, failed, active
    training_config: dict = {}

    def to_storage_record(self) -> ModelVersionRecord:
        """Convert to storage record format."""
        return ModelVersionRecord(
            version=self.version,
            created_at=datetime.fromisoformat(self.created_at),
            base_model=self.base_model,
            dataset_size=self.dataset_size,
            status=self.status,
            training_config=self.training_config,
        )

    @classmethod
    def from_storage_record(cls, record: ModelVersionRecord) -> "VersionInfo":
        """Create from storage record."""
        return cls(
            version=record.version,
            created_at=record.created_at.isoformat() if isinstance(record.created_at, datetime) else record.created_at,
            base_model=record.base_model,
            dataset_size=record.dataset_size,
            status=record.status,
            training_config=record.training_config,
        )


class VersionManager:
    """Manages model versions with rollback capability.

    Uses the storage factory to store model versions either locally
    or on HuggingFace Hub depending on configuration.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        store: Optional[ModelStoreInterface] = None,
    ):
        self._store = store

        # Keep models_dir for backward compatibility
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_version_file = self.models_dir / ".active_version"

    @property
    def store(self) -> ModelStoreInterface:
        """Get the model store (lazy initialization)."""
        if self._store is None:
            self._store = get_model_store()
        return self._store

    def create_version(
        self,
        adapter_path: Path,
        metadata: Optional[dict] = None,
        version_name: Optional[str] = None,
    ) -> str:
        """Create a new model version from adapter."""
        if version_name is None:
            version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create version info
        version_info = VersionInfo(
            version=version_name,
            created_at=datetime.now().isoformat(),
            base_model=metadata.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
            if metadata
            else "Qwen/Qwen2.5-7B-Instruct",
            dataset_size=metadata.get("dataset_size", 0) if metadata else 0,
            status="ready",
            training_config=metadata.get("training_config", {}) if metadata else {},
        )

        # Use storage backend to save version
        self.store.save_version(
            version=version_name,
            adapter_path=Path(adapter_path),
            metadata=version_info.to_storage_record(),
        )

        return version_name

    def activate_version(self, version: str) -> bool:
        """Set a version as active."""
        return self.store.set_active_version(version)

    def get_active_version(self) -> Optional[VersionInfo]:
        """Get currently active model version."""
        version = self.store.get_active_version()
        if not version:
            return None
        record = self.store.get_version(version)
        if record:
            return VersionInfo.from_storage_record(record)
        return None

    def get_active_adapter_path(self) -> Optional[Path]:
        """Get the path to the active adapter."""
        version = self.store.get_active_version()
        if not version:
            return None
        return self.store.get_adapter_path(version)

    def rollback(self, to_version: Optional[str] = None) -> Optional[str]:
        """Rollback to a previous version."""
        if to_version:
            if self.activate_version(to_version):
                return to_version
            return None

        # Find previous stable version
        versions = self.list_versions()
        current = self.store.get_active_version()

        for v in versions:
            if v.version != current and v.status in ("ready", "active"):
                if self.activate_version(v.version):
                    return v.version

        return None

    def mark_failed(self, version: str) -> bool:
        """Mark a version as failed."""
        return self.store.update_status(version, "failed")

    def list_versions(self) -> list[VersionInfo]:
        """List all available versions sorted by creation date (newest first)."""
        records = self.store.list_versions()
        return [VersionInfo.from_storage_record(r) for r in records]

    def cleanup_old_versions(self, keep: int = 5) -> list[str]:
        """Remove old versions, keeping the specified number of newest versions."""
        return self.store.cleanup_old_versions(keep)

    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        record = self.store.get_version(version)
        if record:
            return VersionInfo.from_storage_record(record)
        return None


# Global instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager
