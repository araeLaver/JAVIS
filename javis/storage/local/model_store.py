"""Local file-based model version storage."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.storage.base import ModelStoreInterface, ModelVersionRecord

logger = logging.getLogger(__name__)


class LocalModelStore(ModelStoreInterface):
    """Local file-based implementation of model version storage."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize local model store.

        Args:
            models_dir: Directory for storing model versions.
                        Defaults to data/models.
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent.parent / "data" / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_version_file = self.models_dir / ".active_version"

    def _get_version_dir(self, version: str) -> Path:
        """Get the directory for a version."""
        return self.models_dir / version

    def _get_metadata_path(self, version: str) -> Path:
        """Get the metadata file path for a version."""
        return self._get_version_dir(version) / "metadata.json"

    def _get_adapter_dir(self, version: str) -> Path:
        """Get the adapter directory for a version."""
        return self._get_version_dir(version) / "adapter"

    def save_version(
        self,
        version: str,
        adapter_path: Path,
        metadata: ModelVersionRecord,
    ) -> str:
        """Save a new model version."""
        version_dir = self._get_version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy adapter files
        adapter_dir = self._get_adapter_dir(version)
        if adapter_path.is_dir():
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            shutil.copytree(adapter_path, adapter_dir)
        else:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(adapter_path, adapter_dir / adapter_path.name)

        # Save metadata
        metadata_path = self._get_metadata_path(version)
        meta_data = metadata.model_dump()
        meta_data["created_at"] = meta_data["created_at"].isoformat()
        meta_data["adapter_path"] = str(adapter_dir)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved model version: {version}")
        return version

    def get_version(self, version: str) -> Optional[ModelVersionRecord]:
        """Get version metadata."""
        metadata_path = self._get_metadata_path(version)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data.get("created_at"), str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])

            return ModelVersionRecord(**data)

        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Failed to load version {version}: {e}")
            return None

    def list_versions(self) -> list[ModelVersionRecord]:
        """List all versions."""
        versions = []

        for version_dir in self.models_dir.iterdir():
            if not version_dir.is_dir():
                continue
            if version_dir.name.startswith("."):
                continue

            metadata = self.get_version(version_dir.name)
            if metadata:
                versions.append(metadata)

        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def get_active_version(self) -> Optional[str]:
        """Get the currently active version."""
        if not self._active_version_file.exists():
            return None

        try:
            with open(self._active_version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except IOError:
            return None

    def set_active_version(self, version: str) -> bool:
        """Set the active version."""
        # Verify version exists
        if not self._get_version_dir(version).exists():
            logger.error(f"Version {version} does not exist")
            return False

        # Deactivate previous active version
        current_active = self.get_active_version()
        if current_active and current_active != version:
            self.update_status(current_active, "inactive")

        # Set new active version
        try:
            with open(self._active_version_file, "w", encoding="utf-8") as f:
                f.write(version)

            self.update_status(version, "active")
            logger.info(f"Activated version: {version}")
            return True

        except IOError as e:
            logger.error(f"Failed to set active version: {e}")
            return False

    def get_adapter_path(self, version: str) -> Optional[Path]:
        """Get the path to adapter files for a version."""
        adapter_dir = self._get_adapter_dir(version)
        if adapter_dir.exists():
            return adapter_dir
        return None

    def update_status(self, version: str, status: str) -> bool:
        """Update version status."""
        metadata_path = self._get_metadata_path(version)
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["status"] = status

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Updated version {version} status to {status}")
            return True

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to update status for {version}: {e}")
            return False

    def delete_version(self, version: str) -> bool:
        """Delete a version."""
        version_dir = self._get_version_dir(version)
        if not version_dir.exists():
            return False

        # Don't delete active version
        if self.get_active_version() == version:
            logger.warning(f"Cannot delete active version: {version}")
            return False

        try:
            shutil.rmtree(version_dir)
            logger.info(f"Deleted version: {version}")
            return True
        except IOError as e:
            logger.error(f"Failed to delete version {version}: {e}")
            return False

    def cleanup_old_versions(self, keep: int = 5) -> list[str]:
        """Remove old versions, keeping the most recent ones."""
        versions = self.list_versions()
        deleted = []

        if len(versions) <= keep:
            return deleted

        # Keep the most recent 'keep' versions and the active one
        active_version = self.get_active_version()
        versions_to_delete = versions[keep:]

        for version in versions_to_delete:
            if version.version == active_version:
                continue

            if self.delete_version(version.version):
                deleted.append(version.version)

        return deleted
