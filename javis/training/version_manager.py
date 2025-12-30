"""Model version management with rollback capability."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class VersionInfo(BaseModel):
    """Model version information."""

    version: str
    created_at: str
    base_model: str
    dataset_size: int = 0
    status: str = "ready"  # ready, failed, active
    training_config: dict = {}


class VersionManager:
    """Manages model versions with rollback capability."""

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_version_file = self.models_dir / ".active_version"

    def create_version(
        self,
        adapter_path: Path,
        metadata: Optional[dict] = None,
        version_name: Optional[str] = None,
    ) -> str:
        """Create a new model version from adapter."""
        if version_name is None:
            version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        version_dir = self.models_dir / version_name
        adapter_dest = version_dir / "adapter"

        # Copy adapter files
        if adapter_path.is_dir():
            shutil.copytree(adapter_path, adapter_dest)
        else:
            adapter_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(adapter_path, adapter_dest)

        # Create metadata
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

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(version_info.model_dump(), f, ensure_ascii=False, indent=2)

        return version_name

    def activate_version(self, version: str) -> bool:
        """Set a version as active."""
        version_dir = self.models_dir / version
        if not version_dir.exists():
            return False

        # Update status in metadata
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["status"] = "active"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Deactivate previous active version
        prev_version = self.get_active_version()
        if prev_version and prev_version != version:
            prev_dir = self.models_dir / prev_version
            prev_metadata = prev_dir / "metadata.json"
            if prev_metadata.exists():
                with open(prev_metadata, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                prev_data["status"] = "ready"
                with open(prev_metadata, "w", encoding="utf-8") as f:
                    json.dump(prev_data, f, ensure_ascii=False, indent=2)

        # Write active version pointer
        with open(self.active_version_file, "w") as f:
            f.write(version)

        return True

    def get_active_version(self) -> Optional[str]:
        """Get currently active model version."""
        if not self.active_version_file.exists():
            return None
        with open(self.active_version_file, "r") as f:
            return f.read().strip() or None

    def get_active_adapter_path(self) -> Optional[Path]:
        """Get the path to the active adapter."""
        version = self.get_active_version()
        if not version:
            return None
        adapter_path = self.models_dir / version / "adapter"
        return adapter_path if adapter_path.exists() else None

    def rollback(self, to_version: Optional[str] = None) -> Optional[str]:
        """Rollback to a previous version."""
        if to_version:
            if self.activate_version(to_version):
                return to_version
            return None

        # Find previous stable version
        versions = self.list_versions()
        current = self.get_active_version()

        for v in versions:
            if v.version != current and v.status in ("ready", "active"):
                if self.activate_version(v.version):
                    return v.version

        return None

    def mark_failed(self, version: str) -> bool:
        """Mark a version as failed."""
        version_dir = self.models_dir / version
        metadata_path = version_dir / "metadata.json"

        if not metadata_path.exists():
            return False

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["status"] = "failed"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    def list_versions(self) -> list[VersionInfo]:
        """List all available versions sorted by creation date (newest first)."""
        versions = []

        for version_dir in self.models_dir.iterdir():
            if not version_dir.is_dir() or version_dir.name.startswith("."):
                continue

            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                versions.append(VersionInfo(**data))
            else:
                # Create basic info for versions without metadata
                versions.append(
                    VersionInfo(
                        version=version_dir.name,
                        created_at=datetime.fromtimestamp(
                            version_dir.stat().st_mtime
                        ).isoformat(),
                        base_model="unknown",
                    )
                )

        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def cleanup_old_versions(self, keep: int = 5) -> list[str]:
        """Remove old versions, keeping the specified number of newest versions."""
        versions = self.list_versions()
        active = self.get_active_version()
        removed = []

        # Keep active version and newest N versions
        keep_versions = {active} if active else set()
        for v in versions[:keep]:
            keep_versions.add(v.version)

        for v in versions:
            if v.version not in keep_versions:
                version_dir = self.models_dir / v.version
                shutil.rmtree(version_dir)
                removed.append(v.version)

        return removed

    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        metadata_path = self.models_dir / version / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VersionInfo(**data)


# Global instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager
