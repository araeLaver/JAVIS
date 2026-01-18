"""HuggingFace Hub-based model version storage."""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.storage.base import ModelStoreInterface, ModelVersionRecord
from javis.storage.cloud.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Table name for metadata
MODEL_VERSIONS_TABLE = "model_versions"


class HuggingFaceModelStore(ModelStoreInterface):
    """HuggingFace Hub-based implementation of model version storage.

    Model adapters are stored on HuggingFace Hub, while metadata is stored in Supabase.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize HuggingFace model store.

        Args:
            hf_token: HuggingFace API token. Defaults to HF_TOKEN env var.
            repo_id: HuggingFace repository ID. Defaults to HF_REPO_ID env var.
            cache_dir: Local cache directory for downloaded adapters.
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token is required. Set HF_TOKEN environment variable."
            )

        if not self.repo_id:
            raise ValueError(
                "HuggingFace repository ID is required. Set HF_REPO_ID environment variable."
            )

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "model_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Supabase client for metadata
        self.db_client = get_supabase_client()

        # HuggingFace Hub API (lazy initialization)
        self._hf_api = None

    @property
    def hf_api(self):
        """Get HuggingFace Hub API (lazy initialization)."""
        if self._hf_api is None:
            try:
                from huggingface_hub import HfApi
                self._hf_api = HfApi(token=self.hf_token)
                logger.info("HuggingFace Hub API initialized")
            except ImportError:
                raise ImportError(
                    "huggingface_hub package is required. "
                    "Install it with: pip install huggingface_hub"
                )
        return self._hf_api

    def _get_remote_path(self, version: str) -> str:
        """Get the remote path for a version in the HF repo."""
        return f"adapters/{version}"

    def save_version(
        self,
        version: str,
        adapter_path: Path,
        metadata: ModelVersionRecord,
    ) -> str:
        """Save a new model version to HuggingFace Hub."""
        adapter_path = Path(adapter_path)

        # Upload adapter to HuggingFace Hub
        remote_path = self._get_remote_path(version)

        try:
            if adapter_path.is_dir():
                # Upload directory
                self.hf_api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(adapter_path),
                    path_in_repo=remote_path,
                    commit_message=f"Add adapter version {version}",
                )
            else:
                # Upload single file
                self.hf_api.upload_file(
                    repo_id=self.repo_id,
                    path_or_fileobj=str(adapter_path),
                    path_in_repo=f"{remote_path}/{adapter_path.name}",
                    commit_message=f"Add adapter version {version}",
                )

            logger.info(f"Uploaded adapter to HuggingFace Hub: {remote_path}")

        except Exception as e:
            logger.error(f"Failed to upload adapter to HuggingFace Hub: {e}")
            raise

        # Save metadata to Supabase
        meta_data = {
            "version": version,
            "created_at": metadata.created_at.isoformat(),
            "base_model": metadata.base_model,
            "dataset_size": metadata.dataset_size,
            "status": metadata.status,
            "training_config": metadata.training_config,
            "adapter_path": f"{self.repo_id}/{remote_path}",
        }

        self.db_client.upsert(MODEL_VERSIONS_TABLE, meta_data, on_conflict="version")
        logger.info(f"Saved model version metadata: {version}")

        return version

    def get_version(self, version: str) -> Optional[ModelVersionRecord]:
        """Get version metadata."""
        data = self.db_client.select_one(
            MODEL_VERSIONS_TABLE,
            filters={"version": version},
        )

        if not data:
            return None

        return self._record_from_data(data)

    def _record_from_data(self, data: dict) -> ModelVersionRecord:
        """Convert database record to ModelVersionRecord."""
        return ModelVersionRecord(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            base_model=data["base_model"],
            dataset_size=data["dataset_size"],
            status=data["status"],
            training_config=data.get("training_config", {}),
            adapter_path=data.get("adapter_path"),
        )

    def list_versions(self) -> list[ModelVersionRecord]:
        """List all versions."""
        data = self.db_client.select(
            MODEL_VERSIONS_TABLE,
            order="created_at",
            desc=True,
        )

        return [self._record_from_data(d) for d in data]

    def get_active_version(self) -> Optional[str]:
        """Get the currently active version."""
        data = self.db_client.select_one(
            MODEL_VERSIONS_TABLE,
            filters={"status": "active"},
        )

        return data["version"] if data else None

    def set_active_version(self, version: str) -> bool:
        """Set the active version."""
        # Verify version exists
        if not self.get_version(version):
            logger.error(f"Version {version} does not exist")
            return False

        # Deactivate all other versions
        active_versions = self.db_client.select(
            MODEL_VERSIONS_TABLE,
            filters={"status": "active"},
        )

        for v in active_versions:
            if v["version"] != version:
                self.db_client.update(
                    MODEL_VERSIONS_TABLE,
                    {"status": "inactive"},
                    {"version": v["version"]},
                )

        # Activate the new version
        self.db_client.update(
            MODEL_VERSIONS_TABLE,
            {"status": "active"},
            {"version": version},
        )

        logger.info(f"Activated version: {version}")
        return True

    def get_adapter_path(self, version: str) -> Optional[Path]:
        """Get the path to adapter files for a version.

        Downloads from HuggingFace Hub if not cached locally.
        """
        # Check cache first
        cached_path = self.cache_dir / version
        if cached_path.exists():
            return cached_path

        # Get version info
        version_info = self.get_version(version)
        if not version_info:
            return None

        # Download from HuggingFace Hub
        remote_path = self._get_remote_path(version)

        try:
            from huggingface_hub import snapshot_download

            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=[f"{remote_path}/*"],
                local_dir=str(self.cache_dir / "temp"),
                token=self.hf_token,
            )

            # Move to cache
            source = Path(downloaded_path) / remote_path
            if source.exists():
                if cached_path.exists():
                    shutil.rmtree(cached_path)
                shutil.move(str(source), str(cached_path))

            # Cleanup temp
            temp_dir = self.cache_dir / "temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            logger.info(f"Downloaded adapter from HuggingFace Hub: {version}")
            return cached_path

        except Exception as e:
            logger.error(f"Failed to download adapter from HuggingFace Hub: {e}")
            return None

    def update_status(self, version: str, status: str) -> bool:
        """Update version status."""
        result = self.db_client.update(
            MODEL_VERSIONS_TABLE,
            {"status": status},
            {"version": version},
        )

        if result:
            logger.info(f"Updated version {version} status to {status}")
            return True
        return False

    def delete_version(self, version: str) -> bool:
        """Delete a version."""
        # Don't delete active version
        if self.get_active_version() == version:
            logger.warning(f"Cannot delete active version: {version}")
            return False

        # Delete from HuggingFace Hub
        remote_path = self._get_remote_path(version)
        try:
            self.hf_api.delete_folder(
                repo_id=self.repo_id,
                path_in_repo=remote_path,
                commit_message=f"Delete adapter version {version}",
            )
            logger.info(f"Deleted adapter from HuggingFace Hub: {version}")
        except Exception as e:
            logger.warning(f"Failed to delete from HuggingFace Hub: {e}")
            # Continue to delete metadata anyway

        # Delete metadata from Supabase
        self.db_client.delete(
            MODEL_VERSIONS_TABLE,
            {"version": version},
        )

        # Delete from cache
        cached_path = self.cache_dir / version
        if cached_path.exists():
            shutil.rmtree(cached_path)

        logger.info(f"Deleted version: {version}")
        return True

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
