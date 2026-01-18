#!/usr/bin/env python3
"""Migration script to move local JAVIS data to cloud storage (Supabase + HuggingFace Hub).

Usage:
    python scripts/migrate_to_cloud.py [--dry-run] [--conversations] [--feedback] [--corrections] [--models]

Options:
    --dry-run       Show what would be migrated without actually migrating
    --conversations Migrate only conversations
    --feedback      Migrate only feedback
    --corrections   Migrate only corrections
    --models        Migrate only model versions

Environment Variables Required:
    SUPABASE_URL            Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY   Supabase service role key
    HF_TOKEN                HuggingFace API token (for model migration)
    HF_REPO_ID              HuggingFace repository ID (for model migration)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set storage mode to cloud before imports
os.environ["JAVIS_STORAGE_MODE"] = "cloud"

from javis.storage.cloud.supabase_client import get_supabase_client
from javis.storage.cloud.conversation_store import CloudConversationStore
from javis.storage.cloud.feedback_store import CloudFeedbackStore
from javis.storage.cloud.correction_store import CloudCorrectionStore
from javis.storage.cloud.hf_model_store import HuggingFaceModelStore
from javis.storage.base import (
    ConversationRecord,
    ConversationTurn,
    FeedbackRecord,
    CorrectionRecord,
    ModelVersionRecord,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.conversations_migrated = 0
        self.conversations_failed = 0
        self.feedback_migrated = 0
        self.feedback_failed = 0
        self.corrections_migrated = 0
        self.corrections_failed = 0
        self.models_migrated = 0
        self.models_failed = 0

    def summary(self) -> str:
        return f"""
Migration Summary:
==================
Conversations: {self.conversations_migrated} migrated, {self.conversations_failed} failed
Feedback:      {self.feedback_migrated} migrated, {self.feedback_failed} failed
Corrections:   {self.corrections_migrated} migrated, {self.corrections_failed} failed
Models:        {self.models_migrated} migrated, {self.models_failed} failed
"""


def migrate_conversations(data_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Migrate conversations from local files to Supabase.

    Args:
        data_dir: Path to the data directory
        dry_run: If True, don't actually migrate

    Returns:
        Tuple of (migrated_count, failed_count)
    """
    conversations_dir = data_dir / "conversations"
    if not conversations_dir.exists():
        logger.warning(f"Conversations directory not found: {conversations_dir}")
        return 0, 0

    migrated = 0
    failed = 0

    store = CloudConversationStore() if not dry_run else None

    for json_file in conversations_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to ConversationRecord
            record = ConversationRecord(
                session_id=data.get("session_id", json_file.stem),
                started_at=data.get("started_at", datetime.now().isoformat()),
                turns=[
                    ConversationTurn(
                        role=t.get("role", "user"),
                        content=t.get("content", ""),
                        timestamp=t.get("timestamp", datetime.now().isoformat()),
                    )
                    for t in data.get("turns", [])
                ],
                feedback=data.get("feedback"),
                tags=data.get("tags", []),
            )

            if dry_run:
                logger.info(f"[DRY-RUN] Would migrate conversation: {record.session_id}")
            else:
                store.save(record)
                logger.info(f"Migrated conversation: {record.session_id}")

            migrated += 1

        except Exception as e:
            logger.error(f"Failed to migrate {json_file}: {e}")
            failed += 1

    return migrated, failed


def migrate_feedback(data_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Migrate feedback from local files to Supabase.

    Args:
        data_dir: Path to the data directory
        dry_run: If True, don't actually migrate

    Returns:
        Tuple of (migrated_count, failed_count)
    """
    feedback_dir = data_dir / "feedback"
    if not feedback_dir.exists():
        logger.warning(f"Feedback directory not found: {feedback_dir}")
        return 0, 0

    migrated = 0
    failed = 0

    store = CloudFeedbackStore() if not dry_run else None

    for feedback_type in ["good", "bad", "corrected"]:
        type_dir = feedback_dir / feedback_type
        if not type_dir.exists():
            continue

        for json_file in type_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert to FeedbackRecord
                record = FeedbackRecord(
                    id=data.get("id", json_file.stem),
                    session_id=data.get("session_id", "unknown"),
                    conversation_id=data.get("conversation_id"),
                    timestamp=datetime.fromisoformat(data["timestamp"])
                    if isinstance(data.get("timestamp"), str)
                    else data.get("timestamp", datetime.now()),
                    feedback_type=feedback_type,
                    quality_score=data.get("quality_score"),
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    corrected_response=data.get("corrected_response"),
                    metadata=data.get("metadata", {}),
                )

                if dry_run:
                    logger.info(f"[DRY-RUN] Would migrate feedback: {record.id}")
                else:
                    store.save(record)
                    logger.info(f"Migrated feedback: {record.id}")

                migrated += 1

            except Exception as e:
                logger.error(f"Failed to migrate {json_file}: {e}")
                failed += 1

    return migrated, failed


def migrate_corrections(data_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Migrate corrections from local files to Supabase.

    Args:
        data_dir: Path to the data directory
        dry_run: If True, don't actually migrate

    Returns:
        Tuple of (migrated_count, failed_count)
    """
    corrections_dir = data_dir / "corrections"
    if not corrections_dir.exists():
        logger.warning(f"Corrections directory not found: {corrections_dir}")
        return 0, 0

    migrated = 0
    failed = 0

    store = CloudCorrectionStore() if not dry_run else None

    for json_file in corrections_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to CorrectionRecord
            record = CorrectionRecord(
                id=data.get("id", json_file.stem),
                session_id=data.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(data["timestamp"])
                if isinstance(data.get("timestamp"), str)
                else data.get("timestamp", datetime.now()),
                original_prompt=data.get("original_prompt", ""),
                original_response=data.get("original_response", ""),
                corrected_response=data.get("corrected_response", ""),
                correction_type=data.get("correction_type", "other"),
                notes=data.get("notes"),
                metadata=data.get("metadata", {}),
            )

            if dry_run:
                logger.info(f"[DRY-RUN] Would migrate correction: {record.id}")
            else:
                store.save(record)
                logger.info(f"Migrated correction: {record.id}")

            migrated += 1

        except Exception as e:
            logger.error(f"Failed to migrate {json_file}: {e}")
            failed += 1

    return migrated, failed


def migrate_models(models_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Migrate model versions from local files to HuggingFace Hub.

    Args:
        models_dir: Path to the models directory
        dry_run: If True, don't actually migrate

    Returns:
        Tuple of (migrated_count, failed_count)
    """
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return 0, 0

    migrated = 0
    failed = 0

    store = HuggingFaceModelStore() if not dry_run else None

    for version_dir in models_dir.iterdir():
        if not version_dir.is_dir() or version_dir.name.startswith("."):
            continue

        try:
            metadata_path = version_dir / "metadata.json"
            adapter_path = version_dir / "adapter"

            if not adapter_path.exists():
                logger.warning(f"No adapter found for version {version_dir.name}")
                continue

            # Load or create metadata
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {
                    "version": version_dir.name,
                    "created_at": datetime.fromtimestamp(
                        version_dir.stat().st_mtime
                    ).isoformat(),
                    "base_model": "unknown",
                    "dataset_size": 0,
                    "status": "ready",
                    "training_config": {},
                }

            # Convert to ModelVersionRecord
            record = ModelVersionRecord(
                version=data.get("version", version_dir.name),
                created_at=datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now()),
                base_model=data.get("base_model", "unknown"),
                dataset_size=data.get("dataset_size", 0),
                status=data.get("status", "ready"),
                training_config=data.get("training_config", {}),
            )

            if dry_run:
                logger.info(f"[DRY-RUN] Would migrate model: {record.version}")
            else:
                store.save_version(
                    version=record.version,
                    adapter_path=adapter_path,
                    metadata=record,
                )
                logger.info(f"Migrated model: {record.version}")

            migrated += 1

        except Exception as e:
            logger.error(f"Failed to migrate model {version_dir.name}: {e}")
            failed += 1

    return migrated, failed


def check_environment() -> bool:
    """Check if required environment variables are set.

    Returns:
        True if all required variables are set
    """
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.info("Please set these in your .env file or environment")
        return False

    # Check optional HuggingFace variables for model migration
    hf_vars = ["HF_TOKEN", "HF_REPO_ID"]
    hf_missing = [var for var in hf_vars if not os.environ.get(var)]
    if hf_missing:
        logger.warning(
            f"HuggingFace variables not set ({', '.join(hf_missing)}). "
            "Model migration will be skipped."
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate JAVIS data from local storage to cloud (Supabase + HuggingFace Hub)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating",
    )
    parser.add_argument(
        "--conversations",
        action="store_true",
        help="Migrate only conversations",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Migrate only feedback",
    )
    parser.add_argument(
        "--corrections",
        action="store_true",
        help="Migrate only corrections",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Migrate only model versions",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_root / "models",
        help="Path to the models directory",
    )

    args = parser.parse_args()

    # Load .env file if present
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")

    # Check environment
    if not args.dry_run and not check_environment():
        sys.exit(1)

    # Determine what to migrate
    migrate_all = not any([args.conversations, args.feedback, args.corrections, args.models])

    stats = MigrationStats()

    logger.info("=" * 60)
    logger.info("JAVIS Data Migration")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be actually migrated")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info("=" * 60)

    # Migrate conversations
    if migrate_all or args.conversations:
        logger.info("\n--- Migrating Conversations ---")
        m, f = migrate_conversations(args.data_dir, args.dry_run)
        stats.conversations_migrated = m
        stats.conversations_failed = f

    # Migrate feedback
    if migrate_all or args.feedback:
        logger.info("\n--- Migrating Feedback ---")
        m, f = migrate_feedback(args.data_dir, args.dry_run)
        stats.feedback_migrated = m
        stats.feedback_failed = f

    # Migrate corrections
    if migrate_all or args.corrections:
        logger.info("\n--- Migrating Corrections ---")
        m, f = migrate_corrections(args.data_dir, args.dry_run)
        stats.corrections_migrated = m
        stats.corrections_failed = f

    # Migrate models
    if migrate_all or args.models:
        if os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO_ID"):
            logger.info("\n--- Migrating Models to HuggingFace Hub ---")
            m, f = migrate_models(args.models_dir, args.dry_run)
            stats.models_migrated = m
            stats.models_failed = f
        else:
            logger.warning("Skipping model migration - HuggingFace credentials not set")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info(stats.summary())

    # Exit with error code if any failures
    total_failed = (
        stats.conversations_failed
        + stats.feedback_failed
        + stats.corrections_failed
        + stats.models_failed
    )
    if total_failed > 0:
        logger.warning(f"Migration completed with {total_failed} failures")
        sys.exit(1)
    else:
        logger.info("Migration completed successfully!")


if __name__ == "__main__":
    main()
