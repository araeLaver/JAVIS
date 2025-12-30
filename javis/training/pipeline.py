"""End-to-end training pipeline orchestration."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.utils.config import TrainingConfig, get_config

from .notifications import NotificationService, get_notifier
from .remote import RemoteTrainer, TrainingResult
from .version_manager import VersionManager, get_version_manager

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    """Result of a pipeline run."""

    success: bool
    version: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    duration_seconds: float = 0
    dataset_size: int = 0


class DataStats(BaseModel):
    """Statistics about available training data."""

    total_conversations: int = 0
    good_feedback: int = 0
    bad_feedback: int = 0
    no_feedback: int = 0
    ready_for_training: int = 0


class TrainingPipeline:
    """Orchestrates the complete training workflow."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        conversations_dir: Optional[Path] = None,
    ):
        if config is None:
            config = get_config().training
        self.config = config

        if conversations_dir is None:
            conversations_dir = (
                Path(__file__).parent.parent.parent / "data" / "conversations"
            )
        self.conversations_dir = Path(conversations_dir)

        self.remote_trainer = RemoteTrainer(config.provider)
        self.version_manager = get_version_manager()
        self.notifier = get_notifier()

    def get_data_stats(self) -> DataStats:
        """Get statistics about available training data."""
        stats = DataStats()

        if not self.conversations_dir.exists():
            return stats

        for json_file in self.conversations_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                # Check minimum turns
                if len(conv.get("turns", [])) < 2:
                    continue

                stats.total_conversations += 1
                feedback = conv.get("feedback")

                if feedback == "good":
                    stats.good_feedback += 1
                elif feedback == "bad":
                    stats.bad_feedback += 1
                else:
                    stats.no_feedback += 1

            except (json.JSONDecodeError, IOError):
                continue

        # Ready = good + no feedback (if not excluding bad)
        if self.config.data.exclude_bad_feedback:
            stats.ready_for_training = stats.good_feedback + stats.no_feedback
        else:
            stats.ready_for_training = stats.total_conversations

        return stats

    def check_conditions(self) -> tuple[bool, str]:
        """Check if training conditions are met.

        Returns:
            (can_train, reason) tuple
        """
        stats = self.get_data_stats()

        # Check minimum conversations
        if stats.ready_for_training < self.config.data.min_conversations:
            return (
                False,
                f"Not enough data: {stats.ready_for_training}/{self.config.data.min_conversations} conversations",
            )

        # Check minimum good feedback
        if stats.good_feedback < self.config.data.min_good_feedback:
            return (
                False,
                f"Not enough good feedback: {stats.good_feedback}/{self.config.data.min_good_feedback}",
            )

        return True, "All conditions met"

    def export_training_data(self, output_path: Optional[Path] = None) -> Path:
        """Export conversations to JSONL format for training.

        Args:
            output_path: Output file path (default: auto-generated)

        Returns:
            Path to the exported JSONL file
        """
        if output_path is None:
            export_dir = (
                Path(__file__).parent.parent.parent / "data" / "training" / "exported"
            )
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                export_dir
                / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )

        conversations = []
        cutoff_date = None
        if self.config.data.max_age_days > 0:
            cutoff_date = datetime.now() - timedelta(days=self.config.data.max_age_days)

        for json_file in self.conversations_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                # Skip if not enough turns
                if len(conv.get("turns", [])) < 2:
                    continue

                # Skip bad feedback if configured
                if (
                    self.config.data.exclude_bad_feedback
                    and conv.get("feedback") == "bad"
                ):
                    continue

                # Skip old conversations
                if cutoff_date:
                    started_at = conv.get("started_at", "")
                    if started_at:
                        try:
                            conv_date = datetime.fromisoformat(started_at)
                            if conv_date < cutoff_date:
                                continue
                        except ValueError:
                            pass

                # Extract messages
                messages = []
                for turn in conv.get("turns", []):
                    messages.append(
                        {"role": turn["role"], "content": turn["content"]}
                    )

                conversations.append({"messages": messages})

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {json_file}: {e}")
                continue

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(conversations)} conversations to {output_path}")
        return output_path

    def validate_model(self, adapter_path: Path) -> bool:
        """Validate the trained model with test prompts.

        Args:
            adapter_path: Path to the adapter directory

        Returns:
            True if validation passes
        """
        if not self.config.deployment.validation_required:
            return True

        # Check adapter files exist
        required_files = ["adapter_config.json"]
        for f in required_files:
            if not (adapter_path / f).exists():
                logger.error(f"Missing required file: {f}")
                return False

        # TODO: Add actual model loading and inference validation
        # For now, just check files exist
        logger.info("Model validation passed (file check only)")
        return True

    def run(self, force: bool = False) -> PipelineResult:
        """Execute the full training pipeline.

        Args:
            force: Skip condition checks and run anyway

        Returns:
            PipelineResult with status and version info
        """
        start_time = datetime.now()
        logger.info("Starting training pipeline")

        try:
            # Phase 1: Check conditions
            if not force:
                can_train, reason = self.check_conditions()
                if not can_train:
                    logger.info(f"Skipping training: {reason}")
                    return PipelineResult(
                        success=True, skipped=True, skip_reason=reason
                    )

            # Phase 2: Export training data
            logger.info("Exporting training data...")
            data_path = self.export_training_data()

            # Count lines for dataset size
            with open(data_path, "r", encoding="utf-8") as f:
                dataset_size = sum(1 for _ in f)

            if dataset_size == 0:
                return PipelineResult(
                    success=False,
                    error="No training data after export",
                )

            # Phase 3: Run training
            logger.info(f"Starting remote training with {dataset_size} conversations...")
            training_config = {
                "base_model": self.config.model.base_model,
                "epochs": self.config.model.epochs,
                "batch_size": self.config.model.batch_size,
                "learning_rate": self.config.model.learning_rate,
                "lora_r": self.config.model.lora_r,
                "lora_alpha": self.config.model.lora_alpha,
                "max_seq_length": self.config.model.max_seq_length,
                "gradient_accumulation_steps": self.config.model.gradient_accumulation_steps,
            }

            result = self.remote_trainer.train(data_path, training_config)

            if not result.success:
                logger.error(f"Training failed: {result.error}")
                self.notifier.notify_failure(result.error or "Unknown error", {
                    "version": result.version,
                    "dataset_size": dataset_size,
                })
                return PipelineResult(
                    success=False,
                    version=result.version,
                    error=result.error,
                    dataset_size=dataset_size,
                )

            # Phase 4: Validate model
            logger.info("Validating trained model...")
            if result.adapter_path and not self.validate_model(result.adapter_path):
                logger.error("Model validation failed")
                self.version_manager.mark_failed(result.version)
                self.notifier.notify_failure("Model validation failed", {
                    "version": result.version,
                })
                return PipelineResult(
                    success=False,
                    version=result.version,
                    error="Model validation failed",
                    dataset_size=dataset_size,
                )

            # Phase 5: Create version
            logger.info(f"Creating version {result.version}...")
            if result.adapter_path:
                metadata = result.metadata.copy()
                metadata["dataset_size"] = dataset_size
                metadata["training_config"] = training_config

                self.version_manager.create_version(
                    result.adapter_path, metadata, result.version
                )

            # Phase 6: Deploy if auto_deploy enabled
            if self.config.deployment.auto_deploy:
                logger.info(f"Activating version {result.version}...")
                self.version_manager.activate_version(result.version)

                # Cleanup old versions
                removed = self.version_manager.cleanup_old_versions(
                    keep=self.config.deployment.keep_versions
                )
                if removed:
                    logger.info(f"Cleaned up old versions: {removed}")

            # Phase 7: Notify success
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.notifier.notify_success(result.version, {
                "dataset_size": dataset_size,
                "duration_seconds": duration,
                "auto_deployed": self.config.deployment.auto_deploy,
            })

            logger.info(f"Pipeline complete: version {result.version}")
            return PipelineResult(
                success=True,
                version=result.version,
                duration_seconds=duration,
                dataset_size=dataset_size,
            )

        except Exception as e:
            logger.exception("Pipeline failed with exception")
            self.notifier.notify_failure(str(e), {})
            return PipelineResult(
                success=False,
                error=str(e),
            )


def run_pipeline(force: bool = False) -> PipelineResult:
    """Convenience function to run the training pipeline."""
    pipeline = TrainingPipeline()
    return pipeline.run(force=force)
