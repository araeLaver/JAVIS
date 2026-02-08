"""End-to-end training pipeline orchestration."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.storage import get_conversation_store
from javis.storage.base import ConversationStoreInterface
from javis.utils.config import TrainingConfig, get_config

from .ab_testing import get_ab_test_manager
from .data_quality import get_quality_scorer
from .mlflow_tracker import MLflowConfig, get_mlflow_tracker
from .notifications import get_notifier
from .preference_data import get_preference_generator
from .remote import RemoteTrainer
from .validation import get_validator
from .version_manager import get_version_manager

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
    high_quality: int = 0  # Conversations meeting quality threshold
    avg_quality_score: float = 0.0


class TrainingPipeline:
    """Orchestrates the complete training workflow.

    Uses the storage factory to access conversations from either local files
    or cloud storage (Supabase) depending on configuration.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        conversations_dir: Optional[Path] = None,
        conversation_store: Optional[ConversationStoreInterface] = None,
    ):
        if config is None:
            config = get_config().training
        self.config = config

        # Store reference (lazy initialization)
        self._conversation_store = conversation_store

        # Keep conversations_dir for backward compatibility
        if conversations_dir is None:
            conversations_dir = (
                Path(__file__).parent.parent.parent / "data" / "conversations"
            )
        self.conversations_dir = Path(conversations_dir)

        self.remote_trainer = RemoteTrainer(config.provider)
        self.version_manager = get_version_manager()
        self.notifier = get_notifier()
        self.quality_scorer = get_quality_scorer()
        self.model_validator = get_validator()
        self.ab_test_manager = get_ab_test_manager()
        self.preference_generator = get_preference_generator()

        # MLflow tracker (MLOps)
        mlflow_config = MLflowConfig(
            enabled=config.mlflow.enabled,
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name=config.mlflow.experiment_name,
            artifact_location=config.mlflow.artifact_location,
            log_models=config.mlflow.log_models,
            log_datasets=config.mlflow.log_datasets,
            log_system_metrics=config.mlflow.log_system_metrics,
        )
        self.mlflow_tracker = get_mlflow_tracker(mlflow_config)

    @property
    def conversation_store(self) -> ConversationStoreInterface:
        """Get the conversation store (lazy initialization)."""
        if self._conversation_store is None:
            self._conversation_store = get_conversation_store()
        return self._conversation_store

    def get_data_stats(self) -> DataStats:
        """Get statistics about available training data."""
        stats = DataStats()
        quality_scores: list[float] = []
        min_quality = self.config.data_quality.min_score

        # Get all conversations from storage
        conversations = self.conversation_store.list_all()

        for conv in conversations:
            try:
                # Check minimum turns
                if len(conv.turns) < 2:
                    continue

                stats.total_conversations += 1
                feedback = conv.feedback

                if feedback == "good":
                    stats.good_feedback += 1
                elif feedback == "bad":
                    stats.bad_feedback += 1
                else:
                    stats.no_feedback += 1

                # Calculate quality score
                conv_dict = {
                    "session_id": conv.session_id,
                    "started_at": conv.started_at,
                    "turns": [
                        {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                        for t in conv.turns
                    ],
                    "feedback": conv.feedback,
                    "tags": conv.tags,
                }
                quality = self.quality_scorer.score_conversation(conv_dict)
                quality_scores.append(quality.overall_score)

                if quality.overall_score >= min_quality:
                    stats.high_quality += 1

            except Exception:
                continue

        # Ready = good + no feedback (if not excluding bad)
        if self.config.data.exclude_bad_feedback:
            stats.ready_for_training = stats.good_feedback + stats.no_feedback
        else:
            stats.ready_for_training = stats.total_conversations

        # Calculate average quality score
        if quality_scores:
            stats.avg_quality_score = round(
                sum(quality_scores) / len(quality_scores), 2
            )

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

    def export_training_data(
        self,
        output_path: Optional[Path] = None,
        use_quality_filter: bool = True,
    ) -> Path:
        """Export conversations to JSONL format for training.

        Args:
            output_path: Output file path (default: auto-generated)
            use_quality_filter: Apply quality score filtering

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

        exported_conversations = []
        cutoff_date = None
        if self.config.data.max_age_days > 0:
            cutoff_date = datetime.now() - timedelta(days=self.config.data.max_age_days)

        min_quality = self.config.data_quality.min_score
        filtered_by_quality = 0

        # Get all conversations from storage
        conversations = self.conversation_store.list_all()

        for conv in conversations:
            try:
                # Skip if not enough turns
                if len(conv.turns) < 2:
                    continue

                # Skip bad feedback if configured
                if (
                    self.config.data.exclude_bad_feedback
                    and conv.feedback == "bad"
                ):
                    continue

                # Skip old conversations
                if cutoff_date:
                    started_at = conv.started_at
                    if started_at:
                        try:
                            conv_date = datetime.fromisoformat(started_at)
                            if conv_date < cutoff_date:
                                continue
                        except ValueError:
                            pass

                # Quality filter
                if use_quality_filter:
                    conv_dict = {
                        "session_id": conv.session_id,
                        "started_at": conv.started_at,
                        "turns": [
                            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                            for t in conv.turns
                        ],
                        "feedback": conv.feedback,
                        "tags": conv.tags,
                    }
                    quality = self.quality_scorer.score_conversation(conv_dict)
                    if quality.overall_score < min_quality:
                        filtered_by_quality += 1
                        continue

                # Extract messages
                messages = []
                for turn in conv.turns:
                    messages.append(
                        {"role": turn.role, "content": turn.content}
                    )

                exported_conversations.append({"messages": messages})

            except Exception as e:
                logger.warning(f"Failed to process conversation {conv.session_id}: {e}")
                continue

        # Write JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in exported_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(
            f"Exported {len(exported_conversations)} conversations to {output_path} "
            f"(filtered {filtered_by_quality} by quality)"
        )
        return output_path

    def validate_model(
        self, adapter_path: Path, version: Optional[str] = None
    ) -> bool:
        """Validate the trained model with test prompts.

        Args:
            adapter_path: Path to the adapter directory
            version: Version string for logging

        Returns:
            True if validation passes
        """
        if not self.config.deployment.validation_required:
            return True

        # Use enhanced ModelValidator
        use_inference = self.config.validation.inference_enabled

        if use_inference:
            # Full validation with inference testing
            logger.info("Running model validation with inference testing...")
            result = self.model_validator.validate_with_inference(
                adapter_path=adapter_path,
                base_model=self.config.model.base_model,
            )
        else:
            # File check only
            logger.info("Running model validation (file check only)...")
            result = self.model_validator.validate_files_only(adapter_path)

        if result.passed:
            logger.info(
                f"Model validation passed: {result.passed_tests}/{result.total_tests} tests "
                f"(score: {result.quality_score:.2f})"
            )
            return True
        else:
            logger.error(f"Model validation failed: {result.error_message}")
            for test in result.test_results:
                if not test.get("passed", False):
                    logger.error(f"  - Failed: {test.get('name', 'unknown')}")
            return False

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

            # Phase 3: Run training with MLflow tracking
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

            # Start MLflow run for experiment tracking
            run_name = f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            with self.mlflow_tracker.start_run(
                run_name=run_name,
                tags={"method": "sft", "provider": self.config.provider},
                description=f"SFT training with {dataset_size} conversations",
            ) as mlflow_run_id:
                # Log training configuration
                self.mlflow_tracker.log_training_config(training_config)
                self.mlflow_tracker.log_dataset(data_path, "training_data", "training")
                self.mlflow_tracker.log_metrics({"dataset_size": dataset_size})

                result = self.remote_trainer.train(data_path, training_config)

                if not result.success:
                    logger.error(f"Training failed: {result.error}")
                    self.mlflow_tracker.set_tag("status", "failed")
                    self.mlflow_tracker.log_params({"error": result.error or "Unknown"})
                    self.notifier.notify_failure(result.error or "Unknown error", {
                        "version": result.version,
                        "dataset_size": dataset_size,
                        "mlflow_run_id": mlflow_run_id,
                    })
                    return PipelineResult(
                        success=False,
                        version=result.version,
                        error=result.error,
                        dataset_size=dataset_size,
                    )

                # Log training metrics from result
                if result.metadata:
                    metrics_to_log = {
                        k: v for k, v in result.metadata.items()
                        if isinstance(v, (int, float))
                    }
                    if metrics_to_log:
                        self.mlflow_tracker.log_metrics(metrics_to_log)

                # Phase 4: Validate model
                logger.info("Validating trained model...")
                if result.adapter_path and not self.validate_model(
                    result.adapter_path, result.version
                ):
                    logger.error("Model validation failed")
                    self.mlflow_tracker.set_tag("status", "validation_failed")
                    self.version_manager.mark_failed(result.version)
                    self.notifier.notify_failure("Model validation failed", {
                        "version": result.version,
                        "mlflow_run_id": mlflow_run_id,
                    })
                    return PipelineResult(
                        success=False,
                        version=result.version,
                        error="Model validation failed",
                        dataset_size=dataset_size,
                    )

                # Log model artifacts
                if result.adapter_path:
                    self.mlflow_tracker.log_model(
                        result.adapter_path,
                        model_name="lora_adapter",
                        registered_name=f"javis-{result.version}" if self.config.mlflow.log_models else None,
                    )

                self.mlflow_tracker.set_tag("status", "success")
                self.mlflow_tracker.log_params({"version": result.version})

            # Phase 5: Create version
            logger.info(f"Creating version {result.version}...")
            if result.adapter_path:
                metadata = result.metadata.copy()
                metadata["dataset_size"] = dataset_size
                metadata["training_config"] = training_config
                if mlflow_run_id:
                    metadata["mlflow_run_id"] = mlflow_run_id

                self.version_manager.create_version(
                    result.adapter_path, metadata, result.version
                )

            # Phase 6: Deploy or A/B test
            ab_test_id = None
            current_version = self.version_manager.get_active_version()

            if self.config.ab_testing.auto_create and current_version:
                # Create A/B test between current and new version
                logger.info(
                    f"Creating A/B test: {current_version.version} vs {result.version}"
                )
                ab_test_id = self.ab_test_manager.create_test(
                    version_a=current_version.version,
                    version_b=result.version,
                    description=f"Auto-created: comparing {current_version.version} with {result.version}",
                )
                logger.info(f"A/B test created: {ab_test_id}")

            elif self.config.deployment.auto_deploy:
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

            notify_metadata = {
                "dataset_size": dataset_size,
                "duration_seconds": duration,
                "auto_deployed": self.config.deployment.auto_deploy
                and not ab_test_id,
            }
            if ab_test_id:
                notify_metadata["ab_test_id"] = ab_test_id
                notify_metadata["ab_test_enabled"] = True
            if mlflow_run_id:
                notify_metadata["mlflow_run_id"] = mlflow_run_id
                notify_metadata["mlflow_url"] = self.mlflow_tracker.get_run_url()

            self.notifier.notify_success(result.version, notify_metadata)

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

    def check_dpo_conditions(self) -> tuple[bool, str]:
        """Check if DPO training conditions are met.

        Returns:
            (can_train, reason) tuple
        """
        stats = self.preference_generator.get_statistics()
        min_pairs = self.config.dpo.min_preference_pairs

        if stats.total_pairs < min_pairs:
            return (
                False,
                f"Not enough preference data: {stats.total_pairs}/{min_pairs} pairs",
            )

        return True, "All DPO conditions met"

    def run_dpo(self, force: bool = False) -> PipelineResult:
        """Execute DPO training pipeline.

        Args:
            force: Skip condition checks and run anyway

        Returns:
            PipelineResult with status and version info
        """
        start_time = datetime.now()
        logger.info("Starting DPO training pipeline")

        try:
            # Phase 1: Check conditions
            if not force:
                can_train, reason = self.check_dpo_conditions()
                if not can_train:
                    logger.info(f"Skipping DPO training: {reason}")
                    return PipelineResult(
                        success=True, skipped=True, skip_reason=reason
                    )

            # Phase 2: Export preference data
            logger.info("Exporting preference data...")
            pairs = self.preference_generator.generate_all()
            data_path = self.preference_generator.export_dpo_dataset(pairs=pairs)

            dataset_size = len(pairs)
            if dataset_size == 0:
                return PipelineResult(
                    success=False,
                    error="No preference data after export",
                )

            # Phase 3: Run DPO training
            logger.info(f"Starting remote DPO training with {dataset_size} pairs...")
            dpo_config = self.config.dpo
            training_config = {
                "base_model": self.config.model.base_model,
                "epochs": self.config.model.epochs,
                "batch_size": self.config.model.batch_size,
                "learning_rate": self.config.model.learning_rate,
                "lora_r": self.config.model.lora_r,
                "lora_alpha": self.config.model.lora_alpha,
                "beta": dpo_config.beta,
                "loss_type": dpo_config.loss_type,
                "max_prompt_length": dpo_config.max_prompt_length,
                "max_length": dpo_config.max_response_length,
                "method": "dpo",
            }

            result = self.remote_trainer.train(data_path, training_config)

            if not result.success:
                logger.error(f"DPO Training failed: {result.error}")
                self.notifier.notify_failure(result.error or "Unknown error", {
                    "method": "dpo",
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
            logger.info("Validating DPO-trained model...")
            if result.adapter_path and not self.validate_model(
                result.adapter_path, result.version
            ):
                logger.error("DPO model validation failed")
                self.version_manager.mark_failed(result.version)
                self.notifier.notify_failure("DPO model validation failed", {
                    "method": "dpo",
                    "version": result.version,
                })
                return PipelineResult(
                    success=False,
                    version=result.version,
                    error="DPO model validation failed",
                    dataset_size=dataset_size,
                )

            # Phase 5: Create version
            logger.info(f"Creating DPO version {result.version}...")
            if result.adapter_path:
                metadata = result.metadata.copy()
                metadata["dataset_size"] = dataset_size
                metadata["training_config"] = training_config
                metadata["training_method"] = "dpo"

                self.version_manager.create_version(
                    result.adapter_path, metadata, result.version
                )

            # Phase 6: Deploy or A/B test
            ab_test_id = None
            current_version = self.version_manager.get_active_version()

            if self.config.ab_testing.auto_create and current_version:
                logger.info(
                    f"Creating A/B test: {current_version.version} vs {result.version} (DPO)"
                )
                ab_test_id = self.ab_test_manager.create_test(
                    version_a=current_version.version,
                    version_b=result.version,
                    description=f"Auto-created: DPO {result.version} vs {current_version.version}",
                )
                logger.info(f"A/B test created: {ab_test_id}")

            elif self.config.deployment.auto_deploy:
                logger.info(f"Activating DPO version {result.version}...")
                self.version_manager.activate_version(result.version)

                removed = self.version_manager.cleanup_old_versions(
                    keep=self.config.deployment.keep_versions
                )
                if removed:
                    logger.info(f"Cleaned up old versions: {removed}")

            # Phase 7: Notify success
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            notify_metadata = {
                "method": "dpo",
                "dataset_size": dataset_size,
                "duration_seconds": duration,
                "auto_deployed": self.config.deployment.auto_deploy
                and not ab_test_id,
            }
            if ab_test_id:
                notify_metadata["ab_test_id"] = ab_test_id
                notify_metadata["ab_test_enabled"] = True

            self.notifier.notify_success(result.version, notify_metadata)

            logger.info(f"DPO Pipeline complete: version {result.version}")
            return PipelineResult(
                success=True,
                version=result.version,
                duration_seconds=duration,
                dataset_size=dataset_size,
            )

        except Exception as e:
            logger.exception("DPO Pipeline failed with exception")
            self.notifier.notify_failure(str(e), {"method": "dpo"})
            return PipelineResult(
                success=False,
                error=str(e),
            )


def run_pipeline(force: bool = False) -> PipelineResult:
    """Convenience function to run the training pipeline."""
    pipeline = TrainingPipeline()
    return pipeline.run(force=force)


def run_dpo_pipeline(force: bool = False) -> PipelineResult:
    """Convenience function to run the DPO training pipeline."""
    pipeline = TrainingPipeline()
    return pipeline.run_dpo(force=force)
