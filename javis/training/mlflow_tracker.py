"""MLflow experiment tracking integration for JAVIS training pipeline."""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# MLflow is optional - gracefully handle if not installed
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "javis-training"
    artifact_location: Optional[str] = None
    # Auto-logging settings
    log_models: bool = True
    log_datasets: bool = True
    log_system_metrics: bool = True


class TrainingMetrics(BaseModel):
    """Training metrics to log."""

    loss: float
    learning_rate: float
    epoch: int
    step: int
    grad_norm: Optional[float] = None
    # Evaluation metrics
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    perplexity: Optional[float] = None


class MLflowTracker:
    """MLflow experiment tracking wrapper for JAVIS training.

    Provides a simple interface for:
    - Starting and managing experiments
    - Logging parameters, metrics, and artifacts
    - Tracking model versions
    - Comparing training runs
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow tracker.

        Args:
            config: MLflow configuration. If None, uses defaults.
        """
        self.config = config or MLflowConfig()
        self._client: Optional[Any] = None
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow"
            )
            self.config.enabled = False

        if self.config.enabled and MLFLOW_AVAILABLE:
            self._setup()

    def _setup(self) -> None:
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self._client = MlflowClient(self.config.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )
                logger.info(
                    f"Created MLflow experiment: {self.config.experiment_name}"
                )
            else:
                self._experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing MLflow experiment: {self.config.experiment_name}"
                )

            mlflow.set_experiment(self.config.experiment_name)

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            self.config.enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled and available."""
        return self.config.enabled and MLFLOW_AVAILABLE

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run_id

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """Context manager for MLflow run.

        Args:
            run_name: Name for the run (default: auto-generated)
            tags: Additional tags for the run
            description: Run description

        Yields:
            The run ID
        """
        if not self.is_enabled:
            yield None
            return

        if run_name is None:
            run_name = f"javis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        all_tags = {
            "project": "javis",
            "framework": "unsloth",
        }
        if tags:
            all_tags.update(tags)

        try:
            with mlflow.start_run(
                run_name=run_name,
                tags=all_tags,
                description=description,
            ) as run:
                self._run_id = run.info.run_id
                logger.info(f"Started MLflow run: {run_name} ({self._run_id})")
                yield self._run_id

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            yield None
        finally:
            self._run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log training parameters.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.is_enabled:
            return

        try:
            # Flatten nested dicts and convert values to strings
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.debug(f"Logged {len(flat_params)} parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(
        self,
        metrics: dict[str, float] | TrainingMetrics,
        step: Optional[int] = None,
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metrics or TrainingMetrics object
            step: Training step (optional)
        """
        if not self.is_enabled:
            return

        try:
            if isinstance(metrics, TrainingMetrics):
                metrics_dict = metrics.model_dump(exclude_none=True)
            else:
                metrics_dict = metrics

            # Filter to only numeric values
            numeric_metrics = {
                k: v for k, v in metrics_dict.items()
                if isinstance(v, (int, float))
            }

            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Destination path in artifact store
        """
        if not self.is_enabled:
            return

        try:
            local_path = Path(local_path)
            if local_path.exists():
                mlflow.log_artifact(str(local_path), artifact_path)
                logger.debug(f"Logged artifact: {local_path}")
            else:
                logger.warning(f"Artifact not found: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")

    def log_artifacts(
        self,
        local_dir: str | Path,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a directory of artifacts.

        Args:
            local_dir: Path to the local directory
            artifact_path: Destination path in artifact store
        """
        if not self.is_enabled:
            return

        try:
            local_dir = Path(local_dir)
            if local_dir.exists() and local_dir.is_dir():
                mlflow.log_artifacts(str(local_dir), artifact_path)
                logger.debug(f"Logged artifacts from: {local_dir}")
            else:
                logger.warning(f"Artifact directory not found: {local_dir}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts to MLflow: {e}")

    def log_model(
        self,
        model_path: str | Path,
        model_name: str = "model",
        registered_name: Optional[str] = None,
    ) -> Optional[str]:
        """Log a trained model.

        Args:
            model_path: Path to the model directory
            model_name: Name for the model artifact
            registered_name: Optional name to register in model registry

        Returns:
            Model URI if successful
        """
        if not self.is_enabled or not self.config.log_models:
            return None

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model path not found: {model_path}")
                return None

            # Log as artifacts
            mlflow.log_artifacts(str(model_path), model_name)
            model_uri = f"runs:/{self._run_id}/{model_name}"

            # Register model if name provided
            if registered_name and self._client:
                try:
                    self._client.create_registered_model(registered_name)
                except Exception:
                    pass  # Model may already exist

                self._client.create_model_version(
                    name=registered_name,
                    source=model_uri,
                    run_id=self._run_id,
                )
                logger.info(f"Registered model: {registered_name}")

            logger.info(f"Logged model to MLflow: {model_uri}")
            return model_uri

        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
            return None

    def log_dataset(
        self,
        dataset_path: str | Path,
        name: str = "training_data",
        context: str = "training",
    ) -> None:
        """Log dataset information.

        Args:
            dataset_path: Path to the dataset file
            name: Name for the dataset
            context: Context (training, validation, test)
        """
        if not self.is_enabled or not self.config.log_datasets:
            return

        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                return

            # Log dataset as artifact
            mlflow.log_artifact(str(dataset_path), f"datasets/{context}")

            # Log dataset metadata
            file_size = dataset_path.stat().st_size
            line_count = sum(1 for _ in open(dataset_path, encoding="utf-8"))

            mlflow.log_params({
                f"dataset_{context}_path": str(dataset_path.name),
                f"dataset_{context}_size_bytes": file_size,
                f"dataset_{context}_samples": line_count,
            })

            logger.debug(f"Logged dataset: {name} ({line_count} samples)")

        except Exception as e:
            logger.warning(f"Failed to log dataset to MLflow: {e}")

    def log_training_config(self, config: dict[str, Any]) -> None:
        """Log full training configuration.

        Args:
            config: Training configuration dictionary
        """
        if not self.is_enabled:
            return

        self.log_params(config)

        # Also log as artifact for easy viewing
        try:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(config, f, indent=2, default=str)
                temp_path = f.name

            mlflow.log_artifact(temp_path, "config")
            os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Failed to log config artifact: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        if not self.is_enabled:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set tag: {e}")

    def get_run_url(self) -> Optional[str]:
        """Get the URL for the current run in MLflow UI.

        Returns:
            URL string or None
        """
        if not self.is_enabled or not self._run_id:
            return None

        return f"{self.config.tracking_uri}/#/experiments/{self._experiment_id}/runs/{self._run_id}"

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, str]:
        """Flatten nested dictionary for MLflow params.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary with string values
        """
        items: list[tuple[str, str]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


# Singleton instance
_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker(config: Optional[MLflowConfig] = None) -> MLflowTracker:
    """Get or create the MLflow tracker singleton.

    Args:
        config: Optional configuration (used only on first call)

    Returns:
        MLflowTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(config)
    return _tracker


def reset_mlflow_tracker() -> None:
    """Reset the MLflow tracker singleton (for testing)."""
    global _tracker
    _tracker = None
