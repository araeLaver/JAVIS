"""Optuna-based hyperparameter optimization for JAVIS training pipeline."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from javis.utils.config import OptunaConfig, get_config

logger = logging.getLogger(__name__)

# Optuna is optional - gracefully handle if not installed
try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class HPOResult(BaseModel):
    """Result of hyperparameter optimization."""

    success: bool
    best_params: dict[str, Any] = {}
    best_value: Optional[float] = None
    n_trials: int = 0
    n_complete: int = 0
    n_pruned: int = 0
    study_name: Optional[str] = None
    error: Optional[str] = None


class TrialConfig(BaseModel):
    """Configuration sampled for a single trial."""

    learning_rate: float
    batch_size: int
    lora_r: int
    epochs: int
    lora_alpha: int = 16
    gradient_accumulation_steps: int = 4


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for JAVIS training.

    Provides:
    - Configurable search space
    - Multiple pruning strategies (MedianPruner, HyperbandPruner)
    - Multiple samplers (TPE, Random, CMA-ES)
    - MLflow integration for experiment tracking
    - Study persistence for resumable optimization
    """

    def __init__(self, config: Optional[OptunaConfig] = None):
        """Initialize the optimizer.

        Args:
            config: Optuna configuration. If None, uses defaults from config.yaml.
        """
        if config is None:
            config = get_config().training.optuna
        self.config = config
        self._study: Optional[Any] = None

        if not OPTUNA_AVAILABLE:
            logger.warning(
                "Optuna not installed. Install with: pip install optuna"
            )

    @property
    def is_available(self) -> bool:
        """Check if Optuna is available."""
        return OPTUNA_AVAILABLE

    def _get_sampler(self) -> Any:
        """Get the configured sampler."""
        if not OPTUNA_AVAILABLE:
            return None

        sampler_name = self.config.sampler.lower()
        if sampler_name == "tpe":
            return TPESampler(seed=42)
        elif sampler_name == "random":
            return RandomSampler(seed=42)
        elif sampler_name == "cmaes":
            return CmaEsSampler(seed=42)
        else:
            logger.warning(f"Unknown sampler: {sampler_name}, using TPE")
            return TPESampler(seed=42)

    def _get_pruner(self) -> Any:
        """Get the configured pruner."""
        if not OPTUNA_AVAILABLE:
            return None

        pruner_name = self.config.pruner.lower()
        if pruner_name == "median":
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_name == "hyperband":
            return HyperbandPruner(min_resource=1, max_resource=10)
        elif pruner_name == "nop":
            return NopPruner()
        else:
            logger.warning(f"Unknown pruner: {pruner_name}, using MedianPruner")
            return MedianPruner()

    def create_study(self, study_name: Optional[str] = None) -> Any:
        """Create or load an Optuna study.

        Args:
            study_name: Name for the study. Uses config if not provided.

        Returns:
            Optuna Study object
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not installed")

        if study_name is None:
            study_name = self.config.study_name

        storage = self.config.storage_uri
        direction = self.config.direction

        # Create or load study
        self._study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
        )

        logger.info(
            f"Created/loaded study: {study_name} "
            f"(direction: {direction}, trials: {len(self._study.trials)})"
        )
        return self._study

    def suggest_config(self, trial: Any) -> TrialConfig:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna Trial object

        Returns:
            TrialConfig with suggested parameters
        """
        learning_rate = trial.suggest_float(
            "learning_rate",
            self.config.learning_rate_min,
            self.config.learning_rate_max,
            log=True,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            self.config.batch_size_choices,
        )

        lora_r = trial.suggest_categorical(
            "lora_r",
            self.config.lora_r_choices,
        )

        epochs = trial.suggest_int(
            "epochs",
            self.config.epochs_min,
            self.config.epochs_max,
        )

        # Fixed parameters based on lora_r
        lora_alpha = lora_r // 4 if lora_r >= 32 else 16

        return TrialConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_r=lora_r,
            epochs=epochs,
            lora_alpha=lora_alpha,
        )

    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[list[Callable]] = None,
    ) -> HPOResult:
        """Run hyperparameter optimization.

        Args:
            objective: Objective function that takes a trial and returns a score
            n_trials: Number of trials (uses config if not provided)
            timeout: Timeout in seconds (uses config if not provided)
            callbacks: Optional list of callback functions

        Returns:
            HPOResult with best parameters and statistics
        """
        if not OPTUNA_AVAILABLE:
            return HPOResult(
                success=False,
                error="Optuna is not installed",
            )

        if n_trials is None:
            n_trials = self.config.n_trials
        if timeout is None:
            timeout = self.config.timeout

        try:
            # Create study if not exists
            if self._study is None:
                self.create_study()

            # Run optimization
            logger.info(
                f"Starting HPO: {n_trials} trials, "
                f"timeout: {timeout}s" if timeout else f"Starting HPO: {n_trials} trials"
            )

            self._study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=True,
            )

            # Get results
            best_trial = self._study.best_trial
            all_trials = self._study.trials

            n_complete = sum(1 for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE)
            n_pruned = sum(1 for t in all_trials if t.state == optuna.trial.TrialState.PRUNED)

            logger.info(
                f"HPO complete: best value = {best_trial.value:.4f}, "
                f"trials = {len(all_trials)} (complete: {n_complete}, pruned: {n_pruned})"
            )

            return HPOResult(
                success=True,
                best_params=best_trial.params,
                best_value=best_trial.value,
                n_trials=len(all_trials),
                n_complete=n_complete,
                n_pruned=n_pruned,
                study_name=self._study.study_name,
            )

        except Exception as e:
            logger.exception(f"HPO failed: {e}")
            return HPOResult(
                success=False,
                error=str(e),
            )

    def get_best_params(self) -> dict[str, Any]:
        """Get the best parameters from the study.

        Returns:
            Dictionary of best parameters
        """
        if self._study is None or len(self._study.trials) == 0:
            return {}
        return self._study.best_trial.params

    def get_study_dataframe(self) -> Any:
        """Get study results as a DataFrame.

        Returns:
            pandas DataFrame with trial results
        """
        if not OPTUNA_AVAILABLE or self._study is None:
            return None
        return self._study.trials_dataframe()

    def plot_optimization_history(self, save_path: Optional[Path] = None) -> None:
        """Plot optimization history.

        Args:
            save_path: Optional path to save the plot
        """
        if not OPTUNA_AVAILABLE or self._study is None:
            return

        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self._study)
            if save_path:
                fig.write_image(str(save_path))
            else:
                fig.show()
        except ImportError:
            logger.warning("Plotly not installed for visualization")

    def plot_param_importances(self, save_path: Optional[Path] = None) -> None:
        """Plot parameter importances.

        Args:
            save_path: Optional path to save the plot
        """
        if not OPTUNA_AVAILABLE or self._study is None:
            return

        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self._study)
            if save_path:
                fig.write_image(str(save_path))
            else:
                fig.show()
        except ImportError:
            logger.warning("Plotly not installed for visualization")


def create_training_objective(
    pipeline: Any,
    data_path: Path,
    metric: str = "eval_loss",
) -> Callable[[Any], float]:
    """Create an objective function for training optimization.

    Args:
        pipeline: TrainingPipeline instance
        data_path: Path to training data
        metric: Metric to optimize (default: eval_loss)

    Returns:
        Objective function for Optuna
    """
    optimizer = HyperparameterOptimizer()

    def objective(trial: Any) -> float:
        # Get suggested config
        config = optimizer.suggest_config(trial)

        # Build training config
        training_config = {
            "base_model": pipeline.config.model.base_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_seq_length": pipeline.config.model.max_seq_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }

        logger.info(f"Trial {trial.number}: {training_config}")

        try:
            # Run training
            result = pipeline.remote_trainer.train(data_path, training_config)

            if not result.success:
                raise optuna.TrialPruned(f"Training failed: {result.error}")

            # Get metric from result
            metric_value = result.metadata.get(metric)
            if metric_value is None:
                raise optuna.TrialPruned(f"Metric '{metric}' not found in result")

            # Report intermediate values for pruning
            trial.report(metric_value, step=config.epochs)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return metric_value

        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(str(e))

    return objective


def run_optimization(
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
) -> HPOResult:
    """Convenience function to run hyperparameter optimization.

    Args:
        n_trials: Number of trials (uses config if not provided)
        timeout: Timeout in seconds (uses config if not provided)

    Returns:
        HPOResult with best parameters
    """
    from javis.training.pipeline import TrainingPipeline

    # Initialize pipeline and optimizer
    pipeline = TrainingPipeline()
    optimizer = HyperparameterOptimizer()

    if not optimizer.is_available:
        return HPOResult(
            success=False,
            error="Optuna is not installed. Install with: pip install optuna",
        )

    # Check if we have training data
    can_train, reason = pipeline.check_conditions()
    if not can_train:
        return HPOResult(
            success=False,
            error=f"Cannot run HPO: {reason}",
        )

    # Export training data
    data_path = pipeline.export_training_data()

    # Create objective function
    objective = create_training_objective(pipeline, data_path)

    # Run optimization
    result = optimizer.optimize(
        objective=objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    if result.success:
        logger.info(f"Best parameters: {result.best_params}")
        logger.info(f"Best value: {result.best_value}")

    return result


# Singleton instance
_optimizer: Optional[HyperparameterOptimizer] = None


def get_optimizer(config: Optional[OptunaConfig] = None) -> HyperparameterOptimizer:
    """Get or create the optimizer singleton.

    Args:
        config: Optional configuration (used only on first call)

    Returns:
        HyperparameterOptimizer instance
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = HyperparameterOptimizer(config)
    return _optimizer


def reset_optimizer() -> None:
    """Reset the optimizer singleton (for testing)."""
    global _optimizer
    _optimizer = None
