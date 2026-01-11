"""Training pipeline modules for JAVIS.

This module provides:
- Automatic retraining pipeline with scheduling
- Remote GPU training via Modal.com
- Model version management with rollback
- Discord notifications for training events
- DPO (Direct Preference Optimization) training

Usage:
    # CLI commands
    python -m javis.training.manage train --dry-run  # Check conditions
    python -m javis.training.manage train            # Run training
    python -m javis.training.manage scheduler start  # Start auto-scheduler
    python -m javis.training.manage rollback         # Rollback to previous version

    # Programmatic usage
    from javis.training.pipeline import run_pipeline
    result = run_pipeline(force=True)

    # DPO training
    from javis.training.pipeline import run_dpo_pipeline
    result = run_dpo_pipeline(force=True)

    # Preference data generation
    from javis.training.preference_data import get_preference_generator
    generator = get_preference_generator()
    stats = generator.get_statistics()
"""

from .version_manager import VersionManager, get_version_manager
from .notifications import NotificationService, get_notifier
from .scheduler import TrainingScheduler, get_scheduler, start_scheduler, stop_scheduler
from .pipeline import TrainingPipeline, run_pipeline, run_dpo_pipeline
from .remote import RemoteTrainer, TrainingResult
from .preference_data import (
    PreferencePair,
    PreferenceSource,
    PreferenceStats,
    PreferenceDataGenerator,
    get_preference_generator,
)

__all__ = [
    # Version management
    "VersionManager",
    "get_version_manager",
    # Notifications
    "NotificationService",
    "get_notifier",
    # Scheduler
    "TrainingScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
    # SFT Pipeline
    "TrainingPipeline",
    "run_pipeline",
    # DPO Pipeline
    "run_dpo_pipeline",
    # Preference data
    "PreferencePair",
    "PreferenceSource",
    "PreferenceStats",
    "PreferenceDataGenerator",
    "get_preference_generator",
    # Remote training
    "RemoteTrainer",
    "TrainingResult",
]
