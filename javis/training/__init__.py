"""Training pipeline modules for JAVIS.

This module provides:
- Automatic retraining pipeline with scheduling
- Remote GPU training via Modal.com
- Model version management with rollback
- Discord notifications for training events

Usage:
    # CLI commands
    python -m javis.training.manage train --dry-run  # Check conditions
    python -m javis.training.manage train            # Run training
    python -m javis.training.manage scheduler start  # Start auto-scheduler
    python -m javis.training.manage rollback         # Rollback to previous version

    # Programmatic usage
    from javis.training.pipeline import run_pipeline
    result = run_pipeline(force=True)
"""

from .version_manager import VersionManager, get_version_manager
from .notifications import NotificationService, get_notifier
from .scheduler import TrainingScheduler, get_scheduler, start_scheduler, stop_scheduler
from .pipeline import TrainingPipeline, run_pipeline
from .remote import RemoteTrainer, TrainingResult

__all__ = [
    "VersionManager",
    "get_version_manager",
    "NotificationService",
    "get_notifier",
    "TrainingScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
    "TrainingPipeline",
    "run_pipeline",
    "RemoteTrainer",
    "TrainingResult",
]
