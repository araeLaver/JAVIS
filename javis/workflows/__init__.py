"""Workflow automation module for JAVIS.

Provides YAML-based workflow definition and scheduled execution.

Usage:
    # Get the workflow manager
    from javis.workflows import get_workflow_scheduler

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()
    scheduler.start()

    # Run a workflow manually
    result = scheduler.trigger_now("daily_backup")
    print(result.status)
"""

from javis.workflows.executor import WorkflowExecutor
from javis.workflows.loader import WorkflowLoader
from javis.workflows.models import (
    StepConfig,
    StepResult,
    StepStatus,
    TriggerConfig,
    TriggerType,
    WorkflowConfig,
    WorkflowInfo,
    WorkflowResult,
    WorkflowStatus,
)
from javis.workflows.scheduler import (
    WorkflowScheduler,
    get_workflow_scheduler,
    start_workflow_scheduler,
    stop_workflow_scheduler,
)

__all__ = [
    # Models
    "TriggerType",
    "TriggerConfig",
    "StepConfig",
    "WorkflowConfig",
    "StepStatus",
    "StepResult",
    "WorkflowStatus",
    "WorkflowResult",
    "WorkflowInfo",
    # Classes
    "WorkflowLoader",
    "WorkflowExecutor",
    "WorkflowScheduler",
    # Functions
    "get_workflow_scheduler",
    "start_workflow_scheduler",
    "stop_workflow_scheduler",
]
