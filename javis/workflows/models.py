"""Workflow data models.

Pydantic models for workflow configuration and execution results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Workflow trigger types."""

    CRON = "cron"
    INTERVAL = "interval"
    WEBHOOK = "webhook"
    MANUAL = "manual"


class OnFailureAction(str, Enum):
    """Action to take when a step fails."""

    CONTINUE = "continue"
    STOP = "stop"
    RETRY = "retry"


class TriggerConfig(BaseModel):
    """Workflow trigger configuration."""

    type: TriggerType = TriggerType.MANUAL
    cron: Optional[str] = None  # e.g., "0 2 * * *"
    interval_seconds: Optional[int] = None
    timezone: str = "Asia/Seoul"
    webhook_token: Optional[str] = None  # Auto-generated if webhook type


class StepConfig(BaseModel):
    """Workflow step configuration."""

    name: str
    action: str  # Tool name or built-in action
    params: dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None  # Jinja2 condition
    on_failure: OnFailureAction = OnFailureAction.CONTINUE
    timeout: Optional[int] = None  # Seconds
    retries: int = 0
    retry_delay: int = 5  # Seconds between retries


class NotificationConfig(BaseModel):
    """Notification configuration for workflow events."""

    notify: bool = True
    message: Optional[str] = None
    channel: str = "discord"


class WorkflowConfig(BaseModel):
    """Complete workflow configuration."""

    name: str
    description: str = ""
    enabled: bool = True
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    steps: list[StepConfig] = Field(default_factory=list)
    on_success: Optional[NotificationConfig] = None
    on_failure: Optional[NotificationConfig] = None
    tags: list[str] = Field(default_factory=list)
    timeout: Optional[int] = None  # Overall workflow timeout


# Execution Result Models


class StepStatus(str, Enum):
    """Step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    """Result of a single step execution."""

    name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retries_used: int = 0

    @property
    def step_name(self) -> str:
        """Alias for name (for compatibility)."""
        return self.name

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.duration_ms or 0) / 1000


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowResult(BaseModel):
    """Result of a workflow execution."""

    workflow_name: str
    run_id: str
    status: WorkflowStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    steps: list[StepResult] = Field(default_factory=list)
    error: Optional[str] = None
    trigger_type: str = "manual"
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def completed_at(self) -> Optional[datetime]:
        """Alias for ended_at."""
        return self.ended_at

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.duration_ms or 0) / 1000


class WorkflowInfo(BaseModel):
    """Workflow information for listing."""

    name: str
    description: str
    enabled: bool
    trigger_type: str
    trigger_schedule: Optional[str] = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
