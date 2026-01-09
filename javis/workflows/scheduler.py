"""Workflow scheduler.

Manages scheduled workflow execution using APScheduler.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.utils.config import get_config
from javis.workflows.executor import WorkflowExecutor
from javis.workflows.loader import WorkflowLoader
from javis.workflows.models import (
    TriggerType,
    WorkflowConfig,
    WorkflowInfo,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# Optional APScheduler import
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    SCHEDULER_AVAILABLE = True
except ImportError:
    BackgroundScheduler = None
    CronTrigger = None
    IntervalTrigger = None
    SCHEDULER_AVAILABLE = False


class WorkflowScheduler:
    """Manages workflow scheduling and execution."""

    def __init__(self):
        """Initialize workflow scheduler."""
        self.config = get_config().workflows
        self.loader = WorkflowLoader(Path(self.config.directory))
        self.executor = WorkflowExecutor(default_timeout=self.config.default_timeout)

        self._workflows: dict[str, WorkflowConfig] = {}
        self._history: dict[str, list[WorkflowResult]] = {}
        self._scheduler: Optional[BackgroundScheduler] = None
        self._running = False

    def load_workflows(self) -> int:
        """Load workflows from the configuration directory.

        Returns:
            Number of workflows loaded
        """
        self._workflows = self.loader.load_all()
        logger.info(f"Loaded {len(self._workflows)} workflows")
        return len(self._workflows)

    def start(self) -> bool:
        """Start the workflow scheduler.

        Returns:
            True if started successfully
        """
        if not SCHEDULER_AVAILABLE:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            return False

        if self._running:
            logger.warning("Scheduler already running")
            return True

        # Load workflows
        self.load_workflows()

        # Initialize scheduler
        self._scheduler = BackgroundScheduler(
            timezone=self._get_default_timezone()
        )

        # Schedule enabled workflows
        for name, workflow in self._workflows.items():
            if workflow.enabled:
                self._schedule_workflow(workflow)

        self._scheduler.start()
        self._running = True
        logger.info("Workflow scheduler started")
        return True

    def stop(self) -> bool:
        """Stop the workflow scheduler.

        Returns:
            True if stopped successfully
        """
        if not self._running or self._scheduler is None:
            return True

        self._scheduler.shutdown(wait=True)
        self._scheduler = None
        self._running = False
        logger.info("Workflow scheduler stopped")
        return True

    def _schedule_workflow(self, workflow: WorkflowConfig):
        """Schedule a workflow based on its trigger configuration.

        Args:
            workflow: Workflow to schedule
        """
        if self._scheduler is None:
            return

        trigger_config = workflow.trigger

        if trigger_config.type == TriggerType.CRON and trigger_config.cron:
            # Parse cron expression
            cron_parts = trigger_config.cron.split()
            if len(cron_parts) == 5:
                trigger = CronTrigger(
                    minute=cron_parts[0],
                    hour=cron_parts[1],
                    day=cron_parts[2],
                    month=cron_parts[3],
                    day_of_week=cron_parts[4],
                    timezone=trigger_config.timezone,
                )

                self._scheduler.add_job(
                    self._run_workflow_sync,
                    trigger=trigger,
                    id=f"workflow_{workflow.name}",
                    name=f"Workflow: {workflow.name}",
                    args=[workflow.name],
                    replace_existing=True,
                )
                logger.info(f"Scheduled workflow '{workflow.name}' with cron: {trigger_config.cron}")

        elif trigger_config.type == TriggerType.INTERVAL and trigger_config.interval_seconds:
            trigger = IntervalTrigger(
                seconds=trigger_config.interval_seconds,
                timezone=trigger_config.timezone,
            )

            self._scheduler.add_job(
                self._run_workflow_sync,
                trigger=trigger,
                id=f"workflow_{workflow.name}",
                name=f"Workflow: {workflow.name}",
                args=[workflow.name],
                replace_existing=True,
            )
            logger.info(
                f"Scheduled workflow '{workflow.name}' with interval: {trigger_config.interval_seconds}s"
            )

        # Manual and webhook triggers don't need scheduling

    def _run_workflow_sync(self, workflow_name: str):
        """Synchronous wrapper for async workflow execution.

        Args:
            workflow_name: Name of the workflow to run
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.run_workflow(workflow_name))
                logger.info(f"Scheduled workflow '{workflow_name}' completed: {result.status.value}")
            finally:
                loop.close()
        except Exception as e:
            logger.exception(f"Error running scheduled workflow '{workflow_name}': {e}")

    async def run_workflow(self, workflow_name: str) -> WorkflowResult:
        """Run a workflow by name.

        Args:
            workflow_name: Name of the workflow to run

        Returns:
            WorkflowResult with execution details
        """
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")

        workflow = self._workflows[workflow_name]
        result = await self.executor.execute(workflow)

        # Store in history
        if workflow_name not in self._history:
            self._history[workflow_name] = []
        self._history[workflow_name].append(result)

        # Keep only last 100 runs per workflow
        if len(self._history[workflow_name]) > 100:
            self._history[workflow_name] = self._history[workflow_name][-100:]

        return result

    def trigger_now(self, workflow_name: str) -> WorkflowResult:
        """Trigger a workflow to run immediately (synchronous).

        Args:
            workflow_name: Name of the workflow to run

        Returns:
            WorkflowResult with execution details
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_workflow(workflow_name))
        finally:
            loop.close()

    def get_workflow(self, name: str) -> Optional[WorkflowConfig]:
        """Get a workflow by name.

        Args:
            name: Workflow name

        Returns:
            WorkflowConfig or None
        """
        return self._workflows.get(name)

    def list_workflows(self) -> list[WorkflowInfo]:
        """List all loaded workflows with their status.

        Returns:
            List of WorkflowInfo objects
        """
        infos = []

        for name, workflow in self._workflows.items():
            # Get next run time
            next_run = None
            if self._scheduler and self._running:
                job = self._scheduler.get_job(f"workflow_{name}")
                if job:
                    next_run = job.next_run_time

            # Get last run info
            last_run = None
            last_status = None
            if name in self._history and self._history[name]:
                last_result = self._history[name][-1]
                last_run = last_result.started_at
                last_status = last_result.status.value

            # Determine trigger schedule string
            trigger_schedule = None
            if workflow.trigger.type == TriggerType.CRON:
                trigger_schedule = workflow.trigger.cron
            elif workflow.trigger.type == TriggerType.INTERVAL:
                trigger_schedule = f"every {workflow.trigger.interval_seconds}s"

            infos.append(
                WorkflowInfo(
                    name=name,
                    description=workflow.description,
                    enabled=workflow.enabled,
                    trigger_type=workflow.trigger.type.value,
                    trigger_schedule=trigger_schedule,
                    next_run=next_run,
                    last_run=last_run,
                    last_status=last_status,
                    tags=workflow.tags,
                )
            )

        return infos

    def get_history(self, workflow_name: str, limit: int = 10) -> list[WorkflowResult]:
        """Get execution history for a workflow.

        Args:
            workflow_name: Workflow name
            limit: Maximum number of results

        Returns:
            List of WorkflowResult objects (newest first)
        """
        if workflow_name not in self._history:
            return []
        return list(reversed(self._history[workflow_name][-limit:]))

    def enable_workflow(self, name: str) -> bool:
        """Enable a workflow.

        Args:
            name: Workflow name

        Returns:
            True if successful
        """
        if name not in self._workflows:
            return False

        self._workflows[name].enabled = True

        if self._scheduler and self._running:
            self._schedule_workflow(self._workflows[name])

        return True

    def disable_workflow(self, name: str) -> bool:
        """Disable a workflow.

        Args:
            name: Workflow name

        Returns:
            True if successful
        """
        if name not in self._workflows:
            return False

        self._workflows[name].enabled = False

        if self._scheduler:
            try:
                self._scheduler.remove_job(f"workflow_{name}")
            except Exception:
                pass  # Job might not exist

        return True

    def reload_workflows(self) -> int:
        """Reload workflows from disk.

        Returns:
            Number of workflows loaded
        """
        # Stop existing scheduled jobs
        if self._scheduler:
            for name in self._workflows:
                try:
                    self._scheduler.remove_job(f"workflow_{name}")
                except Exception:
                    pass

        # Reload
        count = self.load_workflows()

        # Reschedule enabled workflows
        if self._scheduler and self._running:
            for name, workflow in self._workflows.items():
                if workflow.enabled:
                    self._schedule_workflow(workflow)

        return count

    def _get_default_timezone(self) -> str:
        """Get default timezone from first workflow or config."""
        for workflow in self._workflows.values():
            return workflow.trigger.timezone
        return "Asia/Seoul"

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Global instance
_scheduler: Optional[WorkflowScheduler] = None


def get_workflow_scheduler() -> WorkflowScheduler:
    """Get the global workflow scheduler instance.

    Returns:
        WorkflowScheduler instance
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = WorkflowScheduler()
    return _scheduler


def start_workflow_scheduler() -> bool:
    """Start the global workflow scheduler.

    Returns:
        True if started successfully
    """
    scheduler = get_workflow_scheduler()
    return scheduler.start()


def stop_workflow_scheduler() -> bool:
    """Stop the global workflow scheduler.

    Returns:
        True if stopped successfully
    """
    scheduler = get_workflow_scheduler()
    return scheduler.stop()
