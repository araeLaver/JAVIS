"""Workflow executor.

Executes workflow steps using the tools system.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from javis.tools.executor import ToolExecutor
from javis.training.notifications import get_notifier
from javis.workflows.models import (
    OnFailureAction,
    StepConfig,
    StepResult,
    StepStatus,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes workflows step by step."""

    # Built-in actions that don't use the tool system
    BUILTIN_ACTIONS = {
        "notify", "wait", "set_variable", "log",
        "system_info", "get_datetime",
    }

    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        default_timeout: int = 300,
    ):
        """Initialize workflow executor.

        Args:
            tool_executor: ToolExecutor instance (creates new if None)
            default_timeout: Default step timeout in seconds
        """
        self.tool_executor = tool_executor or ToolExecutor()
        self.default_timeout = default_timeout
        self.context: dict[str, Any] = {}

    async def execute(self, workflow: WorkflowConfig) -> WorkflowResult:
        """Execute a complete workflow.

        Args:
            workflow: Workflow configuration to execute

        Returns:
            WorkflowResult with execution details
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.now()

        # Initialize context
        self.context = {
            "workflow": {
                "name": workflow.name,
                "run_id": run_id,
            },
            "steps": {},
            "variables": {},
        }

        result = WorkflowResult(
            workflow_name=workflow.name,
            run_id=run_id,
            status=WorkflowStatus.RUNNING,
            started_at=started_at,
            trigger_type=workflow.trigger.type.value,
        )

        logger.info(f"Starting workflow '{workflow.name}' (run_id: {run_id})")

        try:
            # Execute steps
            for step in workflow.steps:
                step_result = await self._execute_step(step, workflow.timeout)
                result.steps.append(step_result)

                # Store step result in context
                self.context["steps"][step.name] = {
                    "success": step_result.status == StepStatus.SUCCESS,
                    "output": step_result.output,
                    "error": step_result.error,
                }

                # Handle step failure
                if step_result.status == StepStatus.FAILED:
                    if step.on_failure == OnFailureAction.STOP:
                        result.status = WorkflowStatus.FAILED
                        result.error = f"Step '{step.name}' failed: {step_result.error}"
                        break

            # Determine final status
            if result.status == WorkflowStatus.RUNNING:
                failed_steps = [s for s in result.steps if s.status == StepStatus.FAILED]
                if failed_steps:
                    result.status = WorkflowStatus.FAILED
                    result.error = f"{len(failed_steps)} step(s) failed"
                else:
                    result.status = WorkflowStatus.SUCCESS

        except asyncio.CancelledError:
            result.status = WorkflowStatus.CANCELLED
            result.error = "Workflow was cancelled"
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = str(e)
            logger.exception(f"Workflow '{workflow.name}' failed with exception")

        # Finalize result
        result.ended_at = datetime.now()
        result.duration_ms = (result.ended_at - started_at).total_seconds() * 1000
        result.context = self.context.get("variables", {})

        # Send notifications
        await self._send_notifications(workflow, result)

        logger.info(
            f"Workflow '{workflow.name}' completed with status: {result.status.value}"
        )

        return result

    async def _execute_step(
        self,
        step: StepConfig,
        workflow_timeout: Optional[int] = None,
    ) -> StepResult:
        """Execute a single step.

        Args:
            step: Step configuration
            workflow_timeout: Workflow-level timeout

        Returns:
            StepResult with execution details
        """
        started_at = datetime.now()
        timeout = step.timeout or workflow_timeout or self.default_timeout

        logger.debug(f"Executing step: {step.name} (action: {step.action})")

        # Check condition
        if step.condition:
            if not self._evaluate_condition(step.condition):
                logger.debug(f"Step '{step.name}' skipped due to condition")
                return StepResult(
                    name=step.name,
                    status=StepStatus.SKIPPED,
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

        # Execute with retries
        last_error = None
        for attempt in range(step.retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retry {attempt}/{step.retries} for step '{step.name}'")
                    await asyncio.sleep(step.retry_delay)

                # Execute action
                output = await asyncio.wait_for(
                    self._execute_action(step.action, step.params),
                    timeout=timeout,
                )

                ended_at = datetime.now()
                return StepResult(
                    name=step.name,
                    status=StepStatus.SUCCESS,
                    output=output,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=(ended_at - started_at).total_seconds() * 1000,
                    retries_used=attempt,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step '{step.name}' attempt {attempt + 1} failed: {e}")

        # All retries failed
        ended_at = datetime.now()
        return StepResult(
            name=step.name,
            status=StepStatus.FAILED,
            error=last_error,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=(ended_at - started_at).total_seconds() * 1000,
            retries_used=step.retries,
        )

    async def _execute_action(self, action: str, params: dict) -> Any:
        """Execute an action (built-in or tool).

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Action output
        """
        # Render template variables in params
        rendered_params = self._render_params(params)

        # Built-in actions
        if action in self.BUILTIN_ACTIONS:
            return await self._execute_builtin(action, rendered_params)

        # Tool actions
        result = await self.tool_executor.execute(action, rendered_params)
        if not result.success:
            raise RuntimeError(result.error or "Tool execution failed")
        return result.output

    async def _execute_builtin(self, action: str, params: dict) -> Any:
        """Execute a built-in action.

        Args:
            action: Built-in action name
            params: Action parameters

        Returns:
            Action output
        """
        if action == "notify":
            message = params.get("message", "")
            channel = params.get("channel", "discord")
            if channel == "discord":
                notifier = get_notifier()
                notifier._send_discord(message, success=True)
            return {"notified": True, "channel": channel}

        elif action == "wait":
            seconds = params.get("seconds", 1)
            await asyncio.sleep(seconds)
            return {"waited": seconds}

        elif action == "set_variable":
            name = params.get("name", "")
            value = params.get("value")
            self.context["variables"][name] = value
            return {"variable": name, "value": value}

        elif action == "log":
            message = params.get("message", "")
            level = params.get("level", "info").lower()
            getattr(logger, level, logger.info)(f"[Workflow] {message}")
            return {"logged": message}

        elif action == "system_info":
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "used": psutil.disk_usage("/").used,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent,
                },
            }

        elif action == "get_datetime":
            from zoneinfo import ZoneInfo
            tz_name = params.get("timezone", "Asia/Seoul")
            fmt = params.get("format", "%Y-%m-%d %H:%M:%S")
            try:
                tz = ZoneInfo(tz_name)
                now = datetime.now(tz)
            except Exception:
                now = datetime.now()
            return now.strftime(fmt)

        else:
            raise ValueError(f"Unknown built-in action: {action}")

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a Jinja2-like condition.

        Args:
            condition: Condition string (e.g., "{{ steps.backup.success }}")

        Returns:
            Boolean result
        """
        # Simple template evaluation
        rendered = self._render_template(condition)

        # Handle common boolean strings
        if rendered.lower() in ("true", "1", "yes"):
            return True
        if rendered.lower() in ("false", "0", "no", "none", ""):
            return False

        # Try to evaluate as Python expression
        try:
            return bool(eval(rendered, {"__builtins__": {}}, {}))
        except Exception:
            return False

    def _render_params(self, params: dict) -> dict:
        """Render template variables in parameters.

        Args:
            params: Parameters with possible template strings

        Returns:
            Rendered parameters
        """
        rendered = {}
        for key, value in params.items():
            if isinstance(value, str):
                rendered[key] = self._render_template(value)
            elif isinstance(value, dict):
                rendered[key] = self._render_params(value)
            elif isinstance(value, list):
                rendered[key] = [
                    self._render_template(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                rendered[key] = value
        return rendered

    def _render_template(self, template: str) -> str:
        """Render a simple Jinja2-like template.

        Args:
            template: Template string with {{ variable }} placeholders

        Returns:
            Rendered string
        """
        # Find all {{ ... }} patterns
        pattern = r"\{\{\s*(.+?)\s*\}\}"

        def replace(match):
            expr = match.group(1)
            try:
                # Navigate context with dot notation
                value = self._get_context_value(expr)
                return str(value) if value is not None else ""
            except Exception:
                return match.group(0)  # Return original if evaluation fails

        return re.sub(pattern, replace, template)

    def _get_context_value(self, expr: str) -> Any:
        """Get a value from context using dot notation.

        Args:
            expr: Expression like "steps.backup.success"

        Returns:
            Value from context
        """
        parts = expr.split(".")
        value = self.context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value

    async def _send_notifications(
        self,
        workflow: WorkflowConfig,
        result: WorkflowResult,
    ):
        """Send notifications based on workflow result.

        Args:
            workflow: Workflow configuration
            result: Workflow execution result
        """
        try:
            notifier = get_notifier()

            if result.status == WorkflowStatus.SUCCESS and workflow.on_success:
                if workflow.on_success.notify:
                    message = workflow.on_success.message or f"Workflow '{workflow.name}' completed successfully"
                    message = self._render_template(message)
                    notifier._send_discord(message, success=True)

            elif result.status == WorkflowStatus.FAILED and workflow.on_failure:
                if workflow.on_failure.notify:
                    self.context["error"] = result.error
                    message = workflow.on_failure.message or f"Workflow '{workflow.name}' failed: {result.error}"
                    message = self._render_template(message)
                    notifier._send_discord(message, success=False)

        except Exception as e:
            logger.warning(f"Failed to send workflow notification: {e}")
