"""Workflow loader.

Loads workflow configurations from YAML files.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from javis.workflows.models import (
    NotificationConfig,
    StepConfig,
    TriggerConfig,
    WorkflowConfig,
)

logger = logging.getLogger(__name__)


class WorkflowLoader:
    """Loads workflow configurations from YAML files."""

    def __init__(self, directory: Optional[Path] = None):
        """Initialize workflow loader.

        Args:
            directory: Directory containing workflow YAML files
        """
        if directory is None:
            from javis.utils.config import get_config

            config = get_config()
            directory = Path(config.workflows.directory)

        self.directory = Path(directory)

    def load_all(self) -> dict[str, WorkflowConfig]:
        """Load all workflow configurations from the directory.

        Returns:
            Dictionary mapping workflow names to configurations
        """
        workflows = {}

        if not self.directory.exists():
            logger.warning(f"Workflow directory does not exist: {self.directory}")
            return workflows

        for yaml_file in self.directory.glob("*.yaml"):
            try:
                workflow = self.load_file(yaml_file)
                if workflow:
                    workflows[workflow.name] = workflow
                    logger.info(f"Loaded workflow: {workflow.name}")
            except Exception as e:
                logger.error(f"Failed to load workflow from {yaml_file}: {e}")

        for yml_file in self.directory.glob("*.yml"):
            try:
                workflow = self.load_file(yml_file)
                if workflow:
                    workflows[workflow.name] = workflow
                    logger.info(f"Loaded workflow: {workflow.name}")
            except Exception as e:
                logger.error(f"Failed to load workflow from {yml_file}: {e}")

        return workflows

    def load_file(self, file_path: Path) -> Optional[WorkflowConfig]:
        """Load a single workflow configuration from a file.

        Args:
            file_path: Path to the YAML file

        Returns:
            WorkflowConfig or None if loading failed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Workflow file not found: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            logger.warning(f"Empty workflow file: {file_path}")
            return None

        return self._parse_workflow(data, file_path)

    def _parse_workflow(self, data: dict, source_file: Path) -> WorkflowConfig:
        """Parse workflow data into a WorkflowConfig.

        Args:
            data: Raw YAML data
            source_file: Source file path (for error messages)

        Returns:
            Parsed WorkflowConfig
        """
        # Parse trigger
        trigger_data = data.get("trigger", {})
        trigger = TriggerConfig(
            type=trigger_data.get("type", "manual"),
            cron=trigger_data.get("cron"),
            interval_seconds=trigger_data.get("interval_seconds"),
            timezone=trigger_data.get("timezone", "Asia/Seoul"),
            webhook_token=trigger_data.get("webhook_token"),
        )

        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            step = StepConfig(
                name=step_data.get("name", "unnamed"),
                action=step_data.get("action", ""),
                params=step_data.get("params", {}),
                condition=step_data.get("condition"),
                on_failure=step_data.get("on_failure", "continue"),
                timeout=step_data.get("timeout"),
                retries=step_data.get("retries", 0),
                retry_delay=step_data.get("retry_delay", 5),
            )
            steps.append(step)

        # Parse notification configs
        on_success = None
        if "on_success" in data:
            on_success_data = data["on_success"]
            on_success = NotificationConfig(
                notify=on_success_data.get("notify", True),
                message=on_success_data.get("message"),
                channel=on_success_data.get("channel", "discord"),
            )

        on_failure = None
        if "on_failure" in data:
            on_failure_data = data["on_failure"]
            on_failure = NotificationConfig(
                notify=on_failure_data.get("notify", True),
                message=on_failure_data.get("message"),
                channel=on_failure_data.get("channel", "discord"),
            )

        return WorkflowConfig(
            name=data.get("name", source_file.stem),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            trigger=trigger,
            steps=steps,
            on_success=on_success,
            on_failure=on_failure,
            tags=data.get("tags", []),
            timeout=data.get("timeout"),
        )

    def reload(self) -> dict[str, WorkflowConfig]:
        """Reload all workflows from disk.

        Returns:
            Dictionary mapping workflow names to configurations
        """
        return self.load_all()
