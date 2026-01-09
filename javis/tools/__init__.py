"""Tool modules for JAVIS.

This module provides a tool system that allows the LLM to perform
various actions like file operations, web requests, and system queries.
"""

import logging
from typing import Optional

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from javis.tools.registry import ToolRegistry, get_registry
from javis.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "get_registry",
    "ToolExecutor",
    "initialize_tools",
    "get_available_categories",
]


def get_available_categories() -> list[str]:
    """사용 가능한 도구 카테고리 목록."""
    return ["file_tools", "web_tools", "system_tools", "calendar_tools", "notion_tools", "slack_tools"]


def initialize_tools(categories: Optional[list[str]] = None) -> None:
    """도구 시스템을 초기화하고 지정된 카테고리의 도구들을 등록 및 활성화합니다."""
    registry = get_registry()

    if categories is None:
        categories = get_available_categories()

    logger.info(f"Initializing tools with categories: {categories}")

    if "file_tools" in categories:
        from javis.tools.file_tools import register_file_tools
        register_file_tools()
        registry.enable_category("file_tools")

    if "web_tools" in categories:
        from javis.tools.web_tools import register_web_tools
        register_web_tools()
        registry.enable_category("web_tools")

    if "system_tools" in categories:
        from javis.tools.system_tools import register_system_tools
        register_system_tools()
        registry.enable_category("system_tools")

    if "calendar_tools" in categories:
        from javis.tools.calendar_tools import register_calendar_tools
        register_calendar_tools()
        registry.enable_category("calendar_tools")

    if "notion_tools" in categories:
        from javis.tools.notion_tools import register_notion_tools
        register_notion_tools()
        registry.enable_category("notion_tools")

    if "slack_tools" in categories:
        from javis.tools.slack_tools import register_slack_tools
        register_slack_tools()
        registry.enable_category("slack_tools")

    logger.info(f"Tool initialization complete: {len(registry.get_enabled_tools())} tools enabled")


def get_tools_info() -> dict:
    """현재 등록된 도구들의 정보를 반환합니다."""
    registry = get_registry()
    return {
        "registered_tools": registry.get_tool_names(),
        "enabled_tools": registry.get_enabled_tool_names(),
        "enabled_categories": list(registry.enabled_categories),
    }
