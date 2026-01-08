"""Tool registry for JAVIS."""

import logging
from typing import Optional

from javis.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """도구 레지스트리 - 싱글톤 패턴."""

    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._enabled_categories: set[str] = set()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """싱글톤 인스턴스 반환."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """레지스트리 초기화 (테스트용)."""
        cls._instance = None

    def register(self, tool: BaseTool, category: str) -> None:
        """도구 등록."""
        name = tool.definition.name
        self._tools[name] = tool
        tool._category = category
        logger.debug(f"Registered tool: {name} (category: {category})")

    def unregister(self, name: str) -> bool:
        """도구 등록 해제."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def enable_category(self, category: str) -> None:
        """카테고리 활성화."""
        self._enabled_categories.add(category)
        logger.info(f"Enabled tool category: {category}")

    def disable_category(self, category: str) -> None:
        """카테고리 비활성화."""
        self._enabled_categories.discard(category)
        logger.info(f"Disabled tool category: {category}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 조회."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[BaseTool]:
        """모든 등록된 도구 반환."""
        return list(self._tools.values())

    def get_enabled_tools(self) -> list[BaseTool]:
        """활성화된 카테고리의 도구들만 반환."""
        return [
            tool for tool in self._tools.values()
            if getattr(tool, "_category", None) in self._enabled_categories
        ]

    def get_openai_tools(self) -> list[dict]:
        """Groq/OpenAI API용 도구 스키마 반환."""
        return [
            tool.definition.to_openai_schema()
            for tool in self.get_enabled_tools()
        ]

    def get_tool_names(self) -> list[str]:
        """등록된 모든 도구 이름 반환."""
        return list(self._tools.keys())

    def get_enabled_tool_names(self) -> list[str]:
        """활성화된 도구 이름만 반환."""
        return [tool.name for tool in self.get_enabled_tools()]

    @property
    def enabled_categories(self) -> set[str]:
        """활성화된 카테고리 집합."""
        return self._enabled_categories.copy()


def get_registry() -> ToolRegistry:
    """레지스트리 싱글톤 인스턴스 반환."""
    return ToolRegistry.get_instance()
