"""Base classes for JAVIS tool system."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# Supported parameter types for OpenAI/Groq function calling
ParameterType = Literal["string", "integer", "boolean", "number", "array", "object"]

# Default value can be string, int, bool, float, list, or dict
DefaultValueType = Union[str, int, bool, float, list, dict, None]


class ToolParameter(BaseModel):
    """도구 파라미터 정의."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: DefaultValueType = None


class ToolDefinition(BaseModel):
    """
    OpenAI/Groq API compatible tool definition.

    Defines the schema for a callable tool that can be used by LLMs.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: List of parameters the tool accepts.

    Example:
        definition = ToolDefinition(
            name="search_web",
            description="Search the web for information",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query")
            ]
        )
    """

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """OpenAI/Groq 호환 스키마로 변환."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# Tool output can be string, dict, list, or None
ToolOutputType = Union[str, dict, list, None]


class ToolResult(BaseModel):
    """도구 실행 결과."""

    success: bool
    output: ToolOutputType = None
    error: Optional[str] = None

    def to_message_content(self) -> str:
        """메시지 콘텐츠로 변환."""
        import json
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, ensure_ascii=False, indent=2)
        return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all JAVIS tools.

    Subclass this to create new tools that can be called by the LLM.
    Each tool must define its schema via the `definition` property
    and implement the `execute` method.

    Attributes:
        _category: Internal category for tool organization and filtering.

    Example:
        class MyTool(BaseTool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="my_tool",
                    description="Does something useful",
                    parameters=[]
                )

            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="Done!")
    """

    _category: Optional[str] = None

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """도구 정의 반환."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """도구 실행. 서브클래스에서 구현."""
        pass

    @property
    def name(self) -> str:
        """도구 이름."""
        return self.definition.name

    @property
    def description(self) -> str:
        """도구 설명."""
        return self.definition.description

    def validate_params(self, params: dict) -> tuple[bool, Optional[str]]:
        """파라미터 유효성 검사."""
        required = [p.name for p in self.definition.parameters if p.required]
        missing = [r for r in required if r not in params]
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, None
