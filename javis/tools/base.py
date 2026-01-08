"""Base classes for JAVIS tool system."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """도구 파라미터 정의."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: Any = None


class ToolDefinition(BaseModel):
    """Groq API 호환 도구 정의."""
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


class ToolResult(BaseModel):
    """도구 실행 결과."""
    success: bool
    output: Any = None
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
    """모든 도구의 기본 추상 클래스."""

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
