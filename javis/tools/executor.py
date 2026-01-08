"""Tool executor for JAVIS."""

import asyncio
import json
import logging
from typing import Any

from javis.tools.base import ToolResult
from javis.tools.registry import get_registry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """도구 실행기 - 타임아웃 및 병렬 실행 지원."""

    def __init__(self, timeout: float = 30.0, max_parallel: int = 5):
        """
        Args:
            timeout: 도구 실행 타임아웃 (초)
            max_parallel: 최대 병렬 실행 수
        """
        self.timeout = timeout
        self.max_parallel = max_parallel
        self.registry = get_registry()

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """단일 도구 실행."""
        tool = self.registry.get_tool(tool_name)

        if tool is None:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}"
            )

        # 파라미터 유효성 검사
        is_valid, error_msg = tool.validate_params(arguments)
        if not is_valid:
            return ToolResult(
                success=False,
                output=None,
                error=error_msg
            )

        try:
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=self.timeout
            )
            logger.info(f"Tool {tool_name} completed: success={result.success}")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} timed out after {self.timeout}s")
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution timed out after {self.timeout}s"
            )
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    async def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]]
    ) -> dict[str, ToolResult]:
        """
        병렬 도구 실행.

        Args:
            tool_calls: Groq API tool_calls 형식의 리스트
                [{"id": "...", "function": {"name": "...", "arguments": "..."}}]

        Returns:
            도구 호출 ID를 키로 하는 결과 딕셔너리
        """
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def limited_execute(call: dict) -> tuple[str, ToolResult]:
            async with semaphore:
                call_id = call["id"]
                tool_name = call["function"]["name"]
                arguments = call["function"]["arguments"]

                # arguments가 문자열이면 JSON 파싱
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as e:
                        return call_id, ToolResult(
                            success=False,
                            output=None,
                            error=f"Invalid JSON arguments: {e}"
                        )

                result = await self.execute(tool_name, arguments)
                return call_id, result

        tasks = [limited_execute(call) for call in tool_calls]
        results = await asyncio.gather(*tasks)

        return {call_id: result for call_id, result in results}

    async def execute_single_call(self, tool_call: dict[str, Any]) -> ToolResult:
        """
        단일 tool_call 객체 실행.

        Args:
            tool_call: Groq API tool_call 형식
                {"id": "...", "function": {"name": "...", "arguments": "..."}}
        """
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid JSON arguments: {e}"
                )

        return await self.execute(tool_name, arguments)
