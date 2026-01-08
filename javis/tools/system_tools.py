"""System information tools for JAVIS."""

import platform
from datetime import datetime
from typing import Optional

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class SystemInfoTool(BaseTool):
    """시스템 정보 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system_info",
            description="현재 시스템의 정보(OS, CPU, 메모리, 디스크 등)를 조회합니다.",
            parameters=[]
        )

    async def execute(self) -> ToolResult:
        try:
            import psutil

            info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "platform_release": platform.release(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "used_percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                    "used_percent": psutil.disk_usage('/').percent
                },
                "current_time": datetime.now().isoformat()
            }
            return ToolResult(success=True, output=info)

        except ImportError:
            # psutil이 없는 경우 기본 정보만 반환
            info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "current_time": datetime.now().isoformat(),
                "note": "psutil not installed - limited system info"
            }
            return ToolResult(success=True, output=info)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class GetDateTimeTool(BaseTool):
    """현재 날짜/시간 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_datetime",
            description="현재 날짜와 시간을 반환합니다. 형식을 지정할 수 있습니다.",
            parameters=[
                ToolParameter(
                    name="format",
                    type="string",
                    description="날짜 형식 (예: %Y-%m-%d %H:%M:%S). 기본값은 ISO 형식입니다.",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="타임존 (예: Asia/Seoul, UTC). 기본값은 시스템 로컬 시간입니다.",
                    required=False,
                    default=None
                )
            ]
        )

    async def execute(
        self,
        format: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> ToolResult:
        try:
            now = datetime.now()

            # 타임존 처리
            if timezone:
                try:
                    from zoneinfo import ZoneInfo
                    now = datetime.now(ZoneInfo(timezone))
                except ImportError:
                    # Python < 3.9 또는 tzdata 없음
                    pass
                except Exception:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Invalid timezone: {timezone}"
                    )

            # 형식 지정
            if format:
                try:
                    formatted = now.strftime(format)
                except Exception as e:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Invalid format string: {e}"
                    )
            else:
                formatted = now.isoformat()

            return ToolResult(
                success=True,
                output={
                    "datetime": formatted,
                    "timestamp": now.timestamp(),
                    "timezone": timezone or "local"
                }
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


def register_system_tools() -> None:
    """시스템 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(SystemInfoTool(), "system_tools")
    registry.register(GetDateTimeTool(), "system_tools")
