"""Google Calendar tools for JAVIS."""

import os
from typing import Optional

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class ListCalendarEventsTool(BaseTool):
    """일정 목록 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_calendar_events",
            description="Google Calendar에서 일정을 조회합니다. 오늘, 내일, 특정 기간의 일정을 확인할 수 있습니다.",
            parameters=[
                ToolParameter(
                    name="time_min",
                    type="string",
                    description="시작 시간 ('today', 'tomorrow', 또는 ISO 형식 날짜/시간)",
                    required=False,
                    default="today"
                ),
                ToolParameter(
                    name="time_max",
                    type="string",
                    description="종료 시간 ('today', 'tomorrow', 또는 ISO 형식 날짜/시간)",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="최대 결과 수 (기본값: 10)",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 키워드",
                    required=False,
                    default=None
                )
            ]
        )

    async def execute(
        self,
        time_min: Optional[str] = "today",
        time_max: Optional[str] = None,
        max_results: int = 10,
        query: Optional[str] = None
    ) -> ToolResult:
        try:
            from javis.integrations.google_calendar import GoogleCalendarClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.google_calendar.enabled:
                return ToolResult(
                    success=False,
                    error="Google Calendar 연동이 비활성화되어 있습니다. config.yaml에서 활성화하세요."
                )

            credentials_path = config.google_credentials_path
            if not credentials_path:
                return ToolResult(
                    success=False,
                    error="GOOGLE_CREDENTIALS_PATH 환경변수가 설정되지 않았습니다."
                )

            client = GoogleCalendarClient(
                credentials_path=credentials_path,
                calendar_id=config.integrations.google_calendar.calendar_id,
            )

            events = await client.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
                query=query
            )

            if not events:
                return ToolResult(
                    success=True,
                    output={"message": "일정이 없습니다.", "events": []}
                )

            return ToolResult(
                success=True,
                output={
                    "count": len(events),
                    "events": events
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Google Calendar 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CreateCalendarEventTool(BaseTool):
    """일정 생성 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_calendar_event",
            description="Google Calendar에 새 일정을 생성합니다.",
            parameters=[
                ToolParameter(
                    name="summary",
                    type="string",
                    description="일정 제목",
                    required=True
                ),
                ToolParameter(
                    name="start",
                    type="string",
                    description="시작 시간 (ISO 형식: 2024-01-15T10:00:00 또는 종일 일정: 2024-01-15)",
                    required=True
                ),
                ToolParameter(
                    name="end",
                    type="string",
                    description="종료 시간 (기본값: 시작 시간 + 1시간)",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="일정 설명",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="location",
                    type="string",
                    description="장소",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="all_day",
                    type="boolean",
                    description="종일 일정 여부",
                    required=False,
                    default=False
                )
            ]
        )

    async def execute(
        self,
        summary: str,
        start: str,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        all_day: bool = False
    ) -> ToolResult:
        try:
            from javis.integrations.google_calendar import GoogleCalendarClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.google_calendar.enabled:
                return ToolResult(
                    success=False,
                    error="Google Calendar 연동이 비활성화되어 있습니다."
                )

            credentials_path = config.google_credentials_path
            if not credentials_path:
                return ToolResult(
                    success=False,
                    error="GOOGLE_CREDENTIALS_PATH 환경변수가 설정되지 않았습니다."
                )

            client = GoogleCalendarClient(
                credentials_path=credentials_path,
                calendar_id=config.integrations.google_calendar.calendar_id,
            )

            event = await client.create_event(
                summary=summary,
                start=start,
                end=end,
                description=description,
                location=location,
                all_day=all_day
            )

            return ToolResult(
                success=True,
                output={
                    "message": f"일정 '{summary}'이(가) 생성되었습니다.",
                    "event": event
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Google Calendar 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetCalendarEventTool(BaseTool):
    """특정 일정 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_calendar_event",
            description="Google Calendar에서 특정 일정의 상세 정보를 조회합니다.",
            parameters=[
                ToolParameter(
                    name="event_id",
                    type="string",
                    description="일정 ID",
                    required=True
                )
            ]
        )

    async def execute(self, event_id: str) -> ToolResult:
        try:
            from javis.integrations.google_calendar import GoogleCalendarClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.google_calendar.enabled:
                return ToolResult(
                    success=False,
                    error="Google Calendar 연동이 비활성화되어 있습니다."
                )

            credentials_path = config.google_credentials_path
            if not credentials_path:
                return ToolResult(
                    success=False,
                    error="GOOGLE_CREDENTIALS_PATH 환경변수가 설정되지 않았습니다."
                )

            client = GoogleCalendarClient(
                credentials_path=credentials_path,
                calendar_id=config.integrations.google_calendar.calendar_id,
            )

            event = await client.get_event(event_id)

            return ToolResult(success=True, output=event)

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Google Calendar 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class UpdateCalendarEventTool(BaseTool):
    """일정 수정 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_calendar_event",
            description="Google Calendar의 기존 일정을 수정합니다.",
            parameters=[
                ToolParameter(
                    name="event_id",
                    type="string",
                    description="수정할 일정 ID",
                    required=True
                ),
                ToolParameter(
                    name="summary",
                    type="string",
                    description="새 제목",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="start",
                    type="string",
                    description="새 시작 시간",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="end",
                    type="string",
                    description="새 종료 시간",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="새 설명",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="location",
                    type="string",
                    description="새 장소",
                    required=False,
                    default=None
                )
            ]
        )

    async def execute(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> ToolResult:
        try:
            from javis.integrations.google_calendar import GoogleCalendarClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.google_calendar.enabled:
                return ToolResult(
                    success=False,
                    error="Google Calendar 연동이 비활성화되어 있습니다."
                )

            credentials_path = config.google_credentials_path
            if not credentials_path:
                return ToolResult(
                    success=False,
                    error="GOOGLE_CREDENTIALS_PATH 환경변수가 설정되지 않았습니다."
                )

            client = GoogleCalendarClient(
                credentials_path=credentials_path,
                calendar_id=config.integrations.google_calendar.calendar_id,
            )

            event = await client.update_event(
                event_id=event_id,
                summary=summary,
                start=start,
                end=end,
                description=description,
                location=location
            )

            return ToolResult(
                success=True,
                output={
                    "message": "일정이 수정되었습니다.",
                    "event": event
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Google Calendar 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DeleteCalendarEventTool(BaseTool):
    """일정 삭제 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_calendar_event",
            description="Google Calendar에서 일정을 삭제합니다.",
            parameters=[
                ToolParameter(
                    name="event_id",
                    type="string",
                    description="삭제할 일정 ID",
                    required=True
                )
            ]
        )

    async def execute(self, event_id: str) -> ToolResult:
        try:
            from javis.integrations.google_calendar import GoogleCalendarClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.google_calendar.enabled:
                return ToolResult(
                    success=False,
                    error="Google Calendar 연동이 비활성화되어 있습니다."
                )

            credentials_path = config.google_credentials_path
            if not credentials_path:
                return ToolResult(
                    success=False,
                    error="GOOGLE_CREDENTIALS_PATH 환경변수가 설정되지 않았습니다."
                )

            client = GoogleCalendarClient(
                credentials_path=credentials_path,
                calendar_id=config.integrations.google_calendar.calendar_id,
            )

            await client.delete_event(event_id)

            return ToolResult(
                success=True,
                output={"message": f"일정이 삭제되었습니다. (ID: {event_id})"}
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Google Calendar 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register_calendar_tools() -> None:
    """캘린더 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(ListCalendarEventsTool(), "calendar_tools")
    registry.register(CreateCalendarEventTool(), "calendar_tools")
    registry.register(GetCalendarEventTool(), "calendar_tools")
    registry.register(UpdateCalendarEventTool(), "calendar_tools")
    registry.register(DeleteCalendarEventTool(), "calendar_tools")
