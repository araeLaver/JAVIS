"""Slack tools for JAVIS."""

from typing import Any, Optional

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class SlackPostMessageTool(BaseTool):
    """Slack 메시지 전송 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slack_post_message",
            description="Slack 채널에 메시지를 전송합니다.",
            parameters=[
                ToolParameter(
                    name="channel",
                    type="string",
                    description="채널 ID 또는 이름 (예: #general, C1234567890)",
                    required=True
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description="전송할 메시지",
                    required=True
                ),
                ToolParameter(
                    name="thread_ts",
                    type="string",
                    description="스레드 타임스탬프 (답글인 경우)",
                    required=False,
                    default=None
                )
            ]
        )

    async def execute(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None
    ) -> ToolResult:
        try:
            from javis.integrations.slack_client import SlackClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.slack.enabled:
                return ToolResult(
                    success=False,
                    error="Slack 연동이 비활성화되어 있습니다. config.yaml에서 활성화하세요."
                )

            bot_token = config.slack_bot_token
            if not bot_token:
                return ToolResult(
                    success=False,
                    error="SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다."
                )

            # Use default channel if not specified with #
            if not channel.startswith("#") and not channel.startswith("C"):
                channel = config.integrations.slack.default_channel

            client = SlackClient(bot_token=bot_token)
            result = await client.post_message(
                channel=channel,
                text=text,
                thread_ts=thread_ts
            )

            return ToolResult(
                success=True,
                output={
                    "message": f"메시지가 {channel}에 전송되었습니다.",
                    **result
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Slack SDK가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SlackListChannelsTool(BaseTool):
    """Slack 채널 목록 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slack_list_channels",
            description="Slack 워크스페이스의 채널 목록을 조회합니다.",
            parameters=[
                ToolParameter(
                    name="types",
                    type="string",
                    description="채널 타입 (public_channel, private_channel)",
                    required=False,
                    default="public_channel,private_channel"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="최대 결과 수",
                    required=False,
                    default=100
                )
            ]
        )

    async def execute(
        self,
        types: str = "public_channel,private_channel",
        limit: int = 100
    ) -> ToolResult:
        try:
            from javis.integrations.slack_client import SlackClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.slack.enabled:
                return ToolResult(
                    success=False,
                    error="Slack 연동이 비활성화되어 있습니다."
                )

            bot_token = config.slack_bot_token
            if not bot_token:
                return ToolResult(
                    success=False,
                    error="SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다."
                )

            client = SlackClient(bot_token=bot_token)
            channels = await client.list_channels(types=types, limit=limit)

            return ToolResult(
                success=True,
                output={
                    "count": len(channels),
                    "channels": channels
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Slack SDK가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SlackGetHistoryTool(BaseTool):
    """Slack 채널 히스토리 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slack_get_history",
            description="Slack 채널의 메시지 히스토리를 조회합니다.",
            parameters=[
                ToolParameter(
                    name="channel",
                    type="string",
                    description="채널 ID",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="조회할 메시지 수",
                    required=False,
                    default=20
                )
            ]
        )

    async def execute(
        self,
        channel: str,
        limit: int = 20
    ) -> ToolResult:
        try:
            from javis.integrations.slack_client import SlackClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.slack.enabled:
                return ToolResult(
                    success=False,
                    error="Slack 연동이 비활성화되어 있습니다."
                )

            bot_token = config.slack_bot_token
            if not bot_token:
                return ToolResult(
                    success=False,
                    error="SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다."
                )

            client = SlackClient(bot_token=bot_token)
            messages = await client.get_channel_history(
                channel=channel,
                limit=limit
            )

            return ToolResult(
                success=True,
                output={
                    "count": len(messages),
                    "messages": messages
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Slack SDK가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SlackSearchTool(BaseTool):
    """Slack 메시지 검색 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slack_search",
            description="Slack 워크스페이스에서 메시지를 검색합니다.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색어",
                    required=True
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="결과 수",
                    required=False,
                    default=20
                )
            ]
        )

    async def execute(
        self,
        query: str,
        count: int = 20
    ) -> ToolResult:
        try:
            from javis.integrations.slack_client import SlackClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.slack.enabled:
                return ToolResult(
                    success=False,
                    error="Slack 연동이 비활성화되어 있습니다."
                )

            bot_token = config.slack_bot_token
            if not bot_token:
                return ToolResult(
                    success=False,
                    error="SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다."
                )

            client = SlackClient(bot_token=bot_token)
            results = await client.search_messages(query=query, count=count)

            return ToolResult(
                success=True,
                output={
                    "count": len(results),
                    "messages": results
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Slack SDK가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SlackAddReactionTool(BaseTool):
    """Slack 리액션 추가 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slack_add_reaction",
            description="Slack 메시지에 리액션(이모지)을 추가합니다.",
            parameters=[
                ToolParameter(
                    name="channel",
                    type="string",
                    description="채널 ID",
                    required=True
                ),
                ToolParameter(
                    name="timestamp",
                    type="string",
                    description="메시지 타임스탬프",
                    required=True
                ),
                ToolParameter(
                    name="emoji",
                    type="string",
                    description="이모지 이름 (콜론 없이, 예: thumbsup)",
                    required=True
                )
            ]
        )

    async def execute(
        self,
        channel: str,
        timestamp: str,
        emoji: str
    ) -> ToolResult:
        try:
            from javis.integrations.slack_client import SlackClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.slack.enabled:
                return ToolResult(
                    success=False,
                    error="Slack 연동이 비활성화되어 있습니다."
                )

            bot_token = config.slack_bot_token
            if not bot_token:
                return ToolResult(
                    success=False,
                    error="SLACK_BOT_TOKEN 환경변수가 설정되지 않았습니다."
                )

            client = SlackClient(bot_token=bot_token)
            success = await client.add_reaction(
                channel=channel,
                timestamp=timestamp,
                emoji=emoji
            )

            return ToolResult(
                success=True,
                output={
                    "message": f":{emoji}: 리액션이 추가되었습니다." if success else "리액션 추가 실패"
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Slack SDK가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register_slack_tools() -> None:
    """Slack 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(SlackPostMessageTool(), "slack_tools")
    registry.register(SlackListChannelsTool(), "slack_tools")
    registry.register(SlackGetHistoryTool(), "slack_tools")
    registry.register(SlackSearchTool(), "slack_tools")
    registry.register(SlackAddReactionTool(), "slack_tools")
