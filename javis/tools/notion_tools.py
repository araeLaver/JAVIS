"""Notion tools for JAVIS."""

from typing import Any, Optional

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class SearchNotionTool(BaseTool):
    """Notion 검색 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_notion",
            description="Notion 워크스페이스에서 페이지나 데이터베이스를 검색합니다.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색어",
                    required=True
                ),
                ToolParameter(
                    name="filter_type",
                    type="string",
                    description="검색 대상 ('page' 또는 'database')",
                    required=False,
                    enum=["page", "database"],
                    default=None
                ),
                ToolParameter(
                    name="page_size",
                    type="integer",
                    description="결과 수 (기본값: 10)",
                    required=False,
                    default=10
                )
            ]
        )

    async def execute(
        self,
        query: str,
        filter_type: Optional[str] = None,
        page_size: int = 10
    ) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다. config.yaml에서 활성화하세요."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            results = await client.search(
                query=query,
                filter_type=filter_type,
                page_size=page_size
            )

            if not results:
                return ToolResult(
                    success=True,
                    output={"message": "검색 결과가 없습니다.", "results": []}
                )

            return ToolResult(
                success=True,
                output={
                    "count": len(results),
                    "results": results
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetNotionPageTool(BaseTool):
    """Notion 페이지 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_notion_page",
            description="Notion 페이지의 상세 정보를 조회합니다.",
            parameters=[
                ToolParameter(
                    name="page_id",
                    type="string",
                    description="페이지 ID",
                    required=True
                )
            ]
        )

    async def execute(self, page_id: str) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            page = await client.get_page(page_id)

            return ToolResult(success=True, output=page)

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CreateNotionPageTool(BaseTool):
    """Notion 페이지 생성 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_notion_page",
            description="Notion에 새 페이지를 생성합니다.",
            parameters=[
                ToolParameter(
                    name="parent_id",
                    type="string",
                    description="부모 페이지 또는 데이터베이스 ID",
                    required=True
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="페이지 제목",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="초기 내용 (선택사항)",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="parent_type",
                    type="string",
                    description="부모 타입 ('page' 또는 'database')",
                    required=False,
                    enum=["page", "database"],
                    default="page"
                )
            ]
        )

    async def execute(
        self,
        parent_id: str,
        title: str,
        content: Optional[str] = None,
        parent_type: str = "page"
    ) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            page = await client.create_page(
                parent_id=parent_id,
                title=title,
                content=content,
                parent_type=parent_type
            )

            return ToolResult(
                success=True,
                output={
                    "message": f"페이지 '{title}'이(가) 생성되었습니다.",
                    "page": page
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class AppendNotionBlockTool(BaseTool):
    """Notion 블록 추가 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="append_notion_block",
            description="Notion 페이지에 내용을 추가합니다.",
            parameters=[
                ToolParameter(
                    name="page_id",
                    type="string",
                    description="페이지 ID",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="추가할 내용",
                    required=True
                ),
                ToolParameter(
                    name="block_type",
                    type="string",
                    description="블록 타입",
                    required=False,
                    enum=["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "code"],
                    default="paragraph"
                )
            ]
        )

    async def execute(
        self,
        page_id: str,
        content: str,
        block_type: str = "paragraph"
    ) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            result = await client.append_blocks(
                page_id=page_id,
                content=content,
                block_type=block_type
            )

            return ToolResult(
                success=True,
                output={
                    "message": "내용이 추가되었습니다.",
                    **result
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetNotionBlocksTool(BaseTool):
    """Notion 블록 내용 조회 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_notion_blocks",
            description="Notion 페이지의 내용(블록)을 조회합니다.",
            parameters=[
                ToolParameter(
                    name="page_id",
                    type="string",
                    description="페이지 ID",
                    required=True
                ),
                ToolParameter(
                    name="page_size",
                    type="integer",
                    description="조회할 블록 수",
                    required=False,
                    default=100
                )
            ]
        )

    async def execute(
        self,
        page_id: str,
        page_size: int = 100
    ) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            blocks = await client.get_block_children(
                block_id=page_id,
                page_size=page_size
            )

            return ToolResult(
                success=True,
                output={
                    "count": len(blocks),
                    "blocks": blocks
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class QueryNotionDatabaseTool(BaseTool):
    """Notion 데이터베이스 쿼리 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="query_notion_database",
            description="Notion 데이터베이스를 쿼리하여 항목을 조회합니다.",
            parameters=[
                ToolParameter(
                    name="database_id",
                    type="string",
                    description="데이터베이스 ID",
                    required=True
                ),
                ToolParameter(
                    name="page_size",
                    type="integer",
                    description="결과 수",
                    required=False,
                    default=100
                )
            ]
        )

    async def execute(
        self,
        database_id: str,
        page_size: int = 100
    ) -> ToolResult:
        try:
            from javis.integrations.notion_client import NotionClient
            from javis.utils.config import get_config

            config = get_config()

            if not config.integrations.notion.enabled:
                return ToolResult(
                    success=False,
                    error="Notion 연동이 비활성화되어 있습니다."
                )

            api_key = config.notion_api_key
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NOTION_API_KEY 환경변수가 설정되지 않았습니다."
                )

            client = NotionClient(api_key=api_key)
            items = await client.query_database(
                database_id=database_id,
                page_size=page_size
            )

            return ToolResult(
                success=True,
                output={
                    "count": len(items),
                    "items": items
                }
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Notion 라이브러리가 설치되지 않았습니다: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register_notion_tools() -> None:
    """Notion 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(SearchNotionTool(), "notion_tools")
    registry.register(GetNotionPageTool(), "notion_tools")
    registry.register(CreateNotionPageTool(), "notion_tools")
    registry.register(AppendNotionBlockTool(), "notion_tools")
    registry.register(GetNotionBlocksTool(), "notion_tools")
    registry.register(QueryNotionDatabaseTool(), "notion_tools")
