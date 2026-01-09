"""Notion API client for JAVIS."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional Notion imports
try:
    from notion_client import Client as NotionSDKClient
    from notion_client import APIResponseError

    NOTION_API_AVAILABLE = True
except ImportError:
    NOTION_API_AVAILABLE = False
    NotionSDKClient = None
    APIResponseError = Exception


class NotionClient:
    """Notion API client.

    Provides methods to interact with Notion pages, databases, and blocks.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Notion client.

        Args:
            api_key: Notion API key (integration token)
        """
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create the Notion client."""
        if not NOTION_API_AVAILABLE:
            raise ImportError(
                "Notion client not installed. Run: pip install notion-client"
            )

        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError("api_key is required")

        self._client = NotionSDKClient(auth=self.api_key)
        return self._client

    async def search(
        self,
        query: str,
        filter_type: Optional[str] = None,
        page_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search Notion workspace.

        Args:
            query: Search query
            filter_type: Filter by type ("page" or "database")
            page_size: Number of results to return

        Returns:
            List of search results
        """
        client = self._get_client()

        search_params: dict[str, Any] = {
            "query": query,
            "page_size": page_size,
        }

        if filter_type:
            search_params["filter"] = {"property": "object", "value": filter_type}

        try:
            response = client.search(**search_params)
            results = response.get("results", [])

            return [
                {
                    "id": item["id"],
                    "type": item["object"],
                    "title": self._extract_title(item),
                    "url": item.get("url"),
                    "created_time": item.get("created_time"),
                    "last_edited_time": item.get("last_edited_time"),
                }
                for item in results
            ]

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Get a page by ID.

        Args:
            page_id: Notion page ID

        Returns:
            Page data
        """
        client = self._get_client()

        try:
            page = client.pages.retrieve(page_id=page_id)

            return {
                "id": page["id"],
                "title": self._extract_title(page),
                "url": page.get("url"),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
                "properties": page.get("properties", {}),
            }

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    async def create_page(
        self,
        parent_id: str,
        title: str,
        content: Optional[str] = None,
        parent_type: str = "page",
        properties: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new page.

        Args:
            parent_id: Parent page or database ID
            title: Page title
            content: Initial content (plain text)
            parent_type: "page" or "database"
            properties: Additional properties (for database items)

        Returns:
            Created page data
        """
        client = self._get_client()

        # Build parent reference
        if parent_type == "database":
            parent = {"database_id": parent_id}
            page_properties = properties or {}
            # Set title for database item
            page_properties["title"] = {
                "title": [{"text": {"content": title}}]
            }
        else:
            parent = {"page_id": parent_id}
            page_properties = {
                "title": {"title": [{"text": {"content": title}}]}
            }

        # Build children (content blocks)
        children = []
        if content:
            # Split content into paragraphs
            paragraphs = content.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": para}}]
                        }
                    })

        try:
            page = client.pages.create(
                parent=parent,
                properties=page_properties,
                children=children if children else None,
            )

            return {
                "id": page["id"],
                "title": title,
                "url": page.get("url"),
                "created_time": page.get("created_time"),
            }

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    async def append_blocks(
        self,
        page_id: str,
        content: str,
        block_type: str = "paragraph",
    ) -> dict[str, Any]:
        """Append content blocks to a page.

        Args:
            page_id: Page ID to append to
            content: Content to append
            block_type: Block type ("paragraph", "heading_1", "heading_2", "bulleted_list_item", etc.)

        Returns:
            Appended blocks info
        """
        client = self._get_client()

        # Build block based on type
        if block_type in ("heading_1", "heading_2", "heading_3"):
            block = {
                "object": "block",
                "type": block_type,
                block_type: {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                }
            }
        elif block_type == "bulleted_list_item":
            # Handle multiple items
            lines = content.split("\n")
            blocks = []
            for line in lines:
                if line.strip():
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{"type": "text", "text": {"content": line.strip()}}]
                        }
                    })
            if blocks:
                result = client.blocks.children.append(block_id=page_id, children=blocks)
                return {
                    "appended_blocks": len(result.get("results", [])),
                    "page_id": page_id
                }
            return {"appended_blocks": 0, "page_id": page_id}
        elif block_type == "code":
            block = {
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": content}}],
                    "language": "plain text"
                }
            }
        else:  # Default to paragraph
            block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                }
            }

        try:
            result = client.blocks.children.append(
                block_id=page_id,
                children=[block]
            )

            return {
                "appended_blocks": len(result.get("results", [])),
                "page_id": page_id
            }

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    async def get_block_children(
        self,
        block_id: str,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Get child blocks of a page or block.

        Args:
            block_id: Page or block ID
            page_size: Number of results

        Returns:
            List of child blocks
        """
        client = self._get_client()

        try:
            response = client.blocks.children.list(
                block_id=block_id,
                page_size=page_size
            )

            blocks = []
            for block in response.get("results", []):
                block_data = {
                    "id": block["id"],
                    "type": block["type"],
                    "has_children": block.get("has_children", False),
                }
                # Extract text content
                block_content = block.get(block["type"], {})
                rich_text = block_content.get("rich_text", [])
                if rich_text:
                    block_data["text"] = "".join(
                        t.get("plain_text", "") for t in rich_text
                    )
                blocks.append(block_data)

            return blocks

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    async def query_database(
        self,
        database_id: str,
        filter_obj: Optional[dict[str, Any]] = None,
        sorts: Optional[list[dict[str, Any]]] = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Query a database.

        Args:
            database_id: Database ID
            filter_obj: Filter object
            sorts: Sort specifications
            page_size: Number of results

        Returns:
            List of database items
        """
        client = self._get_client()

        query_params: dict[str, Any] = {
            "database_id": database_id,
            "page_size": page_size,
        }

        if filter_obj:
            query_params["filter"] = filter_obj
        if sorts:
            query_params["sorts"] = sorts

        try:
            response = client.databases.query(**query_params)

            return [
                {
                    "id": item["id"],
                    "title": self._extract_title(item),
                    "url": item.get("url"),
                    "properties": item.get("properties", {}),
                    "created_time": item.get("created_time"),
                }
                for item in response.get("results", [])
            ]

        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise

    def _extract_title(self, item: dict[str, Any]) -> str:
        """Extract title from a Notion object."""
        properties = item.get("properties", {})

        # Try common title property names
        for prop_name in ["title", "Title", "Name", "name", "이름"]:
            if prop_name in properties:
                prop = properties[prop_name]
                if prop.get("type") == "title":
                    title_content = prop.get("title", [])
                    if title_content:
                        return "".join(t.get("plain_text", "") for t in title_content)

        # Fallback: search all properties for title type
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                title_content = prop.get("title", [])
                if title_content:
                    return "".join(t.get("plain_text", "") for t in title_content)

        return "(제목 없음)"
