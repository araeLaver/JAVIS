"""Slack API client for JAVIS."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional Slack imports
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    SLACK_API_AVAILABLE = True
except ImportError:
    SLACK_API_AVAILABLE = False
    WebClient = None
    SlackApiError = Exception


class SlackClient:
    """Slack Web API client.

    Provides methods to send messages and interact with Slack.
    """

    def __init__(self, bot_token: Optional[str] = None):
        """Initialize Slack client.

        Args:
            bot_token: Slack Bot Token (xoxb-...)
        """
        self.bot_token = bot_token
        self._client = None

    def _get_client(self):
        """Get or create the Slack client."""
        if not SLACK_API_AVAILABLE:
            raise ImportError(
                "Slack SDK not installed. Run: pip install slack-sdk"
            )

        if self._client is not None:
            return self._client

        if not self.bot_token:
            raise ValueError("bot_token is required")

        self._client = WebClient(token=self.bot_token)
        return self._client

    async def post_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        blocks: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Post a message to a channel.

        Args:
            channel: Channel ID or name (e.g., "#general" or "C1234567890")
            text: Message text
            thread_ts: Thread timestamp for replies
            blocks: Block kit blocks for rich formatting

        Returns:
            Message response data
        """
        client = self._get_client()

        try:
            response = client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts,
                blocks=blocks,
            )

            return {
                "ok": response["ok"],
                "channel": response.get("channel"),
                "ts": response.get("ts"),
                "message": response.get("message", {}).get("text"),
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def list_channels(
        self,
        types: str = "public_channel,private_channel",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List channels in the workspace.

        Args:
            types: Channel types (comma-separated)
            limit: Maximum number of channels to return

        Returns:
            List of channel info dictionaries
        """
        client = self._get_client()

        try:
            response = client.conversations_list(
                types=types,
                limit=limit,
            )

            channels = response.get("channels", [])

            return [
                {
                    "id": ch["id"],
                    "name": ch.get("name"),
                    "is_private": ch.get("is_private", False),
                    "is_member": ch.get("is_member", False),
                    "topic": ch.get("topic", {}).get("value"),
                    "purpose": ch.get("purpose", {}).get("value"),
                    "num_members": ch.get("num_members", 0),
                }
                for ch in channels
            ]

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def get_channel_history(
        self,
        channel: str,
        limit: int = 20,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get message history from a channel.

        Args:
            channel: Channel ID
            limit: Number of messages to fetch
            oldest: Start of time range (timestamp)
            latest: End of time range (timestamp)

        Returns:
            List of messages
        """
        client = self._get_client()

        params: dict[str, Any] = {
            "channel": channel,
            "limit": limit,
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest

        try:
            response = client.conversations_history(**params)

            messages = response.get("messages", [])

            return [
                {
                    "ts": msg["ts"],
                    "user": msg.get("user"),
                    "text": msg.get("text"),
                    "type": msg.get("type"),
                    "thread_ts": msg.get("thread_ts"),
                    "reply_count": msg.get("reply_count", 0),
                }
                for msg in messages
            ]

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Get user information.

        Args:
            user_id: User ID

        Returns:
            User info dictionary
        """
        client = self._get_client()

        try:
            response = client.users_info(user=user_id)

            user = response.get("user", {})

            return {
                "id": user.get("id"),
                "name": user.get("name"),
                "real_name": user.get("real_name"),
                "display_name": user.get("profile", {}).get("display_name"),
                "email": user.get("profile", {}).get("email"),
                "is_bot": user.get("is_bot", False),
                "is_admin": user.get("is_admin", False),
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def add_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> bool:
        """Add a reaction to a message.

        Args:
            channel: Channel ID
            timestamp: Message timestamp
            emoji: Emoji name (without colons)

        Returns:
            True if successful
        """
        client = self._get_client()

        try:
            response = client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=emoji,
            )

            return response.get("ok", False)

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def upload_file(
        self,
        channels: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upload a file to Slack.

        Args:
            channels: Channel ID(s), comma-separated
            content: File content (text)
            file_path: Path to file to upload
            filename: Filename to display
            title: File title
            initial_comment: Comment to add

        Returns:
            File upload response
        """
        client = self._get_client()

        params: dict[str, Any] = {
            "channels": channels,
        }

        if content:
            params["content"] = content
        if file_path:
            params["file"] = file_path
        if filename:
            params["filename"] = filename
        if title:
            params["title"] = title
        if initial_comment:
            params["initial_comment"] = initial_comment

        try:
            response = client.files_upload_v2(**params)

            file_info = response.get("file", {})

            return {
                "ok": response.get("ok", False),
                "file_id": file_info.get("id"),
                "name": file_info.get("name"),
                "url": file_info.get("permalink"),
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise

    async def search_messages(
        self,
        query: str,
        count: int = 20,
    ) -> list[dict[str, Any]]:
        """Search messages in the workspace.

        Note: Requires search:read scope.

        Args:
            query: Search query
            count: Number of results

        Returns:
            List of matching messages
        """
        client = self._get_client()

        try:
            response = client.search_messages(
                query=query,
                count=count,
            )

            messages = response.get("messages", {}).get("matches", [])

            return [
                {
                    "ts": msg.get("ts"),
                    "channel": msg.get("channel", {}).get("id"),
                    "channel_name": msg.get("channel", {}).get("name"),
                    "user": msg.get("user"),
                    "username": msg.get("username"),
                    "text": msg.get("text"),
                    "permalink": msg.get("permalink"),
                }
                for msg in messages
            ]

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            raise
