"""External service integrations for JAVIS.

Provides API clients for Google Calendar, Notion, and Slack.
"""

from javis.integrations.google_calendar import GoogleCalendarClient
from javis.integrations.notion_client import NotionClient
from javis.integrations.slack_client import SlackClient

__all__ = [
    "GoogleCalendarClient",
    "NotionClient",
    "SlackClient",
]
