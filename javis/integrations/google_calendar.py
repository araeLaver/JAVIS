"""Google Calendar API client for JAVIS."""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Optional Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    Credentials = None
    ServiceAccountCredentials = None
    build = None
    HttpError = Exception


class GoogleCalendarClient:
    """Google Calendar API client.

    Supports both OAuth2 and service account authentication.
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        calendar_id: str = "primary",
        timezone: str = "Asia/Seoul",
    ):
        """Initialize Google Calendar client.

        Args:
            credentials_path: Path to credentials JSON file
            calendar_id: Calendar ID to use (default: "primary")
            timezone: Default timezone for events
        """
        self.credentials_path = credentials_path
        self.calendar_id = calendar_id
        self.timezone = timezone
        self._service = None

    def _get_service(self):
        """Get or create the Calendar service."""
        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API client not installed. "
                "Run: pip install google-api-python-client google-auth-oauthlib"
            )

        if self._service is not None:
            return self._service

        if not self.credentials_path:
            raise ValueError("credentials_path is required")

        # Try service account first, then OAuth2
        try:
            credentials = ServiceAccountCredentials.from_service_account_file(
                self.credentials_path, scopes=self.SCOPES
            )
        except Exception:
            # Try OAuth2 credentials
            credentials = Credentials.from_authorized_user_file(
                self.credentials_path, scopes=self.SCOPES
            )

        self._service = build("calendar", "v3", credentials=credentials)
        return self._service

    async def list_events(
        self,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 10,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List calendar events.

        Args:
            time_min: Start time (ISO format or "today", "tomorrow")
            time_max: End time (ISO format or relative)
            max_results: Maximum number of events to return
            query: Search query

        Returns:
            List of event dictionaries
        """
        service = self._get_service()
        tz = ZoneInfo(self.timezone)

        # Parse time_min
        if time_min is None or time_min == "today":
            start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_min == "tomorrow":
            start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
            start += timedelta(days=1)
        else:
            start = datetime.fromisoformat(time_min.replace("Z", "+00:00"))

        # Parse time_max
        if time_max is None:
            end = start + timedelta(days=7)  # Default: 1 week
        elif time_max == "today":
            end = datetime.now(tz).replace(hour=23, minute=59, second=59)
        elif time_max == "tomorrow":
            end = datetime.now(tz).replace(hour=23, minute=59, second=59)
            end += timedelta(days=1)
        else:
            end = datetime.fromisoformat(time_max.replace("Z", "+00:00"))

        try:
            events_result = (
                service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=start.isoformat(),
                    timeMax=end.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                    q=query,
                )
                .execute()
            )

            events = events_result.get("items", [])

            return [
                {
                    "id": event["id"],
                    "summary": event.get("summary", "(제목 없음)"),
                    "start": event["start"].get("dateTime", event["start"].get("date")),
                    "end": event["end"].get("dateTime", event["end"].get("date")),
                    "location": event.get("location"),
                    "description": event.get("description"),
                    "status": event.get("status"),
                }
                for event in events
            ]

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            raise

    async def get_event(self, event_id: str) -> dict[str, Any]:
        """Get a specific event.

        Args:
            event_id: Event ID

        Returns:
            Event dictionary
        """
        service = self._get_service()

        try:
            event = (
                service.events()
                .get(calendarId=self.calendar_id, eventId=event_id)
                .execute()
            )

            return {
                "id": event["id"],
                "summary": event.get("summary", "(제목 없음)"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "location": event.get("location"),
                "description": event.get("description"),
                "status": event.get("status"),
                "attendees": event.get("attendees", []),
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            raise

    async def create_event(
        self,
        summary: str,
        start: str,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        all_day: bool = False,
    ) -> dict[str, Any]:
        """Create a calendar event.

        Args:
            summary: Event title
            start: Start time (ISO format or "YYYY-MM-DD" for all-day)
            end: End time (defaults to 1 hour after start)
            description: Event description
            location: Event location
            all_day: Whether this is an all-day event

        Returns:
            Created event dictionary
        """
        service = self._get_service()

        if all_day or len(start) == 10:  # "YYYY-MM-DD" format
            event_body = {
                "summary": summary,
                "start": {"date": start},
                "end": {"date": end or start},
            }
        else:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            if end:
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            else:
                end_dt = start_dt + timedelta(hours=1)

            event_body = {
                "summary": summary,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": self.timezone},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": self.timezone},
            }

        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location

        try:
            event = (
                service.events()
                .insert(calendarId=self.calendar_id, body=event_body)
                .execute()
            )

            return {
                "id": event["id"],
                "summary": event.get("summary"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "htmlLink": event.get("htmlLink"),
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            raise

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update a calendar event.

        Args:
            event_id: Event ID to update
            summary: New title (optional)
            start: New start time (optional)
            end: New end time (optional)
            description: New description (optional)
            location: New location (optional)

        Returns:
            Updated event dictionary
        """
        service = self._get_service()

        # Get existing event
        event = (
            service.events()
            .get(calendarId=self.calendar_id, eventId=event_id)
            .execute()
        )

        # Update fields
        if summary:
            event["summary"] = summary
        if description is not None:
            event["description"] = description
        if location is not None:
            event["location"] = location
        if start:
            if len(start) == 10:
                event["start"] = {"date": start}
            else:
                event["start"] = {
                    "dateTime": start,
                    "timeZone": self.timezone,
                }
        if end:
            if len(end) == 10:
                event["end"] = {"date": end}
            else:
                event["end"] = {
                    "dateTime": end,
                    "timeZone": self.timezone,
                }

        try:
            updated = (
                service.events()
                .update(calendarId=self.calendar_id, eventId=event_id, body=event)
                .execute()
            )

            return {
                "id": updated["id"],
                "summary": updated.get("summary"),
                "start": updated["start"].get("dateTime", updated["start"].get("date")),
                "end": updated["end"].get("dateTime", updated["end"].get("date")),
            }

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            raise

    async def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event.

        Args:
            event_id: Event ID to delete

        Returns:
            True if deleted successfully
        """
        service = self._get_service()

        try:
            service.events().delete(
                calendarId=self.calendar_id, eventId=event_id
            ).execute()
            return True

        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            raise
