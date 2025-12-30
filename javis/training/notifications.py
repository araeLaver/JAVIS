"""Multi-channel notification service for training events."""

import logging
from typing import Optional

import httpx

from javis.utils.config import TrainingNotificationsConfig, get_config

logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications via multiple channels."""

    def __init__(self, config: Optional[TrainingNotificationsConfig] = None):
        if config is None:
            config = get_config().training.notifications
        self.config = config

        # Also check env var for discord webhook
        if not self.config.discord_webhook:
            full_config = get_config()
            if full_config.discord_webhook_url:
                self.config.discord_webhook = full_config.discord_webhook_url

    def notify_success(self, version: str, metadata: dict) -> None:
        """Notify on successful training.

        Args:
            version: The new model version
            metadata: Training metadata (dataset_size, duration, etc.)
        """
        if not self.config.on_success:
            return

        dataset_size = metadata.get("dataset_size", "?")
        duration = metadata.get("duration_seconds", 0)
        duration_min = duration / 60 if duration else 0
        auto_deployed = metadata.get("auto_deployed", False)

        message = f"""**JAVIS Training Complete**

**Version**: `{version}`
**Dataset**: {dataset_size} conversations
**Duration**: {duration_min:.1f} minutes
**Status**: {"Auto-deployed" if auto_deployed else "Ready for deployment"}
"""
        self._send_all(message, success=True)

    def notify_failure(self, error: str, context: dict) -> None:
        """Notify on training failure.

        Args:
            error: Error message
            context: Additional context (version, stage, etc.)
        """
        if not self.config.on_failure:
            return

        version = context.get("version", "unknown")

        message = f"""**JAVIS Training Failed**

**Version**: `{version}`
**Error**: {error}

Please check the logs for details.
"""
        self._send_all(message, success=False)

    def notify_scheduled(self, next_run: str) -> None:
        """Notify when training is scheduled.

        Args:
            next_run: Next scheduled run time
        """
        message = f"""**JAVIS Training Scheduled**

Next training run: {next_run}
"""
        self._send_all(message, success=True)

    def _send_all(self, message: str, success: bool = True) -> None:
        """Send notification to all configured channels.

        Args:
            message: Message to send
            success: Whether this is a success (green) or failure (red) message
        """
        if self.config.discord_webhook:
            self._send_discord(message, success)

    def _send_discord(self, message: str, success: bool = True) -> None:
        """Send notification to Discord webhook.

        Args:
            message: Message to send
            success: Whether this is a success or failure
        """
        if not self.config.discord_webhook:
            return

        # Discord embed color: green for success, red for failure
        color = 0x00FF00 if success else 0xFF0000

        payload = {
            "embeds": [
                {
                    "description": message,
                    "color": color,
                }
            ]
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.config.discord_webhook, json=payload)
                response.raise_for_status()
            logger.info("Discord notification sent")
        except httpx.HTTPError as e:
            logger.error(f"Failed to send Discord notification: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Discord notification: {e}")


# Global instance
_notifier: Optional[NotificationService] = None


def get_notifier() -> NotificationService:
    """Get the global notification service instance."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationService()
    return _notifier
