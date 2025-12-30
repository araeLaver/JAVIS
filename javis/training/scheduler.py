"""Background scheduler for automatic training triggers."""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from javis.utils.config import TrainingConfig, get_config

# APScheduler imports - will be available when installed
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    BackgroundScheduler = None
    CronTrigger = None

logger = logging.getLogger(__name__)


class SchedulerStatus(BaseModel):
    """Scheduler status information."""

    running: bool = False
    enabled: bool = False
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_result: Optional[str] = None
    cron: str = ""
    timezone: str = ""


class TrainingScheduler:
    """APScheduler-based training scheduler."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        if config is None:
            config = get_config().training
        self.config = config

        self._scheduler: Optional[BackgroundScheduler] = None
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[str] = None

    def start(self) -> bool:
        """Start the scheduler with configured cron job.

        Returns:
            True if scheduler started successfully
        """
        if not SCHEDULER_AVAILABLE:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            return False

        if not self.config.schedule.enabled:
            logger.info("Scheduler is disabled in config")
            return False

        if self._scheduler is not None and self._scheduler.running:
            logger.warning("Scheduler is already running")
            return True

        try:
            self._scheduler = BackgroundScheduler(
                timezone=self.config.schedule.timezone
            )

            # Parse cron expression
            cron_parts = self.config.schedule.cron.split()
            if len(cron_parts) == 5:
                trigger = CronTrigger(
                    minute=cron_parts[0],
                    hour=cron_parts[1],
                    day=cron_parts[2],
                    month=cron_parts[3],
                    day_of_week=cron_parts[4],
                    timezone=self.config.schedule.timezone,
                )
            else:
                logger.error(f"Invalid cron expression: {self.config.schedule.cron}")
                return False

            self._scheduler.add_job(
                self._run_training,
                trigger=trigger,
                id="auto_retrain",
                name="JAVIS Auto Retrain",
                replace_existing=True,
            )

            self._scheduler.start()
            logger.info(
                f"Scheduler started with cron: {self.config.schedule.cron} "
                f"(timezone: {self.config.schedule.timezone})"
            )

            # Log next run time
            job = self._scheduler.get_job("auto_retrain")
            if job and job.next_run_time:
                logger.info(f"Next scheduled run: {job.next_run_time}")

            return True

        except Exception as e:
            logger.exception(f"Failed to start scheduler: {e}")
            return False

    def stop(self) -> bool:
        """Stop the scheduler gracefully.

        Returns:
            True if scheduler stopped successfully
        """
        if self._scheduler is None:
            logger.info("Scheduler is not running")
            return True

        try:
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
            logger.info("Scheduler stopped")
            return True
        except Exception as e:
            logger.exception(f"Failed to stop scheduler: {e}")
            return False

    def trigger_now(self) -> bool:
        """Manually trigger a training run immediately.

        Returns:
            True if training was triggered successfully
        """
        logger.info("Manual training trigger requested")
        self._run_training()
        return True

    def _run_training(self) -> None:
        """Execute the training pipeline."""
        from .pipeline import TrainingPipeline

        logger.info("Scheduled training starting...")
        self._last_run = datetime.now()

        try:
            pipeline = TrainingPipeline(self.config)
            result = pipeline.run()

            if result.skipped:
                self._last_result = f"Skipped: {result.skip_reason}"
                logger.info(f"Training skipped: {result.skip_reason}")
            elif result.success:
                self._last_result = f"Success: {result.version}"
                logger.info(f"Training completed: {result.version}")
            else:
                self._last_result = f"Failed: {result.error}"
                logger.error(f"Training failed: {result.error}")

        except Exception as e:
            self._last_result = f"Error: {str(e)}"
            logger.exception(f"Training failed with exception: {e}")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._scheduler is not None and self._scheduler.running

    def get_next_run_time(self) -> Optional[str]:
        """Get the next scheduled run time."""
        if self._scheduler is None:
            return None

        job = self._scheduler.get_job("auto_retrain")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        return SchedulerStatus(
            running=self.is_running(),
            enabled=self.config.schedule.enabled,
            next_run=self.get_next_run_time(),
            last_run=self._last_run.isoformat() if self._last_run else None,
            last_result=self._last_result,
            cron=self.config.schedule.cron,
            timezone=self.config.schedule.timezone,
        )


# Global scheduler instance
_scheduler: Optional[TrainingScheduler] = None


def get_scheduler() -> TrainingScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingScheduler()
    return _scheduler


def start_scheduler() -> bool:
    """Start the global scheduler."""
    return get_scheduler().start()


def stop_scheduler() -> bool:
    """Stop the global scheduler."""
    return get_scheduler().stop()
