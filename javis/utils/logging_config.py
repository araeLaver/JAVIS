"""Logging configuration for JAVIS application.

Provides both text and structured JSON logging formats.
JSON logging is useful for log aggregation systems like ELK, CloudWatch, etc.
"""

import json
import logging
import os
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Optional

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# Log format constants
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging.

    Outputs logs in JSON format with consistent structure:
    {
        "timestamp": "2024-01-15T10:30:00.123456Z",
        "level": "INFO",
        "logger": "javis.core.engine",
        "message": "Request processed",
        "request_id": "abc-123",
        "session_id": "sess-456",
        "user_id": "user-789",
        "module": "engine",
        "function": "process",
        "line": 42,
        "extra": {...}
    }
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_context: bool = True,
        include_location: bool = True,
        extra_fields: Optional[dict[str, Any]] = None,
    ):
        """Initialize JSON formatter.

        Args:
            include_timestamp: Include ISO timestamp
            include_context: Include request/session/user context
            include_location: Include file/function/line info
            extra_fields: Static fields to add to every log
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.include_location = include_location
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: dict[str, Any] = {}

        # Timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()

        # Basic log info
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["message"] = record.getMessage()

        # Context from context variables
        if self.include_context:
            request_id = request_id_var.get()
            session_id = session_id_var.get()
            user_id = user_id_var.get()

            if request_id:
                log_data["request_id"] = request_id
            if session_id:
                log_data["session_id"] = session_id
            if user_id:
                log_data["user_id"] = user_id

        # Location info
        if self.include_location:
            log_data["module"] = record.module
            log_data["function"] = record.funcName
            log_data["line"] = record.lineno

        # Exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Extra fields from record
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            }:
                try:
                    json.dumps(value)  # Test if serializable
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_data["extra"] = extra

        # Static extra fields
        if self.extra_fields:
            log_data.update(self.extra_fields)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ContextFilter(logging.Filter):
    """Filter that adds context variables to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record and allow all records."""
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        record.user_id = user_id_var.get()
        return True


def setup_logging(
    level: int = logging.INFO,
    format_style: str = "default",
    log_file: Optional[str] = None,
    json_logs: Optional[bool] = None,
    service_name: Optional[str] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_style: Format style - "default", "detailed", or "simple"
        log_file: Optional file path to write logs to
        json_logs: Enable JSON logging. If None, reads from JAVIS_JSON_LOGS env var
        service_name: Service name to include in JSON logs
    """
    # Check environment for JSON logging preference
    if json_logs is None:
        json_logs = os.getenv("JAVIS_JSON_LOGS", "false").lower() == "true"

    # Configure root logger
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        extra_fields = {}
        if service_name:
            extra_fields["service"] = service_name
        console_handler.setFormatter(JSONFormatter(extra_fields=extra_fields))
    else:
        format_map = {
            "default": DEFAULT_FORMAT,
            "detailed": DETAILED_FORMAT,
            "simple": SIMPLE_FORMAT,
        }
        log_format = format_map.get(format_style, DEFAULT_FORMAT)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=DATE_FORMAT))

    console_handler.addFilter(ContextFilter())
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        if json_logs:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT))
        file_handler.addFilter(ContextFilter())
        handlers.append(file_handler)

    # Apply configuration
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["httpx", "httpcore", "chromadb", "urllib3", "uvicorn.access"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    This is the preferred way to get a logger in JAVIS modules.
    Always use __name__ as the logger name for proper hierarchy.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        from javis.utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Operation completed")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class that provides a logger property.

    Inherit from this class to add automatic logger to any class.

    Example:
        class MyService(LoggerMixin):
            def do_something(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function entry and exit.

    Args:
        logger: Logger instance to use
        level: Log level for the messages

    Example:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.log(level, f"Exiting {func_name} with exception: {e}")
                raise
        return wrapper
    return decorator


def log_async_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Async decorator to log function entry and exit.

    Args:
        logger: Logger instance to use
        level: Log level for the messages

    Example:
        @log_async_function_call(logger)
        async def my_async_function(arg1):
            return await some_operation()
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Entering {func_name}")
            try:
                result = await func(*args, **kwargs)
                logger.log(level, f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.log(level, f"Exiting {func_name} with exception: {e}")
                raise
        return wrapper
    return decorator


def set_request_context(
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Set context variables for the current request.

    Call this at the start of request handling to add context to all logs.

    Args:
        request_id: Unique request identifier
        session_id: Session identifier
        user_id: User identifier
    """
    if request_id:
        request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    """Clear all context variables.

    Call this at the end of request handling.
    """
    request_id_var.set(None)
    session_id_var.set(None)
    user_id_var.set(None)


class LogContext:
    """Context manager for setting request context.

    Example:
        with LogContext(request_id="abc-123", session_id="sess-456"):
            logger.info("Processing request")  # Includes context
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        self._tokens: list = []

    def __enter__(self) -> "LogContext":
        if self.request_id:
            self._tokens.append(request_id_var.set(self.request_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for token in self._tokens:
            # Reset is not available in all Python versions, so we set to None
            pass
        clear_request_context()


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra: Any,
) -> None:
    """Log a message with extra context fields.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **extra: Additional fields to include in the log

    Example:
        log_with_context(logger, logging.INFO, "User action", action="login", duration_ms=150)
    """
    logger.log(level, message, extra=extra)
