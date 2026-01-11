"""Logging configuration for JAVIS application."""

import logging
import sys
from typing import Optional


# Log format constants
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    format_style: str = "default",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_style: Format style - "default", "detailed", or "simple"
        log_file: Optional file path to write logs to
    """
    # Select format based on style
    format_map = {
        "default": DEFAULT_FORMAT,
        "detailed": DETAILED_FORMAT,
        "simple": SIMPLE_FORMAT,
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)

    # Configure root logger
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=DATE_FORMAT))
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT))
        handlers.append(file_handler)

    # Apply configuration
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["httpx", "httpcore", "chromadb", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

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
    """
    Mixin class that provides a logger property.

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
    """
    Decorator to log function entry and exit.

    Args:
        logger: Logger instance to use
        level: Log level for the messages

    Example:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def decorator(func):
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
    """
    Async decorator to log function entry and exit.

    Args:
        logger: Logger instance to use
        level: Log level for the messages

    Example:
        @log_async_function_call(logger)
        async def my_async_function(arg1):
            return await some_operation()
    """
    def decorator(func):
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
