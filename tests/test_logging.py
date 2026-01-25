"""Tests for logging configuration module."""

import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch

from javis.utils.logging_config import (
    JSONFormatter,
    ContextFilter,
    LoggerMixin,
    LogContext,
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    log_function_call,
    log_async_function_call,
    log_with_context,
    request_id_var,
    session_id_var,
    user_id_var,
)


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_format_without_timestamp(self):
        """Test formatting without timestamp."""
        formatter = JSONFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_format_without_location(self):
        """Test formatting without location info."""
        formatter = JSONFormatter(include_location=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "module" not in data
        assert "function" not in data
        assert "line" not in data

    def test_format_with_extra_fields(self):
        """Test formatting with static extra fields."""
        formatter = JSONFormatter(extra_fields={"service": "javis", "env": "test"})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["service"] == "javis"
        assert data["env"] == "test"

    def test_format_with_context(self):
        """Test formatting with request context."""
        formatter = JSONFormatter()

        # Set context
        request_id_var.set("req-123")
        session_id_var.set("sess-456")
        user_id_var.set("user-789")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            data = json.loads(result)

            assert data["request_id"] == "req-123"
            assert data["session_id"] == "sess-456"
            assert data["user_id"] == "user-789"
        finally:
            clear_request_context()

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]
        assert isinstance(data["exception"]["traceback"], list)


class TestContextFilter:
    """Tests for ContextFilter class."""

    def test_filter_adds_context(self):
        """Test that filter adds context to records."""
        filter_instance = ContextFilter()

        # Set context
        request_id_var.set("req-abc")
        session_id_var.set("sess-def")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = filter_instance.filter(record)

            assert result is True
            assert record.request_id == "req-abc"
            assert record.session_id == "sess-def"
        finally:
            clear_request_context()


class TestContextFunctions:
    """Tests for context management functions."""

    def test_set_and_clear_context(self):
        """Test setting and clearing context."""
        set_request_context(
            request_id="req-123",
            session_id="sess-456",
            user_id="user-789",
        )

        assert request_id_var.get() == "req-123"
        assert session_id_var.get() == "sess-456"
        assert user_id_var.get() == "user-789"

        clear_request_context()

        assert request_id_var.get() is None
        assert session_id_var.get() is None
        assert user_id_var.get() is None

    def test_set_partial_context(self):
        """Test setting partial context."""
        set_request_context(request_id="req-only")

        assert request_id_var.get() == "req-only"
        assert session_id_var.get() is None

        clear_request_context()


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_sets_and_clears(self):
        """Test LogContext sets and clears context."""
        with LogContext(request_id="ctx-req", session_id="ctx-sess"):
            assert request_id_var.get() == "ctx-req"
            assert session_id_var.get() == "ctx-sess"

        assert request_id_var.get() is None
        assert session_id_var.get() is None

    def test_log_context_partial(self):
        """Test LogContext with partial context."""
        with LogContext(user_id="ctx-user"):
            assert user_id_var.get() == "ctx-user"

        assert user_id_var.get() is None


class TestLoggerMixin:
    """Tests for LoggerMixin class."""

    def test_logger_mixin_provides_logger(self):
        """Test LoggerMixin provides logger property."""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        logger = instance.logger

        assert isinstance(logger, logging.Logger)
        assert "TestClass" in logger.name


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_text_format(self):
        """Test setup with text format."""
        # Reset logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        setup_logging(level=logging.DEBUG, format_style="simple", json_logs=False)

        logger = logging.getLogger("test_setup")
        assert logger.level == logging.NOTSET  # Inherits from root

    def test_setup_logging_json_format(self):
        """Test setup with JSON format."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        setup_logging(level=logging.INFO, json_logs=True, service_name="test-service")

        # Check that handler uses JSONFormatter
        has_json_formatter = False
        for handler in logging.root.handlers:
            if isinstance(handler.formatter, JSONFormatter):
                has_json_formatter = True
                break

        assert has_json_formatter

    def test_setup_logging_from_env(self):
        """Test setup reads from environment."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        with patch.dict("os.environ", {"JAVIS_JSON_LOGS": "true"}):
            setup_logging()

        has_json_formatter = any(
            isinstance(h.formatter, JSONFormatter)
            for h in logging.root.handlers
        )
        assert has_json_formatter


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"


class TestFunctionDecorators:
    """Tests for function call decorators."""

    def test_log_function_call(self):
        """Test log_function_call decorator."""
        logger = logging.getLogger("test_decorator")
        log_records = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = TestHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        @log_function_call(logger, logging.DEBUG)
        def test_func(x, y):
            return x + y

        result = test_func(1, 2)

        assert result == 3
        assert len(log_records) >= 2
        assert any("Entering test_func" in r.getMessage() for r in log_records)
        assert any("Exiting test_func" in r.getMessage() for r in log_records)

        logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_log_async_function_call(self):
        """Test log_async_function_call decorator."""
        logger = logging.getLogger("test_async_decorator")
        log_records = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = TestHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        @log_async_function_call(logger, logging.DEBUG)
        async def async_test_func(x):
            return x * 2

        result = await async_test_func(5)

        assert result == 10
        assert any("Entering async_test_func" in r.getMessage() for r in log_records)
        assert any("Exiting async_test_func" in r.getMessage() for r in log_records)

        logger.removeHandler(handler)


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context_adds_extra(self):
        """Test log_with_context adds extra fields."""
        logger = logging.getLogger("test_context_log")
        log_records = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = TestHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_with_context(
            logger,
            logging.INFO,
            "Test message",
            action="test_action",
            duration_ms=100,
        )

        assert len(log_records) == 1
        record = log_records[0]
        assert record.action == "test_action"
        assert record.duration_ms == 100

        logger.removeHandler(handler)
