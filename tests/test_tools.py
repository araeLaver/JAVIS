"""Unit tests for JAVIS tool system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from javis.tools.base import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from javis.tools.registry import ToolRegistry, get_registry
from javis.tools.executor import ToolExecutor


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", delay: float = 0):
        self._name = name
        self._delay = delay

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="A mock tool for testing",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Test message",
                    required=True
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Optional count",
                    required=False,
                    default=1
                ),
            ]
        )

    async def execute(self, message: str, count: int = 1) -> ToolResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return ToolResult(
            success=True,
            output={"message": message, "count": count}
        )


class FailingTool(BaseTool):
    """Tool that always fails for testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failing_tool",
            description="A tool that always fails",
            parameters=[]
        )

    async def execute(self) -> ToolResult:
        raise ValueError("Intentional failure for testing")


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_openai_schema_basic(self):
        """Test OpenAI schema generation."""
        definition = ToolDefinition(
            name="test_tool",
            description="Test description",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                )
            ]
        )

        schema = definition.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "Test description"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_to_openai_schema_with_enum(self):
        """Test schema generation with enum parameter."""
        definition = ToolDefinition(
            name="format_tool",
            description="Format output",
            parameters=[
                ToolParameter(
                    name="format",
                    type="string",
                    description="Output format",
                    required=True,
                    enum=["json", "xml", "text"]
                )
            ]
        )

        schema = definition.to_openai_schema()
        props = schema["function"]["parameters"]["properties"]

        assert props["format"]["enum"] == ["json", "xml", "text"]

    def test_to_openai_schema_optional_param(self):
        """Test schema generation with optional parameter."""
        definition = ToolDefinition(
            name="optional_tool",
            description="Tool with optional param",
            parameters=[
                ToolParameter(
                    name="required_param",
                    type="string",
                    description="Required",
                    required=True
                ),
                ToolParameter(
                    name="optional_param",
                    type="integer",
                    description="Optional",
                    required=False,
                    default=10
                )
            ]
        )

        schema = definition.to_openai_schema()
        required = schema["function"]["parameters"]["required"]

        assert "required_param" in required
        assert "optional_param" not in required


class TestToolResult:
    """Tests for ToolResult."""

    def test_to_message_content_success_string(self):
        """Test message content for string output."""
        result = ToolResult(success=True, output="Hello, World!")
        assert result.to_message_content() == "Hello, World!"

    def test_to_message_content_success_dict(self):
        """Test message content for dict output."""
        result = ToolResult(success=True, output={"key": "value"})
        content = result.to_message_content()
        assert '"key": "value"' in content

    def test_to_message_content_error(self):
        """Test message content for error."""
        result = ToolResult(success=False, error="Something went wrong")
        assert "Error: Something went wrong" in result.to_message_content()


class TestBaseTool:
    """Tests for BaseTool."""

    def test_validate_params_success(self):
        """Test parameter validation with valid params."""
        tool = MockTool()
        is_valid, error = tool.validate_params({"message": "test"})
        assert is_valid is True
        assert error is None

    def test_validate_params_missing_required(self):
        """Test parameter validation with missing required param."""
        tool = MockTool()
        is_valid, error = tool.validate_params({})
        assert is_valid is False
        assert "message" in error

    def test_name_property(self):
        """Test name property."""
        tool = MockTool(name="custom_name")
        assert tool.name == "custom_name"

    def test_description_property(self):
        """Test description property."""
        tool = MockTool()
        assert tool.description == "A mock tool for testing"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        ToolRegistry.reset()
        yield
        ToolRegistry.reset()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = ToolRegistry.get_instance()
        registry2 = ToolRegistry.get_instance()
        assert registry1 is registry2

    def test_register_and_get_tool(self):
        """Test tool registration and retrieval."""
        registry = get_registry()
        tool = MockTool()

        registry.register(tool, "test_category")

        assert registry.get_tool("mock_tool") is tool
        assert tool._category == "test_category"

    def test_unregister_tool(self):
        """Test tool unregistration."""
        registry = get_registry()
        tool = MockTool()

        registry.register(tool, "test")
        assert registry.unregister("mock_tool") is True
        assert registry.get_tool("mock_tool") is None

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist."""
        registry = get_registry()
        assert registry.unregister("nonexistent") is False

    def test_enable_disable_category(self):
        """Test category enable/disable."""
        registry = get_registry()

        registry.enable_category("system")
        assert "system" in registry.enabled_categories

        registry.disable_category("system")
        assert "system" not in registry.enabled_categories

    def test_get_enabled_tools(self):
        """Test getting only enabled tools."""
        registry = get_registry()
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")

        registry.register(tool1, "category_a")
        registry.register(tool2, "category_b")
        registry.enable_category("category_a")

        enabled = registry.get_enabled_tools()
        enabled_names = [t.name for t in enabled]

        assert "tool1" in enabled_names
        assert "tool2" not in enabled_names

    def test_get_all_tools(self):
        """Test getting all registered tools."""
        registry = get_registry()
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")

        registry.register(tool1, "cat_a")
        registry.register(tool2, "cat_b")

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2

    def test_get_openai_tools(self):
        """Test getting OpenAI-format tool schemas."""
        registry = get_registry()
        tool = MockTool()

        registry.register(tool, "test")
        registry.enable_category("test")

        schemas = registry.get_openai_tools()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"

    def test_get_tool_names(self):
        """Test getting all tool names."""
        registry = get_registry()
        registry.register(MockTool(name="alpha"), "cat")
        registry.register(MockTool(name="beta"), "cat")

        names = registry.get_tool_names()
        assert "alpha" in names
        assert "beta" in names


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Setup registry with mock tools."""
        ToolRegistry.reset()
        registry = get_registry()
        registry.register(MockTool(), "test")
        registry.register(FailingTool(), "test")
        registry.enable_category("test")
        yield
        ToolRegistry.reset()

    @pytest.fixture
    def executor(self):
        return ToolExecutor(timeout=5.0, max_parallel=3)

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Test successful tool execution."""
        result = await executor.execute("mock_tool", {"message": "hello"})

        assert result.success is True
        assert result.output["message"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        """Test execution of unknown tool."""
        result = await executor.execute("unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_params(self, executor):
        """Test execution with missing required parameters."""
        result = await executor.execute("mock_tool", {})

        assert result.success is False
        assert "Missing required parameters" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, executor):
        """Test handling of tool execution error."""
        result = await executor.execute("failing_tool", {})

        assert result.success is False
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test tool execution timeout."""
        ToolRegistry.reset()
        registry = get_registry()
        slow_tool = MockTool(name="slow_tool", delay=2.0)
        registry.register(slow_tool, "test")
        registry.enable_category("test")

        executor = ToolExecutor(timeout=0.1)
        result = await executor.execute("slow_tool", {"message": "test"})

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_parallel(self, executor):
        """Test parallel tool execution."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "mock_tool",
                    "arguments": '{"message": "first"}'
                }
            },
            {
                "id": "call_2",
                "function": {
                    "name": "mock_tool",
                    "arguments": '{"message": "second"}'
                }
            }
        ]

        results = await executor.execute_parallel(tool_calls)

        assert len(results) == 2
        assert results["call_1"].success is True
        assert results["call_1"].output["message"] == "first"
        assert results["call_2"].success is True
        assert results["call_2"].output["message"] == "second"

    @pytest.mark.asyncio
    async def test_execute_parallel_with_dict_arguments(self, executor):
        """Test parallel execution with dict arguments (not JSON string)."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "mock_tool",
                    "arguments": {"message": "dict_args"}
                }
            }
        ]

        results = await executor.execute_parallel(tool_calls)

        assert results["call_1"].success is True
        assert results["call_1"].output["message"] == "dict_args"

    @pytest.mark.asyncio
    async def test_execute_parallel_invalid_json(self, executor):
        """Test parallel execution with invalid JSON arguments."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "mock_tool",
                    "arguments": "not valid json {"
                }
            }
        ]

        results = await executor.execute_parallel(tool_calls)

        assert results["call_1"].success is False
        assert "Invalid JSON" in results["call_1"].error

    @pytest.mark.asyncio
    async def test_execute_single_call(self, executor):
        """Test single tool call execution."""
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "mock_tool",
                "arguments": '{"message": "single"}'
            }
        }

        result = await executor.execute_single_call(tool_call)

        assert result.success is True
        assert result.output["message"] == "single"

    @pytest.mark.asyncio
    async def test_execute_single_call_invalid_json(self, executor):
        """Test single call with invalid JSON."""
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "mock_tool",
                "arguments": "{invalid"
            }
        }

        result = await executor.execute_single_call(tool_call)

        assert result.success is False
        assert "Invalid JSON" in result.error
