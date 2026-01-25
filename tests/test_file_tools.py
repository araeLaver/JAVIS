"""Unit tests for file tools security."""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from javis.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    validate_path_security,
)


class TestPathSecurityValidation:
    """Tests for validate_path_security function."""

    def test_valid_path_in_allowed_dir(self, tmp_path):
        """Test that paths within allowed directories are accepted."""
        allowed_dirs = [tmp_path]
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        is_valid, error, resolved = validate_path_security(
            test_file, allowed_dirs, check_exists=True
        )

        assert is_valid is True
        assert error == ""

    def test_path_outside_allowed_dir(self, tmp_path):
        """Test that paths outside allowed directories are rejected."""
        allowed_dirs = [tmp_path / "allowed"]
        test_file = tmp_path / "outside" / "test.txt"

        is_valid, error, resolved = validate_path_security(
            test_file, allowed_dirs, check_exists=False
        )

        assert is_valid is False
        assert "Access denied" in error

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        allowed_dirs = [tmp_path / "allowed"]
        (tmp_path / "allowed").mkdir(exist_ok=True)

        # Create a path that tries to escape
        malicious_path = tmp_path / "allowed" / ".." / "secret.txt"

        is_valid, error, resolved = validate_path_security(
            malicious_path, allowed_dirs, check_exists=False
        )

        # Should be rejected because resolved path is outside allowed dirs
        assert is_valid is False

    def test_symlink_blocked_by_default(self, tmp_path):
        """Test that symlinks are blocked by default."""
        allowed_dirs = [tmp_path]
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")

        symlink = tmp_path / "link.txt"
        try:
            symlink.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlink creation requires elevated privileges on Windows")

        is_valid, error, resolved = validate_path_security(
            symlink, allowed_dirs, check_exists=True, allow_symlinks=False
        )

        assert is_valid is False
        assert "Symbolic links" in error

    def test_nonexistent_path_with_check_exists(self, tmp_path):
        """Test that nonexistent paths fail when check_exists=True."""
        allowed_dirs = [tmp_path]
        nonexistent = tmp_path / "does_not_exist.txt"

        is_valid, error, resolved = validate_path_security(
            nonexistent, allowed_dirs, check_exists=True
        )

        assert is_valid is False
        assert "not found" in error

    def test_nonexistent_path_without_check_exists(self, tmp_path):
        """Test that nonexistent paths pass when check_exists=False."""
        allowed_dirs = [tmp_path]
        new_file = tmp_path / "new_file.txt"

        is_valid, error, resolved = validate_path_security(
            new_file, allowed_dirs, check_exists=False
        )

        assert is_valid is True


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.fixture
    def tool(self):
        return ReadFileTool()

    @pytest.mark.asyncio
    async def test_read_valid_file(self, tool, tmp_path, monkeypatch):
        """Test reading a valid file."""
        # Override ALLOWED_DIRS for testing
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await tool.execute(str(test_file))

        assert result.success is True
        assert result.output["content"] == "Hello, World!"
        assert result.output["size"] == 13

    @pytest.mark.asyncio
    async def test_read_file_outside_allowed_dirs(self, tool):
        """Test reading a file outside allowed directories."""
        result = await tool.execute("/etc/passwd")

        assert result.success is False
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tool, tmp_path, monkeypatch):
        """Test reading a nonexistent file."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        result = await tool.execute(str(tmp_path / "nonexistent.txt"))

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_too_large(self, tool, tmp_path, monkeypatch):
        """Test reading a file that exceeds size limit."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])
        monkeypatch.setattr(tool, "MAX_FILE_SIZE", 10)  # 10 bytes limit

        test_file = tmp_path / "large.txt"
        test_file.write_text("A" * 100)  # 100 bytes

        result = await tool.execute(str(test_file))

        assert result.success is False
        assert "too large" in result.error.lower()


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    @pytest.fixture
    def tool(self):
        return WriteFileTool()

    @pytest.mark.asyncio
    async def test_write_valid_file(self, tool, tmp_path, monkeypatch):
        """Test writing to a valid path."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        test_file = tmp_path / "output.txt"

        result = await tool.execute(str(test_file), "Hello, World!")

        assert result.success is True
        assert test_file.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_file_outside_allowed_dirs(self, tool):
        """Test writing to a path outside allowed directories."""
        result = await tool.execute("/tmp/evil.txt", "malicious content")

        assert result.success is False
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_append_mode(self, tool, tmp_path, monkeypatch):
        """Test appending to an existing file."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        test_file = tmp_path / "append.txt"
        test_file.write_text("Hello, ")

        result = await tool.execute(str(test_file), "World!", append=True)

        assert result.success is True
        assert test_file.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_content_too_large(self, tool, tmp_path, monkeypatch):
        """Test writing content that exceeds size limit."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])
        monkeypatch.setattr(tool, "MAX_CONTENT_SIZE", 10)

        result = await tool.execute(
            str(tmp_path / "large.txt"),
            "A" * 100
        )

        assert result.success is False
        assert "too large" in result.error.lower()


class TestListDirectoryTool:
    """Tests for ListDirectoryTool."""

    @pytest.fixture
    def tool(self):
        return ListDirectoryTool()

    @pytest.mark.asyncio
    async def test_list_valid_directory(self, tool, tmp_path, monkeypatch):
        """Test listing a valid directory."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()

        result = await tool.execute(str(tmp_path))

        assert result.success is True
        assert result.output["count"] == 3
        names = [f["name"] for f in result.output["files"]]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_directory_outside_allowed(self, tool):
        """Test listing a directory outside allowed directories."""
        result = await tool.execute("/etc")

        assert result.success is False
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_list_recursive(self, tool, tmp_path, monkeypatch):
        """Test recursive directory listing."""
        monkeypatch.setattr(tool, "ALLOWED_DIRS", [tmp_path])

        (tmp_path / "file1.txt").write_text("content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")

        result = await tool.execute(str(tmp_path), recursive=True)

        assert result.success is True
        names = [f["name"] for f in result.output["files"]]
        assert any("nested.txt" in name for name in names)
