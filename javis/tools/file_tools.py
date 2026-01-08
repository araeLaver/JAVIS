"""File operation tools for JAVIS."""

import os
from pathlib import Path
from typing import Optional

import aiofiles

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class ReadFileTool(BaseTool):
    """파일 읽기 도구."""

    # 보안: 허용된 디렉토리만 접근 가능
    ALLOWED_DIRS = [
        Path("./data"),
        Path("./workspace"),
        Path("./uploads"),
        Path("./configs"),
    ]

    MAX_FILE_SIZE = 1024 * 1024  # 1MB

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="파일의 내용을 읽습니다. 텍스트 파일만 지원합니다. 보안상 허용된 디렉토리(data, workspace, uploads, configs)의 파일만 읽을 수 있습니다.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="읽을 파일의 경로 (예: ./data/example.txt)"
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="파일 인코딩",
                    required=False,
                    default="utf-8"
                )
            ]
        )

    def _is_path_allowed(self, path: Path) -> bool:
        """경로 보안 검사."""
        try:
            resolved = path.resolve()
            for allowed in self.ALLOWED_DIRS:
                allowed_resolved = allowed.resolve()
                try:
                    resolved.relative_to(allowed_resolved)
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False

    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        file_path = Path(path)

        # 경로 순회 공격 방지
        if ".." in str(file_path):
            return ToolResult(
                success=False,
                output=None,
                error="Path traversal not allowed"
            )

        if not self._is_path_allowed(file_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Access denied: Path not in allowed directories ({', '.join(str(d) for d in self.ALLOWED_DIRS)})"
            )

        if not file_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {path}"
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a file: {path}"
            )

        # 파일 크기 검사
        if file_path.stat().st_size > self.MAX_FILE_SIZE:
            return ToolResult(
                success=False,
                output=None,
                error=f"File too large (max {self.MAX_FILE_SIZE // 1024}KB)"
            )

        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            return ToolResult(
                success=True,
                output={
                    "path": str(file_path),
                    "size": len(content),
                    "content": content
                }
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Cannot decode file with encoding: {encoding}"
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WriteFileTool(BaseTool):
    """파일 쓰기 도구."""

    ALLOWED_DIRS = [
        Path("./data"),
        Path("./workspace"),
    ]

    MAX_CONTENT_SIZE = 1024 * 1024  # 1MB

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="파일에 내용을 씁니다. 보안상 허용된 디렉토리(data, workspace)에만 쓸 수 있습니다.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="쓸 파일의 경로 (예: ./workspace/output.txt)"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="파일에 쓸 내용"
                ),
                ToolParameter(
                    name="append",
                    type="boolean",
                    description="True면 파일 끝에 추가, False면 덮어쓰기",
                    required=False,
                    default=False
                )
            ]
        )

    def _is_path_allowed(self, path: Path) -> bool:
        """경로 보안 검사."""
        try:
            # 부모 디렉토리 기준으로 검사 (파일이 아직 없을 수 있음)
            parent = path.parent.resolve()
            for allowed in self.ALLOWED_DIRS:
                allowed_resolved = allowed.resolve()
                try:
                    parent.relative_to(allowed_resolved)
                    return True
                except ValueError:
                    continue

                # 또는 parent가 allowed와 같거나 하위인지 확인
                if parent == allowed_resolved or allowed_resolved in parent.parents:
                    return True
            return False
        except Exception:
            return False

    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False
    ) -> ToolResult:
        # 크기 제한
        if len(content) > self.MAX_CONTENT_SIZE:
            return ToolResult(
                success=False,
                output=None,
                error=f"Content too large (max {self.MAX_CONTENT_SIZE // 1024}KB)"
            )

        file_path = Path(path)

        # 경로 순회 공격 방지
        if ".." in str(file_path):
            return ToolResult(
                success=False,
                output=None,
                error="Path traversal not allowed"
            )

        if not self._is_path_allowed(file_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Access denied: Path not in allowed directories ({', '.join(str(d) for d in self.ALLOWED_DIRS)})"
            )

        try:
            # 디렉토리 생성
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = 'a' if append else 'w'
            async with aiofiles.open(file_path, mode, encoding='utf-8') as f:
                await f.write(content)

            return ToolResult(
                success=True,
                output={
                    "path": str(file_path),
                    "bytes_written": len(content.encode('utf-8')),
                    "mode": "append" if append else "write"
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ListDirectoryTool(BaseTool):
    """디렉토리 목록 조회 도구."""

    ALLOWED_DIRS = [
        Path("./data"),
        Path("./workspace"),
        Path("./uploads"),
        Path("./configs"),
    ]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="디렉토리의 파일 목록을 반환합니다. 보안상 허용된 디렉토리만 조회할 수 있습니다.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="디렉토리 경로",
                    required=False,
                    default="./workspace"
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="하위 디렉토리도 포함할지 여부",
                    required=False,
                    default=False
                )
            ]
        )

    def _is_path_allowed(self, path: Path) -> bool:
        """경로 보안 검사."""
        try:
            resolved = path.resolve()
            for allowed in self.ALLOWED_DIRS:
                allowed_resolved = allowed.resolve()
                try:
                    resolved.relative_to(allowed_resolved)
                    return True
                except ValueError:
                    pass
                # 정확히 일치하는 경우
                if resolved == allowed_resolved:
                    return True
            return False
        except Exception:
            return False

    async def execute(
        self,
        path: str = "./workspace",
        recursive: bool = False
    ) -> ToolResult:
        dir_path = Path(path)

        # 경로 순회 공격 방지
        if ".." in str(dir_path):
            return ToolResult(
                success=False,
                output=None,
                error="Path traversal not allowed"
            )

        if not self._is_path_allowed(dir_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Access denied: Path not in allowed directories"
            )

        if not dir_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory not found: {path}"
            )

        if not dir_path.is_dir():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a directory: {path}"
            )

        try:
            files = []

            if recursive:
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        files.append({
                            "name": str(item.relative_to(dir_path)),
                            "type": "file",
                            "size": item.stat().st_size
                        })
                    elif item.is_dir():
                        files.append({
                            "name": str(item.relative_to(dir_path)),
                            "type": "directory",
                            "size": None
                        })
            else:
                for item in dir_path.iterdir():
                    files.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None
                    })

            return ToolResult(
                success=True,
                output={
                    "path": str(dir_path),
                    "count": len(files),
                    "files": sorted(files, key=lambda x: (x["type"] == "file", x["name"]))
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


def register_file_tools() -> None:
    """파일 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(ReadFileTool(), "file_tools")
    registry.register(WriteFileTool(), "file_tools")
    registry.register(ListDirectoryTool(), "file_tools")
