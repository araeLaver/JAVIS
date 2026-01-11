"""File operation tools for JAVIS."""

import logging
from pathlib import Path
from typing import Optional

import aiofiles

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """경로 보안 검사 실패 예외."""
    pass


def validate_path_security(
    path: Path,
    allowed_dirs: list[Path],
    check_exists: bool = True,
    allow_symlinks: bool = False
) -> tuple[bool, str, Path]:
    """
    경로 보안을 검증합니다.

    Args:
        path: 검증할 경로
        allowed_dirs: 허용된 디렉토리 목록
        check_exists: 존재 여부 확인 (파일 쓰기 시 False)
        allow_symlinks: 심볼릭 링크 허용 여부

    Returns:
        (성공여부, 에러메시지, 해석된 경로)
    """
    try:
        # 1. 경로 정규화 (심볼릭 링크 해석 포함)
        resolved = path.resolve(strict=False)

        # 2. 심볼릭 링크 검사
        if not allow_symlinks and path.exists():
            if path.is_symlink():
                logger.warning(f"Symlink access blocked: {path}")
                return False, "Symbolic links are not allowed for security reasons", resolved

        # 3. 허용된 디렉토리 내에 있는지 확인
        is_allowed = False
        for allowed in allowed_dirs:
            allowed_resolved = allowed.resolve(strict=False)
            try:
                resolved.relative_to(allowed_resolved)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            allowed_str = ', '.join(str(d) for d in allowed_dirs)
            logger.warning(f"Path outside allowed directories: {path} -> {resolved}")
            return False, f"Access denied: Path not in allowed directories ({allowed_str})", resolved

        # 4. 존재 여부 확인 (선택적)
        if check_exists and not resolved.exists():
            return False, f"Path not found: {path}", resolved

        return True, "", resolved

    except OSError as e:
        logger.error(f"Path validation OS error: {path} - {e}")
        return False, f"Invalid path: {e}", path
    except Exception as e:
        logger.error(f"Path validation error: {path} - {e}")
        return False, "Path validation failed", path


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

    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        file_path = Path(path)

        # 통합 보안 검사
        is_valid, error_msg, resolved_path = validate_path_security(
            file_path,
            self.ALLOWED_DIRS,
            check_exists=True,
            allow_symlinks=False
        )

        if not is_valid:
            return ToolResult(success=False, output=None, error=error_msg)

        if not resolved_path.is_file():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a file: {path}"
            )

        # 파일 크기 검사
        try:
            file_size = resolved_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File too large (max {self.MAX_FILE_SIZE // 1024}KB)"
                )
        except OSError as e:
            return ToolResult(success=False, output=None, error=f"Cannot access file: {e}")

        try:
            async with aiofiles.open(resolved_path, 'r', encoding=encoding) as f:
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
        except OSError as e:
            logger.error(f"File read error: {resolved_path} - {e}")
            return ToolResult(success=False, output=None, error=f"Cannot read file: {e}")


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

        # 통합 보안 검사 (파일이 아직 없을 수 있으므로 check_exists=False)
        is_valid, error_msg, resolved_path = validate_path_security(
            file_path,
            self.ALLOWED_DIRS,
            check_exists=False,
            allow_symlinks=False
        )

        if not is_valid:
            return ToolResult(success=False, output=None, error=error_msg)

        # 기존 파일이 심볼릭 링크인지 확인
        if file_path.exists() and file_path.is_symlink():
            logger.warning(f"Attempted write to symlink: {file_path}")
            return ToolResult(
                success=False,
                output=None,
                error="Cannot write to symbolic links"
            )

        try:
            # 디렉토리 생성
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            mode = 'a' if append else 'w'
            async with aiofiles.open(resolved_path, mode, encoding='utf-8') as f:
                await f.write(content)

            return ToolResult(
                success=True,
                output={
                    "path": str(file_path),
                    "bytes_written": len(content.encode('utf-8')),
                    "mode": "append" if append else "write"
                }
            )
        except PermissionError as e:
            logger.error(f"Permission denied writing to: {resolved_path}")
            return ToolResult(success=False, output=None, error="Permission denied")
        except OSError as e:
            logger.error(f"File write error: {resolved_path} - {e}")
            return ToolResult(success=False, output=None, error=f"Cannot write file: {e}")


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

    async def execute(
        self,
        path: str = "./workspace",
        recursive: bool = False
    ) -> ToolResult:
        dir_path = Path(path)

        # 통합 보안 검사
        is_valid, error_msg, resolved_path = validate_path_security(
            dir_path,
            self.ALLOWED_DIRS,
            check_exists=True,
            allow_symlinks=False
        )

        if not is_valid:
            return ToolResult(success=False, output=None, error=error_msg)

        if not resolved_path.is_dir():
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a directory: {path}"
            )

        try:
            files = []

            if recursive:
                for item in resolved_path.rglob("*"):
                    # 심볼릭 링크 건너뛰기
                    if item.is_symlink():
                        continue
                    try:
                        if item.is_file():
                            files.append({
                                "name": str(item.relative_to(resolved_path)),
                                "type": "file",
                                "size": item.stat().st_size
                            })
                        elif item.is_dir():
                            files.append({
                                "name": str(item.relative_to(resolved_path)),
                                "type": "directory",
                                "size": None
                            })
                    except OSError:
                        # 접근 불가 파일 건너뛰기
                        continue
            else:
                for item in resolved_path.iterdir():
                    # 심볼릭 링크 건너뛰기
                    if item.is_symlink():
                        continue
                    try:
                        files.append({
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None
                        })
                    except OSError:
                        continue

            return ToolResult(
                success=True,
                output={
                    "path": str(dir_path),
                    "count": len(files),
                    "files": sorted(files, key=lambda x: (x["type"] == "file", x["name"]))
                }
            )
        except PermissionError:
            return ToolResult(success=False, output=None, error="Permission denied")
        except OSError as e:
            logger.error(f"Directory listing error: {resolved_path} - {e}")
            return ToolResult(success=False, output=None, error=f"Cannot list directory: {e}")


def register_file_tools() -> None:
    """파일 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(ReadFileTool(), "file_tools")
    registry.register(WriteFileTool(), "file_tools")
    registry.register(ListDirectoryTool(), "file_tools")
