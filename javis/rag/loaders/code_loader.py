"""Codebase loader for JAVIS RAG system."""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.rag.loaders.base import BaseLoader
from javis.rag.loaders.file_loader import FileLoader
from javis.rag.models import Document, DocumentSource

logger = logging.getLogger(__name__)


class CodeLoader(BaseLoader):
    """Load documents from a code directory, respecting .gitignore."""

    DEFAULT_IGNORE_PATTERNS = {
        # Directories
        ".git", ".svn", ".hg",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "node_modules", "bower_components",
        ".venv", "venv", "env", ".env",
        "dist", "build", "target",
        ".idea", ".vscode",
        "*.egg-info",
        # Files
        "*.pyc", "*.pyo", "*.pyd",
        "*.so", "*.dll", "*.dylib",
        "*.class", "*.jar",
        "*.o", "*.obj", "*.exe",
        "*.log", "*.tmp",
        "*.lock", "package-lock.json", "yarn.lock",
    }

    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php",
        ".cs", ".swift", ".kt", ".scala",
        ".sql", ".sh", ".bash",
    }

    def __init__(self, ignore_patterns: Optional[set[str]] = None):
        """
        Initialize code loader.

        Args:
            ignore_patterns: Additional patterns to ignore
        """
        self.ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self.ignore_patterns.update(ignore_patterns)

        self.file_loader = FileLoader()

    async def can_load(self, source: str) -> bool:
        """Check if this loader can handle the source."""
        path = Path(source)
        return path.exists() and path.is_dir()

    async def load(self, source: str, **kwargs) -> Document:
        """
        Load is not supported for directories.
        Use load_directory instead.
        """
        raise NotImplementedError("Use load_directory for code directories")

    async def load_directory(
        self,
        directory: str,
        extensions: Optional[set[str]] = None,
        max_files: int = 100
    ) -> list[Document]:
        """
        Load all code files from a directory.

        Args:
            directory: Directory path
            extensions: File extensions to include (defaults to CODE_EXTENSIONS)
            max_files: Maximum number of files to load

        Returns:
            List of Document objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        extensions = extensions or self.CODE_EXTENSIONS
        gitignore = self._load_gitignore(dir_path)

        documents = []
        files_processed = 0

        for file_path in self._walk_directory(dir_path, extensions, gitignore):
            if files_processed >= max_files:
                logger.warning(f"Reached max files limit: {max_files}")
                break

            try:
                content = self.file_loader._read_file(file_path)
                relative_path = file_path.relative_to(dir_path)

                doc = Document(
                    id=str(uuid.uuid4()),
                    source=DocumentSource.CODEBASE,
                    source_path=str(file_path.absolute()),
                    title=str(relative_path),
                    content=content,
                    created_at=datetime.now(),
                    metadata={
                        "directory": str(dir_path.absolute()),
                        "relative_path": str(relative_path),
                        "file_extension": file_path.suffix,
                        "file_size": file_path.stat().st_size,
                    }
                )
                documents.append(doc)
                files_processed += 1

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} files from {directory}")
        return documents

    def _walk_directory(
        self,
        directory: Path,
        extensions: set[str],
        gitignore: set[str]
    ):
        """Walk directory yielding matching files."""
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not self._should_ignore(d, is_dir=True, gitignore=gitignore)
            ]

            for file in files:
                file_path = root_path / file

                if self._should_ignore(file, is_dir=False, gitignore=gitignore):
                    continue

                if file_path.suffix.lower() not in extensions:
                    continue

                yield file_path

    def _should_ignore(
        self,
        name: str,
        is_dir: bool,
        gitignore: set[str]
    ) -> bool:
        """Check if file/directory should be ignored."""
        # Check exact match
        if name in self.ignore_patterns or name in gitignore:
            return True

        # Check pattern match
        for pattern in self.ignore_patterns | gitignore:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True

        return False

    def _load_gitignore(self, directory: Path) -> set[str]:
        """Load .gitignore patterns from directory."""
        gitignore_path = directory / ".gitignore"
        if not gitignore_path.exists():
            return set()

        patterns = set()
        try:
            content = gitignore_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Simplified pattern handling
                    patterns.add(line.strip("/"))
        except Exception as e:
            logger.warning(f"Failed to read .gitignore: {e}")

        return patterns
