"""Local file loader for JAVIS RAG system."""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from javis.rag.loaders.base import BaseLoader
from javis.rag.models import Document, DocumentSource

logger = logging.getLogger(__name__)


class FileLoader(BaseLoader):
    """Load documents from local files."""

    SUPPORTED_EXTENSIONS = {
        # Text files
        ".txt", ".md", ".markdown", ".rst",
        # Code files
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php",
        ".css", ".scss", ".less",
        ".html", ".htm", ".xml",
        ".sql",
        # Data files
        ".json", ".yaml", ".yml", ".toml",
        ".csv",
        # Config files
        ".env", ".ini", ".cfg",
        ".gitignore", ".dockerignore",
    }

    ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize file loader.

        Args:
            base_path: Optional base path for relative file paths
        """
        self.base_path = Path(base_path) if base_path else None

    async def can_load(self, source: str) -> bool:
        """Check if this loader can handle the source."""
        path = self._resolve_path(source)
        if not path.exists() or not path.is_file():
            return False

        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def load(self, source: str, **kwargs) -> Document:
        """
        Load a document from a local file.

        Args:
            source: File path (absolute or relative)
            **kwargs: Additional options

        Returns:
            Loaded Document object
        """
        path = self._resolve_path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        content = self._read_file(path)
        title = self.get_title(source, content) or path.name

        return Document(
            id=str(uuid.uuid4()),
            source=DocumentSource.LOCAL_FILE,
            source_path=str(path.absolute()),
            title=title,
            content=content,
            created_at=datetime.now(),
            metadata={
                "file_name": path.name,
                "file_extension": path.suffix,
                "file_size": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
        )

    def _resolve_path(self, source: str) -> Path:
        """Resolve source to absolute path."""
        path = Path(source)
        if not path.is_absolute() and self.base_path:
            path = self.base_path / path
        return path

    def _read_file(self, path: Path) -> str:
        """Read file with encoding detection."""
        for encoding in self.ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Last resort: read as binary and decode with errors ignored
        return path.read_bytes().decode("utf-8", errors="ignore")

    def get_title(self, source: str, content: str) -> Optional[str]:
        """Extract title from file content."""
        path = Path(source)
        suffix = path.suffix.lower()

        # For markdown, try to find first heading
        if suffix in {".md", ".markdown"}:
            for line in content.split("\n"):
                if line.startswith("# "):
                    return line[2:].strip()

        # For Python, try to find module docstring
        if suffix == ".py":
            if content.startswith('"""'):
                end = content.find('"""', 3)
                if end > 0:
                    docstring = content[3:end].strip()
                    first_line = docstring.split("\n")[0]
                    if len(first_line) < 100:
                        return first_line

        return None
