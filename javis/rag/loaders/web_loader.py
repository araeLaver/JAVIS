"""Web page loader for JAVIS RAG system."""

import logging
import re
import uuid
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx

from javis.rag.loaders.base import BaseLoader
from javis.rag.models import Document, DocumentSource

logger = logging.getLogger(__name__)


class WebLoader(BaseLoader):
    """Load documents from web pages."""

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    def __init__(self, timeout: float = 30.0):
        """
        Initialize web loader.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    async def can_load(self, source: str) -> bool:
        """Check if this loader can handle the source."""
        try:
            parsed = urlparse(source)
            return parsed.scheme in {"http", "https"}
        except Exception:
            return False

    async def load(self, source: str, **kwargs) -> Document:
        """
        Load a document from a web URL.

        Args:
            source: Web URL
            **kwargs: Additional options

        Returns:
            Loaded Document object
        """
        url = source.strip()

        if not await self.can_load(url):
            raise ValueError(f"Invalid URL: {url}")

        async with httpx.AsyncClient(
            headers=self.DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=self.timeout
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        html_content = response.text

        # Extract text from HTML
        text_content = self._html_to_text(html_content)
        title = self._extract_title(html_content) or self._url_to_title(url)

        return Document(
            id=str(uuid.uuid4()),
            source=DocumentSource.WEB_PAGE,
            source_path=url,
            title=title,
            content=text_content,
            created_at=datetime.now(),
            metadata={
                "url": url,
                "domain": urlparse(url).netloc,
                "content_type": content_type,
                "status_code": response.status_code,
            }
        )

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = []
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    lines.append(line)

            return "\n\n".join(lines)

        except ImportError:
            logger.warning("beautifulsoup4 not installed, using basic HTML stripping")
            return self._basic_html_strip(html)

    def _basic_html_strip(self, html: str) -> str:
        """Basic HTML tag removal without BeautifulSoup."""
        # Remove script and style content
        html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
        html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Decode common HTML entities
        entities = {
            '&nbsp;': ' ', '&lt;': '<', '&gt;': '>',
            '&amp;': '&', '&quot;': '"', '&apos;': "'"
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")

            # Try <title> tag
            if soup.title and soup.title.string:
                return soup.title.string.strip()

            # Try <h1> tag
            h1 = soup.find("h1")
            if h1:
                return h1.get_text(strip=True)

        except ImportError:
            # Regex fallback
            match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _url_to_title(self, url: str) -> str:
        """Generate title from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if path:
            # Use last path segment
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()

        return parsed.netloc
