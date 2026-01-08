"""Web-related tools for JAVIS."""

from typing import Optional
from urllib.parse import urlparse

import httpx

from javis.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult


class HttpRequestTool(BaseTool):
    """HTTP 요청 도구."""

    MAX_RESPONSE_SIZE = 100 * 1024  # 100KB
    TIMEOUT = 10.0  # 10초

    # SSRF 방지: 차단할 호스트/IP
    BLOCKED_HOSTS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "169.254.",  # link-local
        "10.",       # private
        "172.16.",   # private
        "172.17.",
        "172.18.",
        "172.19.",
        "172.20.",
        "172.21.",
        "172.22.",
        "172.23.",
        "172.24.",
        "172.25.",
        "172.26.",
        "172.27.",
        "172.28.",
        "172.29.",
        "172.30.",
        "172.31.",
        "192.168.", # private
    ]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="http_request",
            description="HTTP GET/POST 요청을 보내고 응답을 반환합니다. 보안상 로컬 및 내부 네트워크 주소는 차단됩니다.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="요청할 URL (https:// 권장)"
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP 메서드",
                    required=False,
                    default="GET",
                    enum=["GET", "POST"]
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="HTTP 헤더 (JSON 객체)",
                    required=False
                ),
                ToolParameter(
                    name="body",
                    type="string",
                    description="요청 본문 (POST용)",
                    required=False
                )
            ]
        )

    def _is_url_safe(self, url: str) -> tuple[bool, Optional[str]]:
        """URL 보안 검사."""
        try:
            parsed = urlparse(url)

            # 스킴 검사
            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid scheme: {parsed.scheme}. Only http/https allowed."

            # 호스트 검사
            host = parsed.netloc.lower()
            # 포트 제거
            if ":" in host:
                host = host.split(":")[0]

            for blocked in self.BLOCKED_HOSTS:
                if host == blocked or host.startswith(blocked):
                    return False, f"Blocked host: {host} (security)"

            return True, None

        except Exception as e:
            return False, f"Invalid URL: {e}"

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        body: Optional[str] = None
    ) -> ToolResult:
        # URL 보안 검사
        is_safe, error = self._is_url_safe(url)
        if not is_safe:
            return ToolResult(
                success=False,
                output=None,
                error=error
            )

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers or {})
                elif method.upper() == "POST":
                    response = await client.post(
                        url,
                        headers=headers or {},
                        content=body
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Unsupported method: {method}"
                    )

                # 응답 크기 제한
                content = response.text
                truncated = False
                if len(content) > self.MAX_RESPONSE_SIZE:
                    content = content[:self.MAX_RESPONSE_SIZE]
                    truncated = True

                return ToolResult(
                    success=True,
                    output={
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content": content,
                        "content_length": len(response.text),
                        "truncated": truncated,
                        "url": str(response.url)
                    }
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request timed out after {self.TIMEOUT}s"
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request failed: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class UrlParseTool(BaseTool):
    """URL 파싱 도구."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="parse_url",
            description="URL을 파싱하여 구성 요소(scheme, host, path 등)를 반환합니다.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="파싱할 URL"
                )
            ]
        )

    async def execute(self, url: str) -> ToolResult:
        try:
            parsed = urlparse(url)

            return ToolResult(
                success=True,
                output={
                    "scheme": parsed.scheme,
                    "netloc": parsed.netloc,
                    "hostname": parsed.hostname,
                    "port": parsed.port,
                    "path": parsed.path,
                    "query": parsed.query,
                    "fragment": parsed.fragment,
                    "username": parsed.username,
                    "password": parsed.password
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


def register_web_tools() -> None:
    """웹 도구들을 레지스트리에 등록."""
    from javis.tools.registry import get_registry

    registry = get_registry()
    registry.register(HttpRequestTool(), "web_tools")
    registry.register(UrlParseTool(), "web_tools")
