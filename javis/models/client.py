"""Groq API Client for model inference."""

import httpx
from pydantic import BaseModel

from javis.utils.config import get_config


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    """Chat completion request."""

    messages: list[Message]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] | None = None


class ChatResponse(BaseModel):
    """Chat completion response."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None


class ModelClient:
    """Client for Groq API inference."""

    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.1-8b-instant"

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

    async def chat(self, messages: list[Message]) -> ChatResponse:
        """Send a chat completion request."""
        generation_config = self.config.model.generation

        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": generation_config.max_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
        }

        if generation_config.stop_sequences:
            payload["stop"] = generation_config.stop_sequences

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )

            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            return ChatResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage"),
            )

    def chat_sync(self, messages: list[Message]) -> ChatResponse:
        """Synchronous chat completion (for CLI)."""
        import asyncio

        return asyncio.run(self.chat(messages))
