"""Conversation summarizer for JAVIS memory system."""

import json
import logging
import os
from typing import Optional

import httpx

from javis.memory.models import ConversationSummary

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """LLM-based conversation summarizer."""

    SUMMARY_PROMPT = """다음 대화를 분석해서 JSON 형식으로 요약해줘.

대화:
{conversation}

다음 JSON 형식으로 응답해:
{{
    "summary": "대화 내용을 2-3문장으로 요약",
    "topics": ["토픽1", "토픽2", ...],
    "key_entities": ["중요한 개체나 개념"],
    "sentiment": "positive/neutral/negative"
}}

JSON만 응답하고 다른 텍스트는 포함하지 마."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant"
    ):
        """
        Initialize summarizer.

        Args:
            api_key: Groq API key (defaults to env var)
            model: Model to use for summarization
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def summarize(
        self,
        turns: list[dict],
        session_id: Optional[str] = None
    ) -> ConversationSummary:
        """
        Summarize a conversation.

        Args:
            turns: List of conversation turns (role, content)
            session_id: Optional session ID for logging

        Returns:
            ConversationSummary with summary, topics, entities
        """
        if not turns:
            return ConversationSummary(
                summary="빈 대화",
                topics=[],
                key_entities=[]
            )

        # Format conversation
        conversation_text = self._format_conversation(turns)

        # Generate summary using LLM
        try:
            summary_json = await self._call_llm(conversation_text)
            return self._parse_summary(summary_json)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: simple extraction
            return self._fallback_summary(turns)

    def _format_conversation(self, turns: list[dict]) -> str:
        """Format conversation turns for the prompt."""
        formatted = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            # Skip system messages
            if role == "system":
                continue

            role_kr = "사용자" if role == "user" else "AI"
            formatted.append(f"{role_kr}: {content}")

        return "\n".join(formatted)

    async def _call_llm(self, conversation_text: str) -> str:
        """Call LLM API for summarization."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        prompt = self.SUMMARY_PROMPT.format(conversation=conversation_text)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3  # Low temperature for consistent output
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]

    def _parse_summary(self, json_str: str) -> ConversationSummary:
        """Parse LLM response into ConversationSummary."""
        try:
            # Clean up response (remove markdown code blocks if present)
            json_str = json_str.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])

            data = json.loads(json_str)

            return ConversationSummary(
                summary=data.get("summary", "요약 없음"),
                topics=data.get("topics", []),
                key_entities=data.get("key_entities", []),
                sentiment=data.get("sentiment")
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse summary JSON: {e}")
            # Return the raw text as summary
            return ConversationSummary(
                summary=json_str[:500],
                topics=[],
                key_entities=[]
            )

    def _fallback_summary(self, turns: list[dict]) -> ConversationSummary:
        """Fallback summary when LLM fails."""
        # Extract user messages
        user_messages = [
            t.get("content", "")[:100]
            for t in turns
            if t.get("role") == "user"
        ]

        summary = f"사용자가 {len(user_messages)}개의 질문/요청을 함"
        if user_messages:
            summary += f": {user_messages[0]}"
            if len(user_messages) > 1:
                summary += f" 외 {len(user_messages) - 1}건"

        return ConversationSummary(
            summary=summary,
            topics=[],
            key_entities=[]
        )
