"""Chat engine with tool support for JAVIS."""

import json
import logging
import os
from typing import Any, Optional

import httpx

from javis.models.client import Message, ChatResponse
from javis.tools.registry import get_registry
from javis.tools.executor import ToolExecutor
from javis.utils.config import get_config

logger = logging.getLogger(__name__)

# Lazy imports for memory and RAG to avoid loading heavy dependencies at startup
_memory_manager = None
_rag_manager = None


def _get_memory_manager():
    """Get memory manager with lazy loading."""
    global _memory_manager
    if _memory_manager is None:
        try:
            config = get_config()
            if config.memory.long_term.enabled:
                from javis.memory import initialize_memory
                _memory_manager = initialize_memory(
                    db_path=config.memory.long_term.db_path,
                    collection_name=config.memory.long_term.collection_name,
                    embedding_model=config.memory.long_term.embedding_model
                )
                logger.info("Memory manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize memory manager: {e}")
    return _memory_manager


def _get_rag_manager():
    """Get RAG manager with lazy loading."""
    global _rag_manager
    if _rag_manager is None:
        try:
            config = get_config()
            if config.rag.enabled:
                from javis.rag import initialize_rag
                _rag_manager = initialize_rag(
                    db_path=config.rag.db_path,
                    collection_name=config.rag.collection_name,
                    embedding_model=config.rag.embedding_model
                )
                logger.info("RAG manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG manager: {e}")
    return _rag_manager


class ChatEngine:
    """도구 통합 채팅 엔진.

    Groq API와 도구 시스템을 연결하여 LLM이 도구를 호출할 수 있게 합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_tool_iterations: int = 5,
        tool_timeout: float = 30.0,
        use_memory: bool = True,
        use_rag: bool = True
    ):
        """
        Args:
            api_key: Groq API 키 (없으면 환경변수에서 로드)
            model: 사용할 모델 이름
            max_tool_iterations: 최대 도구 호출 반복 횟수
            tool_timeout: 도구 실행 타임아웃 (초)
            use_memory: 장기 기억 사용 여부
            use_rag: RAG 사용 여부
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.executor = ToolExecutor(timeout=tool_timeout)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.use_memory = use_memory
        self.use_rag = use_rag

    async def _enhance_with_context(
        self,
        messages: list[dict]
    ) -> list[dict]:
        """
        Enhance messages with memory and RAG context.

        Args:
            messages: Current message list (dict format)

        Returns:
            Enhanced message list with context injected
        """
        # Find the last user message for context search
        user_query = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        if not user_query:
            return messages

        context_parts = []

        # Search memory for relevant past conversations
        if self.use_memory:
            memory_manager = _get_memory_manager()
            if memory_manager:
                try:
                    memory_context = await memory_manager.get_relevant_context(
                        user_query, limit=3, max_chars=500
                    )
                    if memory_context:
                        context_parts.append(memory_context)
                        logger.debug("Added memory context to prompt")
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")

        # Search RAG for relevant documents
        if self.use_rag:
            rag_manager = _get_rag_manager()
            if rag_manager:
                try:
                    config = get_config()
                    rag_context = await rag_manager.get_context_for_query(
                        user_query,
                        top_k=config.rag.top_k,
                        max_chars=1000
                    )
                    if rag_context:
                        context_parts.append(rag_context)
                        logger.debug("Added RAG context to prompt")
                except Exception as e:
                    logger.warning(f"RAG search failed: {e}")

        # If we have context, inject it into the system prompt
        if context_parts:
            context_str = "\n\n".join(context_parts)
            enhanced_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    # Append context to system prompt
                    enhanced_content = f"{msg['content']}\n\n{context_str}"
                    enhanced_messages.append({
                        "role": "system",
                        "content": enhanced_content
                    })
                else:
                    enhanced_messages.append(msg)

            # If no system message exists, add one with context
            if not any(m["role"] == "system" for m in enhanced_messages):
                enhanced_messages.insert(0, {
                    "role": "system",
                    "content": context_str
                })

            return enhanced_messages

        return messages

    async def chat_with_tools(
        self,
        messages: list[Message],
        use_tools: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> ChatResponse:
        """
        도구를 사용한 채팅.

        Args:
            messages: 대화 메시지 목록
            use_tools: 도구 사용 여부
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            최종 ChatResponse
        """
        registry = get_registry()

        # 도구 스키마 준비
        tools = None
        if use_tools:
            tools = registry.get_openai_tools()
            if not tools:
                logger.warning("No tools enabled, proceeding without tools")
                tools = None

        iteration = 0
        # Message 객체를 dict로 변환
        current_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Memory/RAG 컨텍스트로 메시지 강화
        current_messages = await self._enhance_with_context(current_messages)

        total_tool_calls = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            # API 요청 페이로드
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": current_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            logger.debug(f"Tool iteration {iteration}, payload keys: {payload.keys()}")

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=60.0
                    )
                    response.raise_for_status()
                    data = response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"Groq API error: {e.response.status_code} - {e.response.text}")
                return ChatResponse(
                    content=f"API 오류가 발생했습니다: {e.response.status_code}",
                    finish_reason="error"
                )
            except Exception as e:
                logger.exception("Unexpected error in chat_with_tools")
                return ChatResponse(
                    content=f"오류가 발생했습니다: {str(e)}",
                    finish_reason="error"
                )

            choice = data["choices"][0]
            message = choice["message"]

            # 도구 호출이 없으면 최종 응답 반환
            if not message.get("tool_calls"):
                return ChatResponse(
                    content=message.get("content", ""),
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage")
                )

            # 도구 호출 처리
            logger.info(f"Processing {len(message['tool_calls'])} tool calls")

            # assistant 메시지 추가 (tool_calls 포함)
            current_messages.append(message)

            tool_calls = message["tool_calls"]
            total_tool_calls += len(tool_calls)

            # 도구 실행
            results = await self.executor.execute_parallel(tool_calls)

            # 도구 결과를 메시지에 추가
            for call in tool_calls:
                call_id = call["id"]
                tool_name = call["function"]["name"]
                result = results[call_id]

                logger.info(f"Tool {tool_name} result: success={result.success}")

                # 도구 결과 메시지 추가
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.to_message_content()
                })

        # 최대 반복 횟수 도달
        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
        return ChatResponse(
            content=f"도구 호출이 최대 횟수({self.max_tool_iterations})에 도달했습니다. 일부 작업이 완료되지 않았을 수 있습니다.",
            finish_reason="max_iterations"
        )

    async def chat_simple(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> ChatResponse:
        """
        도구 없이 단순 채팅.

        Args:
            messages: 대화 메시지 목록
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            ChatResponse
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()

            choice = data["choices"][0]
            return ChatResponse(
                content=choice["message"].get("content", ""),
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage")
            )

        except Exception as e:
            logger.exception("Error in chat_simple")
            return ChatResponse(
                content=f"오류가 발생했습니다: {str(e)}",
                finish_reason="error"
            )

    async def store_conversation_to_memory(
        self,
        session_id: str,
        messages: list[Message],
        metadata: Optional[dict] = None
    ) -> bool:
        """
        대화를 장기 기억에 저장.

        Args:
            session_id: 세션 ID
            messages: 대화 메시지 목록
            metadata: 추가 메타데이터

        Returns:
            저장 성공 여부
        """
        if not self.use_memory:
            return False

        memory_manager = _get_memory_manager()
        if not memory_manager:
            return False

        try:
            # Message 객체를 dict로 변환
            turns = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]

            result = await memory_manager.store_conversation(
                session_id=session_id,
                turns=turns,
                metadata=metadata
            )

            if result:
                logger.info(f"Stored conversation {session_id} to memory")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to store conversation to memory: {e}")
            return False
