"""Chat engine with tool support for JAVIS."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import httpx

from javis.models.client import Message, ChatResponse
from javis.tools.registry import get_registry
from javis.tools.executor import ToolExecutor
from javis.utils.config import get_config
from javis.utils.constants import (
    GROQ_API_BASE_URL,
    Timeouts,
    Defaults,
    Headers,
    EnvVars,
    Roles,
    FinishReasons,
)
from javis.utils.resilience import (
    get_circuit_breaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    retry,
    RetryConfig,
    RESILIENCE_PROFILES,
    CIRCUIT_BREAKER_PROFILES,
)
from javis.utils.metrics import (
    CHAT_REQUESTS,
    CHAT_LATENCY,
    TOOL_EXECUTIONS,
    MODEL_LATENCY,
    ERRORS_TOTAL,
)
from javis.utils.tracing import (
    create_span,
    traced,
    add_span_attributes,
    add_span_event,
    set_span_error,
)

if TYPE_CHECKING:
    from javis.memory.manager import MemoryManager
    from javis.rag.manager import RAGManager

logger = logging.getLogger(__name__)


class ContextManagerRegistry:
    """
    Thread-safe singleton registry for context managers.

    Manages Memory and RAG managers with lazy loading pattern.
    Uses double-checked locking to ensure thread safety during initialization.

    Attributes:
        memory_manager: Lazily loaded MemoryManager instance for long-term memory.
        rag_manager: Lazily loaded RAGManager instance for document retrieval.

    Example:
        registry = get_context_registry()
        if registry.memory_manager:
            context = await registry.memory_manager.get_relevant_context(query)
    """

    _instance: Optional[ContextManagerRegistry] = None
    _lock: threading.Lock = threading.Lock()
    _memory_manager: Optional[MemoryManager]
    _rag_manager: Optional[RAGManager]
    _initialized: bool

    def __new__(cls) -> ContextManagerRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._memory_manager = None
                    cls._instance._rag_manager = None
                    cls._instance._initialized = False
        return cls._instance

    @property
    def memory_manager(self) -> Optional[MemoryManager]:
        """Memory manager를 lazy loading으로 반환합니다."""
        if self._memory_manager is None:
            with self._lock:
                if self._memory_manager is None:
                    self._memory_manager = self._init_memory_manager()
        return self._memory_manager

    @property
    def rag_manager(self) -> Optional[RAGManager]:
        """RAG manager를 lazy loading으로 반환합니다."""
        if self._rag_manager is None:
            with self._lock:
                if self._rag_manager is None:
                    self._rag_manager = self._init_rag_manager()
        return self._rag_manager

    def _init_memory_manager(self) -> Optional[MemoryManager]:
        """Memory manager를 초기화합니다."""
        try:
            config = get_config()
            if config.memory.long_term.enabled:
                from javis.memory import initialize_memory
                manager = initialize_memory(
                    db_path=config.memory.long_term.db_path,
                    collection_name=config.memory.long_term.collection_name,
                    embedding_model=config.memory.long_term.embedding_model
                )
                logger.info("Memory manager initialized")
                return manager
        except Exception as e:
            logger.warning(f"Failed to initialize memory manager: {e}")
        return None

    def _init_rag_manager(self) -> Optional[RAGManager]:
        """RAG manager를 초기화합니다."""
        try:
            config = get_config()
            if config.rag.enabled:
                from javis.rag import initialize_rag
                manager = initialize_rag(
                    db_path=config.rag.db_path,
                    collection_name=config.rag.collection_name,
                    embedding_model=config.rag.embedding_model
                )
                logger.info("RAG manager initialized")
                return manager
        except Exception as e:
            logger.warning(f"Failed to initialize RAG manager: {e}")
        return None

    def reset(self) -> None:
        """테스트용: 레지스트리를 리셋합니다."""
        with self._lock:
            self._memory_manager = None
            self._rag_manager = None
            self._initialized = False


def get_context_registry() -> ContextManagerRegistry:
    """ContextManagerRegistry 싱글톤 인스턴스를 반환합니다."""
    return ContextManagerRegistry()


class ChatEngine:
    """
    Chat engine with integrated tool support.

    Connects Groq API with the tool system to enable LLM tool calling.
    Supports context enhancement through Memory and RAG systems.

    Features:
        - Multi-turn tool calling with automatic result injection
        - Long-term memory integration for conversation context
        - RAG-based document retrieval for knowledge augmentation
        - Configurable timeouts and iteration limits

    Example:
        engine = ChatEngine(api_key="your-api-key")
        response = await engine.chat_with_tools([
            Message(role="user", content="What's the weather?")
        ])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = Defaults.MODEL_NAME,
        max_tool_iterations: int = Defaults.MAX_TOOL_ITERATIONS,
        tool_timeout: float = Timeouts.TOOL_EXECUTION,
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
        self.api_key = api_key or os.getenv(EnvVars.GROQ_API_KEY)
        if not self.api_key:
            raise ValueError(f"{EnvVars.GROQ_API_KEY} not found")

        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.executor = ToolExecutor(timeout=tool_timeout)
        self.base_url = GROQ_API_BASE_URL
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
            registry = get_context_registry()
            memory_manager = registry.memory_manager
            if memory_manager:
                try:
                    memory_context = await memory_manager.get_relevant_context(
                        user_query,
                        limit=Defaults.MEMORY_CONTEXT_LIMIT,
                        max_chars=Defaults.MEMORY_MAX_CHARS
                    )
                    if memory_context:
                        context_parts.append(memory_context)
                        logger.debug("Added memory context to prompt")
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")

        # Search RAG for relevant documents
        if self.use_rag:
            registry = get_context_registry()
            rag_manager = registry.rag_manager
            if rag_manager:
                try:
                    config = get_config()
                    rag_context = await rag_manager.get_context_for_query(
                        user_query,
                        top_k=config.rag.top_k,
                        max_chars=Defaults.RAG_MAX_CHARS
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

    def _build_api_payload(
        self,
        messages: list[dict],
        tools: Optional[list],
        temperature: float,
        max_tokens: int
    ) -> dict[str, Any]:
        """API 요청 페이로드를 생성합니다."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    async def _call_groq_api(self, payload: dict[str, Any]) -> dict:
        """Groq API를 호출합니다 (Circuit Breaker 및 Retry 패턴 적용)."""
        circuit_breaker = get_circuit_breaker(
            "groq-api",
            CIRCUIT_BREAKER_PROFILES["model_inference"]
        )

        async with circuit_breaker:
            return await self._call_groq_api_with_retry(payload)

    @retry(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
        ),
        non_retryable_exceptions=(
            httpx.HTTPStatusError,  # Don't retry 4xx/5xx errors
        ),
    )
    async def _call_groq_api_with_retry(self, payload: dict[str, Any]) -> dict:
        """Groq API 호출 (재시도 로직 포함)."""
        import time
        start_time = time.time()

        with create_span(
            "groq.api.call",
            attributes={
                "model": self.model,
                "max_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature"),
                "has_tools": "tools" in payload,
            }
        ) as span:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=Timeouts.DEFAULT_API
                )
                response.raise_for_status()
                result = response.json()

                # Record model latency
                duration = time.time() - start_time
                MODEL_LATENCY.observe(duration, labels={"model": self.model})

                # Add response info to span
                usage = result.get("usage", {})
                span.set_attribute("response.status_code", response.status_code)
                span.set_attribute("response.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("response.completion_tokens", usage.get("completion_tokens", 0))
                span.set_attribute("response.total_tokens", usage.get("total_tokens", 0))
                span.set_attribute("response.duration_ms", round(duration * 1000, 2))

                return result

    async def _process_tool_calls(
        self,
        tool_calls: list[dict],
        current_messages: list[dict]
    ) -> None:
        """도구 호출을 처리하고 결과를 메시지에 추가합니다."""
        with create_span(
            "tools.execute_batch",
            attributes={"tool_count": len(tool_calls)}
        ) as batch_span:
            results = await self.executor.execute_parallel(tool_calls)

            for call in tool_calls:
                call_id = call["id"]
                tool_name = call["function"]["name"]
                result = results[call_id]

                # Record tool execution metrics
                status = "success" if result.success else "error"
                TOOL_EXECUTIONS.inc(labels={"tool": tool_name, "status": status})

                # Add event for each tool execution
                add_span_event(
                    f"tool.{tool_name}",
                    {
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "success": result.success,
                    }
                )

                logger.info(f"Tool {tool_name} result: success={result.success}")

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.to_message_content()
                })

            # Set batch result attributes
            success_count = sum(1 for r in results.values() if r.success)
            batch_span.set_attribute("tools.success_count", success_count)
            batch_span.set_attribute("tools.error_count", len(results) - success_count)

    async def chat_with_tools(
        self,
        messages: list[Message],
        use_tools: bool = True,
        temperature: float = Defaults.TEMPERATURE,
        max_tokens: int = Defaults.MAX_TOKENS
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
        import time
        start_time = time.time()
        tool_mode = "with_tools" if use_tools else "simple"

        def _record_metrics(status: str) -> None:
            duration = time.time() - start_time
            CHAT_REQUESTS.inc(labels={"model": self.model, "status": status})
            CHAT_LATENCY.observe(duration, labels={"model": self.model, "mode": tool_mode})

        with create_span(
            "chat.with_tools",
            attributes={
                "chat.model": self.model,
                "chat.use_tools": use_tools,
                "chat.message_count": len(messages),
                "chat.temperature": temperature,
                "chat.max_tokens": max_tokens,
            }
        ) as chat_span:
            # 도구 스키마 준비
            tools = None
            if use_tools:
                registry = get_registry()
                tools = registry.get_openai_tools()
                if not tools:
                    logger.warning("No tools enabled, proceeding without tools")
                else:
                    chat_span.set_attribute("chat.available_tools", len(tools))

            # Message 객체를 dict로 변환 및 컨텍스트 강화
            current_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            current_messages = await self._enhance_with_context(current_messages)
            add_span_event("context_enhanced", {"message_count": len(current_messages)})

            # 도구 호출 루프
            for iteration in range(1, self.max_tool_iterations + 1):
                payload = self._build_api_payload(
                    current_messages, tools, temperature, max_tokens
                )
                logger.debug(f"Tool iteration {iteration}")
                add_span_event(f"iteration_{iteration}_start", {"iteration": iteration})

                try:
                    data = await self._call_groq_api(payload)
                except CircuitOpenError as e:
                    logger.warning(f"Circuit breaker open: {e.circuit_name}")
                    _record_metrics("circuit_open")
                    ERRORS_TOTAL.inc(labels={"type": "CircuitOpenError", "component": "chat_engine"})
                    set_span_error(e)
                    chat_span.set_attribute("chat.status", "circuit_open")
                    retry_msg = ""
                    if e.retry_after:
                        retry_msg = f" (약 {int(e.retry_after)}초 후 재시도 가능)"
                    return ChatResponse(
                        content=f"서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.{retry_msg}",
                        finish_reason=FinishReasons.ERROR
                    )
                except httpx.HTTPStatusError as e:
                    logger.error(f"Groq API error: {e.response.status_code}")
                    _record_metrics("api_error")
                    ERRORS_TOTAL.inc(labels={"type": "HTTPStatusError", "component": "chat_engine"})
                    set_span_error(e)
                    chat_span.set_attribute("chat.status", "api_error")
                    return ChatResponse(
                        content=f"API 오류가 발생했습니다: {e.response.status_code}",
                        finish_reason=FinishReasons.ERROR
                    )
                except httpx.TimeoutException as e:
                    logger.error("Groq API timeout")
                    _record_metrics("timeout")
                    ERRORS_TOTAL.inc(labels={"type": "TimeoutException", "component": "chat_engine"})
                    set_span_error(e)
                    chat_span.set_attribute("chat.status", "timeout")
                    return ChatResponse(
                        content="API 요청 시간이 초과되었습니다.",
                        finish_reason=FinishReasons.ERROR
                    )
                except Exception as e:
                    logger.exception("Unexpected error in chat_with_tools")
                    _record_metrics("error")
                    ERRORS_TOTAL.inc(labels={"type": type(e).__name__, "component": "chat_engine"})
                    set_span_error(e)
                    chat_span.set_attribute("chat.status", "error")
                    return ChatResponse(
                        content=f"오류가 발생했습니다: {str(e)}",
                        finish_reason=FinishReasons.ERROR
                    )

                choice = data["choices"][0]
                message = choice["message"]

                # 도구 호출이 없으면 최종 응답 반환
                if not message.get("tool_calls"):
                    _record_metrics("success")
                    chat_span.set_attribute("chat.status", "success")
                    chat_span.set_attribute("chat.iterations", iteration)
                    chat_span.set_attribute("chat.finish_reason", choice.get("finish_reason"))
                    return ChatResponse(
                        content=message.get("content", ""),
                        finish_reason=choice.get("finish_reason"),
                        usage=data.get("usage")
                    )

                # 도구 호출 처리
                tool_calls = message["tool_calls"]
                logger.info(f"Processing {len(tool_calls)} tool calls")

                current_messages.append(message)
                await self._process_tool_calls(tool_calls, current_messages)

            # 최대 반복 횟수 도달
            logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
            _record_metrics("max_iterations")
            chat_span.set_attribute("chat.status", "max_iterations")
            chat_span.set_attribute("chat.iterations", self.max_tool_iterations)
            return ChatResponse(
                content=f"도구 호출이 최대 횟수({self.max_tool_iterations})에 도달했습니다.",
                finish_reason=FinishReasons.MAX_ITERATIONS
            )

    async def chat_simple(
        self,
        messages: list[Message],
        temperature: float = Defaults.TEMPERATURE,
        max_tokens: int = Defaults.MAX_TOKENS
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
        import time
        start_time = time.time()

        def _record_metrics(status: str) -> None:
            duration = time.time() - start_time
            CHAT_REQUESTS.inc(labels={"model": self.model, "status": status})
            CHAT_LATENCY.observe(duration, labels={"model": self.model, "mode": "simple"})

        with create_span(
            "chat.simple",
            attributes={
                "chat.model": self.model,
                "chat.message_count": len(messages),
                "chat.temperature": temperature,
                "chat.max_tokens": max_tokens,
            }
        ) as chat_span:
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
                data = await self._call_groq_api(payload)

                choice = data["choices"][0]
                _record_metrics("success")
                chat_span.set_attribute("chat.status", "success")
                chat_span.set_attribute("chat.finish_reason", choice.get("finish_reason"))
                return ChatResponse(
                    content=choice["message"].get("content", ""),
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage")
                )

            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker open in chat_simple: {e.circuit_name}")
                _record_metrics("circuit_open")
                ERRORS_TOTAL.inc(labels={"type": "CircuitOpenError", "component": "chat_engine"})
                set_span_error(e)
                chat_span.set_attribute("chat.status", "circuit_open")
                retry_msg = ""
                if e.retry_after:
                    retry_msg = f" (약 {int(e.retry_after)}초 후 재시도 가능)"
                return ChatResponse(
                    content=f"서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.{retry_msg}",
                    finish_reason=FinishReasons.ERROR
                )

            except Exception as e:
                logger.exception("Error in chat_simple")
                _record_metrics("error")
                ERRORS_TOTAL.inc(labels={"type": type(e).__name__, "component": "chat_engine"})
                set_span_error(e)
                chat_span.set_attribute("chat.status", "error")
                return ChatResponse(
                    content=f"오류가 발생했습니다: {str(e)}",
                    finish_reason=FinishReasons.ERROR
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

        registry = get_context_registry()
        memory_manager = registry.memory_manager
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
