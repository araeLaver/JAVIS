"""FastAPI server for JAVIS web interface."""

import functools
import logging
import os
import aiofiles
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

# Type variable for generic decorator
T = TypeVar("T")

api_logger = logging.getLogger(__name__)

from javis.utils.config import load_config, get_config
from javis.utils.constants import DEFAULT_CORS_ORIGINS, EnvVars
from javis.models.client import ModelClient, Message
from javis.models.local_client import (
    LocalModelClient,
    Message as LocalMessage,
    get_latest_adapter,
    list_adapters,
)
from javis.models.modal_client import (
    ModalInferenceClient,
    Message as ModalMessage,
    check_modal_available,
    get_latest_adapter_path,
)
from javis.data.conversation_logger import get_logger
from javis.training.scheduler import (
    get_scheduler,
    start_scheduler,
    stop_scheduler,
    SCHEDULER_AVAILABLE,
)
from javis.core.engine import ChatEngine
from javis.tools import initialize_tools

# 업로드 디렉토리
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Error Handling Decorators ---

def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    API 엔드포인트의 공통 에러 핸들링 데코레이터.

    - HTTPException은 그대로 전달
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found
    - PermissionError -> 403 Forbidden
    - 기타 예외 -> 500 Internal Server Error (상세 로깅)
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            api_logger.exception(f"Unexpected error in {func.__name__}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {type(e).__name__}"
            )
    return wrapper


def require_feature(feature_path: str, error_message: str = None):
    """
    특정 기능이 활성화되어 있는지 확인하는 데코레이터.

    Args:
        feature_path: 설정 경로 (예: "memory.long_term.enabled", "rag.enabled")
        error_message: 비활성화 시 표시할 에러 메시지
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = get_config()

            # 설정 경로를 따라가며 값 확인
            value = config
            for attr in feature_path.split("."):
                value = getattr(value, attr, None)
                if value is None:
                    break

            if not value:
                msg = error_message or f"Feature '{feature_path}' is not enabled"
                raise HTTPException(status_code=400, detail=msg)

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Session storage (in-memory, 프로덕션에서는 Redis 등 사용)
sessions: dict[str, list[Message]] = {}

# Local model client (lazy loaded, singleton)
_local_client: Optional[LocalModelClient] = None
_local_client_loading: bool = False

# Modal inference client (lazy loaded)
_modal_client: Optional[ModalInferenceClient] = None


class ChatRequest(BaseModel):
    """Chat request from client."""
    message: str
    session_id: str = "default"
    model: str = "groq"  # "groq", "local", "modal"
    adapter_version: Optional[str] = None  # 특정 어댑터 버전 지정 (local/modal용)
    use_tools: bool = False  # 도구 사용 여부


class ChatResponse(BaseModel):
    """Chat response to client."""
    response: str
    session_id: str
    model_used: str = "groq"  # "groq" or "local"
    tools_used: bool = False  # 도구 사용 여부


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    import logging
    logger = logging.getLogger(__name__)

    # Startup
    load_config()
    config = get_config()

    # Initialize tools if enabled
    if config.tools.enabled:
        initialize_tools(config.tools.available)
        logger.info(f"Tools initialized: {config.tools.available}")

    # Auto-start scheduler if enabled in config
    if config.training.schedule.enabled and SCHEDULER_AVAILABLE:
        if start_scheduler():
            scheduler = get_scheduler()
            logger.info(f"Training scheduler started. Next run: {scheduler.get_next_run_time()}")

    yield

    # Shutdown
    if SCHEDULER_AVAILABLE:
        stop_scheduler()
    sessions.clear()


# API Tags for documentation organization
tags_metadata = [
    {
        "name": "Chat",
        "description": "Chat endpoints for conversing with JAVIS",
    },
    {
        "name": "Session",
        "description": "Session management endpoints",
    },
    {
        "name": "Models",
        "description": "Model and adapter management",
    },
    {
        "name": "Memory",
        "description": "Long-term memory system",
    },
    {
        "name": "RAG",
        "description": "Retrieval-Augmented Generation for document knowledge",
    },
    {
        "name": "Training",
        "description": "Model training and fine-tuning endpoints",
    },
    {
        "name": "Tools",
        "description": "Tool system endpoints",
    },
    {
        "name": "Voice",
        "description": "Voice interface (STT/TTS)",
    },
    {
        "name": "Workflows",
        "description": "Workflow automation",
    },
    {
        "name": "Health",
        "description": "Health check and status endpoints",
    },
]

app = FastAPI(
    title="JAVIS API",
    description="""
## JAVIS - Personal AI Assistant API

JAVIS is a personal AI assistant with custom fine-tuned models and integrated tool support.

### Features
- **Multi-model support**: Groq (cloud), Modal (serverless GPU), Local (fine-tuned)
- **Tool integration**: File operations, web search, calendar, Slack, Notion
- **Long-term memory**: Conversation history with semantic search
- **RAG system**: Document knowledge base with vector search
- **Voice interface**: Speech-to-text and text-to-speech
- **Workflow automation**: Scheduled tasks and triggers

### Authentication
Currently no authentication is required. For production, configure CORS and add API key authentication.
""",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS 설정 - 환경변수로 허용 도메인 관리
# CORS_ORIGINS 환경변수: 쉼표로 구분된 도메인 목록 (예: "https://example.com,https://app.example.com")
# 설정하지 않으면 개발 환경용 기본값 사용
def _get_cors_origins() -> list[str]:
    """환경변수에서 CORS 허용 도메인 목록을 가져옵니다."""
    origins_env = os.getenv(EnvVars.CORS_ORIGINS, "").strip()
    if origins_env:
        return [origin.strip() for origin in origins_env.split(",") if origin.strip()]
    # 개발 환경 기본값 (프로덕션에서는 반드시 CORS_ORIGINS 설정 필요)
    return DEFAULT_CORS_ORIGINS.copy()

_cors_origins = _get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
)


def get_or_create_session(session_id: str) -> list[Message]:
    """Get or create a chat session."""
    if session_id not in sessions:
        config = get_config()
        sessions[session_id] = [
            Message(role="system", content=config.conversation.system_prompt)
        ]
    return sessions[session_id]


def get_adapter_path(adapter_version: Optional[str] = None) -> str:
    """Get adapter path, either specific version or latest."""
    models_dir = Path(__file__).parent.parent.parent / "models"

    if adapter_version:
        adapter_path = models_dir / adapter_version / "adapter"
        if not adapter_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Adapter version not found: {adapter_version}"
            )
        return str(adapter_path)
    else:
        adapter_path = get_latest_adapter(str(models_dir))
        if not adapter_path:
            raise HTTPException(
                status_code=404,
                detail="No trained adapter found. Please train a model first."
            )
        return adapter_path


def get_local_client(adapter_version: Optional[str] = None) -> LocalModelClient:
    """Get or create local model client."""
    global _local_client

    adapter_path = get_adapter_path(adapter_version)

    # Create new client if needed
    if _local_client is None or _local_client.adapter_path != adapter_path:
        _local_client = LocalModelClient(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path=adapter_path,
            load_in_4bit=True,
        )

    return _local_client


def get_modal_client(adapter_version: Optional[str] = None) -> ModalInferenceClient:
    """Get or create Modal inference client."""
    global _modal_client

    if not check_modal_available():
        raise HTTPException(
            status_code=500,
            detail="Modal is not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."
        )

    adapter_path = get_adapter_path(adapter_version)

    # Create new client if needed (always recreate with new adapter)
    _modal_client = ModalInferenceClient(adapter_path=adapter_path)

    return _modal_client


@app.get("/")
async def root():
    """Serve the web interface."""
    static_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "JAVIS API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    """Simple health check endpoint.

    Returns a basic health status. For detailed status, use `/api/status`.
    """
    return {"status": "healthy"}


@app.get("/api/status", tags=["Health"])
async def get_system_status():
    """Get detailed system status.

    Returns comprehensive status of all JAVIS subsystems including:
    - API key availability
    - Memory system status
    - RAG system status
    - Tool system status
    - Training scheduler status
    - Voice system status
    - Workflow scheduler status
    """
    import sys
    from datetime import datetime

    config = get_config()

    status = {
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "python_version": sys.version,
        "components": {
            "groq": {
                "available": bool(config.groq_api_key),
                "model": "llama-3.1-8b-instant",
            },
            "modal": {
                "available": check_modal_available(),
            },
            "local_model": {
                "loaded": _local_client is not None and getattr(_local_client, '_loaded', False),
                "adapter": _local_client.adapter_path if _local_client else None,
            },
            "tools": {
                "enabled": config.tools.enabled,
                "count": len(config.tools.available) if config.tools.enabled else 0,
            },
            "memory": {
                "enabled": config.memory.long_term.enabled,
            },
            "rag": {
                "enabled": config.rag.enabled,
            },
            "voice": {
                "enabled": config.voice.enabled,
            },
            "workflows": {
                "enabled": config.workflows.enabled,
            },
            "training_scheduler": {
                "available": SCHEDULER_AVAILABLE,
                "running": get_scheduler().get_status().running if SCHEDULER_AVAILABLE and get_scheduler() else False,
            },
        },
        "sessions": {
            "active_count": len(sessions),
        },
    }

    return status


@app.get("/api/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes-style readiness probe.

    Returns 200 if the service is ready to accept traffic,
    503 if not ready (e.g., still initializing).
    """
    config = get_config()

    # Check if at least one model backend is available
    groq_ready = bool(config.groq_api_key)
    modal_ready = check_modal_available()
    local_ready = _local_client is not None and getattr(_local_client, '_loaded', False)

    if groq_ready or modal_ready or local_ready:
        return {
            "ready": True,
            "backends": {
                "groq": groq_ready,
                "modal": modal_ready,
                "local": local_ready,
            }
        }

    from fastapi import HTTPException
    raise HTTPException(
        status_code=503,
        detail="No model backend available"
    )


# --- Model Handlers (chat 엔드포인트 리팩토링) ---

async def _handle_modal_chat(
    messages: list[Message],
    adapter_version: Optional[str]
) -> tuple[Any, str]:
    """Modal.com GPU를 통한 파인튜닝 모델 추론."""
    modal_client = get_modal_client(adapter_version)
    modal_messages = [
        ModalMessage(role=m.role, content=m.content) for m in messages
    ]
    response = await modal_client.chat_async(modal_messages)
    return response, "modal"


async def _handle_local_chat(
    messages: list[Message],
    adapter_version: Optional[str]
) -> tuple[Any, str]:
    """로컬 GPU를 통한 파인튜닝 모델 추론."""
    local_client = get_local_client(adapter_version)
    local_messages = [
        LocalMessage(role=m.role, content=m.content) for m in messages
    ]
    response = await local_client.chat_async(local_messages)
    return response, "local"


async def _handle_groq_chat(
    messages: list[Message],
    use_tools: bool,
    config
) -> tuple[Any, str, bool]:
    """Groq API를 통한 추론."""
    if not config.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="Groq API key not configured"
        )

    tools_used = False

    if use_tools and config.tools.enabled:
        engine = ChatEngine()
        response = await engine.chat_with_tools(messages)
        tools_used = True
    else:
        client = ModelClient()
        response = await client.chat(messages)

    return response, "groq", tools_used


def _trim_session_history(
    session_id: str,
    messages: list[Message],
    max_history: int
) -> None:
    """세션 히스토리를 최대 길이로 제한."""
    if len(messages) > max_history + 1:
        sessions[session_id] = [messages[0]] + messages[-(max_history):]


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
@handle_api_errors
async def chat(request: ChatRequest):
    """Chat with JAVIS.

    Send a message and receive an AI response. Supports multiple model backends.

    **Models:**
    - `groq`: Fast inference via Groq API (Llama 3.1)
    - `modal`: Fine-tuned model via Modal.com GPU (recommended)
    - `local`: Fine-tuned model on local GPU (requires CUDA)

    **Tool Usage:**
    Set `use_tools=true` to enable tool calling (Groq only).
    """
    config = get_config()
    conv_logger = get_logger()

    # 세션 및 메시지 준비
    messages = get_or_create_session(request.session_id)
    messages.append(Message(role="user", content=request.message))
    conv_logger.add_turn(request.session_id, "user", request.message)

    # 모델별 핸들러 호출
    tools_used = False

    if request.model == "modal":
        response, model_used = await _handle_modal_chat(
            messages, request.adapter_version
        )
    elif request.model == "local":
        response, model_used = await _handle_local_chat(
            messages, request.adapter_version
        )
    else:
        response, model_used, tools_used = await _handle_groq_chat(
            messages, request.use_tools, config
        )

    # 응답 처리
    messages.append(Message(role="assistant", content=response.content))
    conv_logger.add_turn(request.session_id, "assistant", response.content)
    _trim_session_history(
        request.session_id, messages, config.conversation.max_history
    )

    return ChatResponse(
        response=response.content,
        session_id=request.session_id,
        model_used=model_used,
        tools_used=tools_used
    )


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Streaming chat with JAVIS (Groq only for now).

    Returns Server-Sent Events (SSE) stream.
    """
    config = get_config()
    log = get_logger()

    # 세션 가져오기
    messages = get_or_create_session(request.session_id)

    # 사용자 메시지 추가
    messages.append(Message(role="user", content=request.message))

    # 대화 로깅
    log.add_turn(request.session_id, "user", request.message)

    async def generate():
        """Generate streaming response."""
        full_response = ""

        try:
            if request.model != "groq":
                # Modal/local don't support streaming yet
                yield f"data: {{\"error\": \"Streaming only available for Groq model\"}}\n\n"
                return

            if not config.groq_api_key:
                yield f"data: {{\"error\": \"Groq API key not configured\"}}\n\n"
                return

            client = ModelClient()

            async for chunk in client.chat_stream(messages):
                full_response += chunk
                # SSE format: data: {json}\n\n
                yield f"data: {{\"content\": {repr(chunk)}}}\n\n"

            # Send done signal
            yield f"data: {{\"done\": true}}\n\n"

            # 응답 저장
            messages.append(Message(role="assistant", content=full_response))
            log.add_turn(request.session_id, "assistant", full_response)

            # 히스토리 제한
            max_history = config.conversation.max_history
            if len(messages) > max_history + 1:
                sessions[request.session_id] = [messages[0]] + messages[-(max_history):]

        except Exception as e:
            yield f"data: {{\"error\": {repr(str(e))}}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear", tags=["Session"])
async def clear_session(session_id: str = "default"):
    """Clear a chat session and save conversation log."""
    logger = get_logger()
    config = get_config()
    memory_saved = False

    # Save to long-term memory if enabled
    if session_id in sessions and config.memory.long_term.enabled:
        try:
            from javis.memory import get_memory_manager
            memory_manager = get_memory_manager()
            messages = sessions[session_id]

            # Convert Message objects to dict for memory storage
            turns = [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"  # Exclude system message
            ]

            if len(turns) >= 2:  # At least one exchange
                result = await memory_manager.store_conversation(
                    session_id=session_id,
                    turns=turns
                )
                memory_saved = result is not None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save to memory: {e}")

    # 대화 저장 후 종료
    filepath = logger.end_conversation(session_id)

    if session_id in sessions:
        sessions[session_id] = [
            Message(role="system", content=config.conversation.system_prompt)
        ]

    return {
        "status": "cleared",
        "session_id": session_id,
        "saved_to": str(filepath) if filepath else None,
        "memory_saved": memory_saved
    }


@app.post("/api/feedback", tags=["Session"])
async def add_feedback(session_id: str, feedback: str):
    """Add feedback (good/bad) to current conversation for training."""
    logger = get_logger()

    if feedback not in ["good", "bad"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'good' or 'bad'")

    logger.add_feedback(session_id, feedback)
    return {"status": "feedback_added", "session_id": session_id, "feedback": feedback}


@app.post("/api/export-training")
async def export_training_data(feedback_filter: str = None):
    """Export conversations as training data (JSONL format)."""
    logger = get_logger()

    # 현재 활성 대화들 먼저 저장
    for session_id in list(logger.active_conversations.keys()):
        logger.save_conversation(session_id)

    # JSONL로 내보내기
    output_file = logger.export_for_training(feedback_filter=feedback_filter)

    return {
        "status": "exported",
        "file": str(output_file)
    }


@app.get("/api/sessions", tags=["Session"])
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {"id": sid, "message_count": len(msgs)}
            for sid, msgs in sessions.items()
        ]
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files."""
    uploaded = []
    for file in files:
        # 파일 저장
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        uploaded.append(file.filename)

    return {"files": uploaded, "count": len(uploaded)}


@app.get("/api/models", tags=["Models"])
async def get_models():
    """Get available models and adapters."""
    models_dir = Path(__file__).parent.parent.parent / "models"
    adapters = list_adapters(str(models_dir))

    # Check local model status
    local_status = "not_loaded"
    local_adapter = None
    if _local_client is not None:
        local_status = "loaded" if _local_client._loaded else "initialized"
        local_adapter = _local_client.adapter_path

    return {
        "groq": {
            "available": bool(get_config().groq_api_key),
            "model": "llama-3.1-8b-instant",
            "description": "Fast inference via Groq API",
        },
        "modal": {
            "available": check_modal_available(),
            "model": "Qwen/Qwen2.5-7B-Instruct + LoRA",
            "description": "Fine-tuned model via Modal.com GPU (recommended)",
        },
        "local": {
            "available": False,  # No local GPU
            "status": local_status,
            "current_adapter": local_adapter,
            "description": "Fine-tuned model on local GPU (requires CUDA)",
        },
        "adapters": adapters,
    }


@app.post("/api/models/local/load", tags=["Models"])
async def load_local_model(adapter_version: Optional[str] = None):
    """Pre-load the local model (takes a few minutes)."""
    try:
        client = get_local_client(adapter_version)

        # Load in background thread
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client.load)

        return {
            "status": "loaded",
            "adapter": client.adapter_path,
            "base_model": client.base_model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/local/unload", tags=["Models"])
async def unload_local_model():
    """Unload the local model to free memory."""
    global _local_client

    if _local_client is not None:
        # Clear model from memory
        if _local_client.model is not None:
            del _local_client.model
        if _local_client.tokenizer is not None:
            del _local_client.tokenizer
        _local_client = None

        # Force garbage collection
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status": "unloaded"}

    return {"status": "not_loaded"}


# ==================== Training Scheduler API ====================

@app.get("/api/training/scheduler")
async def get_scheduler_status():
    """Get training scheduler status."""
    if not SCHEDULER_AVAILABLE:
        return {
            "available": False,
            "message": "APScheduler not installed. Run: pip install apscheduler",
        }

    scheduler = get_scheduler()
    status = scheduler.get_status()

    return {
        "available": True,
        "running": status.running,
        "enabled": status.enabled,
        "next_run": status.next_run,
        "last_run": status.last_run,
        "last_result": status.last_result,
        "cron": status.cron,
        "timezone": status.timezone,
    }


@app.post("/api/training/scheduler/start")
async def start_training_scheduler():
    """Start the automatic training scheduler."""
    if not SCHEDULER_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="APScheduler not installed. Run: pip install apscheduler",
        )

    success = start_scheduler()

    if success:
        scheduler = get_scheduler()
        return {
            "status": "started",
            "next_run": scheduler.get_next_run_time(),
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to start scheduler. Check config and logs.",
        )


@app.post("/api/training/scheduler/stop")
async def stop_training_scheduler():
    """Stop the automatic training scheduler."""
    if not SCHEDULER_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="APScheduler not installed",
        )

    success = stop_scheduler()

    return {"status": "stopped" if success else "failed"}


@app.post("/api/training/run")
async def trigger_training():
    """Manually trigger a training run now."""
    from javis.training.pipeline import TrainingPipeline
    from javis.utils.config import get_config

    config = get_config()

    try:
        pipeline = TrainingPipeline(config.training)
        result = pipeline.run()

        return {
            "success": result.success,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "version": result.version,
            "error": result.error,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DPO Training API ====================

@app.get("/api/training/dpo/stats", tags=["Training"])
async def get_dpo_preference_stats():
    """Get statistics about available preference data for DPO training."""
    try:
        from javis.training.preference_data import get_preference_generator

        generator = get_preference_generator()
        stats = generator.get_statistics()

        return {
            "total_pairs": stats.total_pairs,
            "from_corrections": stats.from_corrections,
            "from_feedback": stats.from_feedback,
            "from_quality_score": stats.from_quality_score,
            "ready_for_training": stats.ready_for_training,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/dpo/export", tags=["Training"])
async def export_dpo_preference_data():
    """Export preference data for DPO training as JSONL."""
    try:
        from javis.training.preference_data import get_preference_generator

        generator = get_preference_generator()
        pairs = generator.generate_all()
        output_path = generator.export_dpo_dataset(pairs=pairs)

        return {
            "status": "exported",
            "count": len(pairs),
            "path": str(output_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/dpo/run", tags=["Training"])
async def trigger_dpo_training(force: bool = False):
    """Manually trigger a DPO training run.

    Args:
        force: Skip minimum data requirements check
    """
    from javis.training.pipeline import TrainingPipeline
    from javis.utils.config import get_config

    config = get_config()

    try:
        pipeline = TrainingPipeline(config.training)
        result = pipeline.run_dpo(force=force)

        return {
            "success": result.success,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "version": result.version,
            "error": result.error,
            "dataset_size": result.dataset_size,
            "duration_seconds": result.duration_seconds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/dpo/config", tags=["Training"])
async def get_dpo_config():
    """Get current DPO training configuration."""
    from javis.utils.config import get_config

    config = get_config()
    dpo_config = config.training.dpo

    return {
        "enabled": dpo_config.enabled,
        "beta": dpo_config.beta,
        "loss_type": dpo_config.loss_type,
        "min_preference_pairs": dpo_config.min_preference_pairs,
        "use_reference_model": dpo_config.use_reference_model,
        "reference_model": dpo_config.reference_model,
        "max_prompt_length": dpo_config.max_prompt_length,
        "max_response_length": dpo_config.max_response_length,
    }


# ==================== Tools API ====================

@app.get("/api/tools", tags=["Tools"])
async def get_tools_info():
    """Get information about available tools."""
    from javis.tools import get_tools_info
    config = get_config()

    if not config.tools.enabled:
        return {
            "enabled": False,
            "message": "Tools are disabled in config",
        }

    info = get_tools_info()
    return {
        "enabled": True,
        "registered_tools": info["registered_tools"],
        "enabled_tools": info["enabled_tools"],
        "enabled_categories": info["enabled_categories"],
    }


# ==================== Memory API ====================

class MemorySearchRequest(BaseModel):
    """Memory search request."""
    query: str
    limit: int = 5
    min_score: float = 0.3


@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory system statistics."""
    config = get_config()

    if not config.memory.long_term.enabled:
        return {
            "enabled": False,
            "message": "Long-term memory is disabled in config",
        }

    try:
        from javis.memory import get_memory_manager
        manager = get_memory_manager()
        stats = await manager.get_stats()

        return {
            "enabled": True,
            "count": stats["count"],
            "db_path": stats["db_path"],
            "collection": stats["collection"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search memories."""
    config = get_config()

    if not config.memory.long_term.enabled:
        raise HTTPException(status_code=400, detail="Memory system is disabled")

    try:
        from javis.memory import get_memory_manager
        manager = get_memory_manager()
        results = await manager.search_memories(
            request.query,
            limit=request.limit,
            min_score=request.min_score
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "id": r.entry.id,
                    "session_id": r.entry.session_id,
                    "summary": r.entry.summary,
                    "topics": r.entry.topics,
                    "score": r.score,
                    "created_at": r.entry.created_at.isoformat(),
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory entry."""
    config = get_config()

    if not config.memory.long_term.enabled:
        raise HTTPException(status_code=400, detail="Memory system is disabled")

    try:
        from javis.memory import get_memory_manager
        manager = get_memory_manager()
        success = await manager.delete_memory(memory_id)

        if success:
            return {"status": "deleted", "id": memory_id}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RAG API ====================

class RAGAddDocumentRequest(BaseModel):
    """Request to add a document to RAG."""
    source_type: str  # file, web, code
    source_path: str


class RAGSearchRequest(BaseModel):
    """RAG search request."""
    query: str
    top_k: int = 5
    min_score: float = 0.3


@app.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics."""
    config = get_config()

    if not config.rag.enabled:
        return {
            "enabled": False,
            "message": "RAG system is disabled in config",
        }

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()
        stats = await manager.get_stats()

        return {
            "enabled": True,
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "by_source": stats.by_source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/documents")
async def add_rag_document(request: RAGAddDocumentRequest):
    """Add a document to RAG system."""
    config = get_config()

    if not config.rag.enabled:
        raise HTTPException(status_code=400, detail="RAG system is disabled")

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()

        document = await manager.add_document(
            source_type=request.source_type,
            source_path=request.source_path
        )

        if document:
            return {
                "status": "added",
                "document": {
                    "id": document.id,
                    "title": document.title,
                    "source": document.source.value,
                    "source_path": document.source_path,
                },
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add document")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/documents")
async def list_rag_documents(limit: int = 100, offset: int = 0):
    """List documents in RAG system."""
    config = get_config()

    if not config.rag.enabled:
        raise HTTPException(status_code=400, detail="RAG system is disabled")

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()
        documents = await manager.list_documents(limit=limit, offset=offset)

        return {
            "count": len(documents),
            "documents": documents,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/search")
async def search_rag(request: RAGSearchRequest):
    """Search RAG documents."""
    config = get_config()

    if not config.rag.enabled:
        raise HTTPException(status_code=400, detail="RAG system is disabled")

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()

        results = await manager.search(
            request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "chunk_id": r.chunk.id,
                    "document_id": r.chunk.document_id,
                    "content": r.chunk.content,
                    "score": r.score,
                    "document_title": r.document.title if r.document else None,
                    "source_path": r.document.source_path if r.document else None,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/documents/{document_id}")
async def delete_rag_document(document_id: str):
    """Delete a document from RAG system."""
    config = get_config()

    if not config.rag.enabled:
        raise HTTPException(status_code=400, detail="RAG system is disabled")

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()
        success = await manager.delete_document(document_id)

        if success:
            return {"status": "deleted", "id": document_id}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/codebase")
async def add_codebase(directory: str, max_files: int = 100):
    """Add a codebase directory to RAG system."""
    config = get_config()

    if not config.rag.enabled:
        raise HTTPException(status_code=400, detail="RAG system is disabled")

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()

        documents = await manager.add_codebase(
            directory=directory,
            max_files=max_files
        )

        return {
            "status": "added",
            "count": len(documents),
            "documents": [
                {"id": d.id, "title": d.title}
                for d in documents
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 5: Data Quality API ====================

@app.get("/api/training/quality/stats")
async def get_quality_stats():
    """Get data quality statistics."""
    try:
        from javis.training.data_quality import get_quality_scorer
        scorer = get_quality_scorer()
        stats = scorer.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/quality/score-all")
async def score_all_conversations():
    """Score all conversations for quality."""
    try:
        from javis.training.data_quality import get_quality_scorer
        scorer = get_quality_scorer()
        scores = scorer.score_all_conversations()

        return {
            "count": len(scores),
            "scores": [
                {
                    "conversation_id": s.conversation_id,
                    "overall_score": s.overall_score,
                    "length_score": s.length_score,
                    "coherence_score": s.coherence_score,
                    "feedback_score": s.feedback_score,
                    "recency_score": s.recency_score,
                }
                for s in sorted(scores, key=lambda x: x.overall_score, reverse=True)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 5: Metrics API ====================

@app.get("/api/training/metrics/{version}")
async def get_version_metrics(version: str, days: int = 7):
    """Get performance metrics for a model version."""
    try:
        from javis.training.metrics import get_metrics_collector
        from datetime import datetime, timedelta

        collector = get_metrics_collector()
        end = datetime.now()
        start = end - timedelta(days=days)

        metrics = collector.get_metrics(version, start, end)

        return {
            "version": metrics.version,
            "period_start": metrics.period_start.isoformat(),
            "period_end": metrics.period_end.isoformat(),
            "total_requests": metrics.total_requests,
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "p95_latency_ms": round(metrics.p95_latency_ms, 2),
            "positive_feedback_rate": round(metrics.positive_feedback_rate, 4),
            "negative_feedback_rate": round(metrics.negative_feedback_rate, 4),
            "error_rate": round(metrics.error_rate, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/metrics/compare")
async def compare_version_metrics(version_a: str, version_b: str, days: int = 7):
    """Compare metrics between two versions."""
    try:
        from javis.training.metrics import get_metrics_collector
        collector = get_metrics_collector()
        comparison = collector.compare_versions(version_a, version_b, days)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/metrics/dashboard")
async def get_metrics_dashboard():
    """Get metrics dashboard data."""
    try:
        from javis.training.metrics import get_metrics_collector
        collector = get_metrics_collector()
        return collector.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 5: A/B Testing API ====================

class ABTestCreateRequest(BaseModel):
    """Request to create an A/B test."""
    version_a: str
    version_b: str
    traffic_split: float = 0.5
    description: Optional[str] = None


@app.post("/api/training/ab-tests")
async def create_ab_test(request: ABTestCreateRequest):
    """Create a new A/B test."""
    try:
        from javis.training.ab_testing import get_ab_test_manager
        manager = get_ab_test_manager()

        test_id = manager.create_test(
            version_a=request.version_a,
            version_b=request.version_b,
            traffic_split=request.traffic_split,
            description=request.description,
        )

        return {
            "status": "created",
            "test_id": test_id,
            "version_a": request.version_a,
            "version_b": request.version_b,
            "traffic_split": request.traffic_split,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/ab-tests")
async def list_ab_tests(status: Optional[str] = None):
    """List all A/B tests."""
    try:
        from javis.training.ab_testing import get_ab_test_manager
        manager = get_ab_test_manager()
        tests = manager.list_tests(status=status)

        return {
            "count": len(tests),
            "tests": [
                {
                    "test_id": t.test_id,
                    "version_a": t.version_a,
                    "version_b": t.version_b,
                    "traffic_split": t.traffic_split,
                    "status": t.status,
                    "start_time": t.start_time.isoformat(),
                    "end_time": t.end_time.isoformat() if t.end_time else None,
                }
                for t in tests
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/ab-tests/{test_id}")
async def get_ab_test(test_id: str):
    """Get A/B test details and results."""
    try:
        from javis.training.ab_testing import get_ab_test_manager
        manager = get_ab_test_manager()

        config = manager.get_test(test_id)
        if not config:
            raise HTTPException(status_code=404, detail="Test not found")

        result = manager.calculate_results(test_id)

        return {
            "test_id": test_id,
            "config": {
                "version_a": config.version_a,
                "version_b": config.version_b,
                "traffic_split": config.traffic_split,
                "status": config.status,
                "start_time": config.start_time.isoformat(),
            },
            "results": {
                "version_a": {
                    "requests": result.version_a_metrics.requests,
                    "positive_feedback": result.version_a_metrics.positive_feedback,
                    "feedback_rate": result.version_a_metrics.feedback_rate,
                    "avg_latency_ms": result.version_a_metrics.avg_latency_ms,
                },
                "version_b": {
                    "requests": result.version_b_metrics.requests,
                    "positive_feedback": result.version_b_metrics.positive_feedback,
                    "feedback_rate": result.version_b_metrics.feedback_rate,
                    "avg_latency_ms": result.version_b_metrics.avg_latency_ms,
                },
                "statistical_significance": result.statistical_significance,
                "winner": result.winner,
                "recommendation": result.recommendation,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/ab-tests/{test_id}/complete")
async def complete_ab_test(test_id: str, winner: Optional[str] = None):
    """Complete an A/B test."""
    try:
        from javis.training.ab_testing import get_ab_test_manager
        manager = get_ab_test_manager()
        manager.complete_test(test_id, winner)
        return {"status": "completed", "test_id": test_id, "winner": winner}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/training/ab-tests/{test_id}")
async def delete_ab_test(test_id: str):
    """Cancel/delete an A/B test."""
    try:
        from javis.training.ab_testing import get_ab_test_manager
        manager = get_ab_test_manager()
        manager.cancel_test(test_id)
        return {"status": "cancelled", "test_id": test_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 5: Corrections API ====================

class CorrectionRequest(BaseModel):
    """Request to add a correction."""
    session_id: str
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: str = "other"
    notes: Optional[str] = None


@app.post("/api/training/corrections")
async def add_correction(request: CorrectionRequest):
    """Add a response correction."""
    try:
        from javis.training.corrections import get_correction_manager
        manager = get_correction_manager()

        correction_id = manager.add_correction(
            session_id=request.session_id,
            original_prompt=request.original_prompt,
            original_response=request.original_response,
            corrected_response=request.corrected_response,
            correction_type=request.correction_type,
            notes=request.notes,
        )

        return {
            "status": "added",
            "correction_id": correction_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/corrections")
async def list_corrections(limit: int = 50):
    """List response corrections."""
    try:
        from javis.training.corrections import get_correction_manager
        manager = get_correction_manager()
        corrections = manager.get_corrections(limit=limit)

        return {
            "count": len(corrections),
            "corrections": [
                {
                    "id": c.id,
                    "session_id": c.session_id,
                    "correction_type": c.correction_type,
                    "timestamp": c.timestamp.isoformat(),
                    "original_prompt": c.original_prompt[:100] + "..." if len(c.original_prompt) > 100 else c.original_prompt,
                }
                for c in corrections
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/corrections/{correction_id}")
async def get_correction(correction_id: str):
    """Get correction details."""
    try:
        from javis.training.corrections import get_correction_manager
        manager = get_correction_manager()
        correction = manager.get_correction(correction_id)

        if not correction:
            raise HTTPException(status_code=404, detail="Correction not found")

        return {
            "id": correction.id,
            "session_id": correction.session_id,
            "correction_type": correction.correction_type,
            "timestamp": correction.timestamp.isoformat(),
            "original_prompt": correction.original_prompt,
            "original_response": correction.original_response,
            "corrected_response": correction.corrected_response,
            "notes": correction.notes,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/training/corrections/{correction_id}")
async def delete_correction_endpoint(correction_id: str):
    """Delete a correction."""
    try:
        from javis.training.corrections import get_correction_manager
        manager = get_correction_manager()

        if manager.delete_correction(correction_id):
            return {"status": "deleted", "id": correction_id}
        else:
            raise HTTPException(status_code=404, detail="Correction not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/corrections/stats")
async def get_correction_stats():
    """Get correction statistics."""
    try:
        from javis.training.corrections import get_correction_manager
        manager = get_correction_manager()
        return manager.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 5: Enhanced Feedback API ====================

class EnhancedFeedbackRequest(BaseModel):
    """Request for enhanced feedback."""
    session_id: str
    prompt: str
    response: str
    feedback_type: str  # "good", "bad"
    quality_score: Optional[float] = None
    corrected_response: Optional[str] = None


@app.post("/api/training/feedback/enhanced")
async def add_enhanced_feedback(request: EnhancedFeedbackRequest):
    """Add enhanced feedback with optional quality score and correction."""
    try:
        from javis.training.feedback_store import get_feedback_store
        store = get_feedback_store()

        if request.feedback_type == "good":
            feedback_id = store.store_good_response(
                session_id=request.session_id,
                prompt=request.prompt,
                response=request.response,
                quality_score=request.quality_score,
            )
        else:
            feedback_id = store.store_bad_response(
                session_id=request.session_id,
                prompt=request.prompt,
                response=request.response,
                corrected_response=request.corrected_response,
            )

        return {
            "status": "added",
            "feedback_id": feedback_id,
            "feedback_type": request.feedback_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/feedback/stats")
async def get_feedback_stats():
    """Get enhanced feedback statistics."""
    try:
        from javis.training.feedback_store import get_feedback_store
        store = get_feedback_store()
        return store.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 6: Voice API ====================

class TTSRequest(BaseModel):
    """TTS request."""
    text: str
    voice: Optional[str] = None


@app.get("/api/voice/status")
async def get_voice_status():
    """Get voice system status."""
    config = get_config()

    return {
        "enabled": config.voice.enabled,
        "language": config.voice.language,
        "stt": {
            "provider": config.voice.stt.provider,
            "model": config.voice.stt.groq_whisper.model,
        },
        "tts": {
            "provider": config.voice.tts.provider,
            "voice": config.voice.tts.edge_tts.voice,
        },
    }


@app.post("/api/voice/stt")
async def voice_stt(file: UploadFile = File(...), language: str = "ko"):
    """Convert speech to text.

    Upload an audio file and get transcription.
    Supported formats: wav, mp3, m4a, webm
    """
    config = get_config()

    if not config.voice.enabled:
        raise HTTPException(status_code=400, detail="Voice interface is disabled")

    if not config.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        from javis.voice import get_stt_provider

        stt = get_stt_provider()

        # Read uploaded file
        audio_data = await file.read()

        # Transcribe
        result = await stt.transcribe(
            audio_data=audio_data,
            language=language,
            filename=file.filename or "audio.wav",
        )

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/tts")
async def voice_tts(request: TTSRequest):
    """Convert text to speech.

    Returns audio as MP3 stream.
    """
    config = get_config()

    if not config.voice.enabled:
        raise HTTPException(status_code=400, detail="Voice interface is disabled")

    try:
        from javis.voice import get_tts_provider
        from javis.voice.tts.edge_tts_provider import EdgeTTSProvider

        # Use custom voice if specified
        if request.voice:
            tts = EdgeTTSProvider(voice=request.voice)
        else:
            tts = get_tts_provider()

        # Generate audio
        audio_data = await tts.synthesize(request.text)

        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/tts/stream")
async def voice_tts_stream(request: TTSRequest):
    """Stream text-to-speech audio.

    Returns chunked audio stream.
    """
    config = get_config()

    if not config.voice.enabled:
        raise HTTPException(status_code=400, detail="Voice interface is disabled")

    try:
        from javis.voice import get_tts_provider
        from javis.voice.tts.edge_tts_provider import EdgeTTSProvider

        # Use custom voice if specified
        if request.voice:
            tts = EdgeTTSProvider(voice=request.voice)
        else:
            tts = get_tts_provider()

        async def audio_generator():
            async for chunk in tts.synthesize_stream(request.text):
                yield chunk

        return StreamingResponse(
            audio_generator(),
            media_type="audio/mpeg",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    session_id: str = "default",
    language: str = "ko",
):
    """Voice chat: speech input → text response.

    Upload audio, get AI response as text.
    """
    config = get_config()

    if not config.voice.enabled:
        raise HTTPException(status_code=400, detail="Voice interface is disabled")

    if not config.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        from javis.voice import get_stt_provider

        stt = get_stt_provider()

        # 1. Transcribe audio
        audio_data = await file.read()
        transcription = await stt.transcribe(
            audio_data=audio_data,
            language=language,
            filename=file.filename or "audio.wav",
        )

        if not transcription.text.strip():
            return {
                "transcription": "",
                "response": "음성을 인식하지 못했습니다.",
                "session_id": session_id,
            }

        # 2. Get chat response using existing chat logic
        messages = get_or_create_session(session_id)
        messages.append(Message(role="user", content=transcription.text))

        # Use Groq for response
        client = ModelClient()
        response = await client.chat(messages)

        messages.append(Message(role="assistant", content=response.content))

        # Trim history
        max_history = config.conversation.max_history
        if len(messages) > max_history + 1:
            sessions[session_id] = [messages[0]] + messages[-(max_history):]

        return {
            "transcription": transcription.text,
            "response": response.content,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/tts/voices")
async def list_tts_voices(language: Optional[str] = None):
    """List available TTS voices.

    Args:
        language: Filter by language (e.g., "ko-KR", "en-US")
    """
    try:
        from javis.voice.tts.edge_tts_provider import EdgeTTSProvider

        voices = await EdgeTTSProvider.list_voices(language)

        return {
            "count": len(voices),
            "voices": voices,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase 6-2: Workflow API ====================

@app.get("/api/workflows")
async def list_workflows():
    """List all workflows."""
    config = get_config()

    if not config.workflows.enabled:
        return {
            "enabled": False,
            "message": "Workflows are disabled in config",
        }

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        workflows = scheduler.list_workflows()

        return {
            "enabled": True,
            "count": len(workflows),
            "workflows": [
                {
                    "name": wf.name,
                    "description": wf.description,
                    "enabled": wf.enabled,
                    "trigger_type": wf.trigger_type,
                    "trigger_schedule": wf.trigger_schedule,
                    "next_run": wf.next_run.isoformat() if wf.next_run else None,
                    "last_run": wf.last_run.isoformat() if wf.last_run else None,
                    "last_status": wf.last_status,
                    "tags": wf.tags,
                }
                for wf in workflows
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{name}")
async def get_workflow(name: str):
    """Get workflow details."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        workflow = scheduler.get_workflow(name)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {name}")

        info = next((w for w in scheduler.list_workflows() if w.name == name), None)

        return {
            "name": workflow.name,
            "description": workflow.description,
            "enabled": workflow.enabled,
            "trigger": {
                "type": workflow.trigger.type.value,
                "cron": workflow.trigger.cron,
                "interval_seconds": workflow.trigger.interval_seconds,
                "timezone": workflow.trigger.timezone,
            },
            "steps": [
                {
                    "name": step.name,
                    "action": step.action,
                    "params": step.params,
                    "condition": step.condition,
                    "on_failure": step.on_failure.value,
                    "timeout": step.timeout,
                    "retries": step.retries,
                }
                for step in workflow.steps
            ],
            "on_success": workflow.on_success.model_dump() if workflow.on_success else None,
            "on_failure": workflow.on_failure.model_dump() if workflow.on_failure else None,
            "tags": workflow.tags,
            "last_run": info.last_run.isoformat() if info and info.last_run else None,
            "last_status": info.last_status if info else None,
            "next_run": info.next_run.isoformat() if info and info.next_run else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/{name}/run")
async def run_workflow(name: str):
    """Run a workflow immediately."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        workflow = scheduler.get_workflow(name)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {name}")

        result = scheduler.trigger_now(name)

        return {
            "workflow_name": result.workflow_name,
            "run_id": result.run_id,
            "status": result.status.value,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration": result.duration,
            "steps": [
                {
                    "step_name": step.step_name,
                    "status": step.status.value,
                    "duration": step.duration,
                    "output": step.output,
                    "error": step.error,
                }
                for step in result.steps
            ],
            "error": result.error,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/{name}/enable")
async def enable_workflow(name: str):
    """Enable a workflow."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        if scheduler.enable_workflow(name):
            return {"status": "enabled", "name": name}
        else:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/{name}/disable")
async def disable_workflow(name: str):
    """Disable a workflow."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        if scheduler.disable_workflow(name):
            return {"status": "disabled", "name": name}
        else:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{name}/history")
async def get_workflow_history(name: str, limit: int = 10):
    """Get workflow execution history."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.load_workflows()

        workflow = scheduler.get_workflow(name)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {name}")

        history = scheduler.get_history(name, limit=limit)

        return {
            "workflow_name": name,
            "count": len(history),
            "history": [
                {
                    "run_id": result.run_id,
                    "status": result.status.value,
                    "started_at": result.started_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "duration": result.duration,
                    "steps_succeeded": sum(1 for s in result.steps if s.status.value == "success"),
                    "steps_total": len(result.steps),
                    "error": result.error,
                }
                for result in history
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/reload")
async def reload_workflows():
    """Reload workflows from disk."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        count = scheduler.reload_workflows()

        return {
            "status": "reloaded",
            "count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/scheduler/status")
async def get_workflow_scheduler_status():
    """Get workflow scheduler status."""
    config = get_config()

    if not config.workflows.enabled:
        return {
            "enabled": False,
            "message": "Workflows are disabled in config",
        }

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()

        return {
            "enabled": True,
            "running": scheduler.is_running,
            "workflows_loaded": len(scheduler._workflows),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/scheduler/start")
async def start_workflow_scheduler_api():
    """Start the workflow scheduler."""
    config = get_config()

    if not config.workflows.enabled:
        raise HTTPException(status_code=400, detail="Workflows are disabled")

    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        success = scheduler.start()

        if success:
            return {
                "status": "started",
                "workflows_scheduled": len([w for w in scheduler._workflows.values() if w.enabled]),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start scheduler")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/scheduler/stop")
async def stop_workflow_scheduler_api():
    """Stop the workflow scheduler."""
    try:
        from javis.workflows import get_workflow_scheduler

        scheduler = get_workflow_scheduler()
        scheduler.stop()

        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static files (if exists)
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
