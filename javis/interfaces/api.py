"""FastAPI server for JAVIS web interface."""

import os
import aiofiles
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional

from javis.utils.config import load_config, get_config
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

# 업로드 디렉토리
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


class ChatResponse(BaseModel):
    """Chat response to client."""
    response: str
    session_id: str
    model_used: str = "groq"  # "groq" or "local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    load_config()

    # Auto-start scheduler if enabled in config
    config = get_config()
    if config.training.schedule.enabled and SCHEDULER_AVAILABLE:
        import logging
        logger = logging.getLogger(__name__)
        if start_scheduler():
            scheduler = get_scheduler()
            logger.info(f"Training scheduler started. Next run: {scheduler.get_next_run_time()}")

    yield

    # Shutdown
    if SCHEDULER_AVAILABLE:
        stop_scheduler()
    sessions.clear()


app = FastAPI(
    title="JAVIS API",
    description="Personal AI Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정 (모든 도메인 허용 - 프로덕션에서는 제한 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with JAVIS.

    Models:
    - groq: Fast inference via Groq API (Llama 3.1)
    - modal: Fine-tuned model via Modal.com GPU (recommended)
    - local: Fine-tuned model on local GPU (requires CUDA)
    """
    config = get_config()
    logger = get_logger()

    try:
        # 세션 가져오기
        messages = get_or_create_session(request.session_id)

        # 사용자 메시지 추가
        messages.append(Message(role="user", content=request.message))

        # 대화 로깅 (파인튜닝 데이터 수집)
        logger.add_turn(request.session_id, "user", request.message)

        # 모델 선택 및 호출
        if request.model == "modal":
            # Modal.com GPU를 통한 파인튜닝 모델 추론
            modal_client = get_modal_client(request.adapter_version)

            # Message 타입 변환
            modal_messages = [
                ModalMessage(role=m.role, content=m.content) for m in messages
            ]

            response = await modal_client.chat_async(modal_messages)
            model_used = "modal"

        elif request.model == "local":
            # 로컬 GPU를 통한 파인튜닝 모델 추론
            local_client = get_local_client(request.adapter_version)

            # Message 타입 변환
            local_messages = [
                LocalMessage(role=m.role, content=m.content) for m in messages
            ]

            response = await local_client.chat_async(local_messages)
            model_used = "local"

        else:  # groq (default)
            # Groq API 사용
            if not config.groq_api_key:
                raise HTTPException(
                    status_code=500,
                    detail="Groq API key not configured"
                )
            client = ModelClient()
            response = await client.chat(messages)
            model_used = "groq"

        # 어시스턴트 응답 추가
        messages.append(Message(role="assistant", content=response.content))

        # 어시스턴트 응답 로깅
        logger.add_turn(request.session_id, "assistant", response.content)

        # 히스토리 제한
        max_history = config.conversation.max_history
        if len(messages) > max_history + 1:
            sessions[request.session_id] = [messages[0]] + messages[-(max_history):]

        return ChatResponse(
            response=response.content,
            session_id=request.session_id,
            model_used=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
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


@app.post("/api/clear")
async def clear_session(session_id: str = "default"):
    """Clear a chat session and save conversation log."""
    logger = get_logger()

    # 대화 저장 후 종료
    filepath = logger.end_conversation(session_id)

    if session_id in sessions:
        config = get_config()
        sessions[session_id] = [
            Message(role="system", content=config.conversation.system_prompt)
        ]

    return {
        "status": "cleared",
        "session_id": session_id,
        "saved_to": str(filepath) if filepath else None
    }


@app.post("/api/feedback")
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


@app.get("/api/sessions")
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


@app.get("/api/models")
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


@app.post("/api/models/local/load")
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


@app.post("/api/models/local/unload")
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


# Static files (if exists)
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
