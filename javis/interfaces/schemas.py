"""API request and response schemas for JAVIS.

This module defines Pydantic models for API requests and responses,
providing consistent structure and automatic OpenAPI documentation.
"""

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field


# Generic type for response data
T = TypeVar("T")


# ==================== Base Response Models ====================

class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str = Field(..., description="Machine-readable error code", examples=["AUTH_001"])
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "code": "AUTH_001",
                        "message": "Invalid or expired token",
                        "details": {}
                    }
                }
            ]
        }
    }


class SuccessResponse(BaseModel):
    """Generic success response."""
    status: str = Field(default="success", description="Operation status")
    message: Optional[str] = Field(default=None, description="Optional status message")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated list response."""
    items: list[T]
    total: int = Field(..., description="Total number of items")
    offset: int = Field(default=0, description="Number of items skipped")
    limit: int = Field(..., description="Maximum items per page")


# ==================== Health & Status ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status", examples=["healthy"])


class LivenessResponse(BaseModel):
    """Liveness probe response."""
    alive: bool = Field(default=True, description="Whether service is alive")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ComponentHealthResponse(BaseModel):
    """Detailed component health status."""
    status: str = Field(..., description="Status: healthy, degraded, unhealthy, unknown")
    message: Optional[str] = Field(default=None, description="Status message")
    latency_ms: Optional[float] = Field(default=None, description="Health check latency")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    last_check: Optional[str] = Field(default=None, description="Last check timestamp")


class DetailedHealthResponse(BaseModel):
    """Comprehensive system health response."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Check timestamp (ISO 8601)")
    components: dict[str, ComponentHealthResponse] = Field(
        ..., description="Individual component health statuses"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "uptime_seconds": 3600.5,
                    "timestamp": "2024-01-15T10:30:00Z",
                    "components": {
                        "groq": {
                            "status": "healthy",
                            "message": "API key configured",
                            "latency_ms": 1.5,
                            "details": {"model": "llama-3.1-8b-instant"}
                        }
                    }
                }
            ]
        }
    }


class ComponentStatus(BaseModel):
    """Component status details (legacy)."""
    available: bool = Field(..., description="Whether component is available")
    model: Optional[str] = Field(default=None, description="Model name if applicable")
    loaded: Optional[bool] = Field(default=None, description="Whether model is loaded")
    adapter: Optional[str] = Field(default=None, description="Adapter path if applicable")
    enabled: Optional[bool] = Field(default=None, description="Whether feature is enabled")
    count: Optional[int] = Field(default=None, description="Count of items if applicable")
    running: Optional[bool] = Field(default=None, description="Whether service is running")


class SystemStatusResponse(BaseModel):
    """Detailed system status response (legacy)."""
    timestamp: datetime = Field(..., description="Status check timestamp")
    version: str = Field(..., description="API version")
    python_version: str = Field(..., description="Python version")
    components: dict[str, ComponentStatus] = Field(..., description="Component statuses")
    sessions: dict[str, int] = Field(..., description="Session information")


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Whether service is ready")
    backends: dict[str, bool] = Field(..., description="Backend availability")


# ==================== Chat Models ====================

class ChatRequest(BaseModel):
    """Chat request from client."""
    message: str = Field(
        ...,
        description="User message to send",
        min_length=1,
        max_length=10000,
        examples=["Hello, how are you?"]
    )
    session_id: str = Field(
        default="default",
        description="Session identifier for conversation tracking",
        examples=["user-123-session-1"]
    )
    model: str = Field(
        default="groq",
        description="Model to use for response",
        examples=["groq", "modal", "local"]
    )
    adapter_version: Optional[str] = Field(
        default=None,
        description="Specific adapter version for local/modal models",
        examples=["v1.0.0"]
    )
    use_tools: bool = Field(
        default=False,
        description="Enable tool calling (Groq only)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What's the weather like today?",
                    "session_id": "user-123",
                    "model": "groq",
                    "use_tools": True
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Chat response to client."""
    response: str = Field(..., description="AI response message")
    session_id: str = Field(..., description="Session identifier")
    model_used: str = Field(default="groq", description="Model that generated response")
    tools_used: bool = Field(default=False, description="Whether tools were used")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "I'm doing well, thank you for asking!",
                    "session_id": "user-123",
                    "model_used": "groq",
                    "tools_used": False
                }
            ]
        }
    }


# ==================== Session Models ====================

class SessionInfo(BaseModel):
    """Session information."""
    id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Number of messages in session")


class SessionListResponse(BaseModel):
    """List of active sessions."""
    sessions: list[SessionInfo]


class SessionClearResponse(BaseModel):
    """Session clear response."""
    status: str = Field(default="cleared")
    session_id: str
    saved_to: Optional[str] = Field(default=None, description="File path where conversation was saved")
    memory_saved: bool = Field(default=False, description="Whether saved to long-term memory")


class FeedbackRequest(BaseModel):
    """Feedback request."""
    feedback: str = Field(
        ...,
        description="Feedback type",
        pattern="^(good|bad)$",
        examples=["good", "bad"]
    )


class FeedbackResponse(BaseModel):
    """Feedback response."""
    status: str = Field(default="feedback_added")
    session_id: str
    feedback: str


# ==================== Model Management ====================

class ModelInfo(BaseModel):
    """Model information."""
    available: bool = Field(..., description="Whether model is available")
    model: Optional[str] = Field(default=None, description="Model identifier")
    description: str = Field(..., description="Model description")
    status: Optional[str] = Field(default=None, description="Model status")
    current_adapter: Optional[str] = Field(default=None, description="Current adapter path")


class ModelsResponse(BaseModel):
    """Available models response."""
    groq: ModelInfo
    modal: ModelInfo
    local: ModelInfo
    adapters: list[str] = Field(..., description="Available adapter versions")


class ModelLoadResponse(BaseModel):
    """Model load response."""
    status: str
    adapter: str
    base_model: str


# ==================== Memory Models ====================

class MemorySearchRequest(BaseModel):
    """Memory search request."""
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        examples=["conversation about project planning"]
    )
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results to return")
    min_score: float = Field(default=0.3, ge=0, le=1, description="Minimum similarity score")


class MemoryEntry(BaseModel):
    """Memory entry."""
    id: str
    session_id: str
    summary: str
    topics: list[str]
    score: float
    created_at: datetime


class MemorySearchResponse(BaseModel):
    """Memory search response."""
    query: str
    count: int
    results: list[MemoryEntry]


class MemoryStatsResponse(BaseModel):
    """Memory statistics response."""
    enabled: bool
    count: Optional[int] = None
    db_path: Optional[str] = None
    collection: Optional[str] = None
    message: Optional[str] = None


# ==================== RAG Models ====================

class RAGAddDocumentRequest(BaseModel):
    """Request to add a document to RAG."""
    source_type: str = Field(
        ...,
        description="Document source type",
        examples=["file", "web", "code"]
    )
    source_path: str = Field(
        ...,
        description="Path or URL to document",
        examples=["/path/to/doc.pdf", "https://example.com/page"]
    )


class RAGSearchRequest(BaseModel):
    """RAG search request."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    min_score: float = Field(default=0.3, ge=0, le=1, description="Minimum similarity score")


class RAGSearchResult(BaseModel):
    """RAG search result."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    document_title: Optional[str]
    source_path: Optional[str]


class RAGSearchResponse(BaseModel):
    """RAG search response."""
    query: str
    count: int
    results: list[RAGSearchResult]


class RAGStatsResponse(BaseModel):
    """RAG statistics response."""
    enabled: bool
    total_documents: Optional[int] = None
    total_chunks: Optional[int] = None
    by_source: Optional[dict[str, int]] = None
    message: Optional[str] = None


# ==================== Training Models ====================

class TrainingSchedulerStatus(BaseModel):
    """Training scheduler status."""
    available: bool
    running: Optional[bool] = None
    enabled: Optional[bool] = None
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_result: Optional[str] = None
    cron: Optional[str] = None
    timezone: Optional[str] = None
    message: Optional[str] = None


class TrainingRunResult(BaseModel):
    """Training run result."""
    success: bool
    skipped: bool = False
    skip_reason: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    dataset_size: Optional[int] = None
    duration_seconds: Optional[float] = None


class DPOConfig(BaseModel):
    """DPO training configuration."""
    enabled: bool
    beta: float
    loss_type: str
    min_preference_pairs: int
    use_reference_model: bool
    reference_model: Optional[str]
    max_prompt_length: int
    max_response_length: int


class DPOStatsResponse(BaseModel):
    """DPO preference data statistics."""
    total_pairs: int
    from_corrections: int
    from_feedback: int
    from_quality_score: int
    ready_for_training: bool


# ==================== A/B Testing Models ====================

class ABTestCreateRequest(BaseModel):
    """Request to create an A/B test."""
    version_a: str = Field(..., description="First model version")
    version_b: str = Field(..., description="Second model version")
    traffic_split: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Traffic proportion for version A"
    )
    description: Optional[str] = Field(default=None, description="Test description")


class ABTestInfo(BaseModel):
    """A/B test information."""
    test_id: str
    version_a: str
    version_b: str
    traffic_split: float
    status: str
    start_time: datetime
    end_time: Optional[datetime]


class ABTestListResponse(BaseModel):
    """List of A/B tests."""
    count: int
    tests: list[ABTestInfo]


# ==================== Correction Models ====================

class CorrectionRequest(BaseModel):
    """Request to add a correction."""
    session_id: str = Field(..., description="Session ID")
    original_prompt: str = Field(..., description="Original user prompt")
    original_response: str = Field(..., description="Original model response")
    corrected_response: str = Field(..., description="Corrected response")
    correction_type: str = Field(
        default="other",
        description="Type of correction",
        examples=["factual", "style", "safety", "other"]
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class CorrectionInfo(BaseModel):
    """Correction information."""
    id: str
    session_id: str
    correction_type: str
    timestamp: datetime
    original_prompt: str


class CorrectionListResponse(BaseModel):
    """List of corrections."""
    count: int
    corrections: list[CorrectionInfo]


# ==================== Voice Models ====================

class TTSRequest(BaseModel):
    """TTS request."""
    text: str = Field(
        ...,
        description="Text to synthesize",
        min_length=1,
        max_length=5000
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice ID to use",
        examples=["ko-KR-SunHiNeural", "en-US-JennyNeural"]
    )


class STTResponse(BaseModel):
    """STT response."""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")


class VoiceChatResponse(BaseModel):
    """Voice chat response."""
    transcription: str = Field(..., description="Transcribed user speech")
    response: str = Field(..., description="AI response")
    session_id: str


class VoiceInfo(BaseModel):
    """TTS voice information."""
    name: str
    short_name: str
    gender: str
    locale: str


class VoiceListResponse(BaseModel):
    """List of available voices."""
    count: int
    voices: list[VoiceInfo]


# ==================== Workflow Models ====================

class WorkflowInfo(BaseModel):
    """Workflow information."""
    name: str
    description: Optional[str]
    enabled: bool
    trigger_type: str
    trigger_schedule: Optional[str]
    next_run: Optional[datetime]
    last_run: Optional[datetime]
    last_status: Optional[str]
    tags: list[str]


class WorkflowListResponse(BaseModel):
    """List of workflows."""
    enabled: bool
    count: int
    workflows: list[WorkflowInfo]
    message: Optional[str] = None


class WorkflowStepResult(BaseModel):
    """Workflow step execution result."""
    step_name: str
    status: str
    duration: Optional[float]
    output: Optional[Any]
    error: Optional[str]


class WorkflowRunResult(BaseModel):
    """Workflow run result."""
    workflow_name: str
    run_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    steps: list[WorkflowStepResult]
    error: Optional[str]


# ==================== Tool Models ====================

class ToolsInfoResponse(BaseModel):
    """Tools information response."""
    enabled: bool
    registered_tools: Optional[list[str]] = None
    enabled_tools: Optional[list[str]] = None
    enabled_categories: Optional[list[str]] = None
    message: Optional[str] = None


# ==================== Analytics Models ====================

class DailyMetric(BaseModel):
    """Daily metric data point."""
    date: str
    count: int


class LatencyMetric(BaseModel):
    """Latency by version."""
    version: str
    avg_latency: float


class ToolUsageMetric(BaseModel):
    """Tool usage metric."""
    tool: str
    count: int


class FeedbackTrend(BaseModel):
    """Daily feedback trend."""
    date: str
    positive: int
    negative: int


class AnalyticsDashboardResponse(BaseModel):
    """Analytics dashboard response."""
    period: dict[str, Any]
    metrics: dict[str, Any]
    daily_conversations: list[DailyMetric]
    latency_by_version: list[LatencyMetric]
    tool_usage: list[ToolUsageMetric]
    feedback_trends: list[FeedbackTrend]


# ==================== Integration Models ====================

class IntegrationInfo(BaseModel):
    """Integration information."""
    id: str
    name: str
    icon: str
    description: str
    enabled: bool
    configured: bool
    config_hint: Optional[str]
    status: str = Field(..., description="Status: connected, disabled, not_configured")


class IntegrationsStatusResponse(BaseModel):
    """Integrations status response."""
    integrations: list[IntegrationInfo]


class IntegrationTestResponse(BaseModel):
    """Integration test result."""
    status: str = Field(..., description="Test result: success, warning, error")
    message: str


# ==================== File Upload ====================

class UploadResponse(BaseModel):
    """File upload response."""
    files: list[str] = Field(..., description="List of uploaded file names")
    count: int = Field(..., description="Number of files uploaded")
