"""Tests for API request and response schemas."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from javis.interfaces.schemas import (
    # Base models
    ErrorDetail,
    ErrorResponse,
    SuccessResponse,
    # Health models
    HealthResponse,
    ReadinessResponse,
    # Chat models
    ChatRequest,
    ChatResponse,
    # Session models
    SessionInfo,
    SessionListResponse,
    SessionClearResponse,
    FeedbackRequest,
    FeedbackResponse,
    # Memory models
    MemorySearchRequest,
    MemoryEntry,
    MemorySearchResponse,
    MemoryStatsResponse,
    # RAG models
    RAGAddDocumentRequest,
    RAGSearchRequest,
    RAGSearchResult,
    RAGSearchResponse,
    RAGStatsResponse,
    # Training models
    TrainingRunResult,
    DPOConfig,
    DPOStatsResponse,
    ABTestCreateRequest,
    ABTestInfo,
    # Voice models
    TTSRequest,
    STTResponse,
    VoiceChatResponse,
    # Workflow models
    WorkflowInfo,
    WorkflowStepResult,
    WorkflowRunResult,
    # Tool models
    ToolsInfoResponse,
    # Analytics models
    DailyMetric,
    AnalyticsDashboardResponse,
    # Integration models
    IntegrationInfo,
    IntegrationsStatusResponse,
    IntegrationTestResponse,
    # Upload models
    UploadResponse,
)


class TestBaseModels:
    """Tests for base response models."""

    def test_error_detail(self):
        """Test ErrorDetail model."""
        detail = ErrorDetail(
            code="TEST_001",
            message="Test error message",
            details={"field": "test"}
        )
        assert detail.code == "TEST_001"
        assert detail.message == "Test error message"
        assert detail.details == {"field": "test"}

    def test_error_detail_defaults(self):
        """Test ErrorDetail with default values."""
        detail = ErrorDetail(code="ERR", message="Error")
        assert detail.details == {}

    def test_error_response(self):
        """Test ErrorResponse model."""
        response = ErrorResponse(
            error=ErrorDetail(
                code="AUTH_001",
                message="Unauthorized"
            )
        )
        assert response.error.code == "AUTH_001"

    def test_success_response(self):
        """Test SuccessResponse model."""
        response = SuccessResponse()
        assert response.status == "success"
        assert response.message is None

    def test_success_response_with_message(self):
        """Test SuccessResponse with custom message."""
        response = SuccessResponse(status="ok", message="Operation completed")
        assert response.status == "ok"
        assert response.message == "Operation completed"


class TestHealthModels:
    """Tests for health check models."""

    def test_health_response(self):
        """Test HealthResponse model."""
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"

    def test_readiness_response(self):
        """Test ReadinessResponse model."""
        response = ReadinessResponse(
            ready=True,
            backends={"groq": True, "modal": False, "local": False}
        )
        assert response.ready is True
        assert response.backends["groq"] is True


class TestChatModels:
    """Tests for chat-related models."""

    def test_chat_request_minimal(self):
        """Test ChatRequest with minimal fields."""
        request = ChatRequest(message="Hello")
        assert request.message == "Hello"
        assert request.session_id == "default"
        assert request.model == "groq"
        assert request.use_tools is False

    def test_chat_request_full(self):
        """Test ChatRequest with all fields."""
        request = ChatRequest(
            message="What's the weather?",
            session_id="user-123",
            model="modal",
            adapter_version="v1.0.0",
            use_tools=True
        )
        assert request.message == "What's the weather?"
        assert request.session_id == "user-123"
        assert request.model == "modal"
        assert request.adapter_version == "v1.0.0"
        assert request.use_tools is True

    def test_chat_request_validation(self):
        """Test ChatRequest validation."""
        with pytest.raises(ValidationError):
            ChatRequest(message="")  # Empty message should fail

    def test_chat_response(self):
        """Test ChatResponse model."""
        response = ChatResponse(
            response="Hello! How can I help?",
            session_id="user-123",
            model_used="groq",
            tools_used=False
        )
        assert response.response == "Hello! How can I help?"
        assert response.session_id == "user-123"


class TestSessionModels:
    """Tests for session-related models."""

    def test_session_info(self):
        """Test SessionInfo model."""
        info = SessionInfo(id="sess-1", message_count=5)
        assert info.id == "sess-1"
        assert info.message_count == 5

    def test_session_list_response(self):
        """Test SessionListResponse model."""
        response = SessionListResponse(
            sessions=[
                SessionInfo(id="sess-1", message_count=5),
                SessionInfo(id="sess-2", message_count=3)
            ]
        )
        assert len(response.sessions) == 2

    def test_session_clear_response(self):
        """Test SessionClearResponse model."""
        response = SessionClearResponse(
            session_id="sess-1",
            saved_to="/path/to/file",
            memory_saved=True
        )
        assert response.status == "cleared"
        assert response.memory_saved is True

    def test_feedback_request_valid(self):
        """Test FeedbackRequest with valid feedback."""
        request = FeedbackRequest(feedback="good")
        assert request.feedback == "good"

    def test_feedback_request_invalid(self):
        """Test FeedbackRequest validation."""
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback="invalid")

    def test_feedback_response(self):
        """Test FeedbackResponse model."""
        response = FeedbackResponse(
            session_id="sess-1",
            feedback="good"
        )
        assert response.status == "feedback_added"


class TestMemoryModels:
    """Tests for memory-related models."""

    def test_memory_search_request(self):
        """Test MemorySearchRequest model."""
        request = MemorySearchRequest(query="project planning")
        assert request.query == "project planning"
        assert request.limit == 5
        assert request.min_score == 0.3

    def test_memory_search_request_validation(self):
        """Test MemorySearchRequest validation."""
        # Valid custom values
        request = MemorySearchRequest(query="test", limit=10, min_score=0.5)
        assert request.limit == 10
        assert request.min_score == 0.5

        # Invalid limit
        with pytest.raises(ValidationError):
            MemorySearchRequest(query="test", limit=100)  # Max is 50

        # Invalid min_score
        with pytest.raises(ValidationError):
            MemorySearchRequest(query="test", min_score=1.5)  # Max is 1

    def test_memory_entry(self):
        """Test MemoryEntry model."""
        entry = MemoryEntry(
            id="mem-1",
            session_id="sess-1",
            summary="Discussion about AI",
            topics=["AI", "technology"],
            score=0.85,
            created_at=datetime.now(timezone.utc)
        )
        assert entry.id == "mem-1"
        assert "AI" in entry.topics

    def test_memory_stats_response(self):
        """Test MemoryStatsResponse model."""
        response = MemoryStatsResponse(enabled=True, count=100)
        assert response.enabled is True
        assert response.count == 100


class TestRAGModels:
    """Tests for RAG-related models."""

    def test_rag_add_document_request(self):
        """Test RAGAddDocumentRequest model."""
        request = RAGAddDocumentRequest(
            source_type="file",
            source_path="/path/to/doc.pdf"
        )
        assert request.source_type == "file"
        assert request.source_path == "/path/to/doc.pdf"

    def test_rag_search_request(self):
        """Test RAGSearchRequest model."""
        request = RAGSearchRequest(query="machine learning")
        assert request.query == "machine learning"
        assert request.top_k == 5

    def test_rag_search_result(self):
        """Test RAGSearchResult model."""
        result = RAGSearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="ML is a subset of AI...",
            score=0.9,
            document_title="AI Basics",
            source_path="/docs/ai.pdf"
        )
        assert result.chunk_id == "chunk-1"
        assert result.score == 0.9

    def test_rag_stats_response(self):
        """Test RAGStatsResponse model."""
        response = RAGStatsResponse(
            enabled=True,
            total_documents=50,
            total_chunks=500,
            by_source={"file": 30, "web": 20}
        )
        assert response.total_documents == 50


class TestTrainingModels:
    """Tests for training-related models."""

    def test_training_run_result(self):
        """Test TrainingRunResult model."""
        result = TrainingRunResult(
            success=True,
            skipped=False,
            version="v1.0.0",
            dataset_size=1000,
            duration_seconds=3600.5
        )
        assert result.success is True
        assert result.version == "v1.0.0"

    def test_training_run_result_skipped(self):
        """Test skipped training run."""
        result = TrainingRunResult(
            success=False,
            skipped=True,
            skip_reason="Insufficient data"
        )
        assert result.skipped is True
        assert result.skip_reason == "Insufficient data"

    def test_dpo_config(self):
        """Test DPOConfig model."""
        config = DPOConfig(
            enabled=True,
            beta=0.1,
            loss_type="sigmoid",
            min_preference_pairs=100,
            use_reference_model=True,
            reference_model="base-model",
            max_prompt_length=512,
            max_response_length=1024
        )
        assert config.enabled is True
        assert config.beta == 0.1

    def test_dpo_stats_response(self):
        """Test DPOStatsResponse model."""
        stats = DPOStatsResponse(
            total_pairs=500,
            from_corrections=100,
            from_feedback=200,
            from_quality_score=200,
            ready_for_training=True
        )
        assert stats.total_pairs == 500
        assert stats.ready_for_training is True

    def test_ab_test_create_request(self):
        """Test ABTestCreateRequest model."""
        request = ABTestCreateRequest(
            version_a="v1.0.0",
            version_b="v1.1.0",
            traffic_split=0.5,
            description="Testing new model"
        )
        assert request.version_a == "v1.0.0"
        assert request.traffic_split == 0.5


class TestVoiceModels:
    """Tests for voice-related models."""

    def test_tts_request(self):
        """Test TTSRequest model."""
        request = TTSRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.voice is None

    def test_tts_request_with_voice(self):
        """Test TTSRequest with specific voice."""
        request = TTSRequest(
            text="안녕하세요",
            voice="ko-KR-SunHiNeural"
        )
        assert request.voice == "ko-KR-SunHiNeural"

    def test_tts_request_validation(self):
        """Test TTSRequest validation."""
        with pytest.raises(ValidationError):
            TTSRequest(text="")  # Empty text

    def test_stt_response(self):
        """Test STTResponse model."""
        response = STTResponse(
            text="Hello world",
            language="en",
            duration=2.5
        )
        assert response.text == "Hello world"
        assert response.duration == 2.5

    def test_voice_chat_response(self):
        """Test VoiceChatResponse model."""
        response = VoiceChatResponse(
            transcription="What time is it?",
            response="It's 3 PM.",
            session_id="sess-1"
        )
        assert response.transcription == "What time is it?"


class TestWorkflowModels:
    """Tests for workflow-related models."""

    def test_workflow_info(self):
        """Test WorkflowInfo model."""
        info = WorkflowInfo(
            name="daily-report",
            description="Generate daily report",
            enabled=True,
            trigger_type="cron",
            trigger_schedule="0 9 * * *",
            next_run=datetime.now(timezone.utc),
            last_run=None,
            last_status=None,
            tags=["report", "daily"]
        )
        assert info.name == "daily-report"
        assert info.enabled is True

    def test_workflow_step_result(self):
        """Test WorkflowStepResult model."""
        result = WorkflowStepResult(
            step_name="fetch_data",
            status="success",
            duration=1.5,
            output={"rows": 100},
            error=None
        )
        assert result.step_name == "fetch_data"
        assert result.status == "success"

    def test_workflow_run_result(self):
        """Test WorkflowRunResult model."""
        result = WorkflowRunResult(
            workflow_name="daily-report",
            run_id="run-123",
            status="completed",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration=10.5,
            steps=[],
            error=None
        )
        assert result.workflow_name == "daily-report"
        assert result.status == "completed"


class TestToolModels:
    """Tests for tool-related models."""

    def test_tools_info_response_enabled(self):
        """Test ToolsInfoResponse when enabled."""
        response = ToolsInfoResponse(
            enabled=True,
            registered_tools=["file_read", "web_search"],
            enabled_tools=["file_read"],
            enabled_categories=["file", "web"]
        )
        assert response.enabled is True
        assert "file_read" in response.registered_tools

    def test_tools_info_response_disabled(self):
        """Test ToolsInfoResponse when disabled."""
        response = ToolsInfoResponse(
            enabled=False,
            message="Tools are disabled"
        )
        assert response.enabled is False
        assert response.message == "Tools are disabled"


class TestAnalyticsModels:
    """Tests for analytics-related models."""

    def test_daily_metric(self):
        """Test DailyMetric model."""
        metric = DailyMetric(date="2024-01-15", count=100)
        assert metric.date == "2024-01-15"
        assert metric.count == 100

    def test_analytics_dashboard_response(self):
        """Test AnalyticsDashboardResponse model."""
        response = AnalyticsDashboardResponse(
            period={"days": 7, "start": "2024-01-08", "end": "2024-01-15"},
            metrics={
                "total_conversations": 500,
                "avg_latency_ms": 250
            },
            daily_conversations=[DailyMetric(date="2024-01-15", count=50)],
            latency_by_version=[],
            tool_usage=[],
            feedback_trends=[]
        )
        assert response.period["days"] == 7


class TestIntegrationModels:
    """Tests for integration-related models."""

    def test_integration_info(self):
        """Test IntegrationInfo model."""
        info = IntegrationInfo(
            id="slack",
            name="Slack",
            icon="💬",
            description="Send messages to Slack",
            enabled=True,
            configured=True,
            config_hint=None,
            status="connected"
        )
        assert info.id == "slack"
        assert info.status == "connected"

    def test_integrations_status_response(self):
        """Test IntegrationsStatusResponse model."""
        response = IntegrationsStatusResponse(
            integrations=[
                IntegrationInfo(
                    id="slack",
                    name="Slack",
                    icon="💬",
                    description="Slack integration",
                    enabled=True,
                    configured=True,
                    config_hint=None,
                    status="connected"
                )
            ]
        )
        assert len(response.integrations) == 1

    def test_integration_test_response(self):
        """Test IntegrationTestResponse model."""
        response = IntegrationTestResponse(
            status="success",
            message="Connection successful"
        )
        assert response.status == "success"


class TestUploadModels:
    """Tests for upload-related models."""

    def test_upload_response(self):
        """Test UploadResponse model."""
        response = UploadResponse(
            files=["doc1.pdf", "doc2.txt"],
            count=2
        )
        assert len(response.files) == 2
        assert response.count == 2
