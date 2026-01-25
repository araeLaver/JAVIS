"""Tests for JAVIS custom exceptions."""

import pytest

from javis.utils.exceptions import (
    # Base
    JAVISException,
    # Authentication
    AuthenticationError,
    InvalidTokenError,
    MissingTokenError,
    InsufficientPermissionsError,
    # Validation
    ValidationError,
    InvalidInputError,
    MissingFieldError,
    # Resource
    ResourceNotFoundError,
    ResourceConflictError,
    # Model
    ModelError,
    ModelNotAvailableError,
    ModelLoadError,
    ModelInferenceError,
    # Training
    TrainingError,
    InsufficientDataError,
    TrainingInProgressError,
    TrainingConfigError,
    # Storage
    StorageError,
    StorageConnectionError,
    StorageOperationError,
    # Memory
    MemoryError,
    MemoryNotEnabledError,
    # RAG
    RAGError,
    RAGNotEnabledError,
    DocumentProcessingError,
    # Tool
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    # Integration
    IntegrationError,
    IntegrationNotConfiguredError,
    IntegrationConnectionError,
    # Workflow
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowExecutionError,
    # Voice
    VoiceError,
    VoiceNotEnabledError,
    TranscriptionError,
    SynthesisError,
    # Rate Limiting
    RateLimitError,
    # Service
    ServiceUnavailableError,
    ConfigurationError,
)


class TestValidationExceptions:
    """Tests for validation exception classes."""

    def test_invalid_input_error(self):
        """Test InvalidInputError with field name."""
        exc = InvalidInputError("email", "Invalid format")
        assert exc.status_code == 400
        assert exc.error_code == "VALIDATION_001"
        assert "email" in exc.message
        assert exc.details["field"] == "email"

    def test_missing_field_error(self):
        """Test MissingFieldError."""
        exc = MissingFieldError("username")
        assert exc.status_code == 400
        assert exc.error_code == "VALIDATION_002"
        assert "username" in exc.message
        assert exc.details["field"] == "username"


class TestResourceExceptions:
    """Tests for resource exception classes."""

    def test_resource_not_found_without_id(self):
        """Test ResourceNotFoundError without resource ID."""
        exc = ResourceNotFoundError("Session")
        assert exc.status_code == 404
        assert exc.message == "Session not found"
        assert exc.details["resource_type"] == "Session"

    def test_resource_not_found_with_custom_message(self):
        """Test ResourceNotFoundError with custom message."""
        exc = ResourceNotFoundError("User", message="Custom not found message")
        assert "Custom not found message" in exc.message

    def test_resource_conflict_error(self):
        """Test ResourceConflictError."""
        exc = ResourceConflictError("User already exists")
        assert exc.status_code == 409
        assert exc.error_code == "CONFLICT"
        assert exc.message == "User already exists"


class TestModelExceptions:
    """Tests for model exception classes."""

    def test_model_error_base(self):
        """Test base ModelError."""
        exc = ModelError("Model failed")
        assert exc.status_code == 500
        assert exc.error_code == "MODEL_ERROR"

    def test_model_not_available_with_custom_message(self):
        """Test ModelNotAvailableError with custom message."""
        exc = ModelNotAvailableError("custom-model", "Under maintenance")
        assert exc.message == "Under maintenance"
        assert exc.details["model"] == "custom-model"

    def test_model_load_error(self):
        """Test ModelLoadError."""
        exc = ModelLoadError("llama-7b")
        assert exc.error_code == "MODEL_002"
        assert "llama-7b" in exc.message

    def test_model_load_error_with_reason(self):
        """Test ModelLoadError with reason."""
        exc = ModelLoadError("llama-7b", "Out of memory")
        assert "Out of memory" in exc.message
        assert exc.details["model"] == "llama-7b"

    def test_model_inference_error(self):
        """Test ModelInferenceError."""
        exc = ModelInferenceError()
        assert exc.error_code == "MODEL_003"
        assert exc.message == "Model inference failed"

    def test_model_inference_error_custom(self):
        """Test ModelInferenceError with custom message."""
        exc = ModelInferenceError("Timeout during inference")
        assert exc.message == "Timeout during inference"


class TestTrainingExceptions:
    """Tests for training exception classes."""

    def test_training_error_base(self):
        """Test base TrainingError."""
        exc = TrainingError("Training failed")
        assert exc.status_code == 500
        assert exc.error_code == "TRAINING_ERROR"

    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        exc = InsufficientDataError(required=100, available=50)
        assert exc.error_code == "TRAINING_001"
        assert "50/100" in exc.message
        assert exc.details["required"] == 100
        assert exc.details["available"] == 50

    def test_training_in_progress_error(self):
        """Test TrainingInProgressError."""
        exc = TrainingInProgressError()
        assert exc.status_code == 409
        assert exc.error_code == "TRAINING_002"

    def test_training_config_error(self):
        """Test TrainingConfigError."""
        exc = TrainingConfigError("Invalid learning rate")
        assert exc.status_code == 400
        assert exc.error_code == "TRAINING_003"


class TestStorageExceptions:
    """Tests for storage exception classes."""

    def test_storage_error_base(self):
        """Test base StorageError."""
        exc = StorageError("Storage failed")
        assert exc.status_code == 500
        assert exc.error_code == "STORAGE_ERROR"

    def test_storage_connection_error(self):
        """Test StorageConnectionError."""
        exc = StorageConnectionError("supabase")
        assert exc.error_code == "STORAGE_001"
        assert "supabase" in exc.message
        assert exc.details["backend"] == "supabase"

    def test_storage_connection_error_with_message(self):
        """Test StorageConnectionError with custom message."""
        exc = StorageConnectionError("redis", "Connection refused")
        assert "Connection refused" in exc.message

    def test_storage_operation_error(self):
        """Test StorageOperationError."""
        exc = StorageOperationError("save")
        assert exc.error_code == "STORAGE_002"
        assert "save" in exc.message
        assert exc.details["operation"] == "save"

    def test_storage_operation_error_with_message(self):
        """Test StorageOperationError with custom message."""
        exc = StorageOperationError("delete", "Record not found")
        assert "Record not found" in exc.message


class TestMemoryExceptions:
    """Tests for memory exception classes."""

    def test_memory_error_base(self):
        """Test base MemoryError."""
        exc = MemoryError("Memory error")
        assert exc.status_code == 500
        assert exc.error_code == "MEMORY_ERROR"

    def test_memory_not_enabled_error(self):
        """Test MemoryNotEnabledError."""
        exc = MemoryNotEnabledError()
        assert exc.status_code == 400
        assert exc.error_code == "MEMORY_001"
        assert "not enabled" in exc.message


class TestRAGExceptions:
    """Tests for RAG exception classes."""

    def test_rag_error_base(self):
        """Test base RAGError."""
        exc = RAGError("RAG error")
        assert exc.status_code == 500
        assert exc.error_code == "RAG_ERROR"

    def test_rag_not_enabled_error(self):
        """Test RAGNotEnabledError."""
        exc = RAGNotEnabledError()
        assert exc.status_code == 400
        assert exc.error_code == "RAG_001"

    def test_document_processing_error(self):
        """Test DocumentProcessingError."""
        exc = DocumentProcessingError("document.pdf")
        assert exc.error_code == "RAG_002"
        assert "document.pdf" in exc.message
        assert exc.details["source"] == "document.pdf"

    def test_document_processing_error_with_reason(self):
        """Test DocumentProcessingError with reason."""
        exc = DocumentProcessingError("doc.pdf", "Unsupported format")
        assert "Unsupported format" in exc.message


class TestToolExceptions:
    """Tests for tool exception classes."""

    def test_tool_error_base(self):
        """Test base ToolError."""
        exc = ToolError("Tool error")
        assert exc.status_code == 500
        assert exc.error_code == "TOOL_ERROR"

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        exc = ToolNotFoundError("calculator")
        assert exc.status_code == 404
        assert exc.error_code == "TOOL_001"
        assert "calculator" in exc.message
        assert exc.details["tool"] == "calculator"

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        exc = ToolExecutionError("web_search")
        assert exc.error_code == "TOOL_002"
        assert "web_search" in exc.message

    def test_tool_execution_error_with_reason(self):
        """Test ToolExecutionError with reason."""
        exc = ToolExecutionError("file_write", "Permission denied")
        assert "Permission denied" in exc.message
        assert exc.details["tool"] == "file_write"


class TestIntegrationExceptions:
    """Tests for integration exception classes."""

    def test_integration_error_base(self):
        """Test base IntegrationError."""
        exc = IntegrationError("Integration failed")
        assert exc.status_code == 502
        assert exc.error_code == "INTEGRATION_ERROR"

    def test_integration_not_configured_error(self):
        """Test IntegrationNotConfiguredError."""
        exc = IntegrationNotConfiguredError("slack")
        assert exc.status_code == 400
        assert exc.error_code == "INTEGRATION_001"
        assert "slack" in exc.message

    def test_integration_not_configured_with_hint(self):
        """Test IntegrationNotConfiguredError with config hint."""
        exc = IntegrationNotConfiguredError("notion", "Set NOTION_API_KEY")
        assert exc.details["config_hint"] == "Set NOTION_API_KEY"

    def test_integration_connection_error(self):
        """Test IntegrationConnectionError."""
        exc = IntegrationConnectionError("google_calendar")
        assert exc.error_code == "INTEGRATION_002"
        assert "google_calendar" in exc.message

    def test_integration_connection_error_with_reason(self):
        """Test IntegrationConnectionError with reason."""
        exc = IntegrationConnectionError("slack", "API key invalid")
        assert "API key invalid" in exc.message


class TestWorkflowExceptions:
    """Tests for workflow exception classes."""

    def test_workflow_error_base(self):
        """Test base WorkflowError."""
        exc = WorkflowError("Workflow error")
        assert exc.status_code == 500
        assert exc.error_code == "WORKFLOW_ERROR"

    def test_workflow_not_found_error(self):
        """Test WorkflowNotFoundError."""
        exc = WorkflowNotFoundError("daily_report")
        assert exc.status_code == 404
        assert exc.error_code == "WORKFLOW_001"
        assert "daily_report" in exc.message
        assert exc.details["workflow"] == "daily_report"

    def test_workflow_execution_error(self):
        """Test WorkflowExecutionError."""
        exc = WorkflowExecutionError("backup_workflow")
        assert exc.error_code == "WORKFLOW_002"
        assert "backup_workflow" in exc.message

    def test_workflow_execution_error_with_step(self):
        """Test WorkflowExecutionError with step."""
        exc = WorkflowExecutionError("data_pipeline", step="transform")
        assert "transform" in exc.message
        assert exc.details["step"] == "transform"

    def test_workflow_execution_error_full(self):
        """Test WorkflowExecutionError with step and reason."""
        exc = WorkflowExecutionError("etl", step="load", reason="Database timeout")
        assert "etl" in exc.message
        assert "load" in exc.message
        assert "Database timeout" in exc.message


class TestVoiceExceptions:
    """Tests for voice exception classes."""

    def test_voice_error_base(self):
        """Test base VoiceError."""
        exc = VoiceError("Voice error")
        assert exc.status_code == 500
        assert exc.error_code == "VOICE_ERROR"

    def test_voice_not_enabled_error(self):
        """Test VoiceNotEnabledError."""
        exc = VoiceNotEnabledError()
        assert exc.status_code == 400
        assert exc.error_code == "VOICE_001"

    def test_transcription_error(self):
        """Test TranscriptionError."""
        exc = TranscriptionError()
        assert exc.error_code == "VOICE_002"
        assert "Transcription failed" in exc.message

    def test_transcription_error_with_reason(self):
        """Test TranscriptionError with reason."""
        exc = TranscriptionError("Audio too short")
        assert "Audio too short" in exc.message

    def test_synthesis_error(self):
        """Test SynthesisError."""
        exc = SynthesisError()
        assert exc.error_code == "VOICE_003"
        assert "Speech synthesis failed" in exc.message

    def test_synthesis_error_with_reason(self):
        """Test SynthesisError with reason."""
        exc = SynthesisError("Voice model unavailable")
        assert "Voice model unavailable" in exc.message


class TestRateLimitException:
    """Tests for rate limit exception."""

    def test_rate_limit_error(self):
        """Test RateLimitError without retry_after."""
        exc = RateLimitError()
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT"
        assert "Rate limit exceeded" in exc.message

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after."""
        exc = RateLimitError(retry_after=60)
        assert "60 seconds" in exc.message
        assert exc.details["retry_after"] == 60


class TestServiceExceptions:
    """Tests for service exception classes."""

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        exc = ServiceUnavailableError("database")
        assert exc.status_code == 503
        assert exc.error_code == "SERVICE_UNAVAILABLE"
        assert "database" in exc.message

    def test_service_unavailable_with_reason(self):
        """Test ServiceUnavailableError with reason."""
        exc = ServiceUnavailableError("api", "Under maintenance")
        assert "Under maintenance" in exc.message
        assert exc.details["service"] == "api"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError()
        assert exc.status_code == 500
        assert exc.error_code == "CONFIG_ERROR"

    def test_configuration_error_custom(self):
        """Test ConfigurationError with custom message."""
        exc = ConfigurationError("Missing required config")
        assert exc.message == "Missing required config"


class TestExceptionInheritance:
    """Tests for exception inheritance."""

    def test_authentication_inherits_javis(self):
        """Test AuthenticationError inherits from JAVISException."""
        assert issubclass(AuthenticationError, JAVISException)

    def test_invalid_token_inherits_auth(self):
        """Test InvalidTokenError inherits from AuthenticationError."""
        assert issubclass(InvalidTokenError, AuthenticationError)

    def test_validation_inherits_javis(self):
        """Test ValidationError inherits from JAVISException."""
        assert issubclass(ValidationError, JAVISException)

    def test_model_error_inherits_javis(self):
        """Test ModelError inherits from JAVISException."""
        assert issubclass(ModelError, JAVISException)

    def test_storage_error_inherits_javis(self):
        """Test StorageError inherits from JAVISException."""
        assert issubclass(StorageError, JAVISException)

    def test_tool_error_inherits_javis(self):
        """Test ToolError inherits from JAVISException."""
        assert issubclass(ToolError, JAVISException)


class TestExceptionString:
    """Tests for exception string representation."""

    def test_javis_exception_str(self):
        """Test JAVISException string representation."""
        exc = JAVISException("Test message")
        assert str(exc) == "Test message"

    def test_nested_exception_str(self):
        """Test nested exception string representation."""
        exc = ModelNotAvailableError("gpt-4")
        assert "gpt-4" in str(exc)
