"""Custom exceptions for JAVIS application.

This module defines a hierarchy of custom exceptions for consistent error handling
across the application. All exceptions inherit from JAVISException base class.
"""

from typing import Any, Optional


class JAVISException(Exception):
    """Base exception for all JAVIS errors.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code (e.g., "AUTH_001")
        details: Additional error details
        status_code: HTTP status code for API responses
    """

    status_code: int = 500
    error_code: str = "JAVIS_ERROR"

    def __init__(
        self,
        message: str = "An error occurred",
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


# ==================== Authentication Errors ====================

class AuthenticationError(JAVISException):
    """Authentication related errors."""
    status_code = 401
    error_code = "AUTH_ERROR"


class InvalidTokenError(AuthenticationError):
    """Invalid or expired token."""
    error_code = "AUTH_001"

    def __init__(self, message: str = "Invalid or expired token"):
        super().__init__(message)


class MissingTokenError(AuthenticationError):
    """Missing authentication token."""
    error_code = "AUTH_002"

    def __init__(self, message: str = "Authentication token is required"):
        super().__init__(message)


class InsufficientPermissionsError(JAVISException):
    """User lacks required permissions."""
    status_code = 403
    error_code = "AUTH_003"

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message)


# ==================== Validation Errors ====================

class ValidationError(JAVISException):
    """Input validation errors."""
    status_code = 400
    error_code = "VALIDATION_ERROR"


class InvalidInputError(ValidationError):
    """Invalid input data."""
    error_code = "VALIDATION_001"

    def __init__(self, field: str, message: str = "Invalid input"):
        super().__init__(
            message=f"{field}: {message}",
            details={"field": field}
        )


class MissingFieldError(ValidationError):
    """Required field is missing."""
    error_code = "VALIDATION_002"

    def __init__(self, field: str):
        super().__init__(
            message=f"Required field is missing: {field}",
            details={"field": field}
        )


# ==================== Resource Errors ====================

class ResourceNotFoundError(JAVISException):
    """Requested resource was not found."""
    status_code = 404
    error_code = "NOT_FOUND"

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        message: Optional[str] = None
    ):
        msg = message or f"{resource_type} not found"
        if resource_id:
            msg = f"{resource_type} not found: {resource_id}"
        super().__init__(
            message=msg,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ResourceConflictError(JAVISException):
    """Resource already exists or conflict occurred."""
    status_code = 409
    error_code = "CONFLICT"

    def __init__(self, message: str = "Resource conflict"):
        super().__init__(message)


# ==================== Model Errors ====================

class ModelError(JAVISException):
    """Model-related errors."""
    status_code = 500
    error_code = "MODEL_ERROR"


class ModelNotAvailableError(ModelError):
    """Model is not available or not configured."""
    error_code = "MODEL_001"

    def __init__(self, model_name: str, message: Optional[str] = None):
        msg = message or f"Model not available: {model_name}"
        super().__init__(message=msg, details={"model": model_name})


class ModelLoadError(ModelError):
    """Failed to load model."""
    error_code = "MODEL_002"

    def __init__(self, model_name: str, reason: Optional[str] = None):
        msg = f"Failed to load model: {model_name}"
        if reason:
            msg += f" - {reason}"
        super().__init__(message=msg, details={"model": model_name})


class ModelInferenceError(ModelError):
    """Error during model inference."""
    error_code = "MODEL_003"

    def __init__(self, message: str = "Model inference failed"):
        super().__init__(message)


# ==================== Training Errors ====================

class TrainingError(JAVISException):
    """Training-related errors."""
    status_code = 500
    error_code = "TRAINING_ERROR"


class InsufficientDataError(TrainingError):
    """Not enough data for training."""
    error_code = "TRAINING_001"

    def __init__(self, required: int, available: int):
        super().__init__(
            message=f"Insufficient training data: {available}/{required} required",
            details={"required": required, "available": available}
        )


class TrainingInProgressError(TrainingError):
    """Training is already in progress."""
    status_code = 409
    error_code = "TRAINING_002"

    def __init__(self, message: str = "Training is already in progress"):
        super().__init__(message)


class TrainingConfigError(TrainingError):
    """Invalid training configuration."""
    status_code = 400
    error_code = "TRAINING_003"

    def __init__(self, message: str = "Invalid training configuration"):
        super().__init__(message)


# ==================== Storage Errors ====================

class StorageError(JAVISException):
    """Storage-related errors."""
    status_code = 500
    error_code = "STORAGE_ERROR"


class StorageConnectionError(StorageError):
    """Failed to connect to storage backend."""
    error_code = "STORAGE_001"

    def __init__(self, backend: str, message: Optional[str] = None):
        msg = message or f"Failed to connect to storage: {backend}"
        super().__init__(message=msg, details={"backend": backend})


class StorageOperationError(StorageError):
    """Storage operation failed."""
    error_code = "STORAGE_002"

    def __init__(self, operation: str, message: Optional[str] = None):
        msg = message or f"Storage operation failed: {operation}"
        super().__init__(message=msg, details={"operation": operation})


# ==================== Memory Errors ====================

class MemoryError(JAVISException):
    """Memory system errors."""
    status_code = 500
    error_code = "MEMORY_ERROR"


class MemoryNotEnabledError(MemoryError):
    """Memory system is not enabled."""
    status_code = 400
    error_code = "MEMORY_001"

    def __init__(self, message: str = "Long-term memory is not enabled"):
        super().__init__(message)


# ==================== RAG Errors ====================

class RAGError(JAVISException):
    """RAG system errors."""
    status_code = 500
    error_code = "RAG_ERROR"


class RAGNotEnabledError(RAGError):
    """RAG system is not enabled."""
    status_code = 400
    error_code = "RAG_001"

    def __init__(self, message: str = "RAG system is not enabled"):
        super().__init__(message)


class DocumentProcessingError(RAGError):
    """Failed to process document."""
    error_code = "RAG_002"

    def __init__(self, source: str, reason: Optional[str] = None):
        msg = f"Failed to process document: {source}"
        if reason:
            msg += f" - {reason}"
        super().__init__(message=msg, details={"source": source})


# ==================== Tool Errors ====================

class ToolError(JAVISException):
    """Tool system errors."""
    status_code = 500
    error_code = "TOOL_ERROR"


class ToolNotFoundError(ToolError):
    """Tool not found."""
    status_code = 404
    error_code = "TOOL_001"

    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool not found: {tool_name}",
            details={"tool": tool_name}
        )


class ToolExecutionError(ToolError):
    """Tool execution failed."""
    error_code = "TOOL_002"

    def __init__(self, tool_name: str, reason: Optional[str] = None):
        msg = f"Tool execution failed: {tool_name}"
        if reason:
            msg += f" - {reason}"
        super().__init__(message=msg, details={"tool": tool_name})


# ==================== Integration Errors ====================

class IntegrationError(JAVISException):
    """External integration errors."""
    status_code = 502
    error_code = "INTEGRATION_ERROR"


class IntegrationNotConfiguredError(IntegrationError):
    """Integration is not configured."""
    status_code = 400
    error_code = "INTEGRATION_001"

    def __init__(self, integration: str, config_hint: Optional[str] = None):
        msg = f"Integration not configured: {integration}"
        details = {"integration": integration}
        if config_hint:
            details["config_hint"] = config_hint
        super().__init__(message=msg, details=details)


class IntegrationConnectionError(IntegrationError):
    """Failed to connect to external service."""
    error_code = "INTEGRATION_002"

    def __init__(self, integration: str, reason: Optional[str] = None):
        msg = f"Failed to connect to {integration}"
        if reason:
            msg += f": {reason}"
        super().__init__(message=msg, details={"integration": integration})


# ==================== Workflow Errors ====================

class WorkflowError(JAVISException):
    """Workflow-related errors."""
    status_code = 500
    error_code = "WORKFLOW_ERROR"


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found."""
    status_code = 404
    error_code = "WORKFLOW_001"

    def __init__(self, workflow_name: str):
        super().__init__(
            message=f"Workflow not found: {workflow_name}",
            details={"workflow": workflow_name}
        )


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed."""
    error_code = "WORKFLOW_002"

    def __init__(self, workflow_name: str, step: Optional[str] = None, reason: Optional[str] = None):
        msg = f"Workflow execution failed: {workflow_name}"
        details = {"workflow": workflow_name}
        if step:
            msg += f" at step '{step}'"
            details["step"] = step
        if reason:
            msg += f" - {reason}"
        super().__init__(message=msg, details=details)


# ==================== Voice Errors ====================

class VoiceError(JAVISException):
    """Voice processing errors."""
    status_code = 500
    error_code = "VOICE_ERROR"


class VoiceNotEnabledError(VoiceError):
    """Voice interface is not enabled."""
    status_code = 400
    error_code = "VOICE_001"

    def __init__(self, message: str = "Voice interface is not enabled"):
        super().__init__(message)


class TranscriptionError(VoiceError):
    """Speech-to-text transcription failed."""
    error_code = "VOICE_002"

    def __init__(self, reason: Optional[str] = None):
        msg = "Transcription failed"
        if reason:
            msg += f": {reason}"
        super().__init__(message=msg)


class SynthesisError(VoiceError):
    """Text-to-speech synthesis failed."""
    error_code = "VOICE_003"

    def __init__(self, reason: Optional[str] = None):
        msg = "Speech synthesis failed"
        if reason:
            msg += f": {reason}"
        super().__init__(message=msg)


# ==================== Rate Limiting ====================

class RateLimitError(JAVISException):
    """Rate limit exceeded."""
    status_code = 429
    error_code = "RATE_LIMIT"

    def __init__(self, retry_after: Optional[int] = None):
        msg = "Rate limit exceeded"
        details = {}
        if retry_after:
            msg += f". Retry after {retry_after} seconds"
            details["retry_after"] = retry_after
        super().__init__(message=msg, details=details)


# ==================== Service Errors ====================

class ServiceUnavailableError(JAVISException):
    """Service is temporarily unavailable."""
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"

    def __init__(self, service: str, reason: Optional[str] = None):
        msg = f"Service unavailable: {service}"
        if reason:
            msg += f" - {reason}"
        super().__init__(message=msg, details={"service": service})


class ConfigurationError(JAVISException):
    """Configuration error."""
    status_code = 500
    error_code = "CONFIG_ERROR"

    def __init__(self, message: str = "Configuration error"):
        super().__init__(message)
