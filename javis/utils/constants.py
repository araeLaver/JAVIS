"""Constants for JAVIS application."""

from typing import Final


# =============================================================================
# API Constants
# =============================================================================

GROQ_API_BASE_URL: Final[str] = "https://api.groq.com/openai/v1/chat/completions"


# =============================================================================
# Timeout Constants (in seconds)
# =============================================================================

class Timeouts:
    """Timeout constants in seconds."""

    DEFAULT_API: Final[float] = 60.0
    TOOL_EXECUTION: Final[float] = 30.0
    MODAL_INFERENCE: Final[int] = 300


# =============================================================================
# Size Constants
# =============================================================================

class Sizes:
    """Size-related constants."""

    BYTES_PER_KB: Final[int] = 1024
    BYTES_PER_MB: Final[int] = 1024 ** 2
    BYTES_PER_GB: Final[int] = 1024 ** 3

    # File size limits
    MAX_FILE_SIZE_MB: Final[int] = 10
    MAX_CONTENT_SIZE_MB: Final[int] = 5


# =============================================================================
# Default Values
# =============================================================================

class Defaults:
    """Default configuration values."""

    # Model defaults
    MODEL_NAME: Final[str] = "llama-3.1-8b-instant"
    MAX_TOKENS: Final[int] = 2048
    TEMPERATURE: Final[float] = 0.7
    TOP_P: Final[float] = 0.9

    # Tool defaults
    MAX_TOOL_ITERATIONS: Final[int] = 5
    MAX_PARALLEL_TOOLS: Final[int] = 5

    # Memory/RAG defaults
    MEMORY_CONTEXT_LIMIT: Final[int] = 3
    MEMORY_MAX_CHARS: Final[int] = 500
    RAG_MAX_CHARS: Final[int] = 1000

    # Embedding dimension (for MiniLM-L6-v2)
    EMBEDDING_DIMENSION: Final[int] = 384


# =============================================================================
# HTTP Headers
# =============================================================================

class Headers:
    """HTTP header constants."""

    AUTHORIZATION: Final[str] = "Authorization"
    CONTENT_TYPE: Final[str] = "Content-Type"
    JSON_CONTENT: Final[str] = "application/json"


# =============================================================================
# Environment Variable Names
# =============================================================================

class EnvVars:
    """Environment variable names."""

    GROQ_API_KEY: Final[str] = "GROQ_API_KEY"
    RUNPOD_API_KEY: Final[str] = "RUNPOD_API_KEY"
    RUNPOD_ENDPOINT_ID: Final[str] = "RUNPOD_ENDPOINT_ID"
    HF_TOKEN: Final[str] = "HF_TOKEN"
    MODAL_TOKEN_ID: Final[str] = "MODAL_TOKEN_ID"
    MODAL_TOKEN_SECRET: Final[str] = "MODAL_TOKEN_SECRET"
    CORS_ORIGINS: Final[str] = "CORS_ORIGINS"

    # External integrations
    GOOGLE_CREDENTIALS_PATH: Final[str] = "GOOGLE_CREDENTIALS_PATH"
    NOTION_API_KEY: Final[str] = "NOTION_API_KEY"
    SLACK_BOT_TOKEN: Final[str] = "SLACK_BOT_TOKEN"


# =============================================================================
# CORS Default Origins
# =============================================================================

DEFAULT_CORS_ORIGINS: Final[list[str]] = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]


# =============================================================================
# Date/Time Formats
# =============================================================================

class DateFormats:
    """Date and time format strings."""

    ISO: Final[str] = "%Y-%m-%dT%H:%M:%S"
    DATE_ONLY: Final[str] = "%Y-%m-%d"
    TIME_ONLY: Final[str] = "%H:%M:%S"
    DISPLAY: Final[str] = "%Y-%m-%d %H:%M"
    LOG: Final[str] = "%Y-%m-%d %H:%M:%S.%f"


# =============================================================================
# Message Roles
# =============================================================================

class Roles:
    """Chat message roles."""

    SYSTEM: Final[str] = "system"
    USER: Final[str] = "user"
    ASSISTANT: Final[str] = "assistant"
    TOOL: Final[str] = "tool"


# =============================================================================
# Finish Reasons
# =============================================================================

class FinishReasons:
    """LLM finish reasons."""

    STOP: Final[str] = "stop"
    LENGTH: Final[str] = "length"
    TOOL_CALLS: Final[str] = "tool_calls"
    ERROR: Final[str] = "error"
    MAX_ITERATIONS: Final[str] = "max_iterations"
