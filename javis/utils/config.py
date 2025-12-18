"""Configuration management for JAVIS."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class GenerationConfig(BaseModel):
    """Model generation parameters."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = ["<|im_end|>", "<|endoftext|>"]


class ModelConfig(BaseModel):
    """Model configuration."""

    provider: str = "runpod"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter: str | None = None
    generation: GenerationConfig = GenerationConfig()


class ConversationConfig(BaseModel):
    """Conversation settings."""

    system_prompt: str = "너는 JAVIS, 개발자를 위한 개인 AI 비서다."
    max_history: int = 20
    max_context_tokens: int = 8000


class AppConfig(BaseModel):
    """Application configuration."""

    name: str = "JAVIS"
    version: str = "0.1.0"


class Config(BaseModel):
    """Main configuration container."""

    app: AppConfig = AppConfig()
    model: ModelConfig = ModelConfig()
    conversation: ConversationConfig = ConversationConfig()

    # Environment variables (loaded separately)
    groq_api_key: str | None = None
    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    hf_token: str | None = None


_config: Config | None = None


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    global _config

    # Load environment variables
    load_dotenv()

    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

    config_path = Path(config_path)

    # Load YAML config
    config_data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # Create config object
    _config = Config(
        app=AppConfig(**config_data.get("app", {})),
        model=ModelConfig(**config_data.get("model", {})),
        conversation=ConversationConfig(**config_data.get("conversation", {})),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        runpod_api_key=os.getenv("RUNPOD_API_KEY"),
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        hf_token=os.getenv("HF_TOKEN"),
    )

    return _config


def get_config() -> Config:
    """Get the current configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
