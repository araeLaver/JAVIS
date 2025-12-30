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


# Training Pipeline Configuration
class TrainingScheduleConfig(BaseModel):
    """Training scheduler settings."""

    enabled: bool = False
    cron: str = "0 0 * * 0"  # 매주 일요일 자정
    timezone: str = "Asia/Seoul"


class TrainingDataConfig(BaseModel):
    """Training data requirements."""

    min_conversations: int = 50
    min_good_feedback: int = 10
    exclude_bad_feedback: bool = True
    max_age_days: int = 90


class TrainingModelConfig(BaseModel):
    """Training hyperparameters."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 64
    lora_alpha: int = 16
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4


class TrainingDeploymentConfig(BaseModel):
    """Deployment settings after training."""

    auto_deploy: bool = True
    validation_required: bool = True
    keep_versions: int = 5


class TrainingNotificationsConfig(BaseModel):
    """Notification settings."""

    discord_webhook: str | None = None
    on_success: bool = True
    on_failure: bool = True


class TrainingConfig(BaseModel):
    """Complete training pipeline configuration."""

    schedule: TrainingScheduleConfig = TrainingScheduleConfig()
    provider: str = "modal"  # modal, local
    data: TrainingDataConfig = TrainingDataConfig()
    model: TrainingModelConfig = TrainingModelConfig()
    deployment: TrainingDeploymentConfig = TrainingDeploymentConfig()
    notifications: TrainingNotificationsConfig = TrainingNotificationsConfig()


class AppConfig(BaseModel):
    """Application configuration."""

    name: str = "JAVIS"
    version: str = "0.1.0"


class Config(BaseModel):
    """Main configuration container."""

    app: AppConfig = AppConfig()
    model: ModelConfig = ModelConfig()
    conversation: ConversationConfig = ConversationConfig()
    training: TrainingConfig = TrainingConfig()

    # Environment variables (loaded separately)
    groq_api_key: str | None = None
    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    hf_token: str | None = None
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    discord_webhook_url: str | None = None


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

    # Parse training config if present
    training_data = config_data.get("training", {})
    training_config = TrainingConfig(
        schedule=TrainingScheduleConfig(**training_data.get("schedule", {})),
        provider=training_data.get("provider", "modal"),
        data=TrainingDataConfig(**training_data.get("data", {})),
        model=TrainingModelConfig(**training_data.get("model", {})),
        deployment=TrainingDeploymentConfig(**training_data.get("deployment", {})),
        notifications=TrainingNotificationsConfig(**training_data.get("notifications", {})),
    )

    # Create config object
    _config = Config(
        app=AppConfig(**config_data.get("app", {})),
        model=ModelConfig(**config_data.get("model", {})),
        conversation=ConversationConfig(**config_data.get("conversation", {})),
        training=training_config,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        runpod_api_key=os.getenv("RUNPOD_API_KEY"),
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        hf_token=os.getenv("HF_TOKEN"),
        modal_token_id=os.getenv("MODAL_TOKEN_ID"),
        modal_token_secret=os.getenv("MODAL_TOKEN_SECRET"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
    )

    return _config


def get_config() -> Config:
    """Get the current configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
