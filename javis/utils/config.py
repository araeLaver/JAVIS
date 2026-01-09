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


# Phase 5 Configuration Classes
class DataQualityWeightsConfig(BaseModel):
    """Data quality scoring weights."""

    feedback: float = 0.4
    length: float = 0.2
    coherence: float = 0.2
    recency: float = 0.2


class TrainingDataQualityConfig(BaseModel):
    """Data quality configuration for training."""

    enabled: bool = True
    min_score_for_training: float = 3.5
    weights: DataQualityWeightsConfig = DataQualityWeightsConfig()


class TrainingValidationConfig(BaseModel):
    """Model validation configuration."""

    enabled: bool = True
    prompts_path: str = "./configs/validation_prompts.json"
    max_failures: int = 2
    required_pass_rate: float = 0.8


class TrainingMetricsConfig(BaseModel):
    """Metrics collection configuration."""

    enabled: bool = True
    retention_days: int = 90
    aggregation_interval: str = "daily"


class ABTestingConfig(BaseModel):
    """A/B testing configuration."""

    enabled: bool = False
    default_traffic_split: float = 0.5
    min_sample_size: int = 100
    significance_threshold: float = 0.95


class ToolsConfig(BaseModel):
    """Tools system configuration."""

    enabled: bool = False
    available: list[str] = ["file_tools", "web_tools", "system_tools"]


class LongTermMemoryConfig(BaseModel):
    """Long-term memory configuration."""

    enabled: bool = False
    db_path: str = "./data/vectors/memory"
    collection_name: str = "javis_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    summarize_after_turns: int = 10  # Auto-summarize after N turns


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    long_term: LongTermMemoryConfig = LongTermMemoryConfig()


class RAGConfig(BaseModel):
    """RAG system configuration."""

    enabled: bool = False
    db_path: str = "./data/vectors/documents"
    collection_name: str = "javis_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    min_relevance_score: float = 0.3


# Phase 6: Voice Interface Configuration
class STTGroqWhisperConfig(BaseModel):
    """Groq Whisper STT settings."""

    model: str = "whisper-large-v3"


class STTRecordingConfig(BaseModel):
    """Recording settings for STT."""

    sample_rate: int = 16000
    silence_threshold: int = 500  # ms
    max_duration: int = 30  # seconds


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""

    provider: str = "groq_whisper"
    groq_whisper: STTGroqWhisperConfig = STTGroqWhisperConfig()
    recording: STTRecordingConfig = STTRecordingConfig()


class TTSEdgeTTSConfig(BaseModel):
    """Edge TTS settings."""

    voice: str = "ko-KR-SunHiNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class TTSPlaybackConfig(BaseModel):
    """TTS playback settings."""

    auto_play: bool = True


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    provider: str = "edge_tts"
    edge_tts: TTSEdgeTTSConfig = TTSEdgeTTSConfig()
    playback: TTSPlaybackConfig = TTSPlaybackConfig()


class VoiceConfig(BaseModel):
    """Voice interface configuration."""

    enabled: bool = False
    language: str = "ko-KR"
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()


# Phase 7: External Service Integrations Configuration
class GoogleCalendarConfig(BaseModel):
    """Google Calendar integration settings."""

    enabled: bool = False
    calendar_id: str = "primary"
    max_results: int = 10


class NotionIntegrationConfig(BaseModel):
    """Notion integration settings."""

    enabled: bool = False
    default_database_id: str | None = None


class SlackIntegrationConfig(BaseModel):
    """Slack integration settings."""

    enabled: bool = False
    default_channel: str = "#general"


class IntegrationsConfig(BaseModel):
    """External service integrations configuration."""

    google_calendar: GoogleCalendarConfig = GoogleCalendarConfig()
    notion: NotionIntegrationConfig = NotionIntegrationConfig()
    slack: SlackIntegrationConfig = SlackIntegrationConfig()


# Phase 6-2: Workflow Configuration
class WorkflowNotificationsConfig(BaseModel):
    """Workflow notification settings."""

    on_success: bool = True
    on_failure: bool = True


class WorkflowsConfig(BaseModel):
    """Workflow automation configuration."""

    enabled: bool = False
    directory: str = "./configs/workflows"
    max_concurrent: int = 3
    default_timeout: int = 300  # 5 minutes
    notifications: WorkflowNotificationsConfig = WorkflowNotificationsConfig()


class TrainingConfig(BaseModel):
    """Complete training pipeline configuration."""

    schedule: TrainingScheduleConfig = TrainingScheduleConfig()
    provider: str = "modal"  # modal, local
    data: TrainingDataConfig = TrainingDataConfig()
    model: TrainingModelConfig = TrainingModelConfig()
    deployment: TrainingDeploymentConfig = TrainingDeploymentConfig()
    notifications: TrainingNotificationsConfig = TrainingNotificationsConfig()
    # Phase 5 additions
    data_quality: TrainingDataQualityConfig = TrainingDataQualityConfig()
    validation: TrainingValidationConfig = TrainingValidationConfig()
    metrics: TrainingMetricsConfig = TrainingMetricsConfig()
    ab_testing: ABTestingConfig = ABTestingConfig()


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
    tools: ToolsConfig = ToolsConfig()
    memory: MemoryConfig = MemoryConfig()
    rag: RAGConfig = RAGConfig()
    voice: VoiceConfig = VoiceConfig()  # Phase 6-1
    workflows: WorkflowsConfig = WorkflowsConfig()  # Phase 6-2
    integrations: IntegrationsConfig = IntegrationsConfig()  # Phase 7

    # Environment variables (loaded separately)
    groq_api_key: str | None = None
    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    hf_token: str | None = None
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    discord_webhook_url: str | None = None
    # Phase 7: External service API keys
    google_credentials_path: str | None = None
    notion_api_key: str | None = None
    slack_bot_token: str | None = None


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

    # Parse Phase 5 data quality weights
    data_quality_data = training_data.get("data_quality", {})
    weights_data = data_quality_data.get("weights", {})
    data_quality_config = TrainingDataQualityConfig(
        enabled=data_quality_data.get("enabled", True),
        min_score_for_training=data_quality_data.get("min_score_for_training", 3.5),
        weights=DataQualityWeightsConfig(**weights_data),
    )

    training_config = TrainingConfig(
        schedule=TrainingScheduleConfig(**training_data.get("schedule", {})),
        provider=training_data.get("provider", "modal"),
        data=TrainingDataConfig(**training_data.get("data", {})),
        model=TrainingModelConfig(**training_data.get("model", {})),
        deployment=TrainingDeploymentConfig(**training_data.get("deployment", {})),
        notifications=TrainingNotificationsConfig(**training_data.get("notifications", {})),
        # Phase 5 configs
        data_quality=data_quality_config,
        validation=TrainingValidationConfig(**training_data.get("validation", {})),
        metrics=TrainingMetricsConfig(**training_data.get("metrics", {})),
        ab_testing=ABTestingConfig(**training_data.get("ab_testing", {})),
    )

    # Parse tools config if present
    tools_data = config_data.get("tools", {})
    tools_config = ToolsConfig(**tools_data)

    # Parse memory config if present
    memory_data = config_data.get("memory", {})
    memory_config = MemoryConfig(
        long_term=LongTermMemoryConfig(**memory_data.get("long_term", {}))
    )

    # Parse RAG config if present
    rag_data = config_data.get("rag", {})
    rag_config = RAGConfig(**rag_data)

    # Parse voice config if present (Phase 6)
    voice_data = config_data.get("voice", {})
    stt_data = voice_data.get("stt", {})
    tts_data = voice_data.get("tts", {})

    voice_config = VoiceConfig(
        enabled=voice_data.get("enabled", False),
        language=voice_data.get("language", "ko-KR"),
        stt=STTConfig(
            provider=stt_data.get("provider", "groq_whisper"),
            groq_whisper=STTGroqWhisperConfig(**stt_data.get("groq_whisper", {})),
            recording=STTRecordingConfig(**stt_data.get("recording", {})),
        ),
        tts=TTSConfig(
            provider=tts_data.get("provider", "edge_tts"),
            edge_tts=TTSEdgeTTSConfig(**tts_data.get("edge_tts", {})),
            playback=TTSPlaybackConfig(**tts_data.get("playback", {})),
        ),
    )

    # Parse workflows config if present (Phase 6-2)
    workflows_data = config_data.get("workflows", {})
    workflows_notif_data = workflows_data.get("notifications", {})
    workflows_config = WorkflowsConfig(
        enabled=workflows_data.get("enabled", False),
        directory=workflows_data.get("directory", "./configs/workflows"),
        max_concurrent=workflows_data.get("max_concurrent", 3),
        default_timeout=workflows_data.get("default_timeout", 300),
        notifications=WorkflowNotificationsConfig(**workflows_notif_data),
    )

    # Parse integrations config if present (Phase 7)
    integrations_data = config_data.get("integrations", {})
    integrations_config = IntegrationsConfig(
        google_calendar=GoogleCalendarConfig(**integrations_data.get("google_calendar", {})),
        notion=NotionIntegrationConfig(**integrations_data.get("notion", {})),
        slack=SlackIntegrationConfig(**integrations_data.get("slack", {})),
    )

    # Create config object
    _config = Config(
        app=AppConfig(**config_data.get("app", {})),
        model=ModelConfig(**config_data.get("model", {})),
        conversation=ConversationConfig(**config_data.get("conversation", {})),
        training=training_config,
        tools=tools_config,
        memory=memory_config,
        rag=rag_config,
        voice=voice_config,
        workflows=workflows_config,
        integrations=integrations_config,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        runpod_api_key=os.getenv("RUNPOD_API_KEY"),
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        hf_token=os.getenv("HF_TOKEN"),
        modal_token_id=os.getenv("MODAL_TOKEN_ID"),
        modal_token_secret=os.getenv("MODAL_TOKEN_SECRET"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
        # Phase 7
        google_credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH"),
        notion_api_key=os.getenv("NOTION_API_KEY"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
    )

    return _config


def get_config() -> Config:
    """Get the current configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
