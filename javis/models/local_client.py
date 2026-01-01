"""Local model client for fine-tuned JAVIS model."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.utils.config import get_config


class Message(BaseModel):
    """Chat message."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Chat completion response."""
    content: str
    finish_reason: str = "stop"
    usage: dict = {}


class LocalModelClient:
    """Client for local fine-tuned model inference."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
    ):
        self.config = get_config()
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer."""
        if self._loaded:
            return

        # Lazy imports for heavy dependencies
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading model: {self.base_model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load adapter if specified
        if self.adapter_path:
            print(f"Loading adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path
            )

        self._loaded = True
        print("Model loaded successfully")

    def chat(self, messages: list[Message]) -> ChatResponse:
        """Generate chat response."""
        import torch

        if not self._loaded:
            self.load()

        # Format messages
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate
        generation_config = self.config.model.generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return ChatResponse(
            content=response_text.strip(),
            finish_reason="stop",
            usage={
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": len(response_ids),
                "total_tokens": inputs['input_ids'].shape[1] + len(response_ids),
            }
        )

    async def chat_async(self, messages: list[Message]) -> ChatResponse:
        """Async wrapper for chat (runs sync in thread)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages)


def get_latest_adapter(models_dir: str = "models") -> Optional[str]:
    """Get the path to the latest adapter."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None

    adapters = []
    for version_dir in models_path.iterdir():
        if version_dir.is_dir():
            adapter_dir = version_dir / "adapter"
            if adapter_dir.exists():
                adapters.append((version_dir.name, str(adapter_dir)))

    if not adapters:
        return None

    # Sort by version name (assumes format like v20231218_120000)
    adapters.sort(key=lambda x: x[0], reverse=True)
    return adapters[0][1]


def list_adapters(models_dir: str = "models") -> list[dict]:
    """List all available adapters."""
    import json
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    adapters = []
    for version_dir in models_path.iterdir():
        if version_dir.is_dir():
            metadata_file = version_dir / "metadata.json"
            adapter_dir = version_dir / "adapter"

            if adapter_dir.exists():
                info = {
                    "version": version_dir.name,
                    "path": str(adapter_dir),
                }

                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        info["metadata"] = json.load(f)

                adapters.append(info)

    return sorted(adapters, key=lambda x: x["version"], reverse=True)
