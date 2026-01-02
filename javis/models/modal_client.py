"""Modal.com based inference client for fine-tuned models."""

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# Modal imports
try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None


class Message(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatResponse(BaseModel):
    """Chat completion response."""

    content: str
    finish_reason: str = "stop"
    usage: dict = {}


# Modal app for inference
if MODAL_AVAILABLE:
    inference_app = modal.App("javis-inference")

    # Inference image
    inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy<2.0",
        "torch>=2.1.0,<2.5.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.42.0",
        "accelerate>=0.34.0",
        "scipy",
        "sentencepiece",
    )

    @inference_app.function(
        gpu="T4",
        image=inference_image,
        timeout=300,
    )
    def run_inference(
        messages: list[dict],
        adapter_weights_b64: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Run inference on Modal GPU."""
        import tempfile
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        base_model = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading model: {base_model}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Base model loaded!")

        # Load adapter if provided
        if adapter_weights_b64:
            print("Loading adapter...")
            adapter_data = base64.b64decode(adapter_weights_b64)
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(io.BytesIO(adapter_data), "r") as zf:
                    zf.extractall(tmp_dir)
                model = PeftModel.from_pretrained(model, tmp_dir)
            print("Adapter loaded!")

        # Format messages
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "content": response_text.strip(),
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(response_ids),
                "total_tokens": inputs["input_ids"].shape[1] + len(response_ids),
            },
        }


class ModalInferenceClient:
    """Client for Modal-based inference."""

    def __init__(self, adapter_path: Optional[str] = None):
        if not MODAL_AVAILABLE:
            raise RuntimeError("Modal is not installed. Run: pip install modal")

        self.adapter_path = adapter_path
        self._adapter_weights_b64: Optional[str] = None

        # Pre-load adapter weights if path provided
        if adapter_path:
            self._load_adapter_weights(adapter_path)

    def _load_adapter_weights(self, adapter_path: str):
        """Load adapter weights as base64 encoded zip."""
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # Create zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in adapter_dir.rglob("*"):
                if file_path.is_file():
                    # Skip checkpoint directories
                    if "checkpoint" in str(file_path):
                        continue
                    arcname = file_path.relative_to(adapter_dir)
                    zf.write(file_path, arcname)

        zip_buffer.seek(0)
        self._adapter_weights_b64 = base64.b64encode(zip_buffer.read()).decode("utf-8")
        print(f"Adapter loaded: {len(self._adapter_weights_b64)} bytes")

    def chat(self, messages: list[Message]) -> ChatResponse:
        """Generate chat response using Modal."""
        # Convert messages to dicts
        messages_dict = [{"role": m.role, "content": m.content} for m in messages]

        # Call Modal function
        with inference_app.run():
            result = run_inference.remote(
                messages=messages_dict,
                adapter_weights_b64=self._adapter_weights_b64,
            )

        return ChatResponse(
            content=result["content"],
            finish_reason=result.get("finish_reason", "stop"),
            usage=result.get("usage", {}),
        )

    async def chat_async(self, messages: list[Message]) -> ChatResponse:
        """Async chat (runs sync in thread)."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages)


def get_latest_adapter_path() -> Optional[str]:
    """Get path to latest adapter."""
    from javis.models.local_client import get_latest_adapter

    models_dir = Path(__file__).parent.parent.parent / "models"
    return get_latest_adapter(str(models_dir))


def check_modal_available() -> bool:
    """Check if Modal is available and configured."""
    if not MODAL_AVAILABLE:
        return False

    # Modal stores credentials locally after `modal setup`
    home = Path.home()
    modal_toml = home / ".modal.toml"
    modal_creds = home / ".modal" / "credentials"

    return modal_toml.exists() or modal_creds.exists()
