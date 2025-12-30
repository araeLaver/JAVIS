"""Remote GPU training via Modal.com."""

import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# Modal imports - will be available when modal is installed
try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None


class TrainingResult(BaseModel):
    """Result of a training run."""

    success: bool
    version: str
    adapter_path: Optional[Path] = None
    error: Optional[str] = None
    duration_seconds: float = 0
    metadata: dict = {}


# Modal app definition (only created if modal is available)
if MODAL_AVAILABLE:
    app = modal.App("javis-training")

    # Training image with all dependencies
    # Use compatible versions that work together
    training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy<2.0",  # Avoid NumPy 2.x compatibility issues
        "torch>=2.1.0,<2.5.0",
        "transformers>=4.40.0",
        "datasets>=2.21.0",
        "peft>=0.10.0",
        "trl>=0.12.0",
        "bitsandbytes>=0.42.0",
        "accelerate>=0.34.0",
        "scipy",
        "sentencepiece",
    )

    @app.function(
        gpu="A10G",  # 24GB VRAM, ~$1.10/hr
        image=training_image,
        timeout=7200,  # 2 hours max
        serialized=False,  # Avoid Python version mismatch
    )
    def train_on_modal(training_data_jsonl: str, config: dict) -> dict:
        """Execute QLoRA training on Modal GPU.

        Args:
            training_data_jsonl: JSONL string with training data
            config: Training configuration dict

        Returns:
            dict with adapter files (base64 encoded) and metadata
        """
        import base64
        import io
        import json
        import os
        import zipfile
        from datetime import datetime

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer

        print(f"Starting training with config: {config}")
        start_time = datetime.now()

        # Parse training data
        conversations = []
        for line in training_data_jsonl.strip().split("\n"):
            if line.strip():
                conversations.append(json.loads(line))

        print(f"Loaded {len(conversations)} conversations")

        # Create dataset
        dataset = Dataset.from_list(conversations)

        # Load tokenizer
        base_model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Format conversations
        def format_conversation(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        print(f"Formatted {len(dataset)} examples")

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )

        # LoRA config
        lora_config = LoraConfig(
            r=config.get("lora_r", 64),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
            bias="none",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable_params:,} / {total_params:,} params")

        # Training config
        output_dir = "/tmp/javis-adapter"
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_length=config.get("max_seq_length", 2048),  # renamed from max_seq_length in newer trl
            dataset_text_field="text",
            report_to="none",
        )

        # Train
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
        )

        print("Starting training...")
        trainer.train()

        # Save adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Create metadata
        end_time = datetime.now()
        metadata = {
            "base_model": base_model,
            "created_at": end_time.isoformat(),
            "dataset_size": len(conversations),
            "training_config": config,
            "duration_seconds": (end_time - start_time).total_seconds(),
        }

        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Zip adapter files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in Path(output_dir).rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zf.write(file_path, arcname)

        zip_buffer.seek(0)
        adapter_zip_b64 = base64.b64encode(zip_buffer.read()).decode("utf-8")

        print(f"Training complete in {metadata['duration_seconds']:.1f}s")

        return {
            "success": True,
            "adapter_zip_b64": adapter_zip_b64,
            "metadata": metadata,
        }


class RemoteTrainer:
    """Remote training orchestrator."""

    def __init__(self, provider: str = "modal"):
        self.provider = provider

    def train(
        self, data_path: Path, config: dict, output_dir: Optional[Path] = None
    ) -> TrainingResult:
        """Execute remote training and return results.

        Args:
            data_path: Path to JSONL training data
            config: Training configuration dict
            output_dir: Where to save the adapter (default: models/v{timestamp})

        Returns:
            TrainingResult with success status and adapter path
        """
        if self.provider == "modal":
            return self._train_modal(data_path, config, output_dir)
        elif self.provider == "local":
            return self._train_local(data_path, config, output_dir)
        else:
            return TrainingResult(
                success=False,
                version="",
                error=f"Unknown provider: {self.provider}",
            )

    def _train_modal(
        self, data_path: Path, config: dict, output_dir: Optional[Path] = None
    ) -> TrainingResult:
        """Train on Modal.com GPU."""
        if not MODAL_AVAILABLE:
            return TrainingResult(
                success=False,
                version="",
                error="Modal is not installed. Run: pip install modal",
            )

        import base64

        start_time = datetime.now()
        version = f"v{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Read training data
            with open(data_path, "r", encoding="utf-8") as f:
                training_data_jsonl = f.read()

            # Call Modal function (disable output to avoid Windows encoding issues)
            with app.run():
                result = train_on_modal.remote(training_data_jsonl, config)

            if not result.get("success"):
                return TrainingResult(
                    success=False,
                    version=version,
                    error=result.get("error", "Training failed"),
                )

            # Extract adapter
            if output_dir is None:
                output_dir = Path(__file__).parent.parent.parent / "models" / version

            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            # Decode and extract zip
            adapter_zip_b64 = result["adapter_zip_b64"]
            zip_data = base64.b64decode(adapter_zip_b64)

            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(zip_data)
                tmp_path = tmp.name

            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(adapter_dir)

            Path(tmp_path).unlink()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return TrainingResult(
                success=True,
                version=version,
                adapter_path=adapter_dir,
                duration_seconds=duration,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                version=version,
                error=str(e),
            )

    def _train_local(
        self, data_path: Path, config: dict, output_dir: Optional[Path] = None
    ) -> TrainingResult:
        """Train on local GPU using existing finetune.py logic."""
        import subprocess
        import sys

        start_time = datetime.now()
        version = f"v{start_time.strftime('%Y%m%d_%H%M%S')}"

        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "models" / version

        try:
            # Run finetune.py as subprocess
            cmd = [
                sys.executable,
                "-m",
                "javis.training.finetune",
                "--data",
                str(data_path),
                "--output",
                str(output_dir),
                "--epochs",
                str(config.get("epochs", 3)),
                "--batch-size",
                str(config.get("batch_size", 4)),
                "--learning-rate",
                str(config.get("learning_rate", 2e-4)),
                "--lora-r",
                str(config.get("lora_r", 64)),
                "--lora-alpha",
                str(config.get("lora_alpha", 16)),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Read metadata if exists
            metadata_path = output_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            return TrainingResult(
                success=True,
                version=version,
                adapter_path=output_dir / "adapter",
                duration_seconds=duration,
                metadata=metadata,
            )

        except subprocess.CalledProcessError as e:
            return TrainingResult(
                success=False,
                version=version,
                error=f"Local training failed: {e.stderr}",
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                version=version,
                error=str(e),
            )


def check_modal_available() -> bool:
    """Check if Modal is installed and configured."""
    if not MODAL_AVAILABLE:
        return False

    try:
        # Check if credentials are set
        import os

        return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
    except Exception:
        return False
