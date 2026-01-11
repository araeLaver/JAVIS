"""Run training directly via modal run to see progress."""
import modal

app = modal.App("javis-training-direct")

# Use same versions that worked in test
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy<2.0",
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


def _parse_training_data(training_data_jsonl: str) -> list[dict]:
    """JSONL 형식의 학습 데이터를 파싱합니다."""
    import json
    conversations = []
    for line in training_data_jsonl.strip().split("\n"):
        if line.strip():
            conversations.append(json.loads(line))
    return conversations


def _setup_tokenizer(base_model: str):
    """토크나이저를 로드하고 설정합니다."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _prepare_dataset(conversations: list[dict], tokenizer):
    """데이터셋을 준비하고 포맷팅합니다."""
    from datasets import Dataset

    dataset = Dataset.from_list(conversations)

    def format_conversation(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    print(f"Formatted {len(dataset)} examples")
    return dataset


def _load_model(base_model: str, config: dict):
    """모델을 로드하고 LoRA를 적용합니다."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model {base_model} (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    print("Model loaded successfully!")

    # LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} params")

    return model


def _run_training(model, dataset, tokenizer, config: dict, output_dir: str):
    """학습을 실행합니다."""
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_length=config.get("max_seq_length", 1024),
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    return trainer


def _save_adapter(model, tokenizer, output_dir: str, metadata: dict) -> str:
    """어댑터를 저장하고 압축합니다."""
    import base64
    import io
    import json
    import zipfile
    from pathlib import Path

    print("Saving adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Creating adapter archive...")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in Path(output_dir).rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)

    zip_buffer.seek(0)
    return base64.b64encode(zip_buffer.read()).decode("utf-8")


@app.function(
    gpu="A10G",
    image=training_image,
    timeout=7200,
)
def train_model(training_data_jsonl: str, config: dict) -> dict:
    """Train the model."""
    import torch
    from datetime import datetime

    print(f"Starting training with config: {config}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    start_time = datetime.now()
    output_dir = "/tmp/javis-adapter"
    base_model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")

    # 1. 데이터 파싱
    conversations = _parse_training_data(training_data_jsonl)
    print(f"Loaded {len(conversations)} conversations")

    # 2. 토크나이저 설정
    tokenizer = _setup_tokenizer(base_model)

    # 3. 데이터셋 준비
    dataset = _prepare_dataset(conversations, tokenizer)

    # 4. 모델 로드 및 LoRA 적용
    model = _load_model(base_model, config)

    # 5. 학습 실행
    _run_training(model, dataset, tokenizer, config, output_dir)

    # 6. 메타데이터 생성 및 저장
    end_time = datetime.now()
    metadata = {
        "base_model": base_model,
        "created_at": end_time.isoformat(),
        "dataset_size": len(conversations),
        "training_config": config,
        "duration_seconds": (end_time - start_time).total_seconds(),
    }

    adapter_zip_b64 = _save_adapter(model, tokenizer, output_dir, metadata)

    print(f"Training complete in {metadata['duration_seconds']:.1f}s")

    return {
        "success": True,
        "adapter_zip_b64": adapter_zip_b64,
        "metadata": metadata,
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    # Read training data
    data_dir = Path("data/training/exported")
    data_files = sorted(data_dir.glob("*.jsonl"))

    if not data_files:
        # Export from conversations if no exported data
        conversations_dir = Path("data/conversations")
        if conversations_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            conversations = []
            for json_file in conversations_dir.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        conv = json.load(f)
                    if len(conv.get("turns", [])) >= 2:
                        messages = [{"role": t["role"], "content": t["content"]} for t in conv["turns"]]
                        conversations.append({"messages": messages})
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {json_file}: {e}")
                    continue
                except (KeyError, TypeError) as e:
                    print(f"Warning: Invalid conversation format in {json_file}: {e}")
                    continue
                except OSError as e:
                    print(f"Warning: Cannot read {json_file}: {e}")
                    continue

            if conversations:
                output_path = data_dir / "conversations.jsonl"
                with open(output_path, "w", encoding="utf-8") as f:
                    for conv in conversations:
                        f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                data_files = [output_path]

    if not data_files:
        print("No training data found!")
        return

    # Read JSONL
    with open(data_files[-1], "r", encoding="utf-8") as f:
        training_data = f.read()

    print(f"Loaded training data from {data_files[-1]}")
    print(f"Number of conversations: {len(training_data.strip().split(chr(10)))}")

    # Config
    config = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "lora_r": 64,
        "lora_alpha": 16,
        "max_seq_length": 1024,
        "gradient_accumulation_steps": 4,
    }

    print(f"Starting remote training...")
    result = train_model.remote(training_data, config)

    if result["success"]:
        print(f"Training successful!")
        print(f"Duration: {result['metadata']['duration_seconds']:.1f}s")

        # Save adapter locally
        import base64
        models_dir = Path("models") / f"v{result['metadata']['created_at'][:10].replace('-', '')}"
        adapter_dir = models_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        import zipfile
        import tempfile
        zip_data = base64.b64decode(result["adapter_zip_b64"])
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_data)
            tmp_path = tmp.name

        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(adapter_dir)

        Path(tmp_path).unlink()
        print(f"Adapter saved to {adapter_dir}")
    else:
        print(f"Training failed: {result.get('error', 'Unknown error')}")
