"""QLoRA fine-tuning script for JAVIS model."""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_training_data(data_path: Path) -> Dataset:
    """Load JSONL training data."""
    conversations = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                conversations.append(data)

    return Dataset.from_list(conversations)


def format_conversation(example, tokenizer):
    """Format conversation for Qwen chat template."""
    messages = example['messages']

    # Qwen chat template 적용
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune JAVIS model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/exported/conversations_20251219.jsonl",
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for adapter"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Output 경로 설정
    if args.output is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        args.output = f"models/{version}/adapter"

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=" * 50)
    print(f"JAVIS Fine-tuning")
    print(f"=" * 50)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"=" * 50)

    # 1. 데이터 로드
    print("\n[1/5] Loading training data...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    dataset = load_training_data(data_path)
    print(f"Loaded {len(dataset)} conversations")

    # 2. 토크나이저 로드
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 데이터 포맷팅
    print("\n[3/5] Formatting data...")
    dataset = dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # 4. 모델 로드 (4-bit 양자화)
    print("\n[4/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA 설정
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 5. 학습
    print("\n[5/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer.train()

    # 6. 저장
    print("\n[6/6] Saving adapter...")
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # 메타데이터 저장
    metadata = {
        "version": output_path.parent.name,
        "base_model": args.base_model,
        "created_at": datetime.now().isoformat(),
        "training_data": str(args.data),
        "training_config": {
            "method": "qlora",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_length": args.max_seq_length,
        },
        "dataset_size": len(dataset),
    }

    metadata_path = output_path.parent / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Adapter saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
