"""DPO (Direct Preference Optimization) training script for JAVIS model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


def load_preference_data(data_path: Path) -> Dataset:
    """Load JSONL preference data.

    Expected format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

    Args:
        data_path: Path to JSONL file

    Returns:
        Dataset with prompt, chosen, rejected columns
    """
    data = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append({
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="DPO training for JAVIS model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preference data JSONL (prompt, chosen, rejected)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for adapter",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (KL penalty weight)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help="DPO loss type",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum total sequence length",
    )

    args = parser.parse_args()

    # Output path setup
    if args.output is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        args.output = f"models/{version}/adapter"

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("JAVIS DPO Training")
    print("=" * 50)
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Beta: {args.beta}")
    print(f"Loss type: {args.loss_type}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 50)

    # 1. Load preference data
    print("\n[1/5] Loading preference data...")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    dataset = load_preference_data(data_path)
    print(f"Loaded {len(dataset)} preference pairs")

    # 2. Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding

    # 3. Load model with 4-bit quantization
    print("\n[3/5] Loading model with 4-bit quantization...")
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

    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    # 4. DPO Training
    print("\n[4/5] Starting DPO training...")

    # DPO training configuration
    dpo_config = DPOConfig(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # 5. Save adapter
    print("\n[5/5] Saving adapter...")
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save metadata
    metadata = {
        "version": output_path.parent.name,
        "base_model": args.base_model,
        "created_at": datetime.now().isoformat(),
        "training_data": str(args.data),
        "training_config": {
            "method": "dpo",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_prompt_length": args.max_prompt_length,
            "max_length": args.max_length,
        },
        "dataset_size": len(dataset),
    }

    metadata_path = output_path.parent / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print("DPO Training complete!")
    print(f"Adapter saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
