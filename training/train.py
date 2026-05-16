"""
LoRA fine-tune Qwen2.5-1.5B on the group chat training data.

This script is designed to run on Google Colab (free T4 GPU).

------------------------------------------------------------------
BEFORE RUNNING -- install dependencies in a Colab cell:

    !pip install unsloth
    !pip install --no-deps trl peft accelerate bitsandbytes
    !pip install pyyaml datasets

Then upload to Colab (or mount Drive) and run:
    !python training/train.py
    !python training/train.py --config training/config.yaml
------------------------------------------------------------------

What this script does:
  1. Loads hyperparameters from config.yaml
  2. Loads Qwen2.5-1.5B in 4-bit (fits in free Colab T4 VRAM)
  3. Attaches LoRA adapters to attention layers
  4. Loads train.txt and builds a HuggingFace dataset
  5. Runs causal language model training with SFTTrainer
  6. Saves the LoRA adapter to model/adapter/

Only the adapter weights are saved (~50-150MB).
The base model stays unchanged on HuggingFace.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import yaml


# Config loader
def load_config(path="training/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Dataset builder
def build_dataset(train_file, separator="<|endoftext|>"):
    """
    Read train.txt, split on the separator, return a HuggingFace Dataset.
    Each element in the dataset is one conversation window (one training example).
    """
    from datasets import Dataset

    text = Path(train_file).read_text(encoding="utf-8")
    examples = [ex.strip() for ex in text.split(separator) if ex.strip()]

    print("  Loaded {:,} training examples from {}".format(len(examples), train_file))
    print("  Sample (first example, truncated to 200 chars):")
    print("  " + examples[0][:200].replace("\n", " / ") + "...")

    return Dataset.from_dict({"text": examples})


# Model + LoRA setup
def load_model_and_tokenizer(cfg):
    """
    Load the base model with 4-bit quantization and attach LoRA adapters.
    Uses unsloth for 2x faster training.
    """
    from unsloth import FastLanguageModel

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    print("\nLoading base model: {}".format(model_cfg["name"]))
    print("  4-bit quantization: {}".format(model_cfg["load_in_4bit"]))
    print("  Max sequence length: {}".format(model_cfg["max_seq_length"]))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,  # auto: float16 on T4, bfloat16 on A100
    )

    print("\nAttaching LoRA adapters:")
    print("  Rank (r)         : {}".format(lora_cfg["r"]))
    print("  Alpha            : {}".format(lora_cfg["alpha"]))
    print("  Target modules   : {}".format(lora_cfg["target_modules"]))
    print("  Dropout          : {}".format(lora_cfg["dropout"]))

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",  # saves VRAM at slight speed cost
        random_state=cfg["training"]["seed"],
    )

    # Show trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        "\n  Trainable parameters : {:,}  ({:.2f}% of total)".format(
            trainable, 100 * trainable / total
        )
    )
    print("  Frozen parameters    : {:,}".format(total - trainable))

    return model, tokenizer


# Trainer setup
def build_trainer(model, tokenizer, dataset, cfg):
    """
    Configure and return an SFTTrainer for causal language modelling.
    SFTTrainer (from trl) wraps HuggingFace Trainer with sensible defaults
    for supervised/self-supervised fine-tuning.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,  # keep only the last 2 checkpoints
        seed=train_cfg["seed"],
        fp16=True,  # mixed precision on T4
        bf16=False,
        report_to="none",  # disable wandb/tensorboard
        dataloader_num_workers=0,  # Colab works best with 0
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        packing=data_cfg["packing"],
        args=training_args,
    )

    return trainer


# Training summary
def print_training_plan(cfg, dataset):
    train_cfg = cfg["training"]
    eff_batch = train_cfg["batch_size"] * train_cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(dataset) / eff_batch)
    total_steps = steps_per_epoch * train_cfg["epochs"]

    print("\n" + "=" * 50)
    print("  TRAINING PLAN")
    print("=" * 50)
    print("  Model          : {}".format(cfg["model"]["name"]))
    print("  LoRA rank      : {}".format(cfg["lora"]["r"]))
    print("  Examples       : {:,}".format(len(dataset)))
    print("  Epochs         : {}".format(train_cfg["epochs"]))
    print("  Effective batch: {}".format(eff_batch))
    print("  Steps/epoch    : ~{}".format(steps_per_epoch))
    print("  Total steps    : ~{}".format(total_steps))
    print("  Learning rate  : {}".format(train_cfg["learning_rate"]))
    print("  Output dir     : {}".format(train_cfg["output_dir"]))
    print("=" * 50 + "\n")


# Save adapter
def save_adapter(model, tokenizer, output_dir):
    """
    Save only the LoRA adapter weights -- not the full model.
    The base model stays on HuggingFace and is loaded separately at inference.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Show what was saved and how large
    saved_files = list(out.iterdir())
    total_mb = sum(f.stat().st_size for f in saved_files if f.is_file()) / (1024 * 1024)

    print("\nAdapter saved to: {}".format(output_dir))
    print("  Files saved:")
    for f in sorted(saved_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print("    {} ({:.1f} MB)".format(f.name, size_mb))
    print("  Total adapter size: {:.1f} MB".format(total_mb))
    print("\nNext: python training/sample.py")


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-1.5B on group chat data."
    )
    parser.add_argument(
        "--config",
        default="training/config.yaml",
        help="Path to config.yaml (default: training/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and data, print the training plan, then exit without training",
    )
    args = parser.parse_args()

    # -- Load config
    print("Loading config: {}".format(args.config))
    cfg = load_config(args.config)

    # -- Build dataset
    print("\nBuilding dataset...")
    dataset = build_dataset(
        cfg["data"]["train_file"],
        cfg["data"]["separator"],
    )

    # -- Load model + LoRA
    model, tokenizer = load_model_and_tokenizer(cfg)

    # -- Print plan
    print_training_plan(cfg, dataset)

    if args.dry_run:
        print("Dry run complete. Exiting without training.")
        return

    # -- Build trainer
    print("Setting up trainer...")
    trainer = build_trainer(model, tokenizer, dataset, cfg)

    # -- Train
    print("Starting training...\n")
    result = trainer.train()

    # -- Print final loss
    final_loss = result.training_loss
    print("\n" + "=" * 50)
    print("  Training complete")
    print("  Final loss : {:.4f}".format(final_loss))
    if final_loss < 1.5:
        print("  Loss looks good -- the model has absorbed the style well.")
    elif final_loss < 2.5:
        print("  Loss is reasonable. Consider 1-2 more epochs if time allows.")
    else:
        print("  Loss is still high. Try increasing epochs or checking your data.")
    print("=" * 50)

    # -- Save adapter
    save_adapter(model, tokenizer, cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
