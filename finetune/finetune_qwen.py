"""
Full Finetuning Script for Qwen2.5-7B on ToM Data

Hardware Requirements:
- GPU: 4x A100 (40GB) or 8x A100 (80GB) recommended
- RAM: 128GB+
- Disk: 100GB+

This script uses DeepSpeed ZeRO-3 for distributed training.
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention 2"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="../finetuning_data/train.jsonl",
        metadata={"help": "Path to training data file"}
    )
    val_file: str = field(
        default="../finetuning_data/val.jsonl",
        metadata={"help": "Path to validation data file"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class FinetuningArguments(TrainingArguments):
    output_dir: str = field(default="./qwen_finetuned")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: Optional[str] = field(default="deepspeed_config.json")
    report_to: str = field(default="tensorboard")


def format_chat_template(messages, tokenizer):
    """
    Format messages using Qwen's chat template.
    """
    # Qwen uses special tokens for chat format
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess the dataset for training.
    """
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for messages in examples["messages"]:
        # Format with chat template
        formatted_text = format_chat_template(messages, tokenizer)

        # Tokenize
        tokenized = tokenizer(
            formatted_text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        model_inputs["input_ids"].append(tokenized["input_ids"])
        model_inputs["attention_mask"].append(tokenized["attention_mask"])

        # Labels are the same as input_ids for causal LM
        model_inputs["labels"].append(tokenized["input_ids"].copy())

    return model_inputs


def main():
    # Parse arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = FinetuningArguments()

    logger.info("=" * 80)
    logger.info("Starting Qwen2.5-7B Full Finetuning")
    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Val file: {data_args.val_file}")
    logger.info(f"Output dir: {training_args.output_dir}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info("=" * 80)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",  # Important for training
    )

    # Qwen doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "eager",
    )

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Required for gradient checkpointing

    # Load datasets
    logger.info("Loading datasets...")
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "validation": data_args.val_file
        }
    )

    logger.info(f"Train samples: {len(raw_datasets['train'])}")
    logger.info(f"Validation samples: {len(raw_datasets['validation'])}")

    # Preprocess datasets
    logger.info("Preprocessing datasets...")

    def preprocess(examples):
        return preprocess_function(examples, tokenizer, data_args.max_seq_length)

    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing datasets",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
