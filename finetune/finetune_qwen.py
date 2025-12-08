"""
Full Finetuning Script for Qwen2.5-7B on ToM Data

Hardware Requirements:
- GPU: 1x H100 (94GB) or similar high-memory GPU
- RAM: 64GB+
- Disk: 100GB+

This script uses:
- Standard eager attention (flash-attn disabled for stability)
- Gradient checkpointing and BF16 for memory efficiency
- Single GPU training optimized for H100

For multi-GPU or lower memory setups, see deepspeed_backup/deepspeed_config.json
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
        default=False,
        metadata={"help": "Whether to use flash attention 2"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="./train_r2.jsonl",
        metadata={"help": "Path to training data file"}
    )
    val_file: str = field(
        default="./val_r2.jsonl",
        metadata={"help": "Path to validation data file"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_eval_samples: Optional[int] = field(
        default=100,  # Limit eval to 100 samples during training for speed
        metadata={"help": "Maximum number of validation samples (None = all)"}
    )


@dataclass
class FinetuningArguments(TrainingArguments):
    output_dir: str = field(default="./qwen_finetuned")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=4)
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
    eval_accumulation_steps: int = field(default=1)  # Process eval in smaller chunks
    save_total_limit: int = field(default=3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    # DeepSpeed disabled - single H100 has enough memory (94GB)
    # deepspeed: Optional[str] = field(default="deepspeed_config.json")
    report_to: str = field(default="tensorboard")
    # Generation config for evaluation
    generation_max_length: int = field(default=10)  # Max tokens to generate during eval
    predict_with_generate: bool = field(default=True)  # Use generation during eval


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
    Only compute loss on assistant responses, not on the user prompt.
    """
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for messages in examples["messages"]:
        # Separate user and assistant messages
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "[[A]]"}]

        # Tokenize the user part (we'll mask this in labels)
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        user_formatted = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True  # Adds the assistant prompt token
        )
        user_tokenized = tokenizer(
            user_formatted,
            add_special_tokens=False,
            return_tensors=None
        )
        user_input_ids = user_tokenized["input_ids"]

        # Tokenize the full conversation (user + assistant)
        full_formatted = format_chat_template(messages, tokenizer)
        full_tokenized = tokenizer(
            full_formatted,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        # Create labels: mask user prompt with -100, keep assistant response
        labels = input_ids.copy()
        user_prompt_len = len(user_input_ids)

        # Mask all tokens up to (and including) the user prompt
        # Only compute loss on assistant's response
        for i in range(min(user_prompt_len, len(labels))):
            labels[i] = -100

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

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

    # Optionally subsample validation set to reduce memory usage
    if data_args.max_eval_samples is not None and len(raw_datasets['validation']) > data_args.max_eval_samples:
        logger.info(f"Subsampling validation set: {data_args.max_eval_samples} / {len(raw_datasets['validation'])}")
        raw_datasets['validation'] = raw_datasets['validation'].select(range(data_args.max_eval_samples))

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

    # Clear CUDA cache to maximize available memory
    torch.cuda.empty_cache()

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        # compute_metrics disabled - will eval with separate script after training
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"Training samples: {len(tokenized_datasets['train'])}")
    logger.info(f"Validation samples: {len(tokenized_datasets['validation'])}")
    logger.info(f"Effective train batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Eval batch size: {training_args.per_device_eval_batch_size}")

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
    logger.info("")
    logger.info("To evaluate the model, use the evaluation scripts in the parent directory")
    logger.info("with the same format as other model evaluations.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
