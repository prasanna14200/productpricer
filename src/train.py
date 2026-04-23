"""Model training script for fine-tuning GPT-2 on price prediction."""

import argparse
import os
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from src.loaders import ItemLoader


def load_training_data(train_file: str, val_file: str) -> Dict[str, Any]:
    """Load training and validation datasets.

    Args:
        train_file: Path to training data JSONL file
        val_file: Path to validation data JSONL file

    Returns:
        Dictionary with train and validation datasets
    """
    try:
        train_data = ItemLoader.load_from_jsonl(train_file)
        val_data = ItemLoader.load_from_jsonl(val_file)

        # Convert to datasets format
        train_dataset = load_dataset("json", data_files=train_file, split="train")
        val_dataset = load_dataset("json", data_files=val_file, split="train")

        return {
            "train": train_dataset,
            "validation": val_dataset
        }
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def tokenize_function(examples, tokenizer):
    """Tokenize the examples for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance

    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


def train_model(train_file: str, val_file: str, output_dir: str = "models/fine_tuned"):
    """Train the GPT-2 model on price prediction data.

    Args:
        train_file: Path to training data
        val_file: Path to validation data
        output_dir: Output directory for trained model
    """
    print("Loading model and tokenizer...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading training data...")
    datasets = load_training_data(train_file, val_file)
    if not datasets:
        print("Failed to load training data. Exiting.")
        return

    print("Tokenizing data...")
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for price prediction")
    parser.add_argument("--train-file", default="data/processed/fine_tune_train.jsonl",
                       help="Path to training data JSONL file")
    parser.add_argument("--val-file", default="data/processed/fine_tune_validation.jsonl",
                       help="Path to validation data JSONL file")
    parser.add_argument("--output-dir", default="models/fine_tuned",
                       help="Output directory for trained model")

    args = parser.parse_args()

    # Check if training files exist
    if not os.path.exists(args.train_file):
        print(f"Training file {args.train_file} not found. Running data curation first...")
        from src.data_curation import main as curate_main
        curate_main()

    train_model(args.train_file, args.val_file, args.output_dir)


if __name__ == "__main__":
    main()