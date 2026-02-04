import os
import argparse
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# -----------------------------
# Main training function
# -----------------------------
def main(args):
    print("🚀 Starting iPhone Sentiment Training")

    # Detect environment
    is_sagemaker = os.environ.get("SM_TRAINING_ENV") is not None

    # Resolve paths
    train_file = args.train_file
    validation_file = args.validation_file

    if is_sagemaker:
        output_dir = "/opt/ml/model"
    else:
        output_dir = "./model_output"

    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Train file: {train_file}")
    print(f"📂 Validation file: {validation_file}")
    print(f"📦 Output dir: {output_dir}")

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )

    # -----------------------------
    # Tokenizer & model
    # -----------------------------
    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
        fp16=False,  # SAFE for Python 3.12
        disable_tqdm=False,
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Save model
    # -----------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete")
    print(f"📦 Model saved to: {output_dir}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to train.csv",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        required=True,
        help="Path to validation.csv",
    )

    args = parser.parse_args()
    main(args)
