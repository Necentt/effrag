"""
Training script for the EfficientRAG Labeler model.

Usage:
    python -m EfficientRAG.training.train_labeler \
        --train_data data/efficient_rag/labeler/train.jsonl \
        --val_data data/efficient_rag/labeler/val.jsonl \
        --output_dir checkpoints/labeler \
        --model_name microsoft/deberta-v3-large \
        --num_labels 2 \
        --max_length 384 \
        --epochs 2 \
        --batch_size 32 \
        --lr 5e-6
"""

import argparse
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    Trainer,
    TrainingArguments,
)

from EfficientRAG.config import TAG_MAPPING_TWO, TAG_MAPPING_THREE
from EfficientRAG.data.labeler_dataset import LabelerDataset, compute_token_class_weights
from EfficientRAG.models.labeler import DebertaForSequenceTokenClassification


class LabelerTrainer(Trainer):
    """Custom Trainer that passes token class weights to the model."""

    def __init__(self, token_pos_weight: float = 1.0, token_neg_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.token_pos_weight = token_pos_weight
        self.token_neg_weight = token_neg_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_labels = inputs.pop("token_labels", None)
        sequence_labels = inputs.pop("sequence_labels", None)
        token_label_mask = inputs.pop("token_label_mask", None)

        outputs = model(
            **inputs,
            token_labels=token_labels,
            sequence_labels=sequence_labels,
            token_label_mask=token_label_mask,
            token_pos_weight=self.token_pos_weight,
            token_neg_weight=self.token_neg_weight,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute sequence-level accuracy and F1."""
    predictions, labels = eval_pred
    # predictions is (sequence_logits, token_logits) stacked
    # We only evaluate sequence classification here
    if isinstance(predictions, tuple):
        seq_logits = predictions[0]
    else:
        seq_logits = predictions

    seq_preds = np.argmax(seq_logits, axis=-1)
    acc = accuracy_score(labels, seq_preds)
    f1 = f1_score(labels, seq_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def train_labeler(
    train_data: str,
    val_data: Optional[str],
    output_dir: str,
    model_name: str = "microsoft/deberta-v3-large",
    num_labels: int = 2,
    max_length: int = 384,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 5e-6,
    warmup_steps: int = 200,
    weight_average: bool = True,
    fp16: bool = True,
):
    """Train the Labeler model."""

    tag_mapping = TAG_MAPPING_TWO if num_labels == 2 else TAG_MAPPING_THREE
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    train_dataset = LabelerDataset(
        data_path=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        tag_mapping=tag_mapping,
    )

    val_dataset = None
    if val_data:
        val_dataset = LabelerDataset(
            data_path=val_data,
            tokenizer=tokenizer,
            max_length=max_length,
            tag_mapping=tag_mapping,
        )

    # Compute class weights
    neg_w, pos_w = 1.0, 1.0
    if weight_average:
        print("Computing token class weights...")
        neg_w, pos_w = compute_token_class_weights(train_dataset)
        print(f"Token weights: neg={neg_w:.4f}, pos={pos_w:.4f}")

    # Initialize model
    config = DebertaV2Config.from_pretrained(model_name)
    config.num_sequence_labels = num_labels
    model = DebertaForSequenceTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=fp16,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=bool(val_dataset),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = LabelerTrainer(
        token_pos_weight=pos_w,
        token_neg_weight=neg_w,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Labeler model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientRAG Labeler")
    parser.add_argument("--train_data", required=True, help="Path to training JSONL")
    parser.add_argument("--val_data", default=None, help="Path to validation JSONL")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--num_labels", type=int, default=2, choices=[2, 3])
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--no_weight_average", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    train_labeler(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_labels=args.num_labels,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_average=not args.no_weight_average,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
