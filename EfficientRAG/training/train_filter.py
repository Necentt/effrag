"""
Training script for the EfficientRAG Filter model.

Usage:
    python -m EfficientRAG.training.train_filter \
        --train_data data/efficient_rag/filter/train.jsonl \
        --val_data data/efficient_rag/filter/val.jsonl \
        --output_dir checkpoints/filter \
        --model_name microsoft/mdeberta-v3-base \
        --max_length 128 \
        --epochs 2 \
        --batch_size 32 \
        --lr 1e-5
"""

import argparse
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2ForTokenClassification,
    Trainer,
    TrainingArguments,
)

from EfficientRAG.data.filter_dataset import FilterDataset, compute_filter_class_weights


class FilterTrainer(Trainer):
    """Custom trainer for the Filter with weighted token loss."""

    def __init__(
        self, pos_weight: float = 1.0, neg_weight: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_labels = inputs.pop("token_labels")
        token_label_mask = inputs.pop("token_label_mask")

        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq, 2]

        # Flatten and mask
        flat_logits = logits.view(-1, 2)
        flat_labels = token_labels.view(-1)
        flat_mask = token_label_mask.view(-1)

        active_logits = flat_logits[flat_mask]
        active_labels = flat_labels[flat_mask]

        weight = torch.tensor(
            [self.neg_weight, self.pos_weight],
            device=active_logits.device,
            dtype=active_logits.dtype,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute token-level metrics for the Filter."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    flat_preds = preds.flatten()
    flat_labels = labels.flatten()
    mask = flat_labels != -100
    flat_preds = flat_preds[mask]
    flat_labels = flat_labels[mask]

    return {
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "f1": f1_score(flat_labels, flat_preds, average="binary", zero_division=0),
        "precision": precision_score(flat_labels, flat_preds, zero_division=0),
        "recall": recall_score(flat_labels, flat_preds, zero_division=0),
    }


def train_filter(
    train_data: str,
    val_data: Optional[str],
    output_dir: str,
    model_name: str = "microsoft/mdeberta-v3-base",
    max_length: int = 128,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-5,
    warmup_steps: int = 200,
    weight_average: bool = True,
    fp16: bool = True,
):
    """Train the Filter model."""

    # use_fast=False — для mdeberta-v3 fast tokenizer ломает byte fallback
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_dataset = FilterDataset(
        data_path=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = None
    if val_data:
        val_dataset = FilterDataset(
            data_path=val_data,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # Compute class weights
    neg_w, pos_w = 1.0, 1.0
    if weight_average:
        print("Computing filter token class weights...")
        neg_w, pos_w = compute_filter_class_weights(train_dataset)
        print(f"Filter weights: neg={neg_w:.4f}, pos={pos_w:.4f}")

    # Initialize model
    config = DebertaV2Config.from_pretrained(model_name)
    config.num_labels = 2
    model = DebertaV2ForTokenClassification.from_pretrained(
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
        metric_for_best_model="f1" if val_dataset else None,
        greater_is_better=True if val_dataset else None,
        label_names=["token_labels"],
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = FilterTrainer(
        pos_weight=pos_w,
        neg_weight=neg_w,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if val_dataset else None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Filter model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientRAG Filter")
    parser.add_argument("--train_data", required=True, help="Path to training JSONL")
    parser.add_argument("--val_data", default=None, help="Path to validation JSONL")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--no_weight_average", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    train_filter(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
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
