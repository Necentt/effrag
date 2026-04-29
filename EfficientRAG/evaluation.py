"""
Evaluation utilities for EfficientRAG.

Metrics:
  - Retrieval: Recall (chunk recall against gold supporting paragraphs)
  - QA: Exact Match (EM), Token-level F1
"""

import re
import string
from collections import Counter
from typing import Optional


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def chunk_recall(
    retrieved_chunks: list[dict],
    gold_paragraphs: list[dict],
    match_field: str = "text",
) -> float:
    """
    Compute recall: fraction of gold paragraphs present in retrieved chunks.

    Matching is done by normalized text overlap (at least 80% token overlap).
    """
    if not gold_paragraphs:
        return 1.0

    found = 0
    for gold in gold_paragraphs:
        gold_text = normalize_answer(gold.get(match_field, ""))
        gold_tokens = set(gold_text.split())

        for chunk in retrieved_chunks:
            chunk_text = normalize_answer(chunk.get(match_field, chunk.get("text", "")))
            chunk_tokens = set(chunk_text.split())

            if not gold_tokens:
                continue
            overlap = len(gold_tokens & chunk_tokens) / len(gold_tokens)
            if overlap >= 0.8:
                found += 1
                break

    return found / len(gold_paragraphs)


def evaluate_retrieval(results: list[dict]) -> dict:
    """
    Evaluate retrieval quality across a dataset.

    Args:
        results: list of {
            "retrieved_chunks": [{"text": ...}, ...],
            "gold_paragraphs": [{"text": ...}, ...],
        }

    Returns:
        {"recall": ..., "avg_chunks": ...}
    """
    recalls = []
    total_chunks = 0

    for r in results:
        rc = chunk_recall(r["retrieved_chunks"], r["gold_paragraphs"])
        recalls.append(rc)
        total_chunks += len(r["retrieved_chunks"])

    return {
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "avg_chunks": total_chunks / len(results) if results else 0.0,
    }


def evaluate_qa(predictions: list[str], golds: list[str]) -> dict:
    """
    Evaluate QA quality.

    Args:
        predictions: list of predicted answers
        golds: list of gold answers

    Returns:
        {"exact_match": ..., "f1": ...}
    """
    ems = [exact_match(p, g) for p, g in zip(predictions, golds)]
    f1s = [token_f1(p, g) for p, g in zip(predictions, golds)]

    return {
        "exact_match": sum(ems) / len(ems) if ems else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }
