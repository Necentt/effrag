"""
Process dragon-derec single-hop samples WITHOUT LLM calls.

Single-hop samples (1 unique document) are free to process:
  - Evidence text with answer match → FINISH (token labels via exact match)
  - Random non-matching paragraph → TERMINATE (all tokens = 0)

This gives us ~5600 labeler samples for $0.
"""

import json
import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)


def find_answer_tokens(
    chunk_words: list[str],
    answer: str,
    window_size: int = 150,
) -> list[int]:
    """
    Label tokens in chunk that match the answer via sliding window.
    Also labels tokens that overlap with common question-answer patterns.
    """
    labels = [0] * len(chunk_words)
    chunk_lower = [w.lower().strip(".,;:!?«»\"'()") for w in chunk_words]

    # Parse answer (may be JSON list)
    answers = []
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            answers = [str(a) for a in parsed]
        else:
            answers = [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        answers = [str(answer)]

    for ans in answers:
        ans_words = ans.split()
        ans_lower = [w.lower().strip(".,;:!?«»\"'()") for w in ans_words]

        if not ans_lower:
            continue

        # Sliding window search for answer span
        for i in range(len(chunk_lower) - len(ans_lower) + 1):
            match = True
            for j, aw in enumerate(ans_lower):
                if chunk_lower[i + j] != aw:
                    match = False
                    break
            if match:
                for j in range(len(ans_lower)):
                    labels[i + j] = 1
                break  # found first occurrence

        # If exact span not found, try individual word matching
        if sum(labels) == 0:
            pos = 0
            for aw in ans_lower:
                for j in range(pos, min(pos + window_size, len(chunk_lower))):
                    if chunk_lower[j] == aw and labels[j] == 0:
                        labels[j] = 1
                        pos = j + 1
                        break

    return labels


def process_single_hop_samples(
    dataset_split,
    corpus: Optional[list[dict]] = None,
    max_negatives_per_sample: int = 1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Process single-hop grounded samples from dragon-derec.

    Args:
        dataset_split: HuggingFace dataset split
        corpus: optional list of corpus documents for negative sampling
        max_negatives_per_sample: negatives per positive sample
        seed: random seed

    Returns:
        (positive_samples, negative_samples) — both in labeler JSONL format
    """
    rng = random.Random(seed)
    positives = []
    negatives = []

    # Collect all grounded single-hop
    single_hop = []
    for sample in dataset_split:
        if not sample["is_grounded"]:
            continue
        unique_ids = set(sample["found_ids"])
        if len(unique_ids) == 1:
            single_hop.append(sample)

    logger.info(f"Processing {len(single_hop)} single-hop samples")

    for sample in single_hop:
        question = sample["question"]
        ref_answer = sample["reference_answer"]
        evidence_texts = sample["evidence_texts"]
        found_ids = sample["found_ids"]

        # Take first unique evidence text
        evidence = evidence_texts[0] if evidence_texts else ""
        if not evidence.strip():
            continue

        # Token-label the evidence using answer matching
        chunk_words = evidence.split()
        token_labels = find_answer_tokens(chunk_words, ref_answer)

        # If no tokens matched, label the whole chunk as useful
        # (model still learns the FINISH signal)
        if sum(token_labels) == 0:
            # Try more aggressive matching — any word from answer in chunk
            try:
                parsed = json.loads(ref_answer)
                if isinstance(parsed, list):
                    ans_text = " ".join(str(a) for a in parsed)
                else:
                    ans_text = str(parsed)
            except (json.JSONDecodeError, TypeError):
                ans_text = str(ref_answer)

            ans_words_lower = set(
                w.lower().strip(".,;:!?«»\"'()")
                for w in ans_text.split()
                if len(w) > 2
            )
            for i, w in enumerate(chunk_words):
                if w.lower().strip(".,;:!?«»\"'()") in ans_words_lower:
                    token_labels[i] = 1

        positives.append({
            "question": question,
            "chunk": evidence,
            "token_labels": token_labels,
            "tag": "<FINISH>",
        })

        # Generate negative from corpus or other evidence
        if corpus:
            oracle_ids = set(str(fid) for fid in found_ids)
            candidates = [
                doc for doc in corpus
                if str(doc.get("id", "")) not in oracle_ids
            ]
            if candidates:
                neg_doc = rng.choice(candidates)
                neg_text = neg_doc.get("text", "")
                neg_words = neg_text.split()
                negatives.append({
                    "question": question,
                    "chunk": neg_text,
                    "token_labels": [0] * len(neg_words),
                    "tag": "<TERMINATE>",
                })

    logger.info(
        f"Single-hop: {len(positives)} FINISH + {len(negatives)} TERMINATE"
    )
    return positives, negatives


def process_and_save(
    dataset_split,
    corpus_path: str,
    output_dir: str,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Process single-hop and save to JSONL files.

    Args:
        dataset_split: HuggingFace dataset split
        corpus_path: path to dragon_corpus.jsonl
        output_dir: output directory

    Returns:
        (positives, negatives)
    """
    # Load corpus
    corpus = []
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        logger.info(f"Loaded corpus: {len(corpus)} documents")

    positives, negatives = process_single_hop_samples(
        dataset_split=dataset_split,
        corpus=corpus,
        seed=seed,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)

    pos_path = os.path.join(output_dir, "dragon_single_hop_positive.jsonl")
    with open(pos_path, "w", encoding="utf-8") as f:
        for s in positives:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    neg_path = os.path.join(output_dir, "dragon_single_hop_negative.jsonl")
    with open(neg_path, "w", encoding="utf-8") as f:
        for s in negatives:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {pos_path} and {neg_path}")
    return positives, negatives
