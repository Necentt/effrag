"""
Download pre-synthesized HotpotQA training data from the EfficientRAG authors,
or fall back to generating simple training data from raw HotpotQA.

The authors' data is at: https://box.nju.edu.cn/f/a86b512077c7489b8da3/
If download fails, we generate labeler data from HotpotQA using
exact-match heuristics (no LLM needed).
"""

import json
import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)


def generate_hotpotqa_labeler_data(
    num_samples: int = 10000,
    output_dir: str = "data/efficient_rag",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Generate labeler + filter training data from raw HotpotQA using heuristics.
    No LLM calls — uses exact-match token labeling.

    Args:
        num_samples: number of HotpotQA samples to process
        output_dir: directory to save results
        seed: random seed

    Returns:
        (labeler_samples, filter_samples)
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    logger.info(f"Loading HotpotQA (first {num_samples} from train)...")
    hotpot = load_dataset(
        "hotpot_qa", "distractor",
        split=f"train[:{num_samples}]",
        trust_remote_code=True,
    )

    labeler_samples = []
    filter_samples = []

    for sample in hotpot:
        question = sample["question"]
        answer = sample["answer"]
        supporting_titles = set(sample["supporting_facts"]["title"])

        # Build paragraphs
        all_paragraphs = []
        supporting = []
        distractors = []

        for title, sentences in zip(
            sample["context"]["title"], sample["context"]["sentences"]
        ):
            text = " ".join(sentences)
            para = {"title": title, "text": text}
            all_paragraphs.append(para)
            if title in supporting_titles:
                supporting.append(para)
            else:
                distractors.append(para)

        if not supporting:
            continue

        # --- Labeler positive samples ---
        for i, para in enumerate(supporting):
            chunk_text = para["text"]
            chunk_words = chunk_text.split()

            # Token labels via exact match with answer
            token_labels = _match_answer_tokens(chunk_words, answer)

            is_last = (i == len(supporting) - 1)
            tag = "<FINISH>" if is_last else "<CONTINUE>"

            labeler_samples.append({
                "question": question,
                "chunk": chunk_text,
                "token_labels": token_labels,
                "tag": tag,
            })

        # --- Labeler negative sample (distractor) ---
        if distractors:
            neg = rng.choice(distractors)
            neg_words = neg["text"].split()
            labeler_samples.append({
                "question": question,
                "chunk": neg["text"],
                "token_labels": [0] * len(neg_words),
                "tag": "<TERMINATE>",
            })

        # --- Filter sample (if 2+ supporting paragraphs) ---
        if len(supporting) >= 2:
            # Known info from first paragraph
            first_text = supporting[0]["text"]
            first_words = first_text.split()
            first_labels = _match_answer_tokens(first_words, answer)
            useful = [w for w, l in zip(first_words, first_labels) if l == 1]
            info_str = " ".join(useful) if useful else answer

            # Query info = question + info
            query_info = f"{question} {info_str}"
            qi_words = query_info.split()

            # Target: keep question words + info words that aren't redundant
            # Simple heuristic: keep all words
            filter_labels = [1] * len(qi_words)

            filter_samples.append({
                "query_info": query_info,
                "token_labels": filter_labels,
            })

    # Save
    os.makedirs(output_dir, exist_ok=True)

    lab_path = os.path.join(output_dir, "hotpotqa_labeler.jsonl")
    with open(lab_path, "w", encoding="utf-8") as f:
        for s in labeler_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    fil_path = os.path.join(output_dir, "hotpotqa_filter.jsonl")
    with open(fil_path, "w", encoding="utf-8") as f:
        for s in filter_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(
        f"HotpotQA data: {len(labeler_samples)} labeler, "
        f"{len(filter_samples)} filter → {output_dir}"
    )
    return labeler_samples, filter_samples


def _match_answer_tokens(
    chunk_words: list[str], answer: str
) -> list[int]:
    """Match answer words in chunk via exact match."""
    labels = [0] * len(chunk_words)
    chunk_lower = [w.lower().strip(".,;:!?\"'()") for w in chunk_words]
    ans_words = answer.split()
    ans_lower = [w.lower().strip(".,;:!?\"'()") for w in ans_words]

    # Try exact span match first
    for i in range(len(chunk_lower) - len(ans_lower) + 1):
        if chunk_lower[i : i + len(ans_lower)] == ans_lower:
            for j in range(len(ans_lower)):
                labels[i + j] = 1
            return labels

    # Fallback: individual word match
    pos = 0
    for aw in ans_lower:
        for j in range(pos, len(chunk_lower)):
            if chunk_lower[j] == aw and labels[j] == 0:
                labels[j] = 1
                pos = j + 1
                break

    return labels
