"""
Step 5: Final training data assembly.

Combines positive samples (CONTINUE/FINISH) and negative samples (TERMINATE)
into Labeler and Filter training files.
"""

import json
import logging
import os
from typing import Optional

from EfficientRAG.config import CONTINUE_TAG, FINISH_TAG, TERMINATE_TAG

logger = logging.getLogger(__name__)


def build_labeler_data(
    token_labeled_data: list[dict],
    negative_samples: list[dict],
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Build final Labeler training data.

    Positive samples: from token_labeled_data, tagged CONTINUE or FINISH.
    Negative samples: from negative_sampling, tagged TERMINATE.

    Args:
        token_labeled_data: output from token_labeling step
        negative_samples: output from negative_sampling step
        output_path: optional JSONL output path

    Returns:
        list of labeler training samples
    """
    labeler_data = []

    for sample in token_labeled_data:
        question = sample["question"]
        subs = sample.get("labeled_sub_questions", [])
        paragraphs = sample.get("supporting_paragraphs", [])

        for j, sq in enumerate(subs):
            pidx = sq.get("paragraph_idx", 0)
            if pidx < len(paragraphs):
                para = paragraphs[pidx]
                chunk_text = para.get("text", para.get("content", ""))
            else:
                chunk_words = sq.get("chunk_words", [])
                chunk_text = " ".join(chunk_words)

            token_labels = sq.get("token_labels", [])

            # Last sub-question gets FINISH tag, others get CONTINUE
            is_last = (j == len(subs) - 1)
            tag = FINISH_TAG if is_last else CONTINUE_TAG

            labeler_data.append({
                "question": question,
                "chunk": chunk_text,
                "token_labels": token_labels,
                "tag": tag,
            })

    # Add negatives
    labeler_data.extend(negative_samples)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in labeler_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(
            f"Saved {len(labeler_data)} labeler samples to {output_path} "
            f"({len(labeler_data) - len(negative_samples)} positive, "
            f"{len(negative_samples)} negative)"
        )

    return labeler_data


def build_filter_data(
    filter_samples: list[dict],
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Write Filter training data to JSONL.

    Args:
        filter_samples: output from next_hop_query_filtering step
        output_path: optional JSONL output path

    Returns:
        the same filter_samples list
    """
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in filter_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(filter_samples)} filter samples to {output_path}")

    return filter_samples


def run_full_synthesis(
    token_labeled_data: list[dict],
    negative_samples: list[dict],
    filter_samples: list[dict],
    output_dir: str,
):
    """
    Run the full assembly step, writing all training files.

    Output structure:
        {output_dir}/labeler/train.jsonl
        {output_dir}/filter/train.jsonl
    """
    labeler_path = os.path.join(output_dir, "labeler", "train.jsonl")
    filter_path = os.path.join(output_dir, "filter", "train.jsonl")

    build_labeler_data(token_labeled_data, negative_samples, labeler_path)
    build_filter_data(filter_samples, filter_path)

    logger.info(f"Full synthesis complete. Output in {output_dir}")
