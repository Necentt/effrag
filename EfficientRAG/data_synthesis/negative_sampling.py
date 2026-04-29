"""
Step 4: Negative sampling — retrieve hard negatives for Labeler training.

For each sub-question's filtered next-hop query, retrieve top passages
and select the first one that is NOT a supporting paragraph.
These negatives are tagged <TERMINATE> in the training data.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def find_hard_negative(
    query: str,
    retriever_fn,
    oracle_ids: set[str],
    top_k: int = 10,
) -> Optional[dict]:
    """
    Retrieve top-k passages and return the first non-oracle passage.

    Args:
        query: search query
        retriever_fn: callable(query: str, top_k: int) -> list[dict]
            Each dict has 'id', 'text', optionally 'title', 'score'
        oracle_ids: set of ground-truth supporting passage IDs
        top_k: number of passages to retrieve

    Returns:
        A hard negative passage dict, or None if all are oracles
    """
    passages = retriever_fn(query, top_k)
    for p in passages:
        pid = str(p.get("id", ""))
        if pid not in oracle_ids:
            return p
    return None


def build_negative_samples(
    token_labeled_data: list[dict],
    retriever_fn,
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Build hard negative samples for Labeler training.

    Args:
        token_labeled_data: output from token_labeling step
        retriever_fn: callable(query, top_k) -> list[dict]
        output_path: optional JSONL output path

    Returns:
        list of negative samples: {question, chunk, token_labels=[], tag=<TERMINATE>}
    """
    negatives = []

    for i, sample in enumerate(token_labeled_data):
        question = sample["question"]
        subs = sample.get("labeled_sub_questions", [])

        # Collect oracle paragraph IDs
        oracle_ids = set()
        paragraphs = sample.get("supporting_paragraphs", [])
        for p in paragraphs:
            pid = p.get("id", p.get("title", ""))
            oracle_ids.add(str(pid))

        for sq in subs:
            # Use the sub-question as the retrieval query
            query = sq.get("sub_question", question)

            negative = find_hard_negative(
                query=query,
                retriever_fn=retriever_fn,
                oracle_ids=oracle_ids,
            )

            if negative:
                neg_text = negative.get("text", negative.get("content", ""))
                neg_words = neg_text.split()
                negatives.append({
                    "question": question,
                    "chunk": neg_text,
                    "token_labels": [0] * len(neg_words),
                    "tag": "<TERMINATE>",
                })

        if (i + 1) % 100 == 0:
            logger.info(f"Negative sampling: {i + 1}/{len(token_labeled_data)}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for n in negatives:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(negatives)} negative samples to {output_path}")

    return negatives
