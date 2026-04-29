"""
Step 2: Token-level labeling using LLM + SpaCy alignment.

For each (sub-question, chunk) pair, the LLM extracts essential words,
then we align those words back to the chunk text to produce binary per-word labels.
"""

import json
import logging
import os
from typing import Optional

import spacy

from EfficientRAG.prompts.token_labeling import TOKEN_LABELING_PROMPT

logger = logging.getLogger(__name__)


def _load_spacy(model_name: str = "ru_core_news_sm"):
    """Load spaCy model, downloading if needed."""
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


def split_words(text: str, nlp) -> list[str]:
    """Split text into words using spaCy tokenizer."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def label_words_in_chunk(
    chunk_words: list[str],
    extracted_words: list[str],
    window_size: int = 150,
) -> list[int]:
    """
    Align extracted words to chunk words and produce binary labels.

    Uses a sliding window approach for robust matching: for each extracted word,
    find its best match within a window of the current position in the chunk.

    Args:
        chunk_words: words in the chunk (from spaCy tokenization)
        extracted_words: words identified as useful by the LLM
        window_size: size of the matching window

    Returns:
        list of 0/1 labels, one per chunk word
    """
    labels = [0] * len(chunk_words)
    chunk_lower = [w.lower() for w in chunk_words]

    pos = 0  # current position in chunk
    for ext_word in extracted_words:
        ext_lower = ext_word.lower()
        # Search within window [pos, pos + window_size]
        window_end = min(pos + window_size, len(chunk_lower))

        found = False
        for j in range(pos, window_end):
            if chunk_lower[j] == ext_lower and labels[j] == 0:
                labels[j] = 1
                pos = j + 1
                found = True
                break

        if not found:
            # Backward search
            window_start = max(0, pos - window_size)
            for j in range(pos - 1, window_start - 1, -1):
                if chunk_lower[j] == ext_lower and labels[j] == 0:
                    labels[j] = 1
                    found = True
                    break

    return labels


def label_tokens_for_sample(
    question: str,
    answer: str,
    paragraph: str,
    llm_fn,
    nlp,
) -> tuple[list[str], list[int]]:
    """
    Get token-level labels for a (question, paragraph) pair.

    Args:
        question: sub-question text
        answer: answer to the sub-question
        paragraph: supporting paragraph text
        llm_fn: callable(prompt: str) -> str
        nlp: spaCy model

    Returns:
        (chunk_words, labels) where labels[i] = 1 if word i is useful
    """
    # Split paragraph into words first (no LLM needed)
    chunk_words = split_words(paragraph, nlp)

    prompt = TOKEN_LABELING_PROMPT.format(
        question=question,
        answer=answer,
        paragraph=paragraph,
    )

    try:
        response = llm_fn(prompt)
        extracted_words = [w.strip() for w in response.split(",") if w.strip()]
        labels = label_words_in_chunk(chunk_words, extracted_words)
    except Exception as e:
        # Fallback: label by answer matching if LLM fails (e.g. content filter)
        logger.warning(f"LLM failed for token labeling, using fallback: {e}")
        ans_words = answer.lower().split()
        chunk_lower = [w.lower().strip(".,;:!?«»\"'()") for w in chunk_words]
        labels = [0] * len(chunk_words)
        for aw in ans_words:
            aw_clean = aw.strip(".,;:!?«»\"'()")
            for i, cw in enumerate(chunk_lower):
                if cw == aw_clean and labels[i] == 0:
                    labels[i] = 1
                    break

    return chunk_words, labels


def label_tokens_dataset(
    decomposed_data: list[dict],
    llm_fn,
    spacy_model: str = "ru_core_news_sm",
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Run token labeling for all decomposed samples.

    Args:
        decomposed_data: output from query_decompose step
        llm_fn: callable(prompt: str) -> str
        spacy_model: spaCy model name
        output_path: optional path to save results as JSONL

    Returns:
        list of labeled samples
    """
    nlp = _load_spacy(spacy_model)
    results = []

    for i, sample in enumerate(decomposed_data):
        sub_questions = sample.get("sub_questions", [])
        paragraphs = sample.get("supporting_paragraphs", [])

        labeled_subs = []
        for sq in sub_questions:
            pidx = sq.get("paragraph_idx", 0)
            if pidx is None:
                pidx = 0
            if pidx < len(paragraphs):
                paragraph = paragraphs[pidx]
                para_text = paragraph.get("text", paragraph.get("content", ""))
            else:
                continue

            chunk_words, labels = label_tokens_for_sample(
                question=sq["sub_question"],
                answer=sq["answer"],
                paragraph=para_text,
                llm_fn=llm_fn,
                nlp=nlp,
            )

            labeled_subs.append({
                "sub_question": sq["sub_question"],
                "answer": sq["answer"],
                "paragraph_idx": pidx,
                "depends_on": sq.get("depends_on", []),
                "chunk_words": chunk_words,
                "token_labels": labels,
            })

        result = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "labeled_sub_questions": labeled_subs,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            logger.info(f"Token-labeled {i + 1}/{len(decomposed_data)} samples")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved token-labeled data to {output_path}")

    return results
