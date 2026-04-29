"""
Step 3: Next-hop query construction and Filter training data generation.

For sub-questions with dependencies, constructs the "known information" from
answers to dependent sub-questions, then uses the LLM to produce a refactored
query. The word-level labels for the Filter are computed by matching the
refactored query words against the concatenated query+info input.
"""

import json
import logging
import os
from typing import Optional

import spacy

from EfficientRAG.data_synthesis.token_labeling import _load_spacy, split_words
from EfficientRAG.prompts.token_labeling import QUERY_FILTERING_PROMPT

logger = logging.getLogger(__name__)


def build_query_info_text(question: str, info_tokens: list[str]) -> str:
    """
    Build the Filter's input text: "Query: {question} Info: {info1}; Info: {info2}; ..."

    For simplified format: just concatenate question + extracted info.
    """
    info_str = "; ".join(info_tokens) if info_tokens else ""
    if info_str:
        return f"{question} {info_str}"
    return question


def label_filter_words(
    query_info_words: list[str],
    filtered_query_words: list[str],
    window_size: int = 50,
) -> list[int]:
    """
    Label which words in query_info should be kept to form the filtered query.

    Uses a sliding window approach similar to token_labeling.label_words_in_chunk.

    Args:
        query_info_words: words in the concatenated query+info text
        filtered_query_words: words in the target filtered query
        window_size: matching window size

    Returns:
        list of 0/1 labels, one per query_info word
    """
    labels = [0] * len(query_info_words)
    qi_lower = [w.lower() for w in query_info_words]

    pos = 0
    for fw in filtered_query_words:
        fw_lower = fw.lower()
        window_end = min(pos + window_size, len(qi_lower))

        found = False
        for j in range(pos, window_end):
            if qi_lower[j] == fw_lower and labels[j] == 0:
                labels[j] = 1
                pos = j + 1
                found = True
                break

        if not found:
            # Backward search
            for j in range(pos - 1, max(0, pos - window_size) - 1, -1):
                if qi_lower[j] == fw_lower and labels[j] == 0:
                    labels[j] = 1
                    found = True
                    break

    return labels


def construct_filter_sample(
    question: str,
    sub_question: dict,
    all_sub_questions: list[dict],
    llm_fn,
    nlp,
) -> Optional[dict]:
    """
    Construct one Filter training sample for a sub-question with dependencies.

    Args:
        question: original multi-hop question
        sub_question: the current sub-question dict
        all_sub_questions: all sub-questions for this sample
        llm_fn: callable(prompt: str) -> str
        nlp: spaCy model

    Returns:
        {"query_info": ..., "token_labels": [...]} or None
    """
    depends_on = sub_question.get("depends_on", [])
    if not depends_on:
        return None  # No dependencies = no filtering needed

    # Collect known info from dependent sub-questions
    info_parts = []
    for dep_idx in depends_on:
        if dep_idx < len(all_sub_questions):
            dep = all_sub_questions[dep_idx]
            # Get useful tokens from the dependent chunk
            useful = dep.get("token_labels", [])
            words = dep.get("chunk_words", [])
            useful_words = [w for w, l in zip(words, useful) if l == 1]
            if useful_words:
                info_parts.append(" ".join(useful_words))

    if not info_parts:
        return None

    # Build query_info text
    query_info = build_query_info_text(question, info_parts)
    query_info_words = split_words(query_info, nlp)

    # Ask LLM to produce filtered query
    known_info = "; ".join(info_parts)
    prompt = QUERY_FILTERING_PROMPT.format(
        question=question,
        known_info=known_info,
    )
    try:
        filtered_response = llm_fn(prompt)
    except Exception as e:
        logger.warning(f"LLM failed for filter labeling, skipping: {e}")
        return None
    filtered_query_words = split_words(filtered_response.strip(), nlp)

    if not filtered_query_words:
        return None

    # Align filtered query words to query_info words
    labels = label_filter_words(query_info_words, filtered_query_words)

    return {
        "query_info": query_info,
        "token_labels": labels,
    }


def construct_filter_dataset(
    token_labeled_data: list[dict],
    llm_fn,
    spacy_model: str = "ru_core_news_sm",
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Construct Filter training data from token-labeled decomposed samples.

    Args:
        token_labeled_data: output from token_labeling step
        llm_fn: callable(prompt: str) -> str
        spacy_model: spaCy model name
        output_path: optional JSONL output path

    Returns:
        list of filter training samples
    """
    nlp = _load_spacy(spacy_model)
    filter_samples = []

    for i, sample in enumerate(token_labeled_data):
        question = sample["question"]
        subs = sample.get("labeled_sub_questions", [])

        for sq in subs:
            result = construct_filter_sample(
                question=question,
                sub_question=sq,
                all_sub_questions=subs,
                llm_fn=llm_fn,
                nlp=nlp,
            )
            if result:
                filter_samples.append(result)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(token_labeled_data)} for filter data")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in filter_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(filter_samples)} filter samples to {output_path}")

    return filter_samples
