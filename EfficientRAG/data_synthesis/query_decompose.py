"""
Step 1: Multi-hop question decomposition using LLM.

Decomposes complex multi-hop questions into ordered single-hop sub-questions,
each linked to one supporting paragraph.
"""

import json
import logging
import os
from typing import Optional

from EfficientRAG.prompts.decompose import DATASET_PROMPTS, HOTPOTQA_DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)


def decompose_question(
    question: str,
    answer: str,
    supporting_paragraphs: list[dict],
    llm_fn,
    dataset_name: str = "dragon-derec",
) -> list[dict]:
    """
    Decompose a multi-hop question into single-hop sub-questions.

    Args:
        question: the multi-hop question
        answer: the gold answer
        supporting_paragraphs: list of {"title": ..., "text": ...}
        llm_fn: callable(prompt: str) -> str
        dataset_name: dataset name for prompt selection

    Returns:
        list of {sub_question, answer, paragraph_idx, depends_on}
    """
    prompt_template = DATASET_PROMPTS.get(dataset_name, HOTPOTQA_DECOMPOSE_PROMPT)

    paragraphs_text = ""
    for i, p in enumerate(supporting_paragraphs):
        title = p.get("title", f"Параграф {i}")
        text = p.get("text", p.get("content", ""))
        paragraphs_text += f"[{i}] {title}: {text}\n\n"

    prompt = prompt_template.format(
        question=question,
        answer=answer,
        paragraphs=paragraphs_text.strip(),
    )

    response = llm_fn(prompt)

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            sub_questions = json.loads(response[start:end])
        else:
            logger.warning(f"Could not parse decomposition: {response[:200]}")
            sub_questions = []
    except json.JSONDecodeError:
        logger.warning(f"JSON parse error in decomposition: {response[:200]}")
        sub_questions = []

    return sub_questions


def decompose_dataset(
    dataset: list[dict],
    llm_fn,
    dataset_name: str = "dragon-derec",
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Decompose all questions in a dataset.

    Args:
        dataset: list of {"question", "answer", "supporting_paragraphs" | "evidence_texts"}
        llm_fn: callable(prompt: str) -> str
        dataset_name: dataset name
        output_path: optional path to save results as JSONL

    Returns:
        list of decomposed samples
    """
    results = []

    for i, sample in enumerate(dataset):
        question = sample["question"]
        answer = sample["answer"]
        # Support both formats: dragon-derec (evidence_texts) and hotpotqa (supporting_paragraphs)
        paragraphs = sample.get("supporting_paragraphs",
                     sample.get("evidence_texts",
                     sample.get("context", [])))

        # Normalize to list of dicts if evidence_texts is list of strings
        if paragraphs and isinstance(paragraphs[0], str):
            paragraphs = [{"text": t, "title": f"Документ {j}"} for j, t in enumerate(paragraphs)]

        sub_questions = decompose_question(
            question=question,
            answer=answer,
            supporting_paragraphs=paragraphs,
            llm_fn=llm_fn,
            dataset_name=dataset_name,
        )

        result = {
            "id": sample.get("id", sample.get("question_id", str(i))),
            "question": question,
            "answer": answer,
            "supporting_paragraphs": paragraphs,
            "sub_questions": sub_questions,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            logger.info(f"Decomposed {i + 1}/{len(dataset)} questions")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved decomposed data to {output_path}")

    return results
