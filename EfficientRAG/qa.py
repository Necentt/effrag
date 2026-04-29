"""
Final QA generation: send collected chunks + question to an LLM for answer.

This is the ONLY LLM call in the entire EfficientRAG pipeline.
The retrieval loop uses only the Labeler and Filter (small DeBERTa models).
"""

QA_PROMPT = """Answer the following question based on the provided context.
If the context doesn't contain enough information, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    context: str,
    llm_fn,
    prompt_template: str = QA_PROMPT,
) -> str:
    """
    Generate the final answer using an LLM.

    Args:
        question: the original multi-hop question
        context: formatted context from retrieved chunks
        llm_fn: callable(prompt: str) -> str
        prompt_template: template with {question} and {context} placeholders

    Returns:
        Generated answer string
    """
    prompt = prompt_template.format(question=question, context=context)
    return llm_fn(prompt)
