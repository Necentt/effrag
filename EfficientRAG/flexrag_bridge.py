"""
Bridge between EfficientRAG and FlexRAG retrievers.

Allows using any FlexRAG retriever as the base retriever for
the EfficientRAG multi-hop pipeline.
"""

from typing import Optional

from EfficientRAG.retrieve import EfficientRAGRetriever, EfficientRAGResult


def make_flexrag_retriever_fn(flexrag_retriever):
    """
    Wrap a FlexRAG retriever into a callable compatible with EfficientRAG.

    Args:
        flexrag_retriever: a FlexRAG retriever instance with .search() method

    Returns:
        callable(query: str, top_k: int) -> list[dict]
    """

    def retriever_fn(query: str, top_k: int) -> list[dict]:
        results = flexrag_retriever.search([query], top_k=top_k)
        passages = []
        if results and len(results) > 0:
            for ctx in results[0]:
                passage = {
                    "text": ctx.data.get("text", str(ctx.data)),
                    "id": ctx.context_id or "",
                    "score": ctx.score if hasattr(ctx, "score") else 0.0,
                }
                # Include other fields
                if hasattr(ctx, "data") and isinstance(ctx.data, dict):
                    passage.update({
                        k: v for k, v in ctx.data.items() if k != "text"
                    })
                passages.append(passage)
        return passages

    return retriever_fn


def efficient_rag_with_flexrag(
    question: str,
    flexrag_retriever,
    labeler_path: str,
    filter_path: str,
    config=None,
    device: Optional[str] = None,
) -> EfficientRAGResult:
    """
    Run EfficientRAG retrieval using a FlexRAG retriever.

    Args:
        question: the multi-hop question
        flexrag_retriever: a FlexRAG retriever with .search() method
        labeler_path: path to trained Labeler checkpoint
        filter_path: path to trained Filter checkpoint
        config: optional EfficientRAGConfig
        device: torch device string

    Returns:
        EfficientRAGResult
    """
    retriever_fn = make_flexrag_retriever_fn(flexrag_retriever)

    rag = EfficientRAGRetriever(
        labeler_path=labeler_path,
        filter_path=filter_path,
        config=config,
        device=device,
    )

    return rag.retrieve(question, retriever_fn)
