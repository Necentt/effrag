"""
EfficientRAG: Efficient Retriever for Multi-Hop Question Answering

Implementation based on:
  Paper: arXiv:2408.04259 (EMNLP 2024)
  Official repo: https://github.com/microsoft/EfficientRAG
"""

from EfficientRAG.config import EfficientRAGConfig
from EfficientRAG.models.labeler import DebertaForSequenceTokenClassification
from EfficientRAG.retrieve import EfficientRAGRetriever

__all__ = [
    "EfficientRAGConfig",
    "DebertaForSequenceTokenClassification",
    "EfficientRAGRetriever",
]
