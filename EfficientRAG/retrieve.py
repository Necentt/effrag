"""
EfficientRAG iterative retrieval pipeline.

The core inference loop:
1. Retrieve top-k passages for current query
2. Labeler classifies each chunk (CONTINUE/TERMINATE) + extracts useful tokens
3. Filter constructs next-hop query from original question + extracted info
4. Repeat until TERMINATE or max iterations
5. Return collected relevant chunks for final answer generation
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, DebertaV2ForTokenClassification

from EfficientRAG.config import (
    CONTINUE_TAG,
    FINISH_TAG,
    TERMINATE_TAG,
    EfficientRAGConfig,
)
from EfficientRAG.models.labeler import DebertaForSequenceTokenClassification

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata from EfficientRAG processing."""

    text: str
    score: float = 0.0
    tag: str = ""
    useful_tokens: str = ""
    iteration: int = 0
    chunk_id: Optional[str] = None


@dataclass
class EfficientRAGResult:
    """Result of the EfficientRAG retrieval process."""

    question: str
    collected_chunks: list[RetrievedChunk] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    num_iterations: int = 0


class EfficientRAGRetriever:
    """
    EfficientRAG multi-hop retrieval system.

    Uses a trained Labeler + Filter to iteratively retrieve and refine
    queries without LLM calls during the retrieval loop.
    """

    def __init__(
        self,
        labeler_path: str,
        filter_path: str,
        config: Optional[EfficientRAGConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or EfficientRAGConfig()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load Labeler
        logger.info(f"Loading Labeler from {labeler_path}")
        self.labeler_tokenizer = AutoTokenizer.from_pretrained(labeler_path)
        self.labeler = DebertaForSequenceTokenClassification.from_pretrained(
            labeler_path
        ).to(self.device)
        self.labeler.eval()

        # Load Filter
        logger.info(f"Loading Filter from {filter_path}")
        self.filter_tokenizer = AutoTokenizer.from_pretrained(filter_path)
        self.filter = DebertaV2ForTokenClassification.from_pretrained(
            filter_path
        ).to(self.device)
        self.filter.eval()

    @torch.no_grad()
    def label_chunk(self, query: str, chunk: str) -> tuple[str, str]:
        """
        Run the Labeler on a (query, chunk) pair.

        Returns:
            tag: CONTINUE_TAG, TERMINATE_TAG, or FINISH_TAG
            useful_tokens: extracted useful token text from chunk
        """
        # Tokenize: [CLS] query [SEP] chunk [SEP]
        query_tokens = self.labeler_tokenizer.encode(query, add_special_tokens=False)
        chunk_words = chunk.split()
        chunk_subword_ids = []
        word_boundaries = []  # (start_idx, end_idx) in chunk_subword_ids

        for word in chunk_words:
            word_ids = self.labeler_tokenizer.encode(word, add_special_tokens=False)
            start = len(chunk_subword_ids)
            chunk_subword_ids.extend(word_ids)
            word_boundaries.append((start, len(chunk_subword_ids)))

        cls_id = self.labeler_tokenizer.cls_token_id or self.labeler_tokenizer.bos_token_id
        sep_id = self.labeler_tokenizer.sep_token_id or self.labeler_tokenizer.eos_token_id

        input_ids = [cls_id] + query_tokens + [sep_id] + chunk_subword_ids + [sep_id]

        # Truncate
        max_len = self.config.labeler_max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]

        attention_mask = [1] * len(input_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        outputs = self.labeler(input_ids=input_tensor, attention_mask=attn_tensor)

        # Sequence prediction
        seq_pred = outputs.sequence_logits.argmax(dim=-1).item()
        tag = self.config.id_to_tag.get(seq_pred, TERMINATE_TAG)

        # Token predictions — extract useful tokens from chunk portion
        token_preds = outputs.token_logits.argmax(dim=-1)[0]  # [seq_len]

        # Chunk starts at position: 1 (CLS) + len(query_tokens) + 1 (SEP)
        chunk_start = 1 + len(query_tokens) + 1
        useful_words = []

        for i, (word, (start, end)) in enumerate(zip(chunk_words, word_boundaries)):
            global_start = chunk_start + start
            global_end = chunk_start + end
            if global_end > len(token_preds):
                break
            # A word is useful if majority of its subword tokens are labeled 1
            subword_preds = token_preds[global_start:global_end]
            if subword_preds.float().mean() >= 0.5:
                useful_words.append(word)

        useful_tokens = " ".join(useful_words)
        return tag, useful_tokens

    @torch.no_grad()
    def filter_query(self, query_info: str) -> str:
        """
        Run the Filter to construct the next-hop query.

        Args:
            query_info: concatenation of original question + extracted useful tokens

        Returns:
            Filtered next-hop query string
        """
        words = query_info.split()

        # Word-by-word tokenization for alignment
        all_token_ids = []
        word_boundaries = []

        for word in words:
            word_ids = self.filter_tokenizer.encode(word, add_special_tokens=False)
            start = len(all_token_ids)
            all_token_ids.extend(word_ids)
            word_boundaries.append((start, len(all_token_ids)))

        cls_id = self.filter_tokenizer.cls_token_id or self.filter_tokenizer.bos_token_id
        sep_id = self.filter_tokenizer.sep_token_id or self.filter_tokenizer.eos_token_id

        input_ids = [cls_id] + all_token_ids + [sep_id]

        max_len = self.config.filter_max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]

        attention_mask = [1] * len(input_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        outputs = self.filter(input_ids=input_tensor, attention_mask=attn_tensor)
        token_preds = outputs.logits.argmax(dim=-1)[0]  # [seq_len]

        # Extract kept words (skip CLS at position 0)
        kept_words = []
        for word, (start, end) in zip(words, word_boundaries):
            global_start = 1 + start  # +1 for CLS
            global_end = 1 + end
            if global_end > len(token_preds):
                break
            subword_preds = token_preds[global_start:global_end]
            if subword_preds.float().mean() >= 0.5:
                kept_words.append(word)

        return " ".join(kept_words)

    def retrieve(
        self,
        question: str,
        retriever_fn,
        top_k: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> EfficientRAGResult:
        """
        Run the full EfficientRAG iterative retrieval pipeline.

        Args:
            question: the original multi-hop question
            retriever_fn: callable(query: str, top_k: int) -> list[dict]
                Each dict should have at least 'text' key, optionally 'id', 'score'
            top_k: number of passages to retrieve per iteration
            max_iterations: maximum number of retrieval iterations

        Returns:
            EfficientRAGResult with collected relevant chunks and query history
        """
        top_k = top_k or self.config.top_k
        max_iterations = max_iterations or self.config.max_iterations

        result = EfficientRAGResult(question=question)
        current_query = question
        all_info = []  # accumulated useful tokens across iterations

        for iteration in range(1, max_iterations + 1):
            result.query_history.append(current_query)
            logger.info(f"Iteration {iteration}: query='{current_query[:100]}...'")

            # Step 1: Retrieve passages
            passages = retriever_fn(current_query, top_k)
            if not passages:
                logger.info("No passages retrieved, stopping.")
                break

            # Step 2: Label each passage
            iteration_info = []
            found_useful = False

            for passage in passages:
                chunk_text = passage.get("text", passage.get("content", ""))
                chunk_id = passage.get("id", None)
                chunk_score = passage.get("score", 0.0)

                tag, useful_tokens = self.label_chunk(current_query, chunk_text)

                chunk = RetrievedChunk(
                    text=chunk_text,
                    score=chunk_score,
                    tag=tag,
                    useful_tokens=useful_tokens,
                    iteration=iteration,
                    chunk_id=chunk_id,
                )

                if tag in (CONTINUE_TAG, FINISH_TAG):
                    result.collected_chunks.append(chunk)
                    if useful_tokens.strip():
                        iteration_info.append(useful_tokens)
                        found_useful = True
                # TERMINATE chunks are discarded

            if not found_useful:
                logger.info("No useful information extracted, stopping.")
                break

            # Step 3: Construct next-hop query with Filter
            all_info.extend(iteration_info)
            query_info = question + " " + " ".join(all_info)
            next_query = self.filter_query(query_info)

            if not next_query.strip():
                logger.info("Filter produced empty query, stopping.")
                break

            current_query = next_query
            result.num_iterations = iteration

        return result

    def retrieve_and_format(
        self,
        question: str,
        retriever_fn,
        top_k: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> tuple[str, EfficientRAGResult]:
        """
        Retrieve and format chunks as context string for LLM generation.

        Returns:
            (context_string, result)
        """
        result = self.retrieve(question, retriever_fn, top_k, max_iterations)

        context_parts = []
        for i, chunk in enumerate(result.collected_chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")

        context = "\n\n".join(context_parts)
        return context, result
