"""
PyTorch Dataset for the Labeler model.

Each sample consists of:
  - input: [CLS] query [SEP] chunk [SEP]
  - token_labels: binary labels for each token (only chunk tokens have meaningful labels)
  - sequence_label: CONTINUE(0) / TERMINATE(1) / FINISH(2)
  - token_label_mask: True for chunk tokens that should be labeled
"""

import json
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class LabelerDataset(Dataset):
    """
    Dataset for training the Labeler.

    Expected data format (JSONL, one per line):
    {
        "question": "multi-hop question text",
        "chunk": "retrieved passage text",
        "token_labels": [0, 1, 0, 0, 1, ...],  // word-level binary labels
        "tag": "<CONTINUE>" | "<TERMINATE>" | "<FINISH>"
    }

    token_labels are at the word level and will be aligned to subword tokens
    during tokenization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        tag_mapping: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tag_mapping = tag_mapping or {
            "<CONTINUE>": 0,
            "<TERMINATE>": 1,
            "<FINISH>": 0,
        }
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> list[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        question = sample["question"]
        chunk = sample["chunk"]
        word_labels = sample.get("token_labels", [])
        tag = sample["tag"]

        sequence_label = self.tag_mapping.get(tag, 1)

        # Tokenize question and chunk separately for label alignment
        encoding = self._tokenize_with_labels(question, chunk, word_labels)
        encoding["sequence_labels"] = sequence_label

        return encoding

    def _tokenize_with_labels(
        self, question: str, chunk: str, word_labels: list[int]
    ) -> dict:
        """
        Tokenize [CLS] question [SEP] chunk [SEP] and align word-level labels
        to subword tokens.

        Word-level labels are expanded: if a word splits into N subwords,
        all N subwords get that word's label.
        """
        # Tokenize question
        question_tokens = self.tokenizer.encode(
            question, add_special_tokens=False
        )

        # Tokenize chunk word-by-word for label alignment
        chunk_words = chunk.split()
        chunk_token_ids = []
        chunk_token_labels = []

        for i, word in enumerate(chunk_words):
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            chunk_token_ids.extend(word_tokens)
            label = word_labels[i] if i < len(word_labels) else 0
            chunk_token_labels.extend([label] * len(word_tokens))

        # Build full sequence: [CLS] question [SEP] chunk [SEP]
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = [cls_id] + question_tokens + [sep_id] + chunk_token_ids + [sep_id]

        # -100 marks positions to ignore in loss/metrics; actual labels only for chunk
        token_labels = (
            [-100] * (1 + len(question_tokens) + 1)
            + chunk_token_labels
            + [-100]
        )

        token_label_mask = (
            [False] * (1 + len(question_tokens) + 1)
            + [True] * len(chunk_token_labels)
            + [False]
        )

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            token_labels = token_labels[: self.max_length]
            token_label_mask = token_label_mask[: self.max_length]

        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        token_labels += [-100] * pad_len
        token_label_mask += [False] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_labels": torch.tensor(token_labels, dtype=torch.long),
            "token_label_mask": torch.tensor(token_label_mask, dtype=torch.bool),
        }


def compute_token_class_weights(dataset: LabelerDataset) -> tuple[float, float]:
    """
    Compute class weights for token-level loss balancing.

    Returns (negative_weight, positive_weight) where:
      positive_weight = total / (2 * positive_count)
      negative_weight = total / (2 * negative_count)
    """
    total = 0
    positive = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["token_label_mask"]
        labels = sample["token_labels"]
        active_labels = labels[mask]
        total += active_labels.numel()
        positive += active_labels.sum().item()

    negative = total - positive
    if positive == 0 or negative == 0:
        return 1.0, 1.0

    pos_weight = total / (2.0 * positive)
    neg_weight = total / (2.0 * negative)
    return neg_weight, pos_weight
