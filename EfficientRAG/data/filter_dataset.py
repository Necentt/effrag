"""
PyTorch Dataset for the Filter model.

Each sample consists of:
  - input: [CLS] query_info_tokens [SEP]
    where query_info_tokens = original_question + " " + extracted_info
  - token_labels: binary labels per token (keep=1 / discard=0)
"""

import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FilterDataset(Dataset):
    """
    Dataset for training the Filter.

    Expected data format (JSONL, one per line):
    {
        "query_info": "original question + extracted useful tokens",
        "token_labels": [0, 1, 1, 0, 1, ...],  // word-level binary labels
    }

    token_labels are at the word level and will be aligned to subword tokens.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
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
        query_info = sample["query_info"]
        word_labels = sample["token_labels"]

        return self._tokenize_with_labels(query_info, word_labels)

    def _tokenize_with_labels(self, text: str, word_labels: list[int]) -> dict:
        """
        Tokenize [CLS] text [SEP] with word-to-subword label alignment.
        """
        words = text.split()
        all_token_ids = []
        all_token_labels = []

        for i, word in enumerate(words):
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            all_token_ids.extend(word_tokens)
            label = word_labels[i] if i < len(word_labels) else 0
            all_token_labels.extend([label] * len(word_tokens))

        # Build: [CLS] tokens [SEP]
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        # Use -100 for positions that should not contribute to metrics/loss
        input_ids = [cls_id] + all_token_ids + [sep_id]
        token_labels = [-100] + all_token_labels + [-100]
        token_label_mask = [False] + [True] * len(all_token_labels) + [False]

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


def compute_filter_class_weights(dataset: FilterDataset) -> tuple[float, float]:
    """Compute class weights for filter token-level loss balancing."""
    total = 0
    positive = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["token_label_mask"]
        labels = sample["token_labels"]
        active = labels[mask]
        total += active.numel()
        positive += active.sum().item()

    negative = total - positive
    if positive == 0 or negative == 0:
        return 1.0, 1.0

    return total / (2.0 * negative), total / (2.0 * positive)
