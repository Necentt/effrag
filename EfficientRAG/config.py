"""Configuration for EfficientRAG."""

from dataclasses import dataclass, field
from typing import Optional


# Sequence-level tags for the Labeler
CONTINUE_TAG = "<CONTINUE>"
TERMINATE_TAG = "<TERMINATE>"
FINISH_TAG = "<FINISH>"

# 3-label mapping: CONTINUE=0, TERMINATE=1, FINISH=2
TAG_MAPPING_THREE = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 2,
}

# 2-label mapping: CONTINUE/FINISH=0, TERMINATE=1
TAG_MAPPING_TWO = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 0,
}

# Reverse mappings
ID_TO_TAG_THREE = {v: k for k, v in TAG_MAPPING_THREE.items()}
ID_TO_TAG_TWO = {0: CONTINUE_TAG, 1: TERMINATE_TAG}


@dataclass
class EfficientRAGConfig:
    """Main configuration for the EfficientRAG system."""

    # Base model (multilingual for Russian support)
    model_name: str = "microsoft/mdeberta-v3-base"

    # Labeler settings
    labeler_max_length: int = 384
    labeler_train_max_length: int = 512
    num_sequence_labels: int = 2  # 2 or 3

    # Filter settings
    filter_max_length: int = 128

    # Retrieval settings
    top_k: int = 10
    max_iterations: int = 4

    # Training
    labeler_lr: float = 5e-6
    filter_lr: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 2
    batch_size: int = 32
    warmup_steps: int = 200

    @property
    def tag_mapping(self) -> dict:
        if self.num_sequence_labels == 2:
            return TAG_MAPPING_TWO
        return TAG_MAPPING_THREE

    @property
    def id_to_tag(self) -> dict:
        if self.num_sequence_labels == 2:
            return ID_TO_TAG_TWO
        return ID_TO_TAG_THREE


@dataclass
class DataSynthesisConfig:
    """Configuration for training data synthesis."""

    # LLM for data synthesis (via OpenRouter)
    llm_model: str = "meta-llama/llama-3-70b-instruct"
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None

    # Dataset
    dataset_name: str = "dragon-derec"
    hf_dataset: str = "Makson4ic/dragon-derec-dataset"

    # Paths
    corpus_path: str = "dragon_corpus.jsonl"
    output_path: str = "data/efficient_rag"
    decomposed_path: str = "data/synthesized_decomposed"
    token_labeled_path: str = "data/synthesized_token_labeling"
    negative_sampling_path: str = "data/negative_sampling"

    # Retriever for negative sampling
    retriever_name: str = "dragon"
    retriever_path: Optional[str] = "dragon_retriever"
    passages_path: Optional[str] = None
    index_path: Optional[str] = None

    # SpaCy model
    spacy_model: str = "ru_core_news_sm"
