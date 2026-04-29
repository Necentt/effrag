# EfficientRAG Training

Training pipeline for [EfficientRAG](https://arxiv.org/abs/2408.04259) on Russian + English data.

Two small DeBERTa models replace LLM calls in the multi-hop retrieval loop:
- **Labeler** — classifies retrieved chunks (CONTINUE/TERMINATE) and extracts useful tokens
- **Filter** — constructs the next retrieval query by selecting words from question + extracted tokens

LLM is called **only once** at the end for final answer generation.

## Setup

```bash
git clone <this-repo>
cd effrag
pip install -r requirements.txt
```

## Quick training (V100 GPU)

```bash
# Optional: HuggingFace token if datasets are private
export HF_TOKEN=hf_xxx...

chmod +x train_on_gpu.sh
./train_on_gpu.sh
```

The script:
1. Downloads training data from HuggingFace
   ([labeler](https://huggingface.co/datasets/Necent/efficientrag-labeler-training-data),
   [filter](https://huggingface.co/datasets/Necent/efficientrag-filter-training-data))
2. Trains Labeler (~10-15 min on V100)
3. Trains Filter (~2-3 min on V100)

Checkpoints will appear in `checkpoints/efficientrag_labeler/` and `checkpoints/efficientrag_filter/`.

## Manual training

```bash
# Labeler
python -m EfficientRAG.training.train_labeler \
    --train_data data/efficient_rag/labeler/train.jsonl \
    --output_dir checkpoints/efficientrag_labeler \
    --model_name microsoft/mdeberta-v3-base \
    --num_labels 2 --epochs 2 --batch_size 32 \
    --lr 5e-6 --warmup_steps 200

# Filter
python -m EfficientRAG.training.train_filter \
    --train_data data/efficient_rag/filter/train.jsonl \
    --output_dir checkpoints/efficientrag_filter \
    --model_name microsoft/mdeberta-v3-base \
    --max_length 128 --epochs 2 --batch_size 32 \
    --lr 1e-5 --warmup_steps 200
```

## Hyperparameters

| | Labeler | Filter |
|--|---------|--------|
| Base model | `microsoft/mdeberta-v3-base` (86M, multilingual) | same |
| LR | 5e-6 | 1e-5 |
| Batch size | 32 | 32 |
| Max length | 384 | 128 |
| Epochs | 2 | 2 |
| Warmup steps | 200 | 200 |
| FP16 | yes | yes |

## Hardware estimates

| GPU | Labeler (30K samples) | Filter (5.7K samples) |
|-----|----------------------|----------------------|
| V100 16GB (fp16, batch=32) | ~10-15 min | ~2-3 min |
| Apple M3 Pro (CPU/MPS, batch=4) | ~3.4 hours | ~17 min |

## Upload trained models to HuggingFace

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/efficientrag_labeler',
    repo_id='Necent/efficientrag-labeler-mdeberta-v3-base',
    repo_type='model',
)
api.upload_folder(
    folder_path='checkpoints/efficientrag_filter',
    repo_id='Necent/efficientrag-filter-mdeberta-v3-base',
    repo_type='model',
)
"
```

## Repository layout

```
effrag/
├── EfficientRAG/                        # main package
│   ├── config.py                        # config + tag mappings
│   ├── models/
│   │   └── labeler.py                   # DebertaForSequenceTokenClassification
│   ├── data/
│   │   ├── labeler_dataset.py           # PyTorch Dataset for Labeler
│   │   └── filter_dataset.py            # PyTorch Dataset for Filter
│   ├── training/
│   │   ├── train_labeler.py             # Labeler training script
│   │   └── train_filter.py              # Filter training script
│   ├── data_synthesis/                  # data preparation pipeline
│   ├── retrieve.py                      # inference: EfficientRAGRetriever
│   ├── qa.py                            # final answer generation
│   ├── evaluation.py                    # EM / F1 / recall metrics
│   └── flexrag_bridge.py                # FlexRAG retriever integration
├── train_on_gpu.sh                      # one-command training script
├── requirements.txt
└── README.md
```

## References

- Paper: [EfficientRAG (arXiv:2408.04259)](https://arxiv.org/abs/2408.04259)
- Official repo: [microsoft/EfficientRAG](https://github.com/microsoft/EfficientRAG)
- Trained models: [labeler](https://huggingface.co/Necent/efficientrag-labeler-mdeberta-v3-base) ·
  [filter](https://huggingface.co/Necent/efficientrag-filter-mdeberta-v3-base)
- Training data: [labeler](https://huggingface.co/datasets/Necent/efficientrag-labeler-training-data) ·
  [filter](https://huggingface.co/datasets/Necent/efficientrag-filter-training-data)
