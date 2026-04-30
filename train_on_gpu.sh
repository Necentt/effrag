#!/bin/bash
# Скрипт для обучения EfficientRAG на V100 GPU.
# Запускать на сервере где установлен notebook.

set -e

# === 1. Зависимости ===
# Для V100 (Volta sm_70) ставим torch с CUDA 11.8 wheels — самый стабильный вариант.
echo "=== Installing dependencies ==="
pip install -q --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2
pip install -q -r requirements.txt

# === 2. Скачать код EfficientRAG ===
if [ ! -d "EfficientRAG" ]; then
    echo "ERROR: Папки EfficientRAG/ нет. Скопируй её на сервер."
    exit 1
fi

# === 3. Скачать датасет с HuggingFace ===
echo "=== Downloading datasets from HuggingFace ==="
mkdir -p data/efficient_rag/labeler data/efficient_rag/filter

python -c "
from huggingface_hub import hf_hub_download
import shutil

p = hf_hub_download(
    repo_id='Necent/efficientrag-labeler-training-data',
    filename='train.jsonl',
    repo_type='dataset',
)
shutil.copy(p, 'data/efficient_rag/labeler/train.jsonl')
print('Labeler data downloaded')

p = hf_hub_download(
    repo_id='Necent/efficientrag-filter-training-data',
    filename='train.jsonl',
    repo_type='dataset',
)
shutil.copy(p, 'data/efficient_rag/filter/train.jsonl')
print('Filter data downloaded')
"

wc -l data/efficient_rag/labeler/train.jsonl data/efficient_rag/filter/train.jsonl

# === 4. Обучение Labeler (V100, batch_size=32, fp16) ===
echo "=== Training Labeler ==="
python -m EfficientRAG.training.train_labeler \
    --train_data data/efficient_rag/labeler/train.jsonl \
    --output_dir checkpoints/efficientrag_labeler \
    --model_name microsoft/mdeberta-v3-base \
    --num_labels 2 \
    --max_length 384 \
    --epochs 2 \
    --batch_size 32 \
    --lr 5e-6 \
    --warmup_steps 200

# === 5. Обучение Filter ===
echo "=== Training Filter ==="
python -m EfficientRAG.training.train_filter \
    --train_data data/efficient_rag/filter/train.jsonl \
    --output_dir checkpoints/efficientrag_filter \
    --model_name microsoft/mdeberta-v3-base \
    --max_length 128 \
    --epochs 2 \
    --batch_size 32 \
    --lr 1e-5 \
    --warmup_steps 200

echo "=== Done! Models in checkpoints/ ==="
