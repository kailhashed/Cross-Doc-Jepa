#!/bin/bash
# CrossDoc-JEPA pipeline — run in order (README flow)
# Usage: bash scripts/run_pipeline.sh [step]
#   step: 1=env+data, 2=pretrain, 3=finetune, 4=eval. Default: run all.

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

step="${1:-all}"

echo "=== CrossDoc-JEPA Pipeline (from README) ==="

# ─── Step 0: Environment ─────────────────────────────────────────────────
env_check() {
  echo "[0] Environment check..."
  python -c "from nltk.tokenize import sent_tokenize; sent_tokenize('x. y.')" || { echo "Run: python scripts/setup_nltk.py"; exit 1; }
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
  echo "    NLTK OK, GPU OK"
}

# ─── Step 1: Data preparation ─────────────────────────────────────────────
data_prep() {
  echo "[1] Data preparation..."
  python -c "from datasets import load_dataset; load_dataset('multi_news', split='train'); print('    Multi-News cached.')"
  if [ ! -f "data/pretraining/wikipedia_clusters.jsonl" ]; then
    echo "    Building Wikipedia clusters (2–4 hours, ~20GB download)..."
    python data/build_wiki_clusters.py \
      --output data/pretraining/wikipedia_clusters.jsonl \
      --min_docs_per_cluster 3 \
      --max_docs_per_cluster 8 \
      --min_doc_len 200 \
      --eval_sets multi_news
    echo "    Clusters: $(wc -l < data/pretraining/wikipedia_clusters.jsonl) lines"
  else
    echo "    Pretraining clusters already exist."
  fi
}

# ─── Step 2: Stage 1 — Pretraining ───────────────────────────────────────
pretrain() {
  echo "[2] Stage 1: Pretraining..."
  NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "$NGPU" -lt 2 ]; then
    echo "    Using single-GPU config (batch_size 4)."
    CONFIG="configs/pretrain_config_single_gpu.yaml"
    torchrun --nproc_per_node=1 training/pretrain.py --config "$CONFIG"
  else
    torchrun --nproc_per_node=$NGPU --master_port=29500 training/pretrain.py --config configs/pretrain_config.yaml
  fi
  echo "    Pretrain checkpoints in checkpoints/pretrain/"
}

# ─── Step 3: Stage 2 — Finetuning ────────────────────────────────────────
finetune() {
  echo "[3] Stage 2: Finetuning..."
  python training/finetune.py --config configs/finetune_config.yaml
  echo "    Best model: checkpoints/finetune_multinews/best_model.pt"
}

# ─── Step 4: Evaluation ──────────────────────────────────────────────────
eval_model() {
  echo "[4] Evaluation..."
  CKPT="checkpoints/finetune_multinews/best_model.pt"
  if [ ! -f "$CKPT" ]; then
    echo "    No best_model.pt; use latest epoch_*.pt or run finetune first."
    return 1
  fi
  python evaluation/evaluate.py \
    --config configs/finetune_config.yaml \
    --checkpoint "$CKPT" \
    --split test \
    --max_samples 5000
}

# ─── Run ─────────────────────────────────────────────────────────────────
env_check

case "$step" in
  1) data_prep ;;
  2) data_prep; pretrain ;;
  3) finetune ;;
  4) eval_model ;;
  all)
    data_prep
    pretrain
    finetune
    eval_model
    ;;
  *) echo "Unknown step: $step. Use 1, 2, 3, 4, or all."; exit 1 ;;
esac

echo "=== Pipeline step(s) done ==="
