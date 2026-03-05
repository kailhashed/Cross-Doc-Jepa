#!/usr/bin/env bash
set -e
cd /teamspace/studios/this_studio/Cross-Doc-Jepa
CLUSTER_FILE="data/pretraining/wikipedia_clusters.jsonl"
echo "Waiting for build_wiki_clusters to finish and $CLUSTER_FILE to be ready..."
while true; do
  if pgrep -f "build_wiki_clusters.py" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  if [[ -f "$CLUSTER_FILE" ]]; then
    LINES=$(wc -l < "$CLUSTER_FILE" 2>/dev/null || echo 0)
    if [[ "$LINES" -ge 1000 ]]; then
      echo "Build done. Cluster file has $LINES lines. Starting pretrain..."
      break
    fi
  fi
  sleep 30
done
echo "Running pretrain (single GPU, num_workers=8)..."
torchrun --nproc_per_node=1 training/pretrain.py --config configs/pretrain_config_single_gpu.yaml
echo "Pretrain done. Running finetune..."
python training/finetune.py --config configs/finetune_config.yaml
echo "All done."
