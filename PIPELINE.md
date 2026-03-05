# CrossDoc-JEPA Pipeline (README order)

Run steps in this order. Use `bash scripts/run_pipeline.sh [step]` or run commands manually.

---

## Step 0: Environment ✓

- NLTK: `python scripts/setup_nltk.py` (already done)
- Deps: `pip install -r requirements.txt`
- GPU: 1 GPU detected

---

## Step 1: Data preparation ✓

- **Multi-News:** Cached (auto-download on first use).
- **Pretraining data:** Wikipedia clusters
  - **Command:**  
    `python data/build_wiki_clusters.py --output data/pretraining/wikipedia_clusters.jsonl --min_docs_per_cluster 3 --max_docs_per_cluster 8 --min_doc_len 200 --eval_sets multi_news`
  - **Time:** 2–4 hours (downloads ~20GB Wikipedia, builds leakage filter, writes clusters).
  - **When done:** `wc -l data/pretraining/wikipedia_clusters.jsonl` should show ~2M+ lines.

---

## Step 2: Stage 1 — Pretraining

**After** `data/pretraining/wikipedia_clusters.jsonl` exists:

- **1 GPU:**  
  `torchrun --nproc_per_node=1 training/pretrain.py --config configs/pretrain_config_single_gpu.yaml`
- **4 GPUs:**  
  `torchrun --nproc_per_node=4 --master_port=29500 training/pretrain.py --config configs/pretrain_config.yaml`

- **Time:** ~72h (4 GPUs) or longer on 1 GPU.
- **Checkpoints:** `checkpoints/pretrain/` (e.g. `pretrain_step_50000.pt`).
- **For finetune:** Set `pretrain_checkpoint: "checkpoints/pretrain/pretrain_final.pt"` (or latest) in `configs/finetune_config.yaml`, or copy last checkpoint to `pretrain_final.pt`.

---

## Step 3: Stage 2 — Finetuning

- **Command:**  
  `python training/finetune.py --config configs/finetune_config.yaml`
- **Saves:**  
  - Best by R-2: `checkpoints/finetune_multinews/best_model.pt`  
  - Each epoch: `checkpoints/finetune_multinews/epoch_1.pt`, `epoch_2.pt`, …

---

## Step 4: Evaluation

- **Command:**  
  `python evaluation/evaluate.py --config configs/finetune_config.yaml --checkpoint checkpoints/finetune_multinews/best_model.pt --split test --max_samples 5000`

---

## Optional (after evaluation)

- **Gate plot:**  
  `python evaluation/evaluate.py --config configs/finetune_config.yaml --checkpoint checkpoints/finetune_multinews/best_model.pt --plot_gate --gate_log checkpoints/finetune_multinews/gate_alpha.jsonl`
- **Ablations:**  
  `python experiments/ablations/run_ablations.py --config configs/finetune_config.yaml --checkpoint checkpoints/finetune_multinews/best_model.pt --output experiments/ablations/results.json`
- **Baselines:** See README “Baseline Comparisons”.

---

## Quick reference

| Step | Command |
|------|--------|
| 1 Data | `python data/build_wiki_clusters.py --output data/pretraining/wikipedia_clusters.jsonl --min_docs_per_cluster 3 --max_docs_per_cluster 8 --min_doc_len 200 --eval_sets multi_news` |
| 2 Pretrain (1 GPU) | `torchrun --nproc_per_node=1 training/pretrain.py --config configs/pretrain_config_single_gpu.yaml` |
| 2 Pretrain (4 GPUs) | `torchrun --nproc_per_node=4 --master_port=29500 training/pretrain.py --config configs/pretrain_config.yaml` |
| 3 Finetune | `python training/finetune.py --config configs/finetune_config.yaml` |
| 4 Eval | `python evaluation/evaluate.py --config configs/finetune_config.yaml --checkpoint checkpoints/finetune_multinews/best_model.pt --split test --max_samples 5000` |

Or run script: `bash scripts/run_pipeline.sh [1|2|3|4|all]`
