"""
Ablation study runner. Systematically evaluates each component.

Usage:
  python experiments/ablations/run_ablations.py \
    --config configs/finetune_config.yaml \
    --checkpoint checkpoints/finetune_multinews/best_model.pt
"""

import sys
import yaml
import json
import torch
import logging
import argparse
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models import CrossDocJEPA
from data.dataset import MultiNewsDataset, finetune_collate_fn
from evaluation.evaluate import evaluate_model
from transformers import RobertaTokenizerFast, BartTokenizerFast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)

ABLATION_CONFIGS = {
    "full_model": {},  # No changes — baseline

    "no_jepa_pretrain": {
        # Train from scratch without pretraining
        # (just load the config, no pretrain checkpoint)
        "_pretrain_checkpoint_override": None,
    },

    "no_jse": {
        # Replace JSE with uniform sentence weights
        "_disable_jse": True,
    },

    "no_srd": {
        # Replace SRD bottleneck with direct encoder → decoder
        "_disable_srd": True,
    },

    "no_faith_loss": {
        "lambda_faith": 0.0,
    },

    "no_para_jepa": {
        "prediction_level": "doc",   # Doc-level prediction only, no para
    },

    "low_ema_momentum": {
        "ema_momentum": 0.9,
        "ema_base": 0.9,
    },

    "k8_queries": {
        "num_summary_queries": 8,
    },

    "k64_queries": {
        "num_summary_queries": 64,
    },

    "no_dual_attn": {
        "num_dual_attn_layers": 0,
    },
}


def patch_model_for_ablation(model: CrossDocJEPA, ablation_name: str, ablation_config: dict):
    """Apply runtime ablation patches to the model."""

    if ablation_config.get("_disable_jse"):
        # Override JSE forward with uniform scoring
        import torch.nn.functional as F

        def uniform_jse(sentence_embs, paragraph_embs, document_embs,
                        sent_valid_mask, para_valid_mask, oracle_scores=None, top_k=50):
            B, N, P, S, H = sentence_embs.shape
            flat_embs = sentence_embs.view(B, N * P * S, H)
            mask_flat = sent_valid_mask.view(B, N * P * S).float()
            uniform_scores = mask_flat / mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
            k = min(top_k, flat_embs.size(1))
            topk_scores, topk_idx = uniform_scores.topk(k, dim=1)
            topk_embs = flat_embs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
            topk_weights = F.softmax(topk_scores, dim=1)
            return {
                "salience_scores": uniform_scores.view(B, N, P, S),
                "topk_embs": topk_embs,
                "topk_weights": topk_weights,
                "topk_scores": topk_scores,
            }

        model.jse.forward = uniform_jse
        logger.info("[Ablation] JSE replaced with uniform sentence weighting.")

    if ablation_config.get("_disable_srd"):
        # Replace SRD with mean pooling of document embeddings
        original_srd_forward = model.srd.forward

        def mean_pool_srd(document_embs, paragraph_embs, para_mask,
                          salience_scores=None, predicted_doc_embs=None):
            # Simple mean of document embeddings, expanded to match K queries
            B, N, H = document_embs.shape
            K = model.srd.num_summary_queries
            mean_doc = document_embs.mean(dim=1, keepdim=True)  # (B, 1, H)
            summary_emb = mean_doc.expand(B, K, H)              # (B, K, H)
            return {
                "summary_emb": summary_emb,
                "memory": document_embs,
                "memory_mask": torch.ones(B, N, device=document_embs.device),
            }

        model.srd.forward = mean_pool_srd
        logger.info("[Ablation] SRD replaced with document mean pooling.")


def run_all_ablations(config, checkpoint_path, val_loader, enc_tok, dec_tok, device):
    results = {}

    for ablation_name, ablation_overrides in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {ablation_name}")

        # Build config for this ablation
        abl_config = deepcopy(config)
        for k, v in ablation_overrides.items():
            if not k.startswith("_"):  # Skip runtime patches
                abl_config[k] = v

        # Load model
        model = CrossDocJEPA(abl_config).to(device)

        # Handle pretrain checkpoint override
        ckpt_path = ablation_overrides.get("_pretrain_checkpoint_override", checkpoint_path)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("model", state), strict=False)

        # Apply runtime patches
        patch_model_for_ablation(model, ablation_name, ablation_overrides)
        model.eval()

        # Evaluate
        metrics = evaluate_model(model, val_loader, dec_tok, device, max_samples=200)
        metrics["ablation"] = ablation_name
        results[ablation_name] = metrics

        logger.info(
            f"[{ablation_name}] R-1: {metrics['rouge1']:.4f} | "
            f"R-2: {metrics['rouge2']:.4f} | "
            f"R-L: {metrics['rougeL']:.4f}"
        )

    return results


def print_ablation_table(results):
    print("\n" + "="*80)
    print(f"{'Ablation':<30} {'R-1':>8} {'R-2':>8} {'R-L':>8} {'BertScore':>10}")
    print("-"*80)
    for name, m in results.items():
        print(
            f"{name:<30} "
            f"{m['rouge1']:>8.4f} "
            f"{m['rouge2']:>8.4f} "
            f"{m['rougeL']:>8.4f} "
            f"{m.get('bertscore', 0.):>10.4f}"
        )
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/ablations/results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_tok = RobertaTokenizerFast.from_pretrained(config["encoder_model"])
    dec_tok = BartTokenizerFast.from_pretrained(config["decoder_model"])

    val_dataset = MultiNewsDataset(
        split="validation", encoder_tokenizer=enc_tok, decoder_tokenizer=dec_tok,
        max_docs=config["max_docs"], max_paras=config["max_paras"],
        max_sents_per_para=config["max_sents_per_para"],
        max_sent_len=config["max_sent_len"], max_summary_len=config["max_summary_len"],
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                             collate_fn=finetune_collate_fn, num_workers=2)

    results = run_all_ablations(config, args.checkpoint, val_loader, enc_tok, dec_tok, device)
    print_ablation_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")
