"""
Baseline comparisons for CrossDoc-JEPA paper.

Models evaluated on the FULL Multi-News test set (5,622 examples):
  - LEAD-3          (extractive, no training)
  - BART-Large      (vanilla fine-tuned, no JEPA)
  - PRIMERA         (allenai/PRIMERA-multinews, zero-shot)
  - CrossDoc-JEPA   (ours)

Metrics: ROUGE-1/2/L, BERTScore, FactCC, Novel-2gram

Usage:
  # Run all baselines + CrossDoc-JEPA and print comparison table:
  python experiments/baselines/run_baselines.py \\
    --config configs/finetune_config.yaml \\
    --ours_checkpoint checkpoints/finetune_multinews/best_model.pt \\
    --output results/comparison.json

  # Run a single baseline:
  python experiments/baselines/run_baselines.py \\
    --config configs/finetune_config.yaml \\
    --model primera \\
    --output results/primera.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, BartTokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.dataset import MultiNewsDataset, finetune_collate_fn
from evaluation.evaluate import (
    compute_rouge, compute_bertscore, compute_factcc,
    novel_ngrams, compression_ratio, print_comparison_table,
)

log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: reconstruct source texts from the Multi-News HuggingFace split
# ─────────────────────────────────────────────────────────────────────────────

def get_multinews_raw(split: str = "test", max_samples: int = 5622):
    from datasets import load_dataset
    ds = load_dataset("multi_news", split=split)
    clusters, references = [], []
    for item in ds:
        docs = [d.strip() for d in item["document"].split("|||||") if d.strip()]
        clusters.append(docs)
        references.append(item["summary"].strip())
        if len(clusters) >= max_samples:
            break
    return clusters, references


# ─────────────────────────────────────────────────────────────────────────────
# LEAD-3
# ─────────────────────────────────────────────────────────────────────────────

def run_lead(clusters, references, n=3):
    log.info(f"Running LEAD-{n}...")
    from nltk.tokenize import sent_tokenize
    predictions = [" ".join(sent_tokenize(docs[0])[:n]) for docs in clusters]
    m = compute_rouge(predictions, references)
    m["bertscore"]   = compute_bertscore(predictions, references)
    m["factcc"]      = compute_factcc(predictions, [" ".join(d) for d in clusters])
    m["novel_2gram"] = novel_ngrams(predictions, references, 2)
    m["model"]       = f"LEAD-{n}"
    log.info(f"LEAD-{n}: R-2 = {m['rouge2']:.4f}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# PRIMERA  (allenai/PRIMERA-multinews, zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_primera(clusters, references, device, batch_size=2, max_samples=5622):
    log.info("Running PRIMERA-multinews...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "allenai/PRIMERA-multinews"
        tok   = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        log.error(f"PRIMERA unavailable: {e}")
        return {"model": "PRIMERA-multinews", "note": str(e)}

    DOCSEP_ID = tok.convert_tokens_to_ids("<doc-sep>")
    predictions = []

    for i, docs in enumerate(clusters[:max_samples]):
        if i % 100 == 0:
            log.info(f"  PRIMERA: {i}/{min(len(clusters), max_samples)}")
        text = " <doc-sep> ".join(docs)
        enc  = tok(text, max_length=4096, truncation=True, return_tensors="pt").to(device)

        global_att = torch.zeros_like(enc["input_ids"])
        global_att[:, 0] = 1
        for pos in (enc["input_ids"] == DOCSEP_ID).nonzero():
            global_att[pos[0], pos[1]] = 1

        gen = model.generate(
            **enc, global_attention_mask=global_att,
            max_length=256, num_beams=4, length_penalty=2.0,
            no_repeat_ngram_size=3, early_stopping=True,
        )
        predictions.append(tok.decode(gen[0], skip_special_tokens=True))

    refs = references[:len(predictions)]
    srcs = [" ".join(d) for d in clusters[:len(predictions)]]
    m = compute_rouge(predictions, refs)
    m["bertscore"]   = compute_bertscore(predictions, refs)
    m["factcc"]      = compute_factcc(predictions, srcs)
    m["novel_2gram"] = novel_ngrams(predictions, refs, 2)
    m["model"]       = "PRIMERA-multinews"
    log.info(f"PRIMERA: R-2 = {m['rouge2']:.4f}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# BART-Large baseline  (fine-tuned without JEPA)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_bart_baseline(
    clusters, references, device,
    checkpoint: str = None,
    max_samples: int = 5622,
):
    log.info("Running BART-Large baseline...")
    from transformers import BartForConditionalGeneration, BartTokenizerFast
    model_path = checkpoint if (checkpoint and Path(checkpoint).exists()) \
                 else "facebook/bart-large-cnn"
    tok   = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    predictions = []

    for i, docs in enumerate(clusters[:max_samples]):
        if i % 100 == 0:
            log.info(f"  BART: {i}/{min(len(clusters), max_samples)}")
        text = " ".join(docs)
        enc  = tok(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        gen  = model.generate(**enc, max_length=256, num_beams=4,
                               length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        predictions.append(tok.decode(gen[0], skip_special_tokens=True))

    refs = references[:len(predictions)]
    srcs = [" ".join(d) for d in clusters[:len(predictions)]]
    m = compute_rouge(predictions, refs)
    m["bertscore"]   = compute_bertscore(predictions, refs)
    m["factcc"]      = compute_factcc(predictions, srcs)
    m["novel_2gram"] = novel_ngrams(predictions, refs, 2)
    m["model"]       = "BART-Large" + (" (fine-tuned)" if checkpoint else " (cnn/dm)")
    log.info(f"BART: R-2 = {m['rouge2']:.4f}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CrossDoc-JEPA (ours)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_crossdoc_jepa(config, checkpoint, clusters, references, device, max_samples=5622):
    log.info("Running CrossDoc-JEPA...")
    from models import CrossDocJEPA

    enc_tok = RobertaTokenizerFast.from_pretrained(config["encoder_model"])
    dec_tok = BartTokenizerFast.from_pretrained(config["decoder_model"])

    ds_kw = dict(max_docs=config.get("max_docs", 5), max_paras=config.get("max_paras", 12),
                 max_sents_per_para=config.get("max_sents_per_para", 6),
                 max_sent_len=config.get("max_sent_len", 64),
                 max_summary_len=config.get("max_summary_len", 256))
    test_ds = MultiNewsDataset("test", enc_tok, dec_tok, **ds_kw)
    loader  = DataLoader(test_ds, batch_size=2, shuffle=False,
                          collate_fn=finetune_collate_fn, num_workers=2)

    model = CrossDocJEPA(config).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()

    from evaluation.evaluate import evaluate_model
    m = evaluate_model(model, loader, dec_tok, device,
                        max_samples=max_samples, run_factcc=True)
    m["model"] = "CrossDoc-JEPA (ours)"
    m["novel_2gram"] = m.pop("novel_2gram", 0.0)
    log.info(f"CrossDoc-JEPA: R-2 = {m['rouge2']:.4f}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           required=True)
    parser.add_argument("--ours_checkpoint",  default=None)
    parser.add_argument("--bart_checkpoint",  default=None,
                        help="Path to fine-tuned BART baseline. If None, uses bart-large-cnn.")
    parser.add_argument("--model",            default="all",
                        choices=["all", "lead", "primera", "bart", "ours"])
    parser.add_argument("--split",            default="test")
    parser.add_argument("--max_samples",      type=int, default=5622)
    parser.add_argument("--output",           default="results/comparison.json")
    parser.add_argument("--compare",          action="store_true",
                        help="Load existing result JSONs and print table. "
                             "Pass result files as positional args.")
    parser.add_argument("results_files",      nargs="*")
    args = parser.parse_args()

    # ── Compare existing results ─────────────────────────────────────────────
    if args.compare:
        all_results = {}
        for path in args.results_files:
            with open(path) as f:
                d = json.load(f)
                name = d.get("model", Path(path).stem)
                all_results[name] = d
        print_comparison_table(all_results)
        return

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  max_samples: {args.max_samples}")

    clusters, references = get_multinews_raw(args.split, args.max_samples)
    log.info(f"Loaded {len(clusters)} Multi-News {args.split} examples")

    results = {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.model in ("all", "lead"):
        results["LEAD-3"] = run_lead(clusters, references)

    if args.model in ("all", "primera"):
        results["PRIMERA-multinews"] = run_primera(clusters, references, device, max_samples=args.max_samples)

    if args.model in ("all", "bart"):
        results["BART-Large"] = run_bart_baseline(clusters, references, device,
                                                    checkpoint=args.bart_checkpoint,
                                                    max_samples=args.max_samples)

    if args.model in ("all", "ours"):
        if not args.ours_checkpoint:
            log.error("--ours_checkpoint required for 'ours' model")
        else:
            results["CrossDoc-JEPA"] = run_crossdoc_jepa(
                config, args.ours_checkpoint, clusters, references, device, args.max_samples
            )

    # Save and print
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.output}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
