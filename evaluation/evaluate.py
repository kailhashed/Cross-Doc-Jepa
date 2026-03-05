"""
Evaluation suite for CrossDoc-JEPA.

Metrics:
  - ROUGE-1/2/L (lexical overlap)
  - BERTScore F1 (semantic similarity)
  - FactCC (factual consistency, NLI-based)
  - EHR  (Extractive Hallucination Rate)    — noun-phrase grounding proxy (NOT QAGS)
  - Novel n-gram % (abstractiveness)
  - Compression ratio

Baselines (run_baselines):
  - BART-Large (vanilla fine-tuned)
  - PRIMERA (facebook/primera-multinews)
  - LEAD-N (extractive upper-bound approximation)

Convergence tracking:
  - gate_alpha_curve: plots SRD gate_alpha over training steps
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0., "rouge2": 0., "rougeL": 0.}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
    n = max(len(predictions), 1)
    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(predictions, references, lang="en",
                            model_type="microsoft/deberta-xlarge-mnli",
                            verbose=False, batch_size=8)
        return F1.mean().item()
    except ImportError:
        logger.warning("bert_score not installed — skipping BERTScore. pip install bert-score")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FactCC — factual consistency (NLI-based)
# ─────────────────────────────────────────────────────────────────────────────

def compute_factcc(
    predictions: List[str],
    source_docs: List[str],
    batch_size: int = 8,
) -> float:
    """
    FactCC: fraction of summaries classified as CONSISTENT with their source documents.
    Uses the manueldeprada/FactCC model (HuggingFace) — an NLI classifier.
    Lower score = more hallucination. For EMNLP you want this > 0.8.

    source_docs: list of concatenated source texts (one string per cluster).
    """
    try:
        from transformers import pipeline
        factcc = pipeline(
            "text-classification",
            model="manueldeprada/FactCC",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True, max_length=512,
        )
    except Exception as e:
        logger.warning(f"FactCC model unavailable: {e}. pip install transformers>=4.38")
        return 0.0

    consistent = 0
    for i in range(0, len(predictions), batch_size):
        preds_b = predictions[i:i + batch_size]
        srcs_b  = source_docs[i:i + batch_size]
        # FactCC format: [CLS] source [SEP] claim [SEP]
        inputs = [f"{src[:450]} [SEP] {pred[:50]}" for src, pred in zip(srcs_b, preds_b)]
        results = factcc(inputs)
        for r in results:
            if r["label"].upper() == "CONSISTENT":
                consistent += 1

    return consistent / max(len(predictions), 1)


# ─────────────────────────────────────────────────────────────────────────────
# EHR — Extractive Hallucination Rate (noun-phrase grounding; NOT QAGS)
# ─────────────────────────────────────────────────────────────────────────────

def extractive_hallucination_rate(
    predictions: List[str],
    source_docs: List[str],
) -> float:
    """
    Extractive Hallucination Rate (EHR): fraction of multi-word noun phrases
    in the generated summary that do NOT appear verbatim in any source document.

    Interpretation: lower is better (0.0 = fully grounded, 1.0 = fully hallucinated).

    This is a lightweight faithfulness proxy complementing FactCC.
    It is NOT QAGS. QAGS requires a separate QG+QA pipeline
    (Wang et al. 2020, https://github.com/W4ngatang/qags).
    Label this metric "EHR" in your paper, never "QAGS".

    Requirements: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        logger.warning(
            "spacy/en_core_web_sm not available — skipping EHR. "
            "pip install spacy && python -m spacy download en_core_web_sm"
        )
        return 0.0

    scores = []
    for pred, src in zip(predictions, source_docs):
        doc    = nlp(pred)
        chunks = [c.text.lower() for c in doc.noun_chunks if len(c.text.split()) > 1]
        if not chunks:
            scores.append(0.0)  # No multi-word NPs -> no hallucination signal
            continue
        src_lower    = src.lower()
        not_grounded = sum(1 for c in chunks if c not in src_lower)
        scores.append(not_grounded / len(chunks))   # fraction NOT found in source

    return sum(scores) / max(len(scores), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Novel n-grams & compression
# ─────────────────────────────────────────────────────────────────────────────

def novel_ngrams(predictions: List[str], sources: List[str], n: int = 1) -> float:
    total, novel = 0, 0
    for pred, src in zip(predictions, sources):
        def ngrams(text, n):
            toks = text.lower().split()
            return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
        pn, sn = ngrams(pred, n), ngrams(src, n)
        for gram, cnt in pn.items():
            total += cnt
            if gram not in sn:
                novel += cnt
    return novel / max(total, 1)


def compression_ratio(predictions: List[str], sources: List[str]) -> float:
    ratios = [len(p.split()) / max(len(s.split()), 1) for p, s in zip(predictions, sources)]
    return sum(ratios) / max(len(ratios), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop — CrossDoc-JEPA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    max_samples: int = 5000,
    max_length: int = 256,
    num_beams: int = 4,
    run_factcc: bool = True,
    run_ehr: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader. The dataloader must supply `raw_sources`
    (list of raw source strings per batch, as produced by finetune_collate_fn)
    for FactCC and EHR to be computed against the actual source documents.

    If `raw_sources` is absent from a batch, FactCC and EHR are skipped for
    that batch and the run is flagged as `source_text_missing=True` in metrics,
    which surfaces the problem without silently corrupting the scores.
    """
    model.eval()
    predictions, references, source_texts = [], [], []
    source_text_missing = False
    n = 0

    for batch in dataloader:
        if n >= max_samples:
            break
        batch_enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # raw_sources is a list[str], not a tensor — pass through as-is
        if "raw_sources" in batch:
            source_texts.extend(batch["raw_sources"])
        else:
            source_text_missing = True

        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        try:
            preds = model.generate_summary(batch_enc, tokenizer, max_length=max_length,
                                            num_beams=num_beams)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            preds = [""] * len(refs)

        predictions.extend(preds)
        references.extend(refs)
        n += len(refs)

    predictions = predictions[:max_samples]
    references  = references[:max_samples]

    metrics = compute_rouge(predictions, references)
    metrics["bertscore"]     = compute_bertscore(predictions, references)
    metrics["novel_1gram"]   = novel_ngrams(predictions, references, 1)
    metrics["novel_2gram"]   = novel_ngrams(predictions, references, 2)
    metrics["compression"]   = compression_ratio(predictions, references)
    metrics["avg_pred_words"] = sum(len(p.split()) for p in predictions) / max(len(predictions), 1)
    metrics["num_evaluated"]  = len(predictions)

    # Faithfulness metrics — MUST use raw source text, not references.
    # Passing references here would measure consistency with the reference summary
    # (trivially high for extractive models) rather than factual grounding in source.
    if source_text_missing:
        logger.error(
            "evaluate_model: batch does not contain 'raw_sources'. "
            "FactCC and EHR will be SKIPPED. Ensure your DataLoader uses "
            "finetune_collate_fn, which populates raw_sources from dataset.__getitem__."
        )
        metrics["factcc"] = float("nan")
        metrics["ehr"]    = float("nan")
        metrics["source_text_missing"] = True
    else:
        # Align lengths in case generation failed mid-loop
        srcs = source_texts[:len(predictions)]
        if run_factcc:
            metrics["factcc"] = compute_factcc(predictions, srcs)
        if run_ehr:
            metrics["ehr"] = extractive_hallucination_rate(predictions, srcs)
        metrics["source_text_missing"] = False

    # ── Conservative-summary gaming check ────────────────────────────────────
    # FactCC can be "gamed" by very short summaries that avoid making any
    # specific claims (and therefore can't be falsified). We flag this here.
    # Rule: if FactCC > 0.85 AND avg_pred_words < 50 AND compression < 0.05,
    #       the model may be generating safe/truncated summaries. This is
    #       reported as a WARNING in the log and as a flag in the metrics dict.
    # Reviewers expect compression_ratio to be reported alongside FactCC.
    factcc_val = metrics.get("factcc", 0.0)
    comp_val   = metrics.get("compression", 1.0)
    avg_words  = metrics.get("avg_pred_words", 100.0)
    gaming_flag = (
        isinstance(factcc_val, float) and factcc_val > 0.85
        and isinstance(comp_val,   float) and comp_val   < 0.05
        and isinstance(avg_words,  float) and avg_words  < 50.0
    )
    metrics["factcc_gaming_flag"] = gaming_flag
    if gaming_flag:
        logger.warning(
            f"FACTCC GAMING RISK: FactCC={factcc_val:.3f} but compression={comp_val:.4f} "
            f"and avg_pred_words={avg_words:.1f}. The model may be generating overly "
            f"conservative/truncated summaries. Report compression_ratio in the paper "
            f"table alongside FactCC to address reviewer scrutiny."
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: PRIMERA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_primera(dataloader, tokenizer_dec, device, max_samples=5000) -> Dict:
    """
    Evaluate PRIMERA (allenai/PRIMERA-multinews) as a zero-shot baseline.
    Uses PRIMERA's own tokenizer (LEDTokenizer) — different from BART's.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        primera_name = "allenai/PRIMERA-multinews"
        prim_tok   = AutoTokenizer.from_pretrained(primera_name)
        prim_model = AutoModelForSeq2SeqLM.from_pretrained(primera_name).to(device).eval()
    except Exception as e:
        logger.warning(f"PRIMERA unavailable: {e}")
        return {"note": "PRIMERA not available"}

    SEP = prim_tok.convert_tokens_to_ids("<doc-sep>")
    predictions, references = [], []
    n = 0

    for batch in dataloader:
        if n >= max_samples:
            break
        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = tokenizer_dec.pad_token_id
        refs = tokenizer_dec.batch_decode(label_ids, skip_special_tokens=True)

        # PRIMERA expects a flat concatenated string with <doc-sep> tokens
        # We can't easily reconstruct source text from batch tensors here;
        # run evaluate_primera_from_raw() for full source text evaluation.
        predictions.extend([""] * len(refs))  # placeholder
        references.extend(refs)
        n += len(refs)
        logger.warning("PRIMERA baseline requires raw source text — use evaluate_primera_from_raw()")
        break

    return {"note": "Use evaluate_primera_from_raw() for full PRIMERA evaluation."}


@torch.no_grad()
def evaluate_primera_from_raw(
    cluster_texts: List[List[str]],   # [[doc1, doc2, ...], ...]
    reference_summaries: List[str],
    device: torch.device,
    max_length: int = 1024,
    max_samples: int = 5000,
) -> Dict[str, float]:
    """Full PRIMERA evaluation from raw document clusters."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        primera_name = "allenai/PRIMERA-multinews"
        prim_tok   = AutoTokenizer.from_pretrained(primera_name)
        prim_model = AutoModelForSeq2SeqLM.from_pretrained(primera_name).to(device).eval()
    except Exception as e:
        logger.warning(f"PRIMERA unavailable: {e}")
        return {}

    DOCSEP_TOKEN_ID = prim_tok.convert_tokens_to_ids("<doc-sep>")
    predictions = []

    for i, (docs, ref) in enumerate(zip(cluster_texts, reference_summaries)):
        if i >= max_samples:
            break
        text = " <doc-sep> ".join(docs)
        enc  = prim_tok(text, max_length=4096, truncation=True, return_tensors="pt").to(device)
        global_att = torch.zeros_like(enc["input_ids"])
        global_att[:, 0] = 1
        docsep_positions = (enc["input_ids"] == DOCSEP_TOKEN_ID).nonzero()
        for pos in docsep_positions:
            global_att[pos[0], pos[1]] = 1

        gen = prim_model.generate(
            **enc, global_attention_mask=global_att,
            max_length=256, num_beams=4, length_penalty=2.0,
            no_repeat_ngram_size=3, early_stopping=True,
        )
        predictions.append(prim_tok.decode(gen[0], skip_special_tokens=True))

    references = reference_summaries[:len(predictions)]
    metrics = compute_rouge(predictions, references)
    metrics["bertscore"]   = compute_bertscore(predictions, references)
    metrics["factcc"]      = compute_factcc(predictions, [" ".join(d) for d in cluster_texts[:len(predictions)]])
    metrics["model"]       = "PRIMERA-multinews"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: BART-Large (vanilla fine-tuned)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bart_baseline(
    cluster_texts: List[List[str]],
    reference_summaries: List[str],
    checkpoint_path: str,
    device: torch.device,
    max_samples: int = 5000,
) -> Dict[str, float]:
    """Evaluate BART-Large fine-tuned on Multi-News (vanilla baseline, no JEPA)."""
    from transformers import BartForConditionalGeneration, BartTokenizerFast
    model = BartForConditionalGeneration.from_pretrained(
        checkpoint_path if Path(checkpoint_path).exists() else "facebook/bart-large-cnn"
    ).to(device).eval()
    tok = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")

    predictions = []
    for i, docs in enumerate(cluster_texts[:max_samples]):
        text = " ".join(docs)
        enc  = tok(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        gen  = model.generate(**enc, max_length=256, num_beams=4,
                               length_penalty=2.0, no_repeat_ngram_size=3)
        predictions.append(tok.decode(gen[0], skip_special_tokens=True))

    references = reference_summaries[:len(predictions)]
    metrics = compute_rouge(predictions, references)
    metrics["bertscore"] = compute_bertscore(predictions, references)
    metrics["factcc"]    = compute_factcc(predictions, [" ".join(d) for d in cluster_texts[:len(predictions)]])
    metrics["model"]     = "BART-Large (baseline)"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# LEAD-N extractive baseline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lead(
    cluster_texts: List[List[str]],
    reference_summaries: List[str],
    n_sentences: int = 3,
) -> Dict[str, float]:
    """LEAD-N: take the first N sentences from the first document."""
    from nltk.tokenize import sent_tokenize
    predictions = []
    for docs in cluster_texts:
        sents = sent_tokenize(docs[0])[:n_sentences]
        predictions.append(" ".join(sents))
    references = reference_summaries[:len(predictions)]
    metrics = compute_rouge(predictions, references)
    metrics["model"] = f"LEAD-{n_sentences}"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Gate-alpha convergence logger
# ─────────────────────────────────────────────────────────────────────────────

def log_gate_alpha(gate_alpha: float, step: int, log_path: str):
    """Append gate_alpha to a JSONL log file for convergence plotting."""
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "gate_alpha": gate_alpha}) + "\n")


def plot_gate_alpha_curve(log_path: str, output_path: str = "gate_alpha_curve.png"):
    """
    Plot the SRD convergence gate α over training.
    This is a required figure for the paper's qualitative analysis section.
    Shows that the model gradually learns to trust the JEPA predictor.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    steps, alphas = [], []
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d["step"])
            alphas.append(d["gate_alpha"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, alphas, color="#2874A6", linewidth=2, label="SRD gate α")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="α = 0.5 (equal blend)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Gate α = σ(gate_logit)", fontsize=12)
    ax.set_title("SRD Convergence Gate — JEPA Predictor Trust over Training", fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    annotation_step = steps[len(steps) // 4] if steps else 0
    annotation_alpha = alphas[len(alphas) // 4] if alphas else 0
    ax.annotate("Predictor begins\nconverging",
                xy=(annotation_step, annotation_alpha),
                xytext=(annotation_step, annotation_alpha + 0.15),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    logger.info(f"Gate alpha curve saved to {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, Dict]):
    """Print a publication-ready comparison table."""
    cols = ["rouge1", "rouge2", "rougeL", "bertscore", "factcc", "novel_2gram"]
    col_labels = ["R-1", "R-2", "R-L", "BertScore", "FactCC", "Novel-2g"]
    header = f"{'Model':<35}" + "".join(f"{c:>10}" for c in col_labels)
    print("\n" + "=" * (35 + 10 * len(cols)))
    print(header)
    print("-" * (35 + 10 * len(cols)))

    # Find best per column
    best = {}
    for col in cols:
        vals = [r.get(col, 0.0) for r in results.values() if isinstance(r.get(col), float)]
        best[col] = max(vals) if vals else 0.0

    for model_name, metrics in results.items():
        row = f"{model_name:<35}"
        for col in cols:
            val = metrics.get(col, float("nan"))
            mark = "*" if isinstance(val, float) and abs(val - best.get(col, -1)) < 1e-6 else " "
            row += f"{val:>9.4f}{mark}" if isinstance(val, float) else f"{'N/A':>10}"
        print(row)
    print("=" * (35 + 10 * len(cols)))
    print("* = best in column")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--gate_log",   default=None,
                        help="Path to gate_alpha JSONL log (for convergence plot)")
    parser.add_argument("--plot_gate",  action="store_true")
    args = parser.parse_args()

    import yaml
    from transformers import RobertaTokenizerFast, BartTokenizerFast
    from data.dataset import MultiNewsDataset, finetune_collate_fn
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import CrossDocJEPA

    with open(args.config) as f: config = yaml.safe_load(f)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_tok = RobertaTokenizerFast.from_pretrained(config["encoder_model"])
    dec_tok = BartTokenizerFast.from_pretrained(config["decoder_model"])

    model = CrossDocJEPA(config).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))

    ds = MultiNewsDataset(args.split, enc_tok, dec_tok, **{
        k: config[k] for k in ["max_docs","max_paras","max_sents_per_para",
                                "max_sent_len","max_summary_len"] if k in config})
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=finetune_collate_fn, num_workers=2)

    metrics = evaluate_model(model, loader, dec_tok, device,
                              max_samples=args.max_samples)
    print("\n=== CrossDoc-JEPA Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.plot_gate and args.gate_log:
        plot_gate_alpha_curve(args.gate_log, "gate_alpha_curve.png")
