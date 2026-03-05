"""
JEPA-Guided Salience Estimator (JSE) — v4

Fix: L2 Salience Proxy Bias + Ranking Loss NaN edge case
  When all oracle scores in a batch item fall within a tight band (e.g. all sentences
  have ROUGE recall between 0.20–0.25), `pair_mask` becomes entirely False.
  The original code divided by `pair_mask.float().sum().clamp(1)` per-pair but
  the outer `loss / max(cnt, 1)` was still safe. However the per-item inner
  division `(pair_loss * pair_mask).sum() / pair_mask.sum().clamp(1)` could
  produce NaN if pair_loss contained inf values (from very large margins on
  near-zero denominators with mixed precision fp16).

  Fix: guard against zero-pair batches at BOTH the inner and outer level,
  use `.clamp(min=1)` on the inner denominator, and add `.nan_to_num(0.0)`
  after the division to absorb any residual fp16 overflow.

  Additionally: expose `get_paragraph_salience()` which aggregates sentence-level
  scores to paragraph level. This is consumed by SRD._build_memory() to select
  paragraphs by actual salience rather than L2 norm heuristic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class JEPAGuidedSalienceEstimator(nn.Module):

    def __init__(
        self,
        sent_predictor: nn.Module,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        normalize: bool = True,
        minimum_signal_threshold: float = 0.05,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        super().__init__()
        self.sent_predictor = sent_predictor
        self.hidden_size    = hidden_size
        self.normalize      = normalize
        self.min_signal     = minimum_signal_threshold
        self.temperature    = temperature
        self.default_top_k  = top_k

        self.score_refiner = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    # ── Core salience computation ──────────────────────────────────────────────

    def compute_salience_scores(
        self,
        sentence_embs: torch.Tensor,    # (B, N, P, S, H)
        document_embs: torch.Tensor,    # (B, N, H)
        sent_valid_mask: torch.Tensor,  # (B, N, P, S)
    ) -> torch.Tensor:                  # (B, N, P, S) ∈ [0, 1]
        B, N, P, S, H = sentence_embs.shape
        device = sentence_embs.device
        scores = torch.zeros(B, N, P, S, device=device)

        for doc_i in range(N):
            ctx_idx  = [j for j in range(N) if j != doc_i]
            if not ctx_idx:
                continue
            ctx_embs = document_embs[:, ctx_idx, :]
            ctx_mask = torch.ones(B, len(ctx_idx), device=device)

            for para_j in range(P):
                valid_s = sent_valid_mask[:, doc_i, para_j, :]   # (B, S)
                n_s     = int(valid_s.float().sum(dim=1).max().item())
                if n_s == 0:
                    continue

                actual = sentence_embs[:, doc_i, para_j, :n_s, :]   # (B, S', H)
                pred   = self.sent_predictor(ctx_embs, ctx_mask, n_s)

                # ── Cosine dissimilarity instead of L2 error ─────────────
                # L2 distance in high-dimensional space is dominated by embedding
                # magnitude, which correlates with sentence length/token count —
                # not information density. A long boilerplate sentence will score
                # high purely because its norm is large, not because it is salient.
                #
                # Fix: normalise both vectors to the unit sphere first, then
                # measure 1 - cosine_similarity. This isolates directional divergence
                # (semantic mismatch between prediction and reality) from magnitude.
                # Range: [0, 2] → clamped to [0, 1] via /2 for compatibility
                # with downstream normalization and score_refiner scale.
                actual_n = F.normalize(actual, dim=-1)   # (B, S', H)
                pred_n   = F.normalize(pred,   dim=-1)   # (B, S', H)
                cos_sim  = (actual_n * pred_n).sum(dim=-1)          # (B, S') ∈ [-1, 1]
                cos_dis  = (1.0 - cos_sim).clamp(0.0, 2.0) / 2.0   # (B, S') ∈ [0, 1]

                feat    = torch.cat([actual_n, pred_n, cos_dis.unsqueeze(-1)], dim=-1)
                refined = self.score_refiner(feat).squeeze(-1)

                raw_norm = cos_dis   # already in [0, 1]; no further normalisation needed
                score    = 0.5 * raw_norm + 0.5 * refined
                vmask    = valid_s[:, :n_s].float()
                scores[:, doc_i, para_j, :n_s] = score * vmask

        # ── Robust normalization ───────────────────────────────────────────────
        if self.normalize:
            flat    = scores.view(B, -1)
            s_min   = flat.min(dim=1, keepdim=True)[0]
            s_max   = flat.max(dim=1, keepdim=True)[0]
            spread  = (s_max - s_min)

            normalised = torch.where(
                spread > self.min_signal,
                (flat - s_min) / spread.clamp(min=1e-9),
                (sent_valid_mask.view(B, -1).float()
                 / sent_valid_mask.view(B, -1).float().sum(dim=1, keepdim=True).clamp(min=1)),
            )
            scores = normalised.view(B, N, P, S)

        return scores

    # ── Paragraph-level aggregation (consumed by SRD) ─────────────────────────

    def get_paragraph_salience(
        self,
        sentence_salience: torch.Tensor,  # (B, N, P, S)
        sent_valid_mask: torch.Tensor,    # (B, N, P, S)
    ) -> torch.Tensor:                    # (B, N, P) — max-pooled salience per paragraph
        """
        Aggregates sentence-level salience to paragraph level via max-pool.
        Used by SRD._build_memory() to select paragraphs by salience rather than L2 norm.
        Max-pool preserves the "best sentence in this paragraph" signal.
        """
        masked = sentence_salience.masked_fill(~sent_valid_mask.bool(), -1e9)
        para_sal = masked.max(dim=-1).values   # (B, N, P)
        # Replace -inf (all-padding paragraphs) with 0
        para_sal = para_sal.clamp(min=0.0)
        return para_sal

    # ── Top-k extraction ──────────────────────────────────────────────────────

    def get_top_k(
        self,
        salience: torch.Tensor,     # (B, N, P, S)
        sent_embs: torch.Tensor,    # (B, N, P, S, H)
        valid_mask: torch.Tensor,   # (B, N, P, S)
        k: int,
    ) -> Dict[str, torch.Tensor]:
        B, N, P, S, H = sent_embs.shape
        sc_flat  = salience.view(B, N * P * S)
        emb_flat = sent_embs.view(B, N * P * S, H)
        msk_flat = valid_mask.view(B, N * P * S).float()

        sc_flat  = sc_flat * msk_flat + (1.0 - msk_flat) * -1e9
        k_eff    = min(k, int(msk_flat.sum(dim=1).max().item()))
        topk_sc, topk_idx = sc_flat.topk(k_eff, dim=1)
        topk_emb = emb_flat.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
        topk_w   = F.softmax(topk_sc / self.temperature, dim=1)
        return {"topk_embs": topk_emb, "topk_weights": topk_w, "topk_scores": topk_sc}

    # ── Ranking loss — NaN-safe ────────────────────────────────────────────────

    def ranking_loss(
        self,
        pred: torch.Tensor,
        oracle: torch.Tensor,
        valid: torch.Tensor,
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        Margin-based ranking loss.

        NaN guards:
          1. Skip batch items where fewer than 2 valid sentences exist.
          2. Skip batch items where NO valid (high, low) oracle pairs exist
             (i.e. all oracle scores are within `margin` of each other).
          3. `.nan_to_num(0.0)` absorbs any residual fp16 overflow after division.
        """
        B = pred.size(0)
        flat_p = pred.view(B, -1)
        flat_o = oracle.view(B, -1)
        flat_v = valid.view(B, -1).float()

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        n_valid_items = 0

        for b in range(B):
            idx = flat_v[b].nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue

            p_v = flat_p[b, idx]   # (V,)
            o_v = flat_o[b, idx]   # (V,)

            # Outer product: oracle_i > oracle_j by at least margin → valid pair
            o_high = o_v.unsqueeze(1)   # (V, 1)
            o_low  = o_v.unsqueeze(0)   # (1, V)
            pair_mask = (o_high - o_low) > margin   # (V, V)

            # Skip item if no discriminable pairs exist (all oracle scores clustered)
            n_pairs = pair_mask.float().sum()
            if n_pairs < 1.0:
                continue

            p_high = p_v.unsqueeze(1)
            p_low  = p_v.unsqueeze(0)
            pair_loss = F.relu(margin - (p_high - p_low))   # (V, V)

            item_loss = (pair_loss * pair_mask.float()).sum() / n_pairs
            # Guard against fp16 NaN from very small denominators
            item_loss = torch.nan_to_num(item_loss, nan=0.0, posinf=0.0)

            total_loss  = total_loss + item_loss
            n_valid_items += 1

        if n_valid_items == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return total_loss / n_valid_items

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        sentence_embs: torch.Tensor,
        paragraph_embs: torch.Tensor,
        document_embs: torch.Tensor,
        sent_valid_mask: torch.Tensor,
        para_valid_mask: torch.Tensor,
        oracle_scores: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        k   = top_k or self.default_top_k
        sal = self.compute_salience_scores(sentence_embs, document_embs, sent_valid_mask)

        out = {
            "salience_scores":    sal,
            "paragraph_salience": self.get_paragraph_salience(sal, sent_valid_mask),
        }
        out.update(self.get_top_k(sal, sentence_embs, sent_valid_mask, k))

        if oracle_scores is not None:
            out["salience_loss"] = self.ranking_loss(sal, oracle_scores, sent_valid_mask)

        return out
