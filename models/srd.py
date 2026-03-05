"""
Summary Representation Distiller (SRD) — v4

Fix: Salience-Aware Memory Selection
  v2 selected paragraphs for memory by highest L2 norm. L2 norm correlates
  with sentence LENGTH (more tokens → larger norm), not information value.
  Long boilerplate paragraphs (legal disclaimers, datelines, author bios)
  can dominate the memory simply because they are verbose.

  Fix: _build_memory() now accepts `paragraph_salience` (B, N, P) computed by
  JSE.get_paragraph_salience(), and selects the top-M paragraphs by SALIENCE.
  This creates a principled end-to-end pipeline where:
    JSE predicts sentence salience → aggregated to para salience → SRD uses
    only the most salient paragraphs as cross-attention memory.

  Fallback: if `paragraph_salience` is None (e.g. during pretraining when JSE
  hasn't run yet), the L2-norm heuristic is preserved as a fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SummaryRepresentationDistiller(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        num_summary_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_memory_paragraphs: int = 64,
    ):
        super().__init__()
        self.hidden_size            = hidden_size
        self.num_summary_queries    = num_summary_queries
        self.max_memory_paragraphs  = max_memory_paragraphs

        self.summary_queries = nn.Parameter(
            torch.randn(1, num_summary_queries, hidden_size) * 0.02
        )
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.doc_pos_emb = nn.Embedding(32, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

        # Convergence gate: α = sigmoid(gate_logit), init near 0 (≈0.047)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    # ── Salience-aware memory builder ─────────────────────────────────────────

    def _build_memory(
        self,
        document_embs: torch.Tensor,                        # (B, N, H)
        paragraph_embs: torch.Tensor,                       # (B, N, P, H)
        para_mask: torch.Tensor,                            # (B, N, P)
        paragraph_salience: Optional[torch.Tensor] = None,  # (B, N, P) from JSE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = document_embs.shape
        device   = document_embs.device

        # ── Doc-level tokens ──────────────────────────────────────────────────
        pos_ids    = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        doc_tokens = document_embs + self.doc_pos_emb(pos_ids)   # (B, N, H)
        doc_mask_t = torch.ones(B, N, device=device)

        # ── Paragraph selection ───────────────────────────────────────────────
        B2, N2, P, H2 = paragraph_embs.shape
        para_flat = paragraph_embs.reshape(B2, N2 * P, H2)   # (B, N*P, H)
        mask_flat = para_mask.reshape(B2, N2 * P).float()     # (B, N*P)

        if paragraph_salience is not None:
            # PRIMARY: select by JSE paragraph salience (end-to-end principled)
            sal_flat = paragraph_salience.reshape(B2, N2 * P)   # (B, N*P)
            scores   = sal_flat * mask_flat                       # zero out padding
        else:
            # FALLBACK: uniform random selection over valid paragraphs.
            #
            # v3 used L2 norm as the fallback heuristic, which correlates with
            # paragraph length rather than information value. This was retained
            # only for pretraining when JSE hasn't run. But:
            #   (a) During evaluation the fallback must NEVER trigger — if it does
            #       it will silently degrade FactCC/EHR scores and produce invalid
            #       paper results. We now raise a hard error in eval mode.
            #   (b) During pretraining, uniform random selection is preferable to
            #       L2 norm: it introduces no systematic bias and is equivalent to
            #       random memory dropout, which acts as a regulariser.
            #
            if not self.training:
                raise RuntimeError(
                    "SRD._build_memory() called without paragraph_salience during "
                    "evaluation. This will silently corrupt faithfulness metrics. "
                    "Ensure jse_out['paragraph_salience'] is passed to srd.forward()."
                )
            import warnings, traceback
            warnings.warn(
                "SRD: paragraph_salience is None — using uniform random selection "
                "(pretraining fallback). Ensure JSE is wired in for finetuning.",
                RuntimeWarning, stacklevel=4,
            )
            # Uniform random scores over valid paragraphs — no length bias
            scores = (torch.rand_like(mask_flat) * mask_flat)

        M = min(self.max_memory_paragraphs,
                int(mask_flat.sum(dim=1).max().item()))
        M = max(M, 1)

        topk_scores, topk_idx = scores.topk(M, dim=1)       # (B, M)
        sel_para = para_flat.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, H)     # (B, M, H)
        )
        sel_mask = (topk_scores > 0).float()                  # (B, M)

        memory   = torch.cat([doc_tokens, sel_para],  dim=1)  # (B, N+M, H)
        mem_mask = torch.cat([doc_mask_t, sel_mask],  dim=1)  # (B, N+M)
        return memory, mem_mask

    # ── Convergence-gated predictor blend ─────────────────────────────────────

    def _blend(self, actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate_logit)   # scalar ∈ (0,1), init ≈ 0.047

        cos_sim = F.cosine_similarity(
            actual.reshape(-1, actual.size(-1)),
            predicted.reshape(-1, predicted.size(-1)),
            dim=-1,
        ).reshape(actual.shape[0], actual.shape[1], 1)

        safe  = (cos_sim > 0.0).float()
        return (1.0 - alpha * safe) * actual + (alpha * safe) * predicted

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        document_embs: torch.Tensor,                          # (B, N, H)
        paragraph_embs: torch.Tensor,                         # (B, N, P, H)
        para_mask: torch.Tensor,                              # (B, N, P)
        predicted_doc_embs: Optional[torch.Tensor] = None,   # (B, N, H)
        paragraph_salience: Optional[torch.Tensor] = None,   # (B, N, P) from JSE
    ) -> Dict[str, torch.Tensor]:
        B = document_embs.size(0)

        fused = (self._blend(document_embs, predicted_doc_embs)
                 if predicted_doc_embs is not None else document_embs)

        # Salience-aware memory (falls back to L2 norm if salience is None)
        memory, mem_mask = self._build_memory(fused, paragraph_embs, para_mask,
                                              paragraph_salience=paragraph_salience)
        kpm = ~mem_mask.bool()

        queries = self.summary_queries.expand(B, -1, -1)
        x = queries
        for layer in self.layers:
            x = layer(tgt=x, memory=memory, memory_key_padding_mask=kpm)

        return {
            "summary_emb": self.output_norm(x),            # (B, K, H)
            "memory":      memory,
            "memory_mask": mem_mask,
            "gate_alpha":  torch.sigmoid(self.gate_logit).item(),
        }
