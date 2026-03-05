"""
Cross-Document JEPA (CD-JEPA) — v3

Fix: EMA Update Timing
  v1 called update_target_encoder() inside every forward() pass.
  For large datasets this is expensive and — crucially — wrong:
  the target encoder would move before the optimiser has applied gradients,
  meaning the "stable moving target" property is violated.

  Fix: update_target_encoder() is no longer called inside forward().
  Trainers call it explicitly AFTER optimizer.step() (once per gradient step).
  This matches the V-JEPA / I-JEPA / BYOL convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy
import random
import math


class CrossDocumentPredictor(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.target_pos_emb = nn.Embedding(16, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, context_embs: torch.Tensor, context_mask: torch.Tensor,
                num_targets: int) -> torch.Tensor:
        B = context_embs.size(0)
        pos_ids = torch.arange(num_targets, device=context_embs.device)
        queries = self.target_pos_emb(pos_ids).unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~context_mask.bool()
        x = queries
        for layer in self.layers:
            x = layer(tgt=x, memory=context_embs, memory_key_padding_mask=key_padding_mask)
        return self.out_norm(x)


class CrossDocumentJEPA(nn.Module):
    """
    Pretraining: predict target document representations from context documents.
    EMA update is NOT called here — the trainer calls update_target_encoder()
    once per optimiser step, after gradients have been applied.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int = 768,
                 num_heads: int = 8, predictor_layers: int = 4,
                 dropout: float = 0.1, ema_momentum: float = 0.996,
                 min_context_ratio: float = 0.3, max_context_ratio: float = 0.7,
                 prediction_level: str = "hierarchical"):
        super().__init__()
        self.hidden_size = hidden_size
        self.ema_momentum = ema_momentum
        self.min_ctx = min_context_ratio
        self.max_ctx = max_context_ratio
        self.prediction_level = prediction_level

        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.doc_predictor = CrossDocumentPredictor(hidden_size, num_heads, predictor_layers, dropout)

        if prediction_level in ("para", "hierarchical"):
            self.para_predictor = CrossDocumentPredictor(
                hidden_size, num_heads, max(predictor_layers // 2, 1), dropout)

        # For sentence-level pretraining signal (used by JSE sent_predictor warm-up)
        self.sent_predictor = CrossDocumentPredictor(
            hidden_size, num_heads, max(predictor_layers // 2, 1), dropout)

        self.doc_loss_w = 1.0
        self.para_loss_w = 0.5
        self.sent_loss_w = 0.3  # sentence-level JEPA signal

    # ── EMA update — called by TRAINER after optimizer.step() ─────────────────

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None,
                               broadcast_dist: bool = False):
        """
        EMA update: φ ← τ·φ + (1-τ)·θ
        Called AFTER optimizer.step() by the trainer.

        broadcast_dist: if True, broadcast target_encoder weights from rank 0
            to all other ranks after the EMA update. This is required in DDP
            training to prevent target representations from diverging across GPUs.

            Why divergence happens without this:
              In DDP, each GPU computes EMA independently using its own local copy
              of context_encoder weights (which are gradient-synced via all-reduce)
              and its own local target_encoder weights. Although context_encoder
              weights are identical across GPUs after all-reduce, floating-point
              order of operations in mul_+add_ can produce microscopic differences
              that accumulate over thousands of steps, causing each GPU's target
              encoder to follow a slightly different trajectory. This produces
              inconsistent JEPA prediction targets across the batch, introducing
              gradient noise that destabilises training.

              Fix: after every EMA step, rank 0's target_encoder is the canonical
              source. We broadcast it to all other ranks, guaranteeing all GPUs
              share identical target representations.

            Cost: one all-reduce per EMA update (once per gradient step).
            Overhead is negligible — target_encoder has the same parameter count
            as context_encoder, and EMA updates happen once per step anyway.
        """
        import torch.distributed as _dist
        tau = momentum if momentum is not None else self.ema_momentum
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            p_tgt.data.mul_(tau).add_((1.0 - tau) * p_ctx.data)

        if broadcast_dist and _dist.is_available() and _dist.is_initialized():
            for p_tgt in self.target_encoder.parameters():
                _dist.broadcast(p_tgt.data, src=0)

    # ── Context/target split ──────────────────────────────────────────────────

    def _split(self, N: int) -> Tuple[List[int], List[int]]:
        n_ctx = max(1, int(N * random.uniform(self.min_ctx, self.max_ctx)))
        n_ctx = min(n_ctx, N - 1)
        idx = list(range(N))
        random.shuffle(idx)
        return sorted(idx[:n_ctx]), sorted(idx[n_ctx:])

    # ── Forward (pretraining only) ────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        NOTE: Does NOT call update_target_encoder().
        The trainer is responsible for calling it after optimizer.step().
        """
        B, N = batch["sentence_input_ids"].shape[:2]
        device = batch["sentence_input_ids"].device

        # Encode with context encoder (fθ)
        ctx_docs, ctx_paras, ctx_sents = [], [], []
        for i in range(N):
            out = self.context_encoder(
                batch["sentence_input_ids"][:, i],
                batch["sentence_attn_mask"][:, i],
                batch["sentence_valid_mask"][:, i],
                batch["paragraph_valid_mask"][:, i],
            )
            ctx_docs.append(out["document_embs"])
            ctx_paras.append(out["paragraph_embs"])
            ctx_sents.append(out["sentence_embs"])

        ctx_doc_stack  = torch.stack(ctx_docs,  dim=1)   # (B, N, H)
        ctx_para_stack = torch.stack(ctx_paras, dim=1)   # (B, N, P, H)
        ctx_sent_stack = torch.stack(ctx_sents, dim=1)   # (B, N, P, S, H)

        # Encode with target encoder (fφ) — stop gradient
        with torch.no_grad():
            tgt_docs, tgt_paras, tgt_sents = [], [], []
            for i in range(N):
                out = self.target_encoder(
                    batch["sentence_input_ids"][:, i],
                    batch["sentence_attn_mask"][:, i],
                    batch["sentence_valid_mask"][:, i],
                    batch["paragraph_valid_mask"][:, i],
                )
                tgt_docs.append(out["document_embs"])
                tgt_paras.append(out["paragraph_embs"])
                tgt_sents.append(out["sentence_embs"])

            tgt_doc_stack  = torch.stack(tgt_docs,  dim=1)
            tgt_para_stack = torch.stack(tgt_paras, dim=1)
            tgt_sent_stack = torch.stack(tgt_sents, dim=1)

        ctx_idx, tgt_idx = self._split(N)
        n_tgt = len(tgt_idx)

        # Single-document batches: no context for JEPA; return zero JEPA loss (no architecture change)
        if len(ctx_idx) == 0:
            losses = {
                "doc_loss": torch.tensor(0.0, device=device),
                "total_jepa_loss": torch.tensor(0.0, device=device),
            }
            if self.prediction_level in ("para", "hierarchical"):
                losses["para_loss"] = torch.tensor(0.0, device=device)
            if self.prediction_level == "hierarchical":
                losses["sent_loss"] = torch.tensor(0.0, device=device)
            return losses

        ctx_embs  = ctx_doc_stack[:, ctx_idx, :]   # (B, |C|, H)
        tgt_embs  = tgt_doc_stack[:, tgt_idx, :]   # (B, |T|, H)
        ctx_mask  = torch.ones(B, len(ctx_idx), device=device)

        # ── Document-level prediction ─────────────────────────────────────────
        pred_docs = self.doc_predictor(ctx_embs, ctx_mask, n_tgt)   # (B, |T|, H)
        doc_loss  = F.mse_loss(pred_docs, tgt_embs)
        total     = self.doc_loss_w * doc_loss
        losses    = {"doc_loss": doc_loss}

        # ── Paragraph-level prediction ────────────────────────────────────────
        if self.prediction_level in ("para", "hierarchical"):
            P = ctx_para_stack.shape[2]
            ctx_paras_flat = ctx_para_stack[:, ctx_idx].reshape(B, len(ctx_idx) * P, self.hidden_size)
            ctx_para_mask  = torch.ones(B, len(ctx_idx) * P, device=device)
            para_loss = torch.tensor(0.0, device=device)
            for r, abs_i in enumerate(tgt_idx):
                tgt_paras = tgt_para_stack[:, abs_i]           # (B, P, H)
                pmask     = batch["paragraph_valid_mask"][:, abs_i]  # (B, P)
                n_valid   = int(pmask.float().sum(1).max().item())
                if n_valid == 0:
                    continue
                # Augment context with already-predicted doc embedding
                aug_ctx  = torch.cat([ctx_paras_flat, pred_docs[:, r:r+1, :]], dim=1)
                aug_mask = torch.cat([ctx_para_mask, torch.ones(B, 1, device=device)], dim=1)
                pred_p   = self.para_predictor(aug_ctx, aug_mask, n_valid)
                tgt_p    = tgt_paras[:, :n_valid]
                pmask_v  = pmask[:, :n_valid].float()
                para_loss = para_loss + (
                    F.mse_loss(pred_p * pmask_v.unsqueeze(-1),
                               tgt_p  * pmask_v.unsqueeze(-1),
                               reduction="sum")
                    / (pmask_v.sum() + 1e-9)
                )
            para_loss = para_loss / max(n_tgt, 1)
            losses["para_loss"] = para_loss
            total = total + self.para_loss_w * para_loss

        # ── Sentence-level prediction (warms up JSE's sent_predictor) ─────────
        # Sample one target document; predict sentence reps from context doc embs
        if self.prediction_level == "hierarchical" and n_tgt > 0:
            abs_t = tgt_idx[0]
            tgt_sent_embs = tgt_sent_stack[:, abs_t]       # (B, P, S, H)
            sent_mask     = batch["sentence_valid_mask"][:, abs_t]  # (B, P, S)
            B2, P, S, H   = tgt_sent_embs.shape
            para_valid    = batch["paragraph_valid_mask"][:, abs_t]  # (B, P)

            # ── Sample a RANDOM valid paragraph per batch item ────────────────
            # Hardcoding paragraph 0 introduces "lead bias": the sent_predictor
            # only sees introductory paragraphs and is out-of-distribution for
            # middle/end paragraphs during JSE salience estimation at finetuning.
            # Fix: for each batch item, uniformly sample one paragraph from its
            # set of valid (non-padding) paragraphs.
            selected_para_idx = torch.zeros(B2, dtype=torch.long, device=device)
            for b in range(B2):
                valid_para_indices = para_valid[b].nonzero(as_tuple=True)[0]
                if len(valid_para_indices) > 0:
                    chosen = valid_para_indices[random.randrange(len(valid_para_indices))]
                    selected_para_idx[b] = chosen
                # If no valid paragraph (edge case), stays 0 — handled by mask below

            # Gather selected paragraph's sentences: (B, S, H) and (B, S)
            tgt_flat = tgt_sent_embs[
                torch.arange(B2, device=device), selected_para_idx
            ]   # (B, S, H)
            smask_v  = sent_mask[
                torch.arange(B2, device=device), selected_para_idx
            ].float()   # (B, S)

            n_valid_s = int(smask_v.sum(dim=1).max().item())
            if n_valid_s > 0:
                tgt_flat   = tgt_flat[:, :n_valid_s]      # (B, S', H)
                smask_v    = smask_v[:, :n_valid_s]        # (B, S')
                pred_sents = self.sent_predictor(ctx_embs, ctx_mask, n_valid_s)
                sent_loss  = (
                    F.mse_loss(pred_sents * smask_v.unsqueeze(-1),
                               tgt_flat   * smask_v.unsqueeze(-1),
                               reduction="sum")
                    / (smask_v.sum() + 1e-9)
                )
                losses["sent_loss"] = sent_loss
                total = total + self.sent_loss_w * sent_loss

        losses["total_jepa_loss"] = total
        return losses

    # ── Encode all docs (used during finetuning / inference) ─────────────────

    def get_all_document_representations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        N = batch["sentence_input_ids"].shape[1]
        docs, paras, sents = [], [], []
        for i in range(N):
            out = self.context_encoder(
                batch["sentence_input_ids"][:, i],
                batch["sentence_attn_mask"][:, i],
                batch["sentence_valid_mask"][:, i],
                batch["paragraph_valid_mask"][:, i],
            )
            docs.append(out["document_embs"])
            paras.append(out["paragraph_embs"])
            sents.append(out["sentence_embs"])
        return {
            "document_embs": torch.stack(docs,  dim=1),
            "paragraph_embs": torch.stack(paras, dim=1),
            "sentence_embs": torch.stack(sents, dim=1),
        }

    # ── LOO prediction (used by SRD & JSE) ───────────────────────────────────

    def predict_all_from_all(self, doc_embs: torch.Tensor, *_) -> torch.Tensor:
        B, N, H = doc_embs.shape
        if N == 1:
            # Leave-one-out undefined; return identity (no predictor call with empty context)
            return doc_embs
        preds = []
        for t in range(N):
            ctx = doc_embs[:, [j for j in range(N) if j != t], :]
            mask = torch.ones(B, N - 1, device=doc_embs.device)
            preds.append(self.doc_predictor(ctx, mask, 1).squeeze(1))
        return torch.stack(preds, dim=1)
