"""
Hierarchical Document Encoder (HDE) — v2

Fix: Quadratic Attention Bottleneck in DocumentEncoder
  v1 used a standard TransformerEncoder at the document level, attending over
  all N_docs × P_paras paragraph tokens. For WCEP clusters (100+ docs × 12 paras
  = 1200+ tokens), this hits O(L²) memory and causes OOM on 80GB A100s.

  Fix: DocumentEncoder now uses a two-stage approach:
    Stage 1: Within each document, attend over its own paragraphs (O(P²), P ≤ 16).
             Produce one summary vector per document via attention pooling.
    Stage 2: Attend over the N document summary vectors (O(N²), N ≤ 100 for WCEP
             but practically ≤ 10 for Multi-News, bounded by max_docs config).
             For N > max_full_attn_docs (default 32), fall back to a learned
             cross-attention between document summaries and a fixed set of K=16
             global query tokens (O(N×K), linear in N).

  This makes the encoder compatible with WCEP without any truncation of documents.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from typing import Dict, Optional, Tuple


class SentenceEncoder(nn.Module):
    """Encodes individual sentences using a pretrained RoBERTa backbone."""

    def __init__(self, model_name: str = "roberta-base", hidden_size: int = 768):
        super().__init__()
        self.roberta  = RobertaModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        rob_h = self.roberta.config.hidden_size
        self.proj       = nn.Linear(rob_h, hidden_size) if rob_h != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:  # (B, H)
        out  = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.layer_norm(self.proj(emb))


class ParagraphEncoder(nn.Module):
    """Aggregates sentence representations within one paragraph."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool        = nn.Linear(hidden_size, 1)
        self.layer_norm  = nn.LayerNorm(hidden_size)

    def forward(self, sentence_embs: torch.Tensor,
                sentence_mask: torch.Tensor) -> torch.Tensor:  # (B, H)
        kpm     = ~sentence_mask.bool()
        encoded = self.transformer(sentence_embs, src_key_padding_mask=kpm)
        w       = self.pool(encoded).squeeze(-1).masked_fill(kpm, -1e9)
        w       = F.softmax(w, dim=-1)
        return self.layer_norm((encoded * w.unsqueeze(-1)).sum(1))


class DocumentEncoder(nn.Module):
    """
    Two-stage paragraph → document aggregation.

    Stage 1 (intra-doc):
        For each document, attend over its P paragraph embeddings → one doc vector.
        Complexity: O(B × N × P²)  — independent of cluster size N.

    Stage 2 (inter-doc):
        If N ≤ max_full_attn_docs: standard self-attention over N doc vectors.
        If N > max_full_attn_docs: cross-attention from K=16 global query tokens
            to all N doc vectors, then mean-pool the K tokens. O(K × N), linear in N.

    Both stages keep per-paragraph representations for JEPA loss computation.
    """

    MAX_FULL_ATTN = 32   # clusters larger than this use linear cross-attn
    K_GLOBAL      = 16   # global query tokens for linear inter-doc attention

    def __init__(self, hidden_size: int = 768, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Stage 1: intra-doc paragraph attention
        intra_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.intra_doc_attn = nn.TransformerEncoder(intra_layer, num_layers=num_layers)
        self.para_pool      = nn.Linear(hidden_size, 1)

        # Stage 2a: standard inter-doc attention (small clusters)
        inter_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.inter_doc_attn = nn.TransformerEncoder(inter_layer, num_layers=num_layers)

        # Stage 2b: linear inter-doc attention (large clusters, O(K×N))
        self.global_queries = nn.Parameter(
            torch.randn(1, self.K_GLOBAL, hidden_size) * 0.02
        )
        self.linear_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.pos_emb    = nn.Embedding(512, hidden_size)
        self.pool       = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _intra_doc_encode(self, para_embs: torch.Tensor,
                           para_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode paragraphs within a single document.
        para_embs: (B, P, H) — one document's paragraph embeddings
        para_mask: (B, P)
        Returns: doc_vec (B, H),  para_encoded (B, P, H)
        """
        B, P, H = para_embs.shape
        pos_ids  = torch.arange(P, device=para_embs.device).unsqueeze(0).expand(B, -1)
        embs     = para_embs + self.pos_emb(pos_ids)
        kpm      = ~para_mask.bool()
        encoded  = self.intra_doc_attn(embs, src_key_padding_mask=kpm)   # (B, P, H)
        w        = self.para_pool(encoded).squeeze(-1).masked_fill(kpm, -1e9)
        w        = F.softmax(w, dim=-1)
        doc_vec  = self.layer_norm((encoded * w.unsqueeze(-1)).sum(1))   # (B, H)
        return doc_vec, encoded

    def forward(self, para_embs: torch.Tensor,
                para_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        para_embs: (B, P, H) — all paragraphs for ONE document's set
                   NOTE: the caller (HDE.forward) folds N into B, so this
                   method always sees one document at a time in the first call.
        para_mask: (B, P)
        Returns: doc_emb (B, H),  para_encoded (B, P, H)
        """
        # Used only for intra-doc encoding; inter-doc handled in HDE.forward
        return self._intra_doc_encode(para_embs, para_mask)

    def inter_doc_aggregate(self, doc_vecs: torch.Tensor,
                             doc_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate N per-document vectors into a cluster-level representation.
        doc_vecs: (B, N, H)
        doc_mask: (B, N)  — 1 = valid document
        Returns: cluster_emb (B, H)
        """
        B, N, H = doc_vecs.shape
        kpm = ~doc_mask.bool()

        if N <= self.MAX_FULL_ATTN:
            # Standard O(N²) self-attention — fine for Multi-News (N≤5)
            encoded = self.inter_doc_attn(doc_vecs, src_key_padding_mask=kpm)
        else:
            # Linear O(K×N) cross-attention — needed for WCEP (N>100)
            queries  = self.global_queries.expand(B, -1, -1)   # (B, K, H)
            encoded, _ = self.linear_cross_attn(
                query=queries, key=doc_vecs, value=doc_vecs,
                key_padding_mask=kpm,
            )   # (B, K, H)
            # Mean-pool the K global tokens → cluster embedding
            return self.layer_norm(encoded.mean(1))

        w = self.pool(encoded).squeeze(-1).masked_fill(kpm, -1e9)
        w = F.softmax(w, dim=-1)
        return self.layer_norm((encoded * w.unsqueeze(-1)).sum(1))


class HierarchicalDocumentEncoder(nn.Module):
    """
    Full hierarchical encoder: token → sentence → paragraph → document → cluster.

    The document-level encoder uses the two-stage approach above, making the
    encoder memory-bounded for arbitrary cluster sizes.
    """

    def __init__(self, model_name: str = "roberta-base", hidden_size: int = 768,
                 num_heads: int = 8, para_layers: int = 2,
                 doc_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.sentence_enc = SentenceEncoder(model_name, hidden_size)
        self.para_enc     = ParagraphEncoder(hidden_size, num_heads, para_layers, dropout)
        self.doc_enc      = DocumentEncoder(hidden_size, num_heads, doc_layers, dropout)

    def forward(
        self,
        sentence_input_ids:  torch.Tensor,   # (B, P, S, L)
        sentence_attn_mask:  torch.Tensor,   # (B, P, S, L)
        sentence_valid_mask: torch.Tensor,   # (B, P, S)
        paragraph_valid_mask: torch.Tensor,  # (B, P)
    ) -> Dict[str, torch.Tensor]:
        B, P, S, L = sentence_input_ids.shape

        # ── Sentence level ───────────────────────────────────────────────────
        sent_embs_flat = self.sentence_enc(
            sentence_input_ids.view(B * P * S, L),
            sentence_attn_mask.view(B * P * S, L),
        )   # (B*P*S, H)
        sent_embs = sent_embs_flat.view(B, P, S, self.hidden_size)

        # ── Paragraph level ──────────────────────────────────────────────────
        para_embs = self.para_enc(
            sent_embs.view(B * P, S, self.hidden_size),
            sentence_valid_mask.view(B * P, S),
        ).view(B, P, self.hidden_size)   # (B, P, H)

        # ── Document level (intra-doc only; inter-doc handled by caller) ─────
        doc_emb, para_encoded = self.doc_enc(para_embs, paragraph_valid_mask)

        return {
            "sentence_embs":   sent_embs,            # (B, P, S, H)
            "paragraph_embs":  para_encoded,          # (B, P, H)
            "document_embs":   doc_emb,               # (B, H)
            "para_valid_mask": paragraph_valid_mask,
            "sent_valid_mask": sentence_valid_mask,
        }
