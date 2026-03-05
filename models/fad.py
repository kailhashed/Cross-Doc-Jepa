"""
Faithfulness-Aware Decoder (FAD) — v5

Fix: Thread-Safety Violation in Inference Cache
═══════════════════════════════════════════════

v4 stored projected KV tensors as attributes on nn.Module instances:
    layer._cached_sk = sk

This is not thread-safe. If two threads call generate() concurrently on the
same model (e.g. multi-threaded evaluation, DataParallel inference, or any
batched eval harness that parallelizes over examples), they will overwrite each
other's `_cached_*` tensors, producing silently wrong outputs.

Fix: All per-call state lives exclusively in thread-local storage (`_tls`),
keyed by the Python `id()` of each patched layer. The nn.Module is never
mutated at inference time — it remains stateless and purely functional.

Thread-local context layout
    _tls.mode          : "train" | "infer" | None
    _tls.layer_caches  : dict[int, (sk, sv, gk, gv)]  keyed by id(layer)
    _tls.train_*       : raw encoder tensors for the training path
    _tls.orig_B        : original batch size (training path)

Inference path (generate):
    _prepare_inference_cache() projects + expands tensors into _tls.layer_caches
    PatchedBartDecoderLayer.forward() reads from _tls.layer_caches[id(self)]
    No nn.Module attribute is ever set or read.

Training path (forward):
    _set_train_ctx() stores raw encoder embs in _tls.train_*
    Each layer projects them independently (once per grad step).
    No _tls.layer_caches is created.

Concurrency guarantee:
    Two threads running generate() simultaneously each have their own _tls,
    so their caches are completely isolated. The model weights are never written.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartDecoderLayer
from typing import Dict, List, Optional, Tuple
import threading


# ─────────────────────────────────────────────────────────────────────────────
# Thread-local context — ALL per-call state lives here, never on nn.Module
# ─────────────────────────────────────────────────────────────────────────────

_tls = threading.local()

_CacheEntry = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # sk,sv,gk,gv


def _set_infer_ctx(layer_caches: Dict[int, _CacheEntry]):
    """
    Store pre-projected, pre-expanded KV tensors for all patched layers.
    layer_caches: {id(layer): (sk, sv, gk, gv)}  — all in query_dim space
    """
    _tls.mode         = "infer"
    _tls.layer_caches = layer_caches
    _tls.train_summary  = None
    _tls.train_salience = None
    _tls.train_sal_w    = None
    _tls.orig_B         = None


def _set_train_ctx(summary_raw: torch.Tensor,
                   salience_raw: torch.Tensor,
                   salience_w: torch.Tensor,
                   orig_B: int):
    _tls.mode           = "train"
    _tls.layer_caches   = None
    _tls.train_summary  = summary_raw
    _tls.train_salience = salience_raw
    _tls.train_sal_w    = salience_w
    _tls.orig_B         = orig_B


def _clear_ctx():
    _tls.mode           = None
    _tls.layer_caches   = None
    _tls.train_summary  = None
    _tls.train_salience = None
    _tls.train_sal_w    = None
    _tls.orig_B         = None


def _has_ctx() -> bool:
    return getattr(_tls, "mode", None) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Dual Cross-Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class DualCrossAttentionBlock(nn.Module):
    """
    query_dim  = BART decoder hidden (1024 for bart-large)
    memory_dim = HDE encoder hidden  (768 for roberta-base)
    Separate k/v projections handle the dimension mismatch.
    """

    def __init__(self, query_dim: int, memory_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert query_dim % num_heads == 0

        self.sk_proj = nn.Linear(memory_dim, query_dim, bias=False)
        self.sv_proj = nn.Linear(memory_dim, query_dim, bias=False)
        self.gk_proj = nn.Linear(memory_dim, query_dim, bias=False)
        self.gv_proj = nn.Linear(memory_dim, query_dim, bias=False)

        self.summary_attn  = nn.MultiheadAttention(query_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.salience_attn = nn.MultiheadAttention(query_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.summary_norm  = nn.LayerNorm(query_dim)
        self.salience_norm = nn.LayerNorm(query_dim)
        self.gate          = nn.Linear(query_dim * 2, query_dim)
        self.drop          = nn.Dropout(dropout)

    @torch.no_grad()
    def project(self, summary_emb: torch.Tensor,
                salience_emb: torch.Tensor,
                salience_w: torch.Tensor) -> _CacheEntry:
        """
        Project raw encoder embs → query_dim and pre-weight salience values.
        Called ONCE in _prepare_inference_cache; result stored in _tls, not on self.
        """
        sk = self.sk_proj(summary_emb)
        sv = self.sv_proj(summary_emb)
        gk = self.gk_proj(salience_emb)
        gv = self.gv_proj(salience_emb) * salience_w.unsqueeze(-1)
        return sk, sv, gk, gv

    def forward(self, h: torch.Tensor,
                sk: torch.Tensor, sv: torch.Tensor,
                gk: torch.Tensor, gv: torch.Tensor) -> torch.Tensor:
        out_s, _ = self.summary_attn(query=h, key=sk, value=sv)
        h_s = self.summary_norm(h + self.drop(out_s))

        out_g, _ = self.salience_attn(query=h, key=gk, value=gv)
        h_g = self.salience_norm(h + self.drop(out_g))

        gate = torch.sigmoid(self.gate(torch.cat([h_s, h_g], dim=-1)))
        return gate * h_s + (1.0 - gate) * h_g


# ─────────────────────────────────────────────────────────────────────────────
# Patched BART Decoder Layer — reads ALL context from _tls, never from self.*
# ─────────────────────────────────────────────────────────────────────────────

class PatchedBartDecoderLayer(nn.Module):
    """
    Wraps BartDecoderLayer, applies DualCrossAttentionBlock after it.

    The module itself carries NO mutable inference state.
    Both inference and training context are read from _tls (thread-local).
    This makes the module safe for concurrent generate() calls from multiple threads.

    Inference path:
        _tls.layer_caches[id(self)] holds (sk, sv, gk, gv) pre-projected and
        pre-expanded to (B*num_beams, ...). The forward reads and applies them
        with zero overhead — no projection, no expansion.

    Training path:
        _tls.train_* holds raw encoder embs. This layer projects them on the fly
        (one matmul per grad step — negligible).
    """

    def __init__(self, original: BartDecoderLayer, dual: DualCrossAttentionBlock):
        super().__init__()
        self.orig = original
        self.dual = dual

    def forward(self, hidden_states, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                layer_head_mask=None, cross_attn_layer_head_mask=None,
                past_key_value=None, output_attentions=False, use_cache=True,
                **kwargs):
        # Pass through any new transformers API kwargs (e.g. past_key_values, cache_position)
        out = self.orig(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        if not _has_ctx():
            return out

        h   = out[0]
        dev = h.device
        mode = _tls.mode

        # ── INFERENCE: read pre-projected, pre-expanded tensors from _tls ─────
        if mode == "infer":
            layer_id = id(self)
            cache = _tls.layer_caches.get(layer_id)
            if cache is None:
                return out   # This layer was not patched (shouldn't happen)
            sk, sv, gk, gv = (t.to(dev) for t in cache)
            # Tensors are already expanded to (B*num_beams, ...) — no expand needed
            h = self.dual.forward(h, sk, sv, gk, gv)

        # ── TRAINING: project raw encoder embs (once per grad step) ───────────
        elif mode == "train":
            summary_raw  = _tls.train_summary.to(dev)
            salience_raw = _tls.train_salience.to(dev)
            sal_w        = _tls.train_sal_w.to(dev)
            B_eff        = h.size(0)
            # Handle DDP micro-batch split: B_eff may be < orig_B on small batches
            # but in the forward pass it's always orig_B; repeat_interleave handles
            # the unlikely case where pipeline parallel splits the batch differently.
            if B_eff != _tls.orig_B:
                mul = B_eff // _tls.orig_B
                summary_raw  = summary_raw.repeat_interleave(mul, dim=0)
                salience_raw = salience_raw.repeat_interleave(mul, dim=0)
                sal_w        = sal_w.repeat_interleave(mul, dim=0)
            sk, sv, gk, gv = self.dual.project(summary_raw, salience_raw, sal_w)
            h = self.dual.forward(h, sk, sv, gk, gv)

        return (h,) + out[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness-Aware Decoder
# ─────────────────────────────────────────────────────────────────────────────

class FaithfulnessAwareDecoder(nn.Module):

    def __init__(
        self,
        bart_model_name: str = "facebook/bart-large-cnn",
        encoder_hidden_size: int = 768,
        num_dual_attn_layers: int = 4,
        faithfulness_margin: float = 0.85,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart_hidden = self.bart.config.d_model   # 1024 for bart-large
        self.enc_hidden  = encoder_hidden_size        # 768 for roberta-base
        self.faithfulness_margin = faithfulness_margin

        total_layers = len(self.bart.model.decoder.layers)
        patch_idx    = list(range(total_layers - num_dual_attn_layers, total_layers))

        self.dual_blocks = nn.ModuleList([
            DualCrossAttentionBlock(
                query_dim=self.bart_hidden,
                memory_dim=encoder_hidden_size,
                num_heads=8,
                dropout=dropout,
            )
            for _ in patch_idx
        ])

        # Register patched layers — store their Python ids for _tls keying
        self._patched_layer_ids: List[int] = []
        for i, li in enumerate(patch_idx):
            orig    = self.bart.model.decoder.layers[li]
            patched = PatchedBartDecoderLayer(orig, self.dual_blocks[i])
            self.bart.model.decoder.layers[li] = patched
            self._patched_layer_ids.append(id(patched))

        self.faith_proj = nn.Linear(self.bart_hidden, encoder_hidden_size, bias=False)
        # Project encoder (768) to BART d_model (1024) for encoder_outputs
        self.enc_to_bart = nn.Linear(encoder_hidden_size, self.bart_hidden)

    # ── Inference context setup ───────────────────────────────────────────────

    def _prepare_inference_cache(
        self,
        summary_emb: torch.Tensor,
        salient_sent_embs: torch.Tensor,
        salient_weights: torch.Tensor,
        num_beams: int,
    ):
        """
        Pre-project and pre-expand KV tensors for all patched layers.
        Result is stored in _tls.layer_caches — NO nn.Module attribute is written.

        After this call, each thread has its own isolated cache, making concurrent
        generate() calls on the same model completely safe.
        """
        layer_caches: Dict[int, _CacheEntry] = {}

        with torch.no_grad():
            for layer in self.bart.model.decoder.layers:
                if not isinstance(layer, PatchedBartDecoderLayer):
                    continue
                sk, sv, gk, gv = layer.dual.project(
                    summary_emb, salient_sent_embs, salient_weights
                )
                # Beam-expand once; the decode loop reads these directly
                if num_beams > 1:
                    sk = sk.repeat_interleave(num_beams, dim=0)
                    sv = sv.repeat_interleave(num_beams, dim=0)
                    gk = gk.repeat_interleave(num_beams, dim=0)
                    gv = gv.repeat_interleave(num_beams, dim=0)

                # Key by Python id of the layer instance — unique and stable
                layer_caches[id(layer)] = (sk, sv, gk, gv)

        _set_infer_ctx(layer_caches)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        summary_emb: torch.Tensor,
        salient_sent_embs: torch.Tensor,
        salient_weights: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, K = summary_emb.shape[:2]

        _set_train_ctx(summary_emb, salient_sent_embs, salient_weights, B)
        try:
            encoder_hidden = self.enc_to_bart(summary_emb)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
            outputs = self.bart(
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            _clear_ctx()

        lm_logits   = outputs.logits
        last_hidden = outputs.decoder_hidden_states[-1]

        result: Dict[str, torch.Tensor] = {"logits": lm_logits}

        if labels is not None:
            result["ce_loss"] = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Faithfulness regulariser: average over non-padding positions only
        if labels is not None:
            pad_mask = (labels != -100).float().unsqueeze(-1)
            dec_repr = (last_hidden * pad_mask).sum(1) / pad_mask.sum(1).clamp(1)
        else:
            dec_repr = last_hidden.mean(1)

        cos = F.cosine_similarity(
            self.faith_proj(dec_repr), summary_emb.mean(1), dim=-1
        )
        result["faithfulness_loss"] = F.relu(self.faithfulness_margin - cos).mean()
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        summary_emb: torch.Tensor,
        salient_sent_embs: torch.Tensor,
        salient_weights: torch.Tensor,
        tokenizer,
        max_length: int = 256,
        num_beams: int = 4,
        min_length: int = 30,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> List[str]:
        B        = summary_emb.size(0)
        enc_mask = torch.ones(B, summary_emb.size(1), device=summary_emb.device)

        # All cached tensors go into _tls — the nn.Module is never mutated.
        # Concurrent calls from other threads will have their own _tls.
        self._prepare_inference_cache(
            summary_emb, salient_sent_embs, salient_weights, num_beams
        )
        try:
            encoder_hidden = self.enc_to_bart(summary_emb)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
            ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=enc_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        finally:
            _clear_ctx()   # Wipe _tls for this thread — another call starts fresh

        return tokenizer.batch_decode(ids, skip_special_tokens=True)
