"""
CrossDocJEPA — full model (v3)
Wires JSE paragraph_salience → SRD for principled memory selection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .hde import HierarchicalDocumentEncoder
from .cd_jepa import CrossDocumentJEPA
from .srd import SummaryRepresentationDistiller
from .jse import JEPAGuidedSalienceEstimator
from .fad import FaithfulnessAwareDecoder


class CrossDocJEPA(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        H = config["hidden_size"]

        self.hde = HierarchicalDocumentEncoder(
            model_name=config.get("encoder_model", "roberta-base"),
            hidden_size=H,
            num_heads=config.get("num_heads", 8),
            para_layers=config.get("para_layers", 2),
            doc_layers=config.get("doc_layers", 2),
            dropout=config.get("dropout", 0.1),
        )
        self.cd_jepa = CrossDocumentJEPA(
            encoder=self.hde,
            hidden_size=H,
            num_heads=config.get("num_heads", 8),
            predictor_layers=config.get("predictor_layers", 4),
            dropout=config.get("dropout", 0.1),
            ema_momentum=config.get("ema_momentum", 0.996),
            min_context_ratio=config.get("min_context_ratio", 0.3),
            max_context_ratio=config.get("max_context_ratio", 0.7),
            prediction_level=config.get("prediction_level", "hierarchical"),
        )
        self.srd = SummaryRepresentationDistiller(
            hidden_size=H,
            num_summary_queries=config.get("num_summary_queries", 32),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("srd_layers", 4),
            dropout=config.get("dropout", 0.1),
            max_memory_paragraphs=config.get("max_memory_paragraphs", 64),
        )
        # JSE shares sent_predictor with CD-JEPA (warmed up in pretraining)
        self.jse = JEPAGuidedSalienceEstimator(
            sent_predictor=self.cd_jepa.sent_predictor,
            hidden_size=H,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            minimum_signal_threshold=config.get("min_signal_threshold", 0.05),
            temperature=config.get("salience_temperature", 1.0),
            top_k=config.get("top_k_sentences", 50),
        )
        self.fad = FaithfulnessAwareDecoder(
            bart_model_name=config.get("decoder_model", "facebook/bart-large-cnn"),
            encoder_hidden_size=H,
            num_dual_attn_layers=config.get("num_dual_attn_layers", 4),
            faithfulness_margin=config.get("faithfulness_margin", 0.85),
            dropout=config.get("dropout", 0.1),
        )
        self.lam_ce    = config.get("lambda_ce",    1.0)
        self.lam_jepa  = config.get("lambda_jepa",  0.5)
        self.lam_sal   = config.get("lambda_sal",   0.3)
        self.lam_faith = config.get("lambda_faith", 0.2)

    def pretrain_forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self.cd_jepa(batch)

    def finetune_forward(
        self,
        batch: Dict,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        oracle_salience: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        repr_d   = self.cd_jepa.get_all_document_representations(batch)
        doc_embs  = repr_d["document_embs"]
        para_embs = repr_d["paragraph_embs"]
        sent_embs = repr_d["sentence_embs"]
        B, N      = doc_embs.shape[:2]

        jepa_losses = self.cd_jepa(batch)
        jepa_loss   = jepa_losses["total_jepa_loss"]

        # JSE — also produces paragraph_salience for SRD
        jse_out = self.jse(
            sentence_embs=sent_embs,
            paragraph_embs=para_embs,
            document_embs=doc_embs,
            sent_valid_mask=batch["sentence_valid_mask"],
            para_valid_mask=batch["paragraph_valid_mask"],
            oracle_scores=oracle_salience,
        )
        sal_loss = jse_out.get("salience_loss",
                               torch.tensor(0.0, device=doc_embs.device))

        # SRD — receives JSE paragraph salience for principled memory selection
        pred_doc_embs = self.cd_jepa.predict_all_from_all(doc_embs)
        srd_out = self.srd(
            document_embs=doc_embs,
            paragraph_embs=para_embs,
            para_mask=batch["paragraph_valid_mask"].view(B, N, -1),
            predicted_doc_embs=pred_doc_embs,
            paragraph_salience=jse_out["paragraph_salience"],   # ← end-to-end salience
        )
        summary_emb = srd_out["summary_emb"]

        fad_out = self.fad(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            summary_emb=summary_emb,
            salient_sent_embs=jse_out["topk_embs"],
            salient_weights=jse_out["topk_weights"],
            labels=labels,
        )
        ce_loss    = fad_out["ce_loss"]
        faith_loss = fad_out["faithfulness_loss"]

        total = (self.lam_ce    * ce_loss
                 + self.lam_jepa * jepa_loss
                 + self.lam_sal  * sal_loss
                 + self.lam_faith * faith_loss)

        return {
            "loss": total, "ce_loss": ce_loss,
            "jepa_loss": jepa_loss, "salience_loss": sal_loss,
            "faithfulness_loss": faith_loss, "logits": fad_out["logits"],
            "gate_alpha": srd_out.get("gate_alpha", 0.0),
        }

    @torch.no_grad()
    def generate_summary(self, batch: Dict, tokenizer,
                          max_length: int = 256, num_beams: int = 4,
                          min_length: int = 10, length_penalty: float = 2.0) -> List[str]:
        repr_d   = self.cd_jepa.get_all_document_representations(batch)
        doc_embs  = repr_d["document_embs"]
        para_embs = repr_d["paragraph_embs"]
        sent_embs = repr_d["sentence_embs"]
        B, N      = doc_embs.shape[:2]

        jse_out = self.jse(
            sentence_embs=sent_embs, paragraph_embs=para_embs,
            document_embs=doc_embs,
            sent_valid_mask=batch["sentence_valid_mask"],
            para_valid_mask=batch["paragraph_valid_mask"],
        )
        pred_doc_embs = self.cd_jepa.predict_all_from_all(doc_embs)
        srd_out = self.srd(
            document_embs=doc_embs, paragraph_embs=para_embs,
            para_mask=batch["paragraph_valid_mask"].view(B, N, -1),
            predicted_doc_embs=pred_doc_embs,
            paragraph_salience=jse_out["paragraph_salience"],
        )
        return self.fad.generate(
            summary_emb=srd_out["summary_emb"],
            salient_sent_embs=jse_out["topk_embs"],
            salient_weights=jse_out["topk_weights"],
            tokenizer=tokenizer,
            max_length=max_length,
            num_beams=num_beams,
            min_length=min_length,
            length_penalty=length_penalty,
        )
