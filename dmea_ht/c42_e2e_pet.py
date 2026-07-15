from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c41_melr import C41MELRModel, HEAD_PREFIXES as C41_HEAD_PREFIXES
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


HEAD_PREFIXES = (
    "sources.",
    "node_trajectory_encoder.",
    "node_type_embedding",
    "evidence_graph.",
    "graph_norm.",
    "patient_query",
    "patient_attention.",
    "patient_norm.",
    "classifier.",
)


class C42E2EPETModel(C41MELRModel):
    """End-to-end patient evidence graph initialized from C17 sources."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        for name in (
            "trajectory_encoders",
            "modality_heads",
            "router",
            "patient_readout",
            "consensus_head",
        ):
            delattr(self, name)
        hidden_dim = self.hidden_dim
        dropout = float(config["model"]["dropout"])
        heads = int(config["model"]["mea_num_heads"])
        self.node_trajectory_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.node_type_embedding = nn.Parameter(torch.randn(14, hidden_dim) * 0.02)
        self.evidence_graph = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.patient_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.patient_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.patient_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        for parameter in self.sources.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "C42E2EPETModel":
        nn.Module.train(self, mode)
        for module in (
            self.sources.image_encoder,
            self.sources.text_encoder,
            self.sources.bio_encoder,
            self.sources.image_projector,
            self.sources.text_projector,
            self.sources.bio_projector,
        ):
            module.train(mode)
        return self

    def _source_node_states(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        report_ids = batch["report_input_ids"].flatten(0, 1)
        report_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        image_tokens, _ = self.sources.image_encoder(images, image_mask)
        text_tokens, _ = self.sources.text_encoder(report_ids, report_mask)
        bio_tokens, _, _, _ = self.sources.bio_encoder(bio_values, bio_missing, bio_abnormal)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        image = self.sources.image_projector(image_tokens, image_mask)
        text = self.sources.text_projector(text_tokens, report_mask, text_masks)
        bio = self.sources.bio_projector(bio_tokens, bio_missing)
        image_nodes = image["nodes"].reshape(batch_size, visits, 5, self.hidden_dim)
        text_nodes = text["nodes"].reshape(batch_size, visits, 6, self.hidden_dim)
        bio_nodes = bio["nodes"].reshape(batch_size, visits, 3, self.hidden_dim)
        image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
        text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
        bio_valid = bio["valid"].reshape(batch_size, visits, 3).bool()
        visit_mask = batch["visit_mask"].bool()
        return {
            "nodes": torch.cat([image_nodes, text_nodes, bio_nodes], dim=2),
            "valid": torch.cat([image_valid, text_valid, bio_valid], dim=2) & visit_mask.unsqueeze(-1),
            "visit_mask": visit_mask,
        }

    @staticmethod
    def _trajectory_statistics(
        states: torch.Tensor, valid: torch.Tensor, visit_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(states.dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(states.dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(states.dtype)
        latest_effective = latest_weights.unsqueeze(-1) * valid.to(states.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (states * latest_effective.unsqueeze(-1)).sum(dim=1) / latest_denominator.clamp_min(1.0).unsqueeze(-1)
        latest_valid = latest_denominator > 0.0
        history_effective = history_weights.unsqueeze(-1) * history_mask.unsqueeze(-1).to(states.dtype) * valid.to(states.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (states * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0).unsqueeze(-1)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(states.dtype)
        variance = ((states - history.unsqueeze(1)).pow(2) * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0).unsqueeze(-1)
        dispersion = variance.sqrt() * history_valid.unsqueeze(-1).to(states.dtype)
        return torch.cat([latest, history, delta, dispersion], dim=-1), latest_valid | history_valid

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = self._source_node_states(batch)
        summary, available = self._trajectory_statistics(
            source["nodes"], source["valid"], source["visit_mask"]
        )
        evidence_tokens = self.node_trajectory_encoder(summary) + self.node_type_embedding.view(1, 14, -1)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        evidence_tokens = evidence_tokens * safe_available.unsqueeze(-1).to(evidence_tokens.dtype)
        key_padding_mask = ~safe_available
        graph_tokens, graph_attention = self.evidence_graph(
            evidence_tokens,
            evidence_tokens,
            evidence_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        graph_tokens = self.graph_norm(evidence_tokens + graph_tokens)
        query = self.patient_query.expand(graph_tokens.shape[0], -1, -1)
        patient_state, patient_attention = self.patient_attention(
            query,
            graph_tokens,
            graph_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        patient_state = self.patient_norm(patient_state.squeeze(1))
        logit = self.classifier(patient_state).squeeze(-1)
        modality_tokens = graph_tokens[:, (0, 5, 11)]
        modality_logits = logit.new_zeros(logit.shape[0], 3)
        routing_weights = patient_attention.squeeze(1)[:, (0, 5, 11)]
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "modality_tokens": modality_tokens,
            "modality_logits": modality_logits,
            "routing_weights": routing_weights,
            "trajectory_available": available[:, (0, 5, 11)],
            "evidence_logit": logit,
            "consensus_logit": logit.new_zeros(logit.shape[0]),
            "evidence_tokens": graph_tokens,
            "graph_attention": graph_attention,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
