from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c41_melr import C41MELRModel


HEAD_PREFIXES = (
    "visit_encoder.",
    "history_self_attention.",
    "history_norm.",
    "latest_query.",
    "query_norm.",
    "patient_readout.",
    "classifier.",
)


class C44MISEModel(C41MELRModel):
    """Patient-level multi-instance evidence set encoder."""

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
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        heads = int(model_cfg["mea_num_heads"])
        self.visit_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.history_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.latest_query = nn.MultiheadAttention(
            hidden_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def train(self, mode: bool = True) -> "C44MISEModel":
        nn.Module.train(self, mode)
        self.sources.eval()
        return self

    @staticmethod
    def _fixed_buckets(
        visit_mask: torch.Tensor, valid: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_effective = latest_mask & valid
        history_effective = history_mask & valid
        latest_weight = latest_effective.to(dtype)
        history_weight = history_weights * history_effective.to(dtype)
        return latest_mask, history_mask, latest_weight, history_weight, history_effective

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        states = source["states"]
        modality_valid = source["valid"]
        image_state = states[:, :, 0]
        text_state = states[:, :, 1]
        bio_state = states[:, :, 2]
        image_valid = modality_valid[:, :, 0]
        text_valid = modality_valid[:, :, 1]
        bio_valid = modality_valid[:, :, 2]
        visit_valid = (image_valid | text_valid | bio_valid) & source["visit_mask"]
        visit_input = torch.cat(
            [image_state, text_state, bio_state, image_state * text_state], dim=-1
        )
        visit_tokens = self.visit_encoder(visit_input)
        visit_tokens = visit_tokens * visit_valid.unsqueeze(-1).to(visit_tokens.dtype)
        latest_mask, history_mask, latest_weight, history_weight, history_effective = self._fixed_buckets(
            source["visit_mask"], visit_valid, visit_tokens.dtype
        )
        latest_denominator = latest_weight.sum(dim=1, keepdim=True)
        latest_state = (visit_tokens * latest_weight.unsqueeze(-1)).sum(dim=1)
        latest_state = latest_state / latest_denominator.clamp_min(1.0)
        latest_valid = latest_denominator.squeeze(-1) > 0.0
        history_denominator = history_weight.sum(dim=1, keepdim=True)
        history_mean = (visit_tokens * history_weight.unsqueeze(-1)).sum(dim=1)
        history_mean = history_mean / history_denominator.clamp_min(1.0)
        history_valid = history_denominator.squeeze(-1) > 0.0
        centered = (visit_tokens - history_mean.unsqueeze(1)).pow(2)
        variance = (centered * history_weight.unsqueeze(-1)).sum(dim=1)
        variance = variance / history_denominator.clamp_min(1.0)
        dispersion = variance.clamp_min(1e-8).sqrt() * history_valid.unsqueeze(-1).to(visit_tokens.dtype)
        safe_history = history_effective.clone()
        no_history = ~safe_history.any(dim=1)
        if bool(no_history.any().item()):
            safe_history[no_history, 0] = True
        history_input = visit_tokens * safe_history.unsqueeze(-1).to(visit_tokens.dtype)
        history_attended, history_attention = self.history_self_attention(
            history_input,
            history_input,
            history_input,
            key_padding_mask=~safe_history,
            need_weights=True,
        )
        history_tokens = self.history_norm(history_input + history_attended)
        query = latest_state.unsqueeze(1)
        queried_history, query_attention = self.latest_query(
            query,
            history_tokens,
            history_tokens,
            key_padding_mask=~safe_history,
            need_weights=True,
        )
        history_state = self.query_norm(queried_history.squeeze(1))
        history_state = history_state * history_valid.unsqueeze(-1).to(history_state.dtype)
        delta_state = (latest_state - history_mean) * (latest_valid & history_valid).unsqueeze(-1).to(latest_state.dtype)
        patient_input = torch.cat([latest_state, history_state, history_mean, delta_state, dispersion], dim=-1)
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack([latest_state, history_state, delta_state, dispersion], dim=1)
        evidence_valid = torch.stack(
            [latest_valid, history_valid, history_valid, latest_valid & history_valid], dim=1
        )
        evidence_attention = evidence_valid.to(visit_tokens.dtype)
        evidence_attention = evidence_attention / evidence_attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_patient_state = bio_state.mean(dim=1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": history_state,
            "bio_state": bio_patient_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_mask.to(visit_tokens.dtype),
            "history_weights": history_weight,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": evidence_attention,
            "trajectory_available": latest_valid | history_valid,
            "visit_state": visit_tokens,
            "history_attention": history_attention,
            "query_attention": query_attention,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
