from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "support_encoder.",
    "opposition_encoder.",
    "signed_readout.",
    "classifier.",
)


class C48SPEModel(C47DRFEModel):
    """Signed support/opposition evidence readout over fixed C17 streams."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        self.support_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.opposition_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.signed_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        support_tokens = self.support_encoder(summary)
        opposition_tokens = self.opposition_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(support_tokens.dtype)
        support_tokens = support_tokens * weights.unsqueeze(-1)
        opposition_tokens = opposition_tokens * weights.unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        support_state = (support_tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        opposition_state = (opposition_tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        signed_state = support_state - opposition_state
        uncertainty = (support_state - opposition_state).abs()
        support_discordance = (
            (support_tokens - support_state.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        opposition_discordance = (
            (opposition_tokens - opposition_state.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        discordance = 0.5 * (support_discordance + opposition_discordance)
        patient_state = self.signed_readout(
            torch.cat([support_state, opposition_state, signed_state, discordance], dim=-1)
        )
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack(
            [support_state, opposition_state, signed_state, discordance], dim=1
        )
        evidence_valid = available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(support_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = source["states"][:, :, 5]
        bio_valid = source["valid"][:, :, 5]
        bio_weights = bio_valid.to(bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = (bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": signed_state,
            "bio_state": bio_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": available,
            "support_tokens": support_tokens,
            "opposition_tokens": opposition_tokens,
            "signed_state": signed_state,
            "uncertainty_state": uncertainty,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
