from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "binding_encoder.",
    "bound_stream_encoder.",
    "patient_readout.",
    "classifier.",
)


class C49CREBModel(C47DRFEModel):
    """Bind raw and aligned states within modality before fixed pooling."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c49"])
        source_config["c47"]["stream_order"] = [
            "raw_image",
            "raw_text",
            "raw_bio",
            "aligned_image",
            "aligned_text",
            "aligned_bio",
        ]
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        self.binding_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.bound_stream_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        raw = source["states"][:, :, :3]
        aligned = source["states"][:, :, 3:]
        raw_valid = source["valid"][:, :, :3]
        aligned_valid = source["valid"][:, :, 3:]
        bound_input = torch.cat([raw, aligned, (raw - aligned).abs(), raw * aligned], dim=-1)
        bound_visit = self.binding_encoder(bound_input)
        bound_valid = raw_valid | aligned_valid
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            bound_visit, bound_valid, source["visit_mask"]
        )
        bound_tokens = self.bound_stream_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(bound_tokens.dtype)
        bound_tokens = bound_tokens * weights.unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        consensus = (bound_tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        discordance = (
            (bound_tokens - consensus.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        patient_state = self.patient_readout(
            torch.cat([bound_tokens.flatten(start_dim=1), consensus, discordance], dim=-1)
        )
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack(
            [consensus, discordance, consensus * discordance, bound_tokens.mean(dim=1)], dim=1
        )
        evidence_valid = available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(bound_tokens.dtype)
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
            "attended_evidence": consensus,
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
            "bound_tokens": bound_tokens,
            "bound_valid": available,
            "consensus_state": consensus,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
