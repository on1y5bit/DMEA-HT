from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "distribution_encoder.",
    "patient_readout.",
    "classifier.",
)


class C51HRMEModel(C47DRFEModel):
    """Robust distributional set readout over fixed patient evidence instances."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c51"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        self.distribution_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_readout = nn.Sequential(
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
        expected_order = [
            "raw_image",
            "raw_text",
            "raw_bio",
            "aligned_image",
            "aligned_text",
            "aligned_bio",
        ]
        if list(config["c51"]["stream_order"]) != expected_order:
            raise RuntimeError("C51 stream order is fixed to raw then aligned image/text/bio")

    @staticmethod
    def _distributional_summary(
        states: torch.Tensor,
        valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        base_summary, available, latest_mask, history_mask, latest_weights, history_weights = C47DRFEModel._fixed_trajectory_statistics(
            states, valid, visit_mask
        )
        history_valid = history_mask.unsqueeze(-1) & valid
        history_values = states.masked_fill(~history_valid.unsqueeze(-1), float("nan"))
        history_q25 = torch.nanquantile(history_values, 0.25, dim=1)
        history_q50 = torch.nanquantile(history_values, 0.50, dim=1)
        history_q75 = torch.nanquantile(history_values, 0.75, dim=1)
        history_q25 = torch.nan_to_num(history_q25, nan=0.0, posinf=0.0, neginf=0.0)
        history_q50 = torch.nan_to_num(history_q50, nan=0.0, posinf=0.0, neginf=0.0)
        history_q75 = torch.nan_to_num(history_q75, nan=0.0, posinf=0.0, neginf=0.0)
        history_iqr = history_q75 - history_q25
        summary = torch.cat([base_summary, history_q50, history_iqr], dim=-1)
        return summary, available, latest_mask, history_mask, latest_weights, history_weights

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._distributional_summary(
            source["states"], source["valid"], source["visit_mask"]
        )
        stream_tokens = self.distribution_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(stream_tokens.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_tokens = stream_tokens * weights.unsqueeze(-1)
        set_mean = masked_tokens.sum(dim=1) / denominator
        set_max = stream_tokens.masked_fill(~safe_available.unsqueeze(-1), -float("inf")).amax(dim=1)
        set_max = torch.nan_to_num(set_max, nan=0.0, posinf=0.0, neginf=0.0)
        set_deviation = (
            (stream_tokens - set_mean.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        raw_weights = weights[:, :3]
        aligned_weights = weights[:, 3:]
        raw_denominator = raw_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        aligned_denominator = aligned_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        raw_state = (stream_tokens[:, :3] * raw_weights.unsqueeze(-1)).sum(dim=1) / raw_denominator
        aligned_state = (stream_tokens[:, 3:] * aligned_weights.unsqueeze(-1)).sum(dim=1) / aligned_denominator
        resolution_gap = (raw_state - aligned_state).abs()
        patient_state = self.patient_readout(
            torch.cat([set_mean, set_max, set_deviation, resolution_gap], dim=-1)
        )
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack([set_mean, set_max, set_deviation, resolution_gap], dim=1)
        evidence_valid = safe_available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(stream_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_weights = weights[:, (2, 5)]
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = (stream_tokens[:, (2, 5)] * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": set_mean,
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
            "stream_tokens": stream_tokens,
            "stream_valid": safe_available,
            "set_mean": set_mean,
            "set_max": set_max,
            "set_deviation": set_deviation,
            "resolution_gap": resolution_gap,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
