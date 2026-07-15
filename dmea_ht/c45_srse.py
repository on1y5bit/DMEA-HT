from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c41_melr import C41MELRModel


HEAD_PREFIXES = (
    "shared_evidence_encoder.",
    "patient_readout.",
    "classifier.",
)


class C45SRSEModel(C41MELRModel):
    """Shared robust evidence state with training-only modality dropout."""

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
        c45_cfg = dict(config["c45"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        self.modality_dropout = float(c45_cfg["modality_dropout"])
        self.shared_evidence_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
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

    def train(self, mode: bool = True) -> "C45SRSEModel":
        nn.Module.train(self, mode)
        self.sources.eval()
        return self

    @staticmethod
    def _trajectory_statistics(
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
        torch.Tensor,
    ]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(states.dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(states.dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(states.dtype)

        latest_effective = latest_weights.unsqueeze(-1) * valid.to(states.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (states * latest_effective.unsqueeze(-1)).sum(dim=1)
        latest = latest / latest_denominator.clamp_min(1.0).unsqueeze(-1)
        latest_valid = latest_denominator > 0.0

        history_effective = history_weights.unsqueeze(-1) * valid.to(states.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (states * history_effective.unsqueeze(-1)).sum(dim=1)
        history = history / history_denominator.clamp_min(1.0).unsqueeze(-1)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(states.dtype)

        centered = (states - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective.unsqueeze(-1)).sum(dim=1)
        variance = variance / history_denominator.clamp_min(1.0).unsqueeze(-1)
        dispersion = variance.clamp_min(1e-8).sqrt()
        dispersion = dispersion * history_valid.unsqueeze(-1).to(states.dtype)
        summary = torch.cat([latest, history, delta, dispersion], dim=-1)
        available = latest_valid | history_valid
        return (
            summary,
            available,
            latest_mask,
            history_mask,
            latest_weights,
            history_weights,
            latest_valid | history_valid,
        )

    def _drop_modalities(self, available: torch.Tensor) -> torch.Tensor:
        effective = available.clone()
        if self.training and self.modality_dropout > 0.0:
            keep = torch.rand(
                available.shape,
                device=available.device,
                dtype=torch.float32,
            ) >= self.modality_dropout
            effective = available & keep
            no_kept = available.any(dim=1) & ~effective.any(dim=1)
            if bool(no_kept.any().item()):
                first_available = available.to(torch.int64).argmax(dim=1)
                fallback = torch.zeros_like(available)
                fallback.scatter_(1, first_available.unsqueeze(1), True)
                effective = effective | (fallback & no_kept.unsqueeze(1))
        no_available = ~effective.any(dim=1)
        if bool(no_available.any().item()):
            effective[no_available, 0] = True
        return effective

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights, _ = self._trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        raw_tokens = self.shared_evidence_encoder(summary)
        effective_available = self._drop_modalities(available)
        tokens = raw_tokens * effective_available.unsqueeze(-1).to(raw_tokens.dtype)
        weights = effective_available.to(tokens.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        consensus = (tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        discordance = (
            (tokens - consensus.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        consensus_product = consensus * discordance
        patient_input = torch.cat([consensus, discordance, consensus_product], dim=-1)
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)

        consensus_valid = available.any(dim=1)
        evidence_tokens = torch.stack(
            [consensus, discordance, consensus_product, tokens.mean(dim=1)], dim=1
        )
        evidence_valid = consensus_valid.unsqueeze(1).expand(-1, 4)
        evidence_attention = evidence_valid.to(tokens.dtype)
        evidence_attention = evidence_attention / evidence_attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = source["states"][:, :, 2]
        bio_valid = source["valid"][:, :, 2]
        bio_denominator = bio_valid.to(bio_state.dtype).sum(dim=1, keepdim=True)
        bio_state = (bio_state * bio_valid.unsqueeze(-1).to(bio_state.dtype)).sum(dim=1)
        bio_state = bio_state / bio_denominator.clamp_min(1.0)
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
            "attention": evidence_attention,
            "trajectory_available": available,
            "modality_tokens": tokens,
            "modality_valid": available,
            "consensus_state": consensus,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
