from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c41_melr import C41MELRModel


HEAD_PREFIXES = (
    "bio_conditioner.",
    "image_film.",
    "text_film.",
    "cross_modal_fusion.",
    "trajectory_encoder.",
    "patient_readout.",
    "classifier.",
)


class C43MCFEModel(C41MELRModel):
    """Within-visit biochemical conditioning followed by patient-level fusion."""

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
        bio_dim = int(model_cfg["bio_dim"])
        dropout = float(model_cfg["dropout"])
        self.bio_conditioner = nn.Sequential(
            nn.LayerNorm(bio_dim),
            nn.Linear(bio_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim * 4),
        )
        self.image_film = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
        )
        self.text_film = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
        )
        self.cross_modal_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def train(self, mode: bool = True) -> "C43MCFEModel":
        nn.Module.train(self, mode)
        self.sources.eval()
        return self

    @staticmethod
    def _trajectory_statistics(
        states: torch.Tensor,
        valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        latest_effective = latest_weights * valid.to(states.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (states * latest_effective.unsqueeze(-1)).sum(dim=1)
        latest = latest / latest_denominator.clamp_min(1.0).unsqueeze(-1)
        latest_valid = latest_denominator > 0.0

        history_effective = history_weights * history_mask.to(states.dtype) * valid.to(states.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (states * history_effective.unsqueeze(-1)).sum(dim=1)
        history = history / history_denominator.clamp_min(1.0).unsqueeze(-1)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(states.dtype)
        centered = (states - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective.unsqueeze(-1)).sum(dim=1)
        variance = variance / history_denominator.clamp_min(1.0).unsqueeze(-1)
        dispersion = variance.clamp_min(1e-8).sqrt() * history_valid.unsqueeze(-1).to(states.dtype)
        summary = torch.cat([latest, history, delta, dispersion], dim=-1)
        available = latest_valid | history_valid
        return summary, available, latest_mask, history_mask, history_weights

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        states = source["states"]
        valid = source["valid"]
        image_state = states[:, :, 0]
        text_state = states[:, :, 1]
        bio_state = states[:, :, 2]
        image_valid = valid[:, :, 0]
        text_valid = valid[:, :, 1]
        bio_valid = valid[:, :, 2]
        observed = (~batch["bio_missing_mask"].bool()).to(batch["bio_values"].dtype)
        bio_values = batch["bio_values"] * observed
        condition = self.bio_conditioner(bio_values)
        image_params = self.image_film(condition[..., : self.hidden_dim * 2])
        text_params = self.text_film(condition[..., self.hidden_dim * 2 :])
        image_gamma, image_beta = image_params.chunk(2, dim=-1)
        text_gamma, text_beta = text_params.chunk(2, dim=-1)
        image_conditioned = (
            image_state * (1.0 + 0.10 * image_gamma) + 0.10 * image_beta
        ) * image_valid.unsqueeze(-1).to(image_state.dtype)
        text_conditioned = (
            text_state * (1.0 + 0.10 * text_gamma) + 0.10 * text_beta
        ) * text_valid.unsqueeze(-1).to(text_state.dtype)
        cross_term = image_conditioned * text_conditioned
        fused_input = torch.cat([image_conditioned, text_conditioned, bio_state, cross_term], dim=-1)
        visit_state = self.cross_modal_fusion(fused_input)
        visit_valid = (image_valid | text_valid | bio_valid) & source["visit_mask"]
        visit_state = visit_state * visit_valid.unsqueeze(-1).to(visit_state.dtype)
        summary, available, latest_mask, history_mask, history_weights = self._trajectory_statistics(
            visit_state, visit_valid, source["visit_mask"]
        )
        patient_state = self.trajectory_encoder(summary)
        patient_state = self.patient_readout(patient_state)
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = summary.reshape(summary.shape[0], 4, self.hidden_dim)
        evidence_valid = available.unsqueeze(1).expand(-1, 4)
        evidence_attention = evidence_valid.to(summary.dtype)
        evidence_attention = evidence_attention / evidence_attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        latest_bio_valid = (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1)
        history_bio_valid = (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1)
        bio_patient_state = bio_state.mean(dim=1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": patient_state,
            "bio_state": bio_patient_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": latest_bio_valid,
            "history_bio_valid": history_bio_valid,
            "latest_weights": latest_mask.to(summary.dtype),
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": evidence_attention,
            "trajectory_available": available,
            "visit_state": visit_state,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
