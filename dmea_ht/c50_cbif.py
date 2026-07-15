from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "trajectory_encoder.",
    "resolution_fusion.",
    "image_text_fusion.",
    "bio_state_encoder.",
    "bio_conditioner.",
    "conditional_norm.",
    "patient_readout.",
    "classifier.",
)


class C50CBIFModel(C47DRFEModel):
    """Patient-level biochemical conditioning of fixed cross-modal evidence."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c50"])
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
        bio_dim = int(model_cfg["bio_dim"])
        dropout = float(model_cfg["dropout"])
        self.trajectory_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.resolution_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.image_text_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.bio_state_encoder = nn.Sequential(
            nn.LayerNorm(bio_dim * 4),
            nn.Linear(bio_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.bio_conditioner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.conditional_norm = nn.LayerNorm(hidden_dim)
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, hidden_dim),
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
        if list(config["c50"]["stream_order"]) != ["image", "text", "bio"]:
            raise RuntimeError("C50 modality order is fixed to image, text, bio")

    @staticmethod
    def _fixed_numeric_trajectory(
        values: torch.Tensor,
        missing: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[
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
        age = (latest_index - positions).clamp_min(0).to(values.dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(values.dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(values.dtype)
        observed = (~missing.bool()) & visit_mask.unsqueeze(-1)
        latest_effective = latest_weights.unsqueeze(-1) * observed.to(values.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (values * latest_effective).sum(dim=1)
        latest = latest / latest_denominator.clamp_min(1.0)
        latest_valid = latest_denominator > 0.0
        history_effective = history_weights.unsqueeze(-1) * observed.to(values.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (values * history_effective).sum(dim=1)
        history = history / history_denominator.clamp_min(1.0)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).to(values.dtype)
        centered = (values - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective).sum(dim=1)
        variance = variance / history_denominator.clamp_min(1.0)
        dispersion = variance.clamp_min(1e-8).sqrt()
        dispersion = dispersion * history_valid.to(values.dtype)
        summary = torch.cat([latest, history, delta, dispersion], dim=-1)
        available = latest_valid | history_valid
        return (
            summary,
            available,
            latest_mask,
            history_mask,
            latest_weights,
            history_weights,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        trajectory_tokens = self.trajectory_encoder(summary)
        raw_tokens = trajectory_tokens[:, :3]
        aligned_tokens = trajectory_tokens[:, 3:]
        raw_valid = available[:, :3]
        aligned_valid = available[:, 3:]
        modality_tokens = []
        modality_valid = []
        for modality_index in range(3):
            raw = raw_tokens[:, modality_index] * raw_valid[:, modality_index].unsqueeze(-1).to(trajectory_tokens.dtype)
            aligned = aligned_tokens[:, modality_index] * aligned_valid[:, modality_index].unsqueeze(-1).to(trajectory_tokens.dtype)
            fused = self.resolution_fusion(
                torch.cat([raw, aligned, (raw - aligned).abs(), raw * aligned], dim=-1)
            )
            valid = raw_valid[:, modality_index] | aligned_valid[:, modality_index]
            modality_tokens.append(fused * valid.unsqueeze(-1).to(fused.dtype))
            modality_valid.append(valid)
        modality_states = torch.stack(modality_tokens, dim=1)
        modality_available = torch.stack(modality_valid, dim=1)
        image_state = modality_states[:, 0]
        text_state = modality_states[:, 1]
        bio_evidence_state = modality_states[:, 2]
        image_text = self.image_text_fusion(
            torch.cat(
                [image_state, text_state, (image_state - text_state).abs(), image_state * text_state],
                dim=-1,
            )
        )
        image_text_valid = modality_available[:, 0] & modality_available[:, 1]
        image_text = image_text * image_text_valid.unsqueeze(-1).to(image_text.dtype)
        bio_summary, _, _, _, _, _ = self._fixed_numeric_trajectory(
            batch["bio_values"], batch["bio_missing_mask"], batch["visit_mask"].bool()
        )
        bio_state = self.bio_state_encoder(bio_summary)
        condition = self.bio_conditioner(bio_state)
        scale, shift = condition.chunk(2, dim=-1)
        conditioned_image_text = self.conditional_norm(
            image_text * (1.0 + 0.5 * torch.tanh(scale)) + 0.5 * torch.tanh(shift)
        )
        safe_available = modality_available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(modality_states.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        consensus = (modality_states * weights.unsqueeze(-1)).sum(dim=1) / denominator
        discordance = (
            (modality_states - consensus.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        patient_state = self.patient_readout(
            torch.cat(
                [
                    image_state,
                    text_state,
                    bio_evidence_state,
                    image_text,
                    conditioned_image_text,
                    bio_state,
                    consensus,
                    discordance,
                ],
                dim=-1,
            )
        )
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack(
            [consensus, discordance, image_text, conditioned_image_text], dim=1
        )
        evidence_valid = safe_available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(modality_states.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": conditioned_image_text,
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
            "modality_states": modality_states,
            "modality_available": modality_available,
            "image_text_state": image_text,
            "conditioned_image_text_state": conditioned_image_text,
            "consensus_state": consensus,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
