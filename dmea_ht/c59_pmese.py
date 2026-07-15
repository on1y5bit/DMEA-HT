from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "instance_encoder.",
    "patient_readout.",
    "classifier.",
)

STREAM_ORDER = [
    "raw_image",
    "raw_text",
    "raw_bio",
    "aligned_image",
    "aligned_text",
    "aligned_bio",
]

MODALITY_PAIRS = ((0, 3), (1, 4), (2, 5))


class C59PMESEModel(C47DRFEModel):
    """Patient-level multimodal evidence-set encoder over frozen C17 streams."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c59"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)

        model_cfg = dict(config["model"])
        c59_cfg = dict(config["c59"])
        self.instance_dim = int(c59_cfg["instance_dim"])
        patient_dim = int(c59_cfg["patient_dim"])
        classifier_dim = int(c59_cfg["classifier_hidden_dim"])
        feature_dim = self.hidden_dim * 9

        self.instance_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.instance_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(self.instance_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(self.instance_dim * 6),
            nn.Linear(self.instance_dim * 6, patient_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(patient_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(patient_dim),
            nn.Linear(patient_dim, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, 1),
        )
        if list(c59_cfg["stream_order"]) != STREAM_ORDER:
            raise RuntimeError("C59 stream order is fixed to raw then aligned image/text/bio")
        if int(c59_cfg["instance_dim"]) != 128:
            raise RuntimeError("C59 instance dimension is fixed to 128")

    @staticmethod
    def _modality_instance_states(
        states: torch.Tensor, valid: torch.Tensor, raw_index: int, aligned_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = states[:, :, raw_index]
        aligned = states[:, :, aligned_index]
        raw_valid = valid[:, :, raw_index].bool()
        aligned_valid = valid[:, :, aligned_index].bool()
        raw_weight = raw_valid.to(states.dtype)
        aligned_weight = aligned_valid.to(states.dtype)
        denominator = (raw_weight + aligned_weight).clamp_min(1.0).unsqueeze(-1)
        mean = (raw * raw_weight.unsqueeze(-1) + aligned * aligned_weight.unsqueeze(-1)) / denominator
        paired = raw_valid & aligned_valid
        gap = (raw - aligned).abs() * paired.to(states.dtype).unsqueeze(-1)
        return mean, gap, raw_valid | aligned_valid

    @staticmethod
    def _fixed_patient_set_statistics(
        tokens: torch.Tensor, valid: torch.Tensor, visit_mask: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        valid = valid.bool() & visit_mask.bool()
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask.bool() & positions.eq(latest_index)
        history_mask = visit_mask.bool() & ~latest_mask

        latest_effective = latest_mask & valid
        latest_denominator = latest_effective.sum(dim=1, keepdim=True).to(tokens.dtype).clamp_min(1.0)
        latest = (tokens * latest_effective.to(tokens.dtype).unsqueeze(-1)).sum(dim=1) / latest_denominator
        latest_valid = latest_effective.any(dim=1)

        age = (latest_index - positions).clamp_min(0).to(tokens.dtype)
        history_kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(tokens.dtype)
        history_effective = history_kernel * valid.to(tokens.dtype)
        history_denominator = history_effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
        history = (tokens * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator
        history_valid = history_effective.sum(dim=1) > 0.0
        history_weights = history_effective / history_denominator

        both_valid = latest_valid & history_valid
        delta = (latest - history) * both_valid.unsqueeze(-1).to(tokens.dtype)
        centered = (tokens - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator
        dispersion = variance.clamp_min(1e-8).sqrt() * history_valid.unsqueeze(-1).to(tokens.dtype)

        recency_effective = (latest_mask.to(tokens.dtype) + history_kernel) * valid.to(tokens.dtype)
        recency_denominator = recency_effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
        patient_mean = (tokens * recency_effective.unsqueeze(-1)).sum(dim=1) / recency_denominator

        masked_tokens = tokens.masked_fill(~valid.unsqueeze(-1), -float("inf"))
        set_max = masked_tokens.amax(dim=1)
        set_max = torch.nan_to_num(set_max, nan=0.0, posinf=0.0, neginf=0.0)
        patient_available = valid.any(dim=1)
        latest_weights = latest_effective.to(tokens.dtype)
        return (
            latest,
            history,
            delta,
            dispersion,
            set_max,
            patient_mean,
            latest_weights,
            history_weights,
            latest_mask,
            history_mask,
            latest_valid,
            history_valid,
            patient_available,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)

        states = F.layer_norm(source["states"], (self.hidden_dim,))
        valid = source["valid"].bool()
        modality_states = []
        resolution_gaps = []
        modality_available = []
        for raw_index, aligned_index in MODALITY_PAIRS:
            modality_state, gap, available = self._modality_instance_states(
                states, valid, raw_index, aligned_index
            )
            modality_states.append(modality_state)
            resolution_gaps.append(gap)
            modality_available.append(available)

        image_state, text_state, bio_state = modality_states
        image_gap, text_gap, bio_gap = resolution_gaps
        image_available, text_available, bio_available = modality_available
        image_text = image_state * text_state
        image_bio = image_state * bio_state
        text_bio = text_state * bio_state
        visit_features = torch.cat(
            [
                image_state,
                text_state,
                bio_state,
                image_text,
                image_bio,
                text_bio,
                image_gap,
                text_gap,
                bio_gap,
            ],
            dim=-1,
        )
        visit_tokens = self.instance_encoder(visit_features)
        visit_valid = (
            image_available | text_available | bio_available
        ) & source["visit_mask"].bool()
        visit_tokens = visit_tokens * visit_valid.to(visit_tokens.dtype).unsqueeze(-1)

        latest, history, delta, dispersion, set_max, patient_mean, latest_weights, history_weights, latest_mask, history_mask, latest_valid, history_valid, patient_available = (
            self._fixed_patient_set_statistics(visit_tokens, visit_valid, source["visit_mask"])
        )
        patient_input = torch.cat(
            [latest, history, delta, dispersion, set_max, patient_mean], dim=-1
        )
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)

        evidence_tokens = torch.stack([latest, history, delta, dispersion], dim=1)
        evidence_valid = torch.stack(
            [latest_valid, history_valid, patient_available, history_valid],
            dim=1,
        )
        attention = evidence_valid.to(evidence_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)

        source_bio_state = source["states"][:, :, 5]
        source_bio_valid = source["valid"][:, :, 5].bool()
        bio_weights = source_bio_valid.to(source_bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        source_bio_state = (source_bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator

        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": latest,
            "bio_state": source_bio_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": visit_valid,
            "stream_tokens": evidence_tokens,
            "stream_valid": evidence_valid,
            "consensus_state": patient_mean,
            "discordance_state": dispersion,
            "instance_tokens": visit_tokens,
            "instance_valid": visit_valid,
            "fusion_state": patient_input,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
