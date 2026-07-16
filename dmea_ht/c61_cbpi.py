from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c59_pmese import C59PMESEModel, MODALITY_PAIRS


HEAD_PREFIXES = (
    "multimodal_encoder.",
    "continuous_bio_encoder.",
    "joint_instance_encoder.",
    "patient_readout.",
    "classifier.",
)


class C61CBPIModel(C59PMESEModel):
    """Per-visit continuous-bio fusion followed by patient-level evidence-set encoding."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        translated = dict(config)
        translated["c59"] = dict(config["c61"])
        super().__init__(translated, seed)
        self.end_to_end = bool(config.get("end_to_end", False))
        delattr(self, "instance_encoder")

        model_cfg = dict(config["model"])
        c61_cfg = dict(config["c61"])
        projection_dim = int(c61_cfg["continuous_bio_projection_dim"])
        feature_dim = self.hidden_dim * 9
        bio_basis_dim = int(model_cfg["bio_dim"]) * 3

        self.multimodal_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.instance_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(self.instance_dim),
        )
        self.continuous_bio_encoder = nn.Sequential(
            nn.LayerNorm(bio_basis_dim),
            nn.Linear(bio_basis_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(projection_dim),
        )
        self.joint_instance_encoder = nn.Sequential(
            nn.LayerNorm(self.instance_dim * 3),
            nn.Linear(self.instance_dim * 3, self.instance_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(self.instance_dim),
        )
        if projection_dim != self.instance_dim:
            raise RuntimeError("C61 continuous-bio projection must match the instance dimension")
        if int(c61_cfg["bio_basis_order"]) != 3:
            raise RuntimeError("C61 continuous-bio basis is fixed to x, tanh(x), x*tanh(x)")

    def _multimodal_features(
        self, source: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        features = torch.cat(
            [
                image_state,
                text_state,
                bio_state,
                image_state * text_state,
                image_state * bio_state,
                text_state * bio_state,
                image_gap,
                text_gap,
                bio_gap,
            ],
            dim=-1,
        )
        available = image_available | text_available | bio_available
        return features, available & source["visit_mask"].bool()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.end_to_end:
            source = self._source_states(batch)
        else:
            with torch.no_grad():
                source = self._source_states(batch)

        multimodal_features, visit_valid = self._multimodal_features(source)
        multimodal_token = self.multimodal_encoder(multimodal_features)

        bio_values = torch.nan_to_num(
            batch["bio_values"].float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        bio_observed = ~batch["bio_missing_mask"].bool()
        bio_values = bio_values * bio_observed.to(bio_values.dtype)
        bio_nonlinear = torch.cat(
            [bio_values, torch.tanh(bio_values), bio_values * torch.tanh(bio_values)], dim=-1
        )
        continuous_bio_token = self.continuous_bio_encoder(bio_nonlinear)
        joint_features = torch.cat(
            [multimodal_token, continuous_bio_token, multimodal_token * continuous_bio_token], dim=-1
        )
        visit_tokens = self.joint_instance_encoder(joint_features)
        visit_tokens = visit_tokens * visit_valid.to(visit_tokens.dtype).unsqueeze(-1)

        (
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
        ) = self._fixed_patient_set_statistics(visit_tokens, visit_valid, source["visit_mask"])
        patient_input = torch.cat(
            [latest, history, delta, dispersion, set_max, patient_mean], dim=-1
        )
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)

        evidence_tokens = torch.stack([latest, history, delta, dispersion], dim=1)
        evidence_valid = torch.stack(
            [latest_valid, history_valid, patient_available, history_valid], dim=1
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
            "continuous_bio_token": continuous_bio_token,
            "fusion_state": patient_input,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
