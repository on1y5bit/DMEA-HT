from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "image_text_projection.",
    "bio_projection.",
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


class C57CBIEModel(C47DRFEModel):
    """Patient-level nonlinear continuous-biochemical/image-text interaction model."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c57"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        c57_cfg = dict(config["c57"])
        interaction_dim = int(c57_cfg["interaction_dim"])
        patient_dim = int(c57_cfg["patient_dim"])
        classifier_dim = int(c57_cfg["classifier_hidden_dim"])
        image_text_feature_dim = self.hidden_dim * 24
        bio_feature_dim = int(model_cfg["bio_dim"]) * 4 * 3
        self.image_text_projection = nn.Sequential(
            nn.LayerNorm(image_text_feature_dim),
            nn.Linear(image_text_feature_dim, interaction_dim),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.bio_projection = nn.Sequential(
            nn.LayerNorm(bio_feature_dim),
            nn.Linear(bio_feature_dim, interaction_dim),
            nn.GELU(),
            nn.LayerNorm(interaction_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(interaction_dim * 3),
            nn.Linear(interaction_dim * 3, patient_dim),
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
        if list(c57_cfg["stream_order"]) != STREAM_ORDER:
            raise RuntimeError("C57 stream order is fixed to raw then aligned image/text/bio")

    @staticmethod
    def _availability_weighted_mean(
        states: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = valid.to(states.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=2).clamp_min(1.0)
        mean = (states * weights).sum(dim=2) / denominator
        return mean, valid.any(dim=2)

    @staticmethod
    def _safe_statistic_tokens(
        summary: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = available.to(summary.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        tokens = (summary * weights.unsqueeze(-1)).sum(dim=1) / denominator.unsqueeze(-1)
        return tokens, available.any(dim=1)

    def _image_text_kernel(
        self, source: Dict[str, torch.Tensor]
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
    ]:
        states = source["states"]
        valid = source["valid"]
        visit_mask = source["visit_mask"]
        image_states = torch.stack([states[:, :, 0], states[:, :, 3]], dim=2)
        text_states = torch.stack([states[:, :, 1], states[:, :, 4]], dim=2)
        image_valid = torch.stack([valid[:, :, 0], valid[:, :, 3]], dim=2)
        text_valid = torch.stack([valid[:, :, 1], valid[:, :, 4]], dim=2)
        image_summary, image_available, latest_mask, history_mask, latest_weights, history_weights = (
            self._fixed_trajectory_statistics(image_states, image_valid, visit_mask)
        )
        text_summary, text_available, _, _, _, _ = self._fixed_trajectory_statistics(
            text_states, text_valid, visit_mask
        )
        normalized_image, normalized_image_valid = self._availability_weighted_mean(
            torch.stack([F.layer_norm(states[:, :, 0], (self.hidden_dim,)), F.layer_norm(states[:, :, 3], (self.hidden_dim,))], dim=2),
            image_valid,
        )
        normalized_text, normalized_text_valid = self._availability_weighted_mean(
            torch.stack([F.layer_norm(states[:, :, 1], (self.hidden_dim,)), F.layer_norm(states[:, :, 4], (self.hidden_dim,))], dim=2),
            text_valid,
        )
        pair_valid = normalized_image_valid & normalized_text_valid
        product = normalized_image * normalized_text
        disagreement = (normalized_image - normalized_text).abs()
        product_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            product.unsqueeze(2), pair_valid.unsqueeze(2), visit_mask
        )
        disagreement_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            disagreement.unsqueeze(2), pair_valid.unsqueeze(2), visit_mask
        )
        features = torch.cat(
            [
                image_summary.flatten(start_dim=1),
                text_summary.flatten(start_dim=1),
                product_summary.flatten(start_dim=1),
                disagreement_summary.flatten(start_dim=1),
            ],
            dim=-1,
        )
        return (
            features,
            image_summary,
            text_summary,
            image_available,
            text_available,
            latest_mask,
            history_mask,
            latest_weights,
            history_weights,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        (
            image_text_features,
            image_summary,
            text_summary,
            image_available,
            text_available,
            latest_mask,
            history_mask,
            latest_weights,
            history_weights,
        ) = self._image_text_kernel(source)
        visit_mask = source["visit_mask"]
        bio_values = torch.nan_to_num(batch["bio_values"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        bio_valid = ~batch["bio_missing_mask"].bool()
        bio_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            bio_values.unsqueeze(-1), bio_valid, visit_mask
        )
        bio_nonlinear = torch.cat(
            [bio_summary, torch.tanh(bio_summary), bio_summary * torch.tanh(bio_summary)], dim=-1
        ).flatten(start_dim=1)
        image_text_state = self.image_text_projection(image_text_features)
        bio_state = self.bio_projection(bio_nonlinear)
        joint_state = image_text_state * bio_state
        patient_state = self.patient_readout(torch.cat([image_text_state, bio_state, joint_state], dim=-1))
        logit = self.classifier(patient_state).squeeze(-1)

        image_stats = image_summary.reshape(image_summary.shape[0], 2, 4, self.hidden_dim)
        text_stats = text_summary.reshape(text_summary.shape[0], 2, 4, self.hidden_dim)
        representative_summary = torch.cat([image_stats, text_stats], dim=1)
        representative_valid = torch.cat([image_available, text_available], dim=1)
        evidence_tokens, evidence_present = self._safe_statistic_tokens(representative_summary, representative_valid)
        evidence_valid = evidence_present.unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(evidence_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        stream_tokens = torch.cat([image_stats[:, :, 0], text_stats[:, :, 0]], dim=1)
        safe_valid = representative_valid.clone()
        no_evidence = ~safe_valid.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_valid[no_evidence, 0] = True
        stream_tokens = stream_tokens * safe_valid.to(stream_tokens.dtype).unsqueeze(-1)
        consensus = evidence_tokens[:, 0]
        discordance = evidence_tokens[:, 3]
        source_bio_state = source["states"][:, :, 5]
        source_bio_valid = source["valid"][:, :, 5]
        bio_weights = source_bio_valid.to(source_bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        source_bio_state = (source_bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": consensus,
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
            "trajectory_available": representative_valid,
            "stream_tokens": stream_tokens,
            "stream_valid": safe_valid,
            "consensus_state": consensus,
            "discordance_state": discordance,
            "bio_nonlinear_state": bio_nonlinear,
            "joint_state": joint_state,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
