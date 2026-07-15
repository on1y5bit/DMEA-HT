from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "kernel_readout.",
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

MODALITY_STREAMS = ((0, 3), (1, 4), (2, 5))
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))


class C55PESKModel(C47DRFEModel):
    """Patient-level fixed evidence-set kernel over frozen C17 streams."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c55"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        c55_cfg = dict(config["c55"])
        readout_dim = int(c55_cfg["readout_dim"])
        classifier_dim = int(c55_cfg["classifier_hidden_dim"])
        feature_dim = self.hidden_dim * 48
        self.kernel_readout = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, readout_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(readout_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, 1),
        )
        if list(c55_cfg["stream_order"]) != STREAM_ORDER:
            raise RuntimeError("C55 stream order is fixed to raw then aligned image/text/bio")
        if [tuple(pair) for pair in c55_cfg["interaction_pairs"]] != list(PAIR_ORDER):
            raise RuntimeError("C55 interaction pairs are fixed to image-text, image-bio, text-bio")

    @staticmethod
    def _availability_weighted_modality_mean(
        states: torch.Tensor, valid: torch.Tensor, raw_index: int, aligned_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_states = torch.stack([states[:, :, raw_index], states[:, :, aligned_index]], dim=2)
        selected_valid = torch.stack([valid[:, :, raw_index], valid[:, :, aligned_index]], dim=2)
        weights = selected_valid.to(selected_states.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=2).clamp_min(1.0)
        mean = (selected_states * weights).sum(dim=2) / denominator
        return mean, selected_valid.any(dim=2)

    @staticmethod
    def _safe_fixed_statistic_tokens(
        summary: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = available.to(summary.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        tokens = (summary * weights.unsqueeze(-1)).sum(dim=1) / denominator.unsqueeze(-1)
        return tokens, available.any(dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        states = source["states"]
        valid = source["valid"]
        visit_mask = source["visit_mask"]
        base_summary, available, latest_mask, history_mask, latest_weights, history_weights = (
            self._fixed_trajectory_statistics(states, valid, visit_mask)
        )

        normalized_states = F.layer_norm(states, (self.hidden_dim,))
        modality_states = []
        modality_valid = []
        for raw_index, aligned_index in MODALITY_STREAMS:
            modality_state, modality_available = self._availability_weighted_modality_mean(
                normalized_states, valid, raw_index, aligned_index
            )
            modality_states.append(modality_state)
            modality_valid.append(modality_available)
        modality_states_tensor = torch.stack(modality_states, dim=2)
        modality_valid_tensor = torch.stack(modality_valid, dim=2)

        interaction_states = []
        interaction_valid = []
        disagreement_states = []
        disagreement_valid = []
        for left, right in PAIR_ORDER:
            left_state = modality_states_tensor[:, :, left]
            right_state = modality_states_tensor[:, :, right]
            left_valid = modality_valid_tensor[:, :, left]
            right_valid = modality_valid_tensor[:, :, right]
            pair_valid = left_valid & right_valid
            interaction_states.append(left_state * right_state)
            interaction_valid.append(pair_valid)
            disagreement_states.append((left_state - right_state).abs())
            disagreement_valid.append(pair_valid)
        interaction_tensor = torch.stack(interaction_states, dim=2)
        interaction_valid_tensor = torch.stack(interaction_valid, dim=2)
        disagreement_tensor = torch.stack(disagreement_states, dim=2)
        disagreement_valid_tensor = torch.stack(disagreement_valid, dim=2)
        interaction_summary, interaction_available, _, _, _, _ = self._fixed_trajectory_statistics(
            interaction_tensor, interaction_valid_tensor, visit_mask
        )
        disagreement_summary, disagreement_available, _, _, _, _ = self._fixed_trajectory_statistics(
            disagreement_tensor, disagreement_valid_tensor, visit_mask
        )

        features = torch.cat(
            [
                base_summary.flatten(start_dim=1),
                interaction_summary.flatten(start_dim=1),
                disagreement_summary.flatten(start_dim=1),
            ],
            dim=-1,
        )
        patient_state = self.kernel_readout(features)
        logit = self.classifier(patient_state).squeeze(-1)

        base_stats = base_summary.view(base_summary.shape[0], 6, 4, self.hidden_dim)
        evidence_tokens, evidence_present = self._safe_fixed_statistic_tokens(base_stats, available)
        evidence_valid = evidence_present.unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(evidence_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        stream_tokens = base_stats[:, :, 0]
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        stream_tokens = stream_tokens * safe_available.to(stream_tokens.dtype).unsqueeze(-1)
        consensus = evidence_tokens[:, 0]
        discordance = evidence_tokens[:, 3]
        bio_state = states[:, :, 5]
        bio_valid = valid[:, :, 5]
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
            "stream_tokens": stream_tokens,
            "stream_valid": safe_available,
            "consensus_state": consensus,
            "discordance_state": discordance,
            "interaction_available": interaction_available,
            "disagreement_available": disagreement_available,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
