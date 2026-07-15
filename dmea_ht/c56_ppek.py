from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "support_readout.",
    "support_classifier.",
    "opposition_head.",
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


class C56PPEKModel(C47DRFEModel):
    """Polarity-preserving support/opposition readout over frozen evidence kernels."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c56"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        c56_cfg = dict(config["c56"])
        support_dim = self.hidden_dim * 36
        opposition_dim = self.hidden_dim * 12
        support_dim_hidden = int(c56_cfg["support_hidden_dim"])
        classifier_dim = int(c56_cfg["support_classifier_dim"])
        opposition_hidden = int(c56_cfg["opposition_hidden_dim"])
        self.support_readout = nn.Sequential(
            nn.LayerNorm(support_dim),
            nn.Linear(support_dim, support_dim_hidden),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(support_dim_hidden),
        )
        self.support_classifier = nn.Sequential(
            nn.LayerNorm(support_dim_hidden),
            nn.Linear(support_dim_hidden, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, 1),
        )
        self.opposition_head = nn.Sequential(
            nn.LayerNorm(opposition_dim),
            nn.Linear(opposition_dim, opposition_hidden),
            nn.GELU(),
            nn.Linear(opposition_hidden, 1),
        )
        if list(c56_cfg["stream_order"]) != STREAM_ORDER:
            raise RuntimeError("C56 stream order is fixed to raw then aligned image/text/bio")
        if [tuple(pair) for pair in c56_cfg["interaction_pairs"]] != list(PAIR_ORDER):
            raise RuntimeError("C56 interaction pairs are fixed to image-text, image-bio, text-bio")
        if abs(float(c56_cfg["opposition_penalty_bound"]) - 0.75) > 1e-12:
            raise RuntimeError("C56 opposition penalty bound is fixed to 0.75")

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

    def _fixed_kernel_features(
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
    ]:
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
            pair_valid = modality_valid_tensor[:, :, left] & modality_valid_tensor[:, :, right]
            interaction_states.append(left_state * right_state)
            interaction_valid.append(pair_valid)
            disagreement_states.append((left_state - right_state).abs())
            disagreement_valid.append(pair_valid)
        interaction_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            torch.stack(interaction_states, dim=2), torch.stack(interaction_valid, dim=2), visit_mask
        )
        disagreement_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            torch.stack(disagreement_states, dim=2), torch.stack(disagreement_valid, dim=2), visit_mask
        )
        return (
            base_summary,
            interaction_summary,
            disagreement_summary,
            available,
            latest_mask,
            history_mask,
            latest_weights,
            history_weights,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        base_summary, interaction_summary, disagreement_summary, available, latest_mask, history_mask, latest_weights, history_weights = (
            self._fixed_kernel_features(source)
        )
        support_features = torch.cat([base_summary.flatten(start_dim=1), interaction_summary.flatten(start_dim=1)], dim=-1)
        opposition_features = disagreement_summary.flatten(start_dim=1)
        support_state = self.support_readout(support_features)
        support_logit = self.support_classifier(support_state).squeeze(-1)
        opposition_score = self.opposition_head(opposition_features).squeeze(-1)
        bound = float(self._c56_penalty_bound)
        opposition_penalty = bound * torch.tanh(F.softplus(opposition_score))
        logit = support_logit - opposition_penalty
        base_stats = base_summary.reshape(base_summary.shape[0], 6, 4, self.hidden_dim)
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
        bio_state = source["states"][:, :, 5]
        bio_valid = source["valid"][:, :, 5]
        bio_weights = bio_valid.to(bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = (bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": torch.cat([support_state, opposition_penalty.unsqueeze(-1)], dim=-1),
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
            "opposition_penalty": opposition_penalty,
        }

    @property
    def _c56_penalty_bound(self) -> float:
        return 0.75


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
