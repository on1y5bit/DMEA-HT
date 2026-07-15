from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "image_projection.",
    "text_projection.",
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


class C58MLTFModel(C47DRFEModel):
    """Low-rank three-way patient-level fusion of image, text, and continuous bio evidence."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c58"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        c58_cfg = dict(config["c58"])
        rank = int(c58_cfg["fusion_rank"])
        patient_dim = int(c58_cfg["patient_dim"])
        classifier_dim = int(c58_cfg["classifier_hidden_dim"])
        image_text_dim = self.hidden_dim * 8
        bio_dim = int(model_cfg["bio_dim"]) * 4 * 3
        self.image_projection = nn.Sequential(
            nn.LayerNorm(image_text_dim),
            nn.Linear(image_text_dim, rank),
            nn.GELU(),
            nn.LayerNorm(rank),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(image_text_dim),
            nn.Linear(image_text_dim, rank),
            nn.GELU(),
            nn.LayerNorm(rank),
        )
        self.bio_projection = nn.Sequential(
            nn.LayerNorm(bio_dim),
            nn.Linear(bio_dim, rank),
            nn.GELU(),
            nn.LayerNorm(rank),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(rank * 7),
            nn.Linear(rank * 7, patient_dim),
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
        if list(c58_cfg["stream_order"]) != STREAM_ORDER:
            raise RuntimeError("C58 stream order is fixed to raw then aligned image/text/bio")
        if rank != 32:
            raise RuntimeError("C58 fusion rank is fixed to 32")

    @staticmethod
    def _safe_statistic_tokens(
        summary: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = available.to(summary.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        tokens = (summary * weights.unsqueeze(-1)).sum(dim=1) / denominator.unsqueeze(-1)
        return tokens, available.any(dim=1)

    def _modality_summary(
        self, source: Dict[str, torch.Tensor], first: int, second: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = source["states"]
        valid = source["valid"]
        visit_mask = source["visit_mask"]
        selected_states = torch.stack([states[:, :, first], states[:, :, second]], dim=2)
        selected_valid = torch.stack([valid[:, :, first], valid[:, :, second]], dim=2)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = (
            self._fixed_trajectory_statistics(selected_states, selected_valid, visit_mask)
        )
        return summary, available, latest_mask, history_mask, latest_weights, history_weights

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        image_summary, image_available, latest_mask, history_mask, latest_weights, history_weights = self._modality_summary(source, 0, 3)
        text_summary, text_available, _, _, _, _ = self._modality_summary(source, 1, 4)
        image_stats = image_summary.reshape(image_summary.shape[0], 2, 4, self.hidden_dim)
        text_stats = text_summary.reshape(text_summary.shape[0], 2, 4, self.hidden_dim)
        image_features = image_summary.flatten(start_dim=1)
        text_features = text_summary.flatten(start_dim=1)

        bio_values = torch.nan_to_num(batch["bio_values"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        bio_valid = ~batch["bio_missing_mask"].bool()
        bio_summary, _, _, _, _, _ = self._fixed_trajectory_statistics(
            bio_values.unsqueeze(-1), bio_valid, source["visit_mask"]
        )
        bio_features = torch.cat([bio_summary, torch.tanh(bio_summary), bio_summary * torch.tanh(bio_summary)], dim=-1).flatten(start_dim=1)

        image_state = self.image_projection(image_features)
        text_state = self.text_projection(text_features)
        bio_state = self.bio_projection(bio_features)
        pair_image_text = image_state * text_state
        pair_image_bio = image_state * bio_state
        pair_text_bio = text_state * bio_state
        triple_state = pair_image_text * bio_state
        fused = torch.cat(
            [image_state, text_state, bio_state, pair_image_text, pair_image_bio, pair_text_bio, triple_state], dim=-1
        )
        patient_state = self.patient_readout(fused)
        logit = self.classifier(patient_state).squeeze(-1)

        representative_summary = torch.cat([image_stats, text_stats], dim=1)
        representative_valid = torch.cat([image_available, text_available], dim=1)
        evidence_tokens, evidence_present = self._safe_statistic_tokens(representative_summary, representative_valid)
        evidence_valid = evidence_present.unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(evidence_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        stream_tokens = representative_summary[:, :, 0]
        safe_valid = representative_valid.clone()
        no_evidence = ~safe_valid.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_valid[no_evidence, 0] = True
        stream_tokens = stream_tokens * safe_valid.to(stream_tokens.dtype).unsqueeze(-1)
        consensus = evidence_tokens[:, 0]
        discordance = evidence_tokens[:, 3]
        source_bio_state = source["states"][:, :, 5]
        source_bio_valid = source["valid"][:, :, 5]
        source_bio_weights = source_bio_valid.to(source_bio_state.dtype)
        source_bio_denominator = source_bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        source_bio_state = (source_bio_state * source_bio_weights.unsqueeze(-1)).sum(dim=1) / source_bio_denominator
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
            "fusion_state": fused,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
