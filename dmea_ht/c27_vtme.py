from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.mechanism_evidence_alignment import (
    BioEvidenceProjector,
    ImageMorphologyEvidenceProjector,
    TEXT_MASK_KEYS,
    TextEvidenceRoleProjector,
)
from dmea_ht.models import BioEncoder, ImageEncoder, TextEncoder


MECHANISM_NAMES = ("M1", "M2", "M3", "M4", "M5")


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported C27 checkpoint payload: {path}")
    return state


def masked_mean(states: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weights = valid.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    safe = valid.bool()
    masked = scores.masked_fill(~safe, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    return weights * safe.to(weights.dtype)


def prefixed_state(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    selected = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        raise KeyError(f"No C27 frozen source state found for prefix {prefix}")
    return selected


class FrozenC17EvidenceBackbone(nn.Module):
    """Only the C17 source encoders and pre-propagation projectors."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c27"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(int(model_cfg["text_vocab_size"]), hidden_dim, dropout)
        self.bio_encoder = BioEncoder(int(model_cfg["bio_dim"]), hidden_dim, dropout)
        self.image_projector = ImageMorphologyEvidenceProjector(
            hidden_dim, dropout, num_heads=int(model_cfg["mea_num_heads"])
        )
        self.text_projector = TextEvidenceRoleProjector(hidden_dim, dropout)
        self.bio_projector = BioEvidenceProjector(hidden_dim, dropout)

        checkpoint = Path(str(phase_cfg["c17_checkpoint"]).replace("{seed}", str(seed)))
        state = checkpoint_state(checkpoint)
        self.image_encoder.load_state_dict(prefixed_state(state, "base_model.image_encoder."), strict=True)
        self.text_encoder.load_state_dict(prefixed_state(state, "base_model.text_encoder."), strict=True)
        self.bio_encoder.load_state_dict(prefixed_state(state, "base_model.bio_encoder."), strict=True)
        self.image_projector.load_state_dict(
            prefixed_state(state, "mechanism_evidence_alignment.image."), strict=True
        )
        self.text_projector.load_state_dict(
            prefixed_state(state, "mechanism_evidence_alignment.text."), strict=True
        )
        self.bio_projector.load_state_dict(
            prefixed_state(state, "mechanism_evidence_alignment.bio."), strict=True
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "FrozenC17EvidenceBackbone":
        super().train(False)
        return self


class VisitTemporalMechanismCore(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, recency_prior_log_odds: float) -> None:
        super().__init__()
        self.empty_slot_tokens = nn.Parameter(torch.randn(len(MECHANISM_NAMES), hidden_dim) * 0.02)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_linear = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_output = nn.Linear(hidden_dim, 1)
        self.patient_projection = nn.Sequential(
            nn.Linear(hidden_dim * 6 + len(MECHANISM_NAMES), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.recency_prior_log_odds = float(recency_prior_log_odds)

    def _recency(self, visit_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, visits = visit_mask.shape
        positions = torch.arange(visits, device=visit_mask.device, dtype=dtype).view(1, visits)
        counts = visit_mask.sum(dim=1, keepdim=True)
        recency = positions / (counts.to(dtype) - 1.0).clamp_min(1.0)
        recency = torch.where(counts == 1, torch.ones_like(recency), recency)
        return recency * visit_mask.to(dtype)

    def _conflicts(
        self,
        visit_states: torch.Tensor,
        source_valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, mechanisms, hidden = visit_states.shape
        conflicts = visit_states.new_zeros(batch, mechanisms)
        history_available = torch.zeros(batch, mechanisms, dtype=torch.bool, device=visit_states.device)
        valid = source_valid & visit_mask.unsqueeze(-1)
        for batch_index in range(batch):
            for mechanism_index in range(mechanisms):
                indices = torch.nonzero(valid[batch_index, :, mechanism_index], as_tuple=False).flatten()
                if indices.numel() < 2:
                    continue
                latest = visit_states[batch_index, indices[-1], mechanism_index]
                history = visit_states[batch_index, indices[:-1], mechanism_index].mean(dim=0)
                latest_norm = F.layer_norm(latest, (hidden,))
                history_norm = F.layer_norm(history, (hidden,))
                conflicts[batch_index, mechanism_index] = 1.0 - F.cosine_similarity(
                    latest_norm.unsqueeze(0), history_norm.unsqueeze(0), dim=-1
                ).squeeze(0)
                history_available[batch_index, mechanism_index] = True
        return conflicts, history_available

    def forward(
        self,
        source_states: torch.Tensor,
        source_valid: torch.Tensor,
        visit_mask: torch.Tensor,
        fallback_bio_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        empty = self.empty_slot_tokens.view(1, 1, len(MECHANISM_NAMES), -1)
        visit_states = torch.where(source_valid.unsqueeze(-1), source_states, empty)
        content_scores = self.temporal_output(
            torch.tanh(self.temporal_linear(self.temporal_norm(visit_states)))
        ).squeeze(-1)
        recency = self._recency(visit_mask, content_scores.dtype)
        scores = content_scores + self.recency_prior_log_odds * recency.unsqueeze(-1)
        temporal_valid = visit_mask.unsqueeze(-1).expand_as(scores)
        temporal_weights = masked_softmax(scores, temporal_valid, dim=1)
        mechanism_states = torch.einsum("bvk,bvkh->bkh", temporal_weights, visit_states)
        conflicts, history_available = self._conflicts(visit_states, source_valid, visit_mask)
        patient_input = torch.cat(
            [mechanism_states.flatten(start_dim=1), conflicts, fallback_bio_context], dim=-1
        )
        patient_state = self.patient_projection(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        counts = visit_mask.sum(dim=1)
        latest_index = (counts - 1).clamp_min(0)
        latest_weights = temporal_weights[
            torch.arange(len(visit_mask), device=visit_mask.device), latest_index
        ]
        entropy = -(temporal_weights.clamp_min(1e-12) * temporal_weights.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = torch.where(
            counts.unsqueeze(-1) > 1,
            entropy / counts.to(entropy.dtype).log().unsqueeze(-1).clamp_min(1e-6),
            torch.ones_like(entropy),
        )
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "visit_states": visit_states,
            "mechanism_states": mechanism_states,
            "patient_state": patient_state,
            "temporal_weights": temporal_weights,
            "temporal_latest_weights": latest_weights,
            "temporal_entropy": entropy,
            "temporal_normalized_entropy": normalized_entropy,
            "conflicts": conflicts,
            "history_available": history_available,
            "recency": recency,
        }


class C27VTMEModel(nn.Module):
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c27"])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.frozen_sources = FrozenC17EvidenceBackbone(config, seed)
        self.core = VisitTemporalMechanismCore(
            hidden_dim=hidden_dim,
            dropout=float(model_cfg["dropout"]),
            recency_prior_log_odds=float(phase_cfg["recency_prior_log_odds"]),
        )
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C27VTMEModel":
        super().train(mode)
        self.frozen_sources.eval()
        self.core.train(mode)
        return self

    def _frozen_visit_sources(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        flat = batch_size * visits
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        input_ids = batch["report_input_ids"].flatten(0, 1)
        attention_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        image_tokens, _ = self.frozen_sources.image_encoder(images, image_mask)
        text_tokens, _ = self.frozen_sources.text_encoder(input_ids, attention_mask)
        bio_tokens, _, _, _ = self.frozen_sources.bio_encoder(bio_values, bio_missing, bio_abnormal)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        image = self.frozen_sources.image_projector(image_tokens, image_mask)
        text = self.frozen_sources.text_projector(text_tokens, attention_mask, text_masks)
        bio = self.frozen_sources.bio_projector(bio_tokens, bio_missing)

        image_available = image["valid"].any(dim=-1)
        image_morphology = masked_mean(image["nodes"], image["valid"], dim=1)
        text_available = batch["visit_text_valid"].flatten(0, 1)
        text_morphology = text["nodes"][:, (0, 3)].mean(dim=1)
        m1_sources = torch.stack([image_morphology, text_morphology], dim=1)
        m1_valid = torch.stack([image_available, text_available], dim=1)
        m1 = masked_mean(m1_sources, m1_valid, dim=1)
        m2 = bio["nodes"][:, 1]
        m3 = bio["nodes"][:, 2]
        m4 = text["nodes"][:, 1]
        m5 = text["nodes"][:, (2, 4)].mean(dim=1)
        source_states = torch.stack([m1, m2, m3, m4, m5], dim=1)
        source_valid = torch.stack(
            [m1_valid.any(dim=1), bio["valid"][:, 1], bio["valid"][:, 2], text_available, text_available],
            dim=1,
        )
        source_states = source_states * source_valid.unsqueeze(-1).to(source_states.dtype)
        morphology_alignment_valid = image_available & text_available

        fallback_values = batch["fallback_bio_values"]
        fallback_missing = batch["fallback_bio_missing_mask"]
        fallback_abnormal = torch.zeros_like(fallback_values)
        _, fallback_global, _, _ = self.frozen_sources.bio_encoder(
            fallback_values, fallback_missing, fallback_abnormal
        )
        fallback_global = fallback_global * batch["fallback_bio_valid"].unsqueeze(-1).to(fallback_global.dtype)
        return {
            "source_states": source_states.view(batch_size, visits, len(MECHANISM_NAMES), -1),
            "source_valid": source_valid.view(batch_size, visits, len(MECHANISM_NAMES)),
            "fallback_bio_context": fallback_global,
            "morphology_alignment_valid": morphology_alignment_valid.view(batch_size, visits),
            "image_morphology": image_morphology.view(batch_size, visits, -1),
            "text_morphology": text_morphology.view(batch_size, visits, -1),
        }

    @staticmethod
    def _alignment_summaries(
        image_morphology: torch.Tensor,
        text_morphology: torch.Tensor,
        alignment_valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, visits = visit_mask.shape
        same_mean = image_morphology.new_zeros(batch)
        cross_mean = image_morphology.new_zeros(batch)
        latest_same = image_morphology.new_zeros(batch)
        history_same = image_morphology.new_zeros(batch)
        same_count = image_morphology.new_zeros(batch)
        cross_pair_count = image_morphology.new_zeros(batch)
        for index in range(batch):
            valid_indices = torch.nonzero(alignment_valid[index] & visit_mask[index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            same_count[index] = valid_indices.numel()
            image_states = image_morphology[index, valid_indices]
            text_states = text_morphology[index, valid_indices]
            values = F.cosine_similarity(image_states, text_states, dim=-1)
            same_mean[index] = values.mean()
            latest_same[index] = values[-1]
            history_same[index] = values[:-1].mean() if values.numel() > 1 else values.new_tensor(0.0)
            if values.numel() > 1:
                cross_pair_count[index] = values.numel() * (values.numel() - 1)
                pairwise = F.cosine_similarity(
                    image_states.unsqueeze(1), text_states.unsqueeze(0), dim=-1
                )
                off_diagonal = ~torch.eye(values.numel(), dtype=torch.bool, device=values.device)
                cross_mean[index] = pairwise[off_diagonal].mean()
        return {
            "same_visit_alignment_mean": same_mean,
            "cross_visit_alignment_mean": cross_mean,
            "latest_same_visit_alignment": latest_same,
            "history_same_visit_alignment": history_same,
            "same_visit_alignment_count": same_count,
            "cross_visit_alignment_pair_count": cross_pair_count,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            frozen = self._frozen_visit_sources(batch)
        outputs = self.core(
            frozen["source_states"],
            frozen["source_valid"],
            batch["visit_mask"],
            frozen["fallback_bio_context"],
        )
        outputs.update(
            self._alignment_summaries(
                frozen["image_morphology"],
                frozen["text_morphology"],
                frozen["morphology_alignment_valid"],
                batch["visit_mask"],
            )
        )
        outputs["mechanism_source_valid"] = frozen["source_valid"]
        return outputs


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
