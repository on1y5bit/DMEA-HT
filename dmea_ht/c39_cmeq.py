from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import (
    BioEvidenceProjector,
    ImageMorphologyEvidenceProjector,
    TEXT_MASK_KEYS,
    TextEvidenceRoleProjector,
)
from dmea_ht.models import BioEncoder, ImageEncoder, TextEncoder


MODALITIES = ("image", "text", "bio")
PAIR_NAMES = ("image_text", "image_bio", "text_bio")
HEAD_PREFIXES = (
    "image_source_fusion.",
    "text_source_fusion.",
    "bio_source_fusion.",
    "trajectory_fusions.",
    "pair_relations.",
    "patient_readout.",
    "classifier.",
)


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return state


def _prefixed_state(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    selected = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        raise KeyError(f"No state found for checkpoint prefix {prefix}")
    return selected


class FrozenC17ModalitySources(nn.Module):
    """Frozen C17 encoders and the three pre-propagation evidence projectors."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
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

        checkpoint = Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        state = _checkpoint_state(checkpoint)
        self.image_encoder.load_state_dict(
            _prefixed_state(state, "base_model.image_encoder."), strict=True
        )
        self.text_encoder.load_state_dict(
            _prefixed_state(state, "base_model.text_encoder."), strict=True
        )
        self.bio_encoder.load_state_dict(
            _prefixed_state(state, "base_model.bio_encoder."), strict=True
        )
        self.image_projector.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment.image."), strict=True
        )
        self.text_projector.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment.text."), strict=True
        )
        self.bio_projector.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment.bio."), strict=True
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenC17ModalitySources":
        super().train(False)
        return self


class C39CMEQModel(nn.Module):
    """Cross-modal patient evidence model with fixed longitudinal integration.

    Visit order is used only by a fixed recency kernel. The predictor has no
    learned visit score, no patient structure feature, and no evidence-count
    feature. Cross-modal relations are formed only after each modality has a
    patient-level latest/history trajectory.
    """

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.hidden_dim = hidden_dim
        self.seed = int(seed)
        self.sources = FrozenC17ModalitySources(config, seed)

        self.image_source_fusion = self._source_fusion(hidden_dim * 5, hidden_dim, dropout)
        self.text_source_fusion = self._source_fusion(hidden_dim * 6, hidden_dim, dropout)
        self.bio_source_fusion = self._source_fusion(hidden_dim * 3, hidden_dim, dropout)
        self.trajectory_fusions = nn.ModuleList(
            [self._trajectory_fusion(hidden_dim, dropout) for _ in MODALITIES]
        )
        self.pair_relations = nn.ModuleList(
            [self._pair_relation(hidden_dim, dropout) for _ in PAIR_NAMES]
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _source_fusion(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _trajectory_fusion(hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _pair_relation(hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def train(self, mode: bool = True) -> "C39CMEQModel":
        super().train(mode)
        self.sources.eval()
        return self

    def _frozen_source_nodes(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        report_ids = batch["report_input_ids"].flatten(0, 1)
        report_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        image_tokens, _ = self.sources.image_encoder(images, image_mask)
        text_tokens, _ = self.sources.text_encoder(report_ids, report_mask)
        bio_tokens, _, _, _ = self.sources.bio_encoder(bio_values, bio_missing, bio_abnormal)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        image = self.sources.image_projector(image_tokens, image_mask)
        text = self.sources.text_projector(text_tokens, report_mask, text_masks)
        bio = self.sources.bio_projector(bio_tokens, bio_missing)
        image_nodes = image["nodes"].reshape(batch_size, visits, 5, self.hidden_dim)
        image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
        text_nodes = text["nodes"].reshape(batch_size, visits, 6, self.hidden_dim)
        text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
        bio_nodes = bio["nodes"].reshape(batch_size, visits, 3, self.hidden_dim)
        bio_valid = bio["valid"].reshape(batch_size, visits, 3).bool()
        return {
            "image_nodes": image_nodes,
            "image_valid": image_valid,
            "text_nodes": text_nodes,
            "text_valid": text_valid,
            "bio_nodes": bio_nodes,
            "bio_valid": bio_valid,
        }

    def _source_visit_states(
        self, source: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_valid = source["image_valid"]
        text_valid = source["text_valid"]
        bio_valid = source["bio_valid"]
        image_available = image_valid.any(dim=-1)
        text_available = text_valid.any(dim=-1)
        bio_available = bio_valid.any(dim=-1)
        image_input = source["image_nodes"].flatten(start_dim=2)
        text_input = source["text_nodes"].flatten(start_dim=2)
        bio_input = source["bio_nodes"].flatten(start_dim=2)
        image_state = self.image_source_fusion(image_input) * image_available.unsqueeze(-1)
        text_state = self.text_source_fusion(text_input) * text_available.unsqueeze(-1)
        bio_state = self.bio_source_fusion(bio_input) * bio_available.unsqueeze(-1)
        states = torch.stack([image_state, text_state, bio_state], dim=2)
        valid = torch.stack([image_available, text_available, bio_available], dim=2)
        return states, valid

    @staticmethod
    def _partition_weights(
        visit_mask: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(dtype)
        return latest_mask, history_mask, latest_weights, history_weights

    @staticmethod
    def _pool(
        states: torch.Tensor,
        valid: torch.Tensor,
        bucket_mask: torch.Tensor,
        bucket_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective = bucket_weights * bucket_mask.to(bucket_weights.dtype) * valid.to(bucket_weights.dtype)
        denominator = effective.sum(dim=1, keepdim=True)
        pooled = (states * effective.unsqueeze(-1)).sum(dim=1) / denominator.clamp_min(1e-8)
        return pooled, denominator.squeeze(-1) > 0.0

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._frozen_source_nodes(batch)
        source["image_valid"] = source["image_valid"] & batch["visit_mask"].bool().unsqueeze(-1)
        source["text_valid"] = source["text_valid"] & batch["visit_mask"].bool().unsqueeze(-1)
        source["bio_valid"] = source["bio_valid"] & batch["visit_mask"].bool().unsqueeze(-1)
        source_states, source_valid = self._source_visit_states(source)
        latest_mask, history_mask, latest_weights, history_weights = self._partition_weights(
            batch["visit_mask"].bool(), source_states.dtype
        )
        latest_states = []
        history_states = []
        trajectory_states = []
        latest_valid = []
        history_valid = []
        for modality_index, trajectory_fusion in enumerate(self.trajectory_fusions):
            latest, latest_ok = self._pool(
                source_states[:, :, modality_index],
                source_valid[:, :, modality_index],
                latest_mask,
                latest_weights,
            )
            history, history_ok = self._pool(
                source_states[:, :, modality_index],
                source_valid[:, :, modality_index],
                history_mask,
                history_weights,
            )
            trajectory = trajectory_fusion(torch.cat([latest, history, latest - history], dim=-1))
            trajectory = trajectory * (latest_ok | history_ok).unsqueeze(-1).to(trajectory.dtype)
            latest_states.append(latest)
            history_states.append(history)
            trajectory_states.append(trajectory)
            latest_valid.append(latest_ok)
            history_valid.append(history_ok)
        latest_states_tensor = torch.stack(latest_states, dim=1)
        history_states_tensor = torch.stack(history_states, dim=1)
        modality_states = torch.stack(trajectory_states, dim=1)
        modality_valid = torch.stack(latest_valid, dim=1) | torch.stack(history_valid, dim=1)
        modality_states = modality_states * modality_valid.unsqueeze(-1).to(modality_states.dtype)
        pair_indices = ((0, 1), (0, 2), (1, 2))
        pair_states = []
        pair_valid = []
        for relation, (left, right) in zip(self.pair_relations, pair_indices):
            valid = modality_valid[:, left] & modality_valid[:, right]
            state = relation(torch.cat([modality_states[:, left], modality_states[:, right]], dim=-1))
            pair_states.append(state * valid.unsqueeze(-1).to(state.dtype))
            pair_valid.append(valid)
        pair_states_tensor = torch.stack(pair_states, dim=1)
        pair_valid_tensor = torch.stack(pair_valid, dim=1)
        patient_input = torch.cat(
            [modality_states.flatten(start_dim=1), pair_states_tensor.flatten(start_dim=1)], dim=-1
        )
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "modality_states": modality_states,
            "pair_states": pair_states_tensor,
            "source_valid": source_valid,
            "modality_valid": modality_valid,
            "pair_valid": pair_valid_tensor,
            "latest_states": latest_states_tensor,
            "history_states": history_states_tensor,
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
