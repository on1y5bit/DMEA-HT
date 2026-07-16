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


MODALITY_NAMES = ("image", "text", "bio")
HEAD_PREFIXES = (
    "trajectory_encoders.",
    "modality_heads.",
    "router.",
    "patient_readout.",
    "consensus_head.",
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
        raise KeyError(f"No checkpoint state found for prefix {prefix}")
    return selected


def _masked_mean(
    states: torch.Tensor, valid: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valid.to(states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=dim)
    pooled = (states * weights).sum(dim=dim) / denominator.clamp_min(1.0)
    return pooled, denominator.squeeze(-1) > 0.0


class FrozenC17ModalitySources(nn.Module):
    """C17 source encoders/projectors with a compatibility trainable mode."""

    def __init__(self, config: Dict[str, Any], seed: int, trainable: bool = False) -> None:
        super().__init__()
        self.predictive_trainable = bool(trainable)
        initialization = dict(config.get("initialization", {}))
        self.from_base = bool(
            config.get("from_base", False) or initialization.get("mode") == "from_base"
        )
        self.initialization_type = "random_task_specific" if self.from_base else "task_checkpoint_compatibility"
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

        if not self.from_base:
            checkpoint = Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))
            state = _checkpoint_state(checkpoint)
            self.image_encoder.load_state_dict(_prefixed_state(state, "base_model.image_encoder."), strict=True)
            self.text_encoder.load_state_dict(_prefixed_state(state, "base_model.text_encoder."), strict=True)
            self.bio_encoder.load_state_dict(_prefixed_state(state, "base_model.bio_encoder."), strict=True)
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
            parameter.requires_grad_(self.predictive_trainable)
        if not self.predictive_trainable:
            self.eval()

    def train(self, mode: bool = True) -> "FrozenC17ModalitySources":
        super().train(mode if self.predictive_trainable else False)
        return self


class C41MELRModel(nn.Module):
    """Fixed trajectory statistics with single-model evidence-logit routing."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.hidden_dim = hidden_dim
        self.seed = int(seed)
        self.sources = FrozenC17ModalitySources(config, seed)
        self.trajectory_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim * 4),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in MODALITY_NAMES
            ]
        )
        self.modality_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in MODALITY_NAMES
            ]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.consensus_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def train(self, mode: bool = True) -> "C41MELRModel":
        super().train(mode)
        self.sources.eval()
        return self

    def _source_states(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        text_nodes = text["nodes"].reshape(batch_size, visits, 6, self.hidden_dim)
        bio_nodes = bio["nodes"].reshape(batch_size, visits, 3, self.hidden_dim)
        image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
        text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
        bio_valid = bio["valid"].reshape(batch_size, visits, 3).bool()
        image_state, image_available = _masked_mean(image_nodes, image_valid, dim=2)
        text_state, text_available = _masked_mean(text_nodes, text_valid, dim=2)
        bio_state, bio_available = _masked_mean(bio_nodes, bio_valid, dim=2)
        visit_mask = batch["visit_mask"].bool()
        return {
            "states": torch.stack([image_state, text_state, bio_state], dim=2),
            "valid": torch.stack([image_available, text_available, bio_available], dim=2) & visit_mask.unsqueeze(-1),
            "visit_mask": visit_mask,
        }

    @staticmethod
    def _trajectory_statistics(
        states: torch.Tensor,
        valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(states.dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(states.dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(states.dtype)

        latest_effective = latest_weights.unsqueeze(-1) * valid.to(states.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (states * latest_effective.unsqueeze(-1)).sum(dim=1) / latest_denominator.clamp_min(1.0).unsqueeze(-1)
        latest_valid = latest_denominator > 0.0

        history_effective = history_weights.unsqueeze(-1) * history_mask.unsqueeze(-1).to(states.dtype) * valid.to(states.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (states * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0).unsqueeze(-1)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(states.dtype)
        centered = (states - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0).unsqueeze(-1)
        dispersion = variance.sqrt() * history_valid.unsqueeze(-1).to(states.dtype)
        summary = torch.cat([latest, history, delta, dispersion], dim=-1)
        available = latest_valid | history_valid
        return summary, available

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available = self._trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        tokens = []
        for index, encoder in enumerate(self.trajectory_encoders):
            token = encoder(summary[:, index])
            tokens.append(token * available[:, index].unsqueeze(-1).to(token.dtype))
        modality_tokens = torch.stack(tokens, dim=1)
        route_logits = self.router(modality_tokens).squeeze(-1)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        route_logits = route_logits.masked_fill(~safe_available, torch.finfo(route_logits.dtype).min)
        routing_weights = torch.softmax(route_logits, dim=1) * safe_available.to(route_logits.dtype)
        modality_logits = torch.cat(
            [head(modality_tokens[:, index]) for index, head in enumerate(self.modality_heads)], dim=1
        )
        routed_state = (modality_tokens * routing_weights.unsqueeze(-1)).sum(dim=1)
        mean_state = (modality_tokens * safe_available.unsqueeze(-1).to(modality_tokens.dtype)).sum(dim=1)
        mean_state = mean_state / safe_available.sum(dim=1, keepdim=True).clamp_min(1).to(mean_state.dtype)
        masked_tokens = modality_tokens.masked_fill(~safe_available.unsqueeze(-1), torch.finfo(modality_tokens.dtype).min)
        max_state = masked_tokens.max(dim=1).values
        patient_state = self.patient_readout(torch.cat([routed_state, mean_state, max_state], dim=-1))
        evidence_logit = (routing_weights * modality_logits).sum(dim=1)
        consensus_logit = self.consensus_head(patient_state).squeeze(-1)
        logit = evidence_logit + consensus_logit
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "modality_tokens": modality_tokens,
            "modality_logits": modality_logits,
            "routing_weights": routing_weights,
            "trajectory_available": available,
            "evidence_logit": evidence_logit,
            "consensus_logit": consensus_logit,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
