from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import (
    TEXT_MASK_KEYS,
    BioEvidenceProjector,
    ImageMorphologyEvidenceProjector,
    TextEvidenceRoleProjector,
)
from dmea_ht.models import BioEncoder, ImageEncoder, TextEncoder


MECHANISM_NAMES = ("M1", "M2", "M3", "M4", "M5")
MECHANISM_LABELS = (
    "M1_morphology",
    "M2_immune",
    "M3_function",
    "M4_opposition",
    "M5_temporal_text",
)
TRAJECTORY_COMPONENTS = ("latest", "history", "delta")
MECHANISM_SOURCE_MAP = {
    "M1": ("image_morphology", "text_support", "text_nonspecific_morphology"),
    "M2": ("bio_immune",),
    "M3": ("bio_function",),
    "M4": ("text_opposition",),
    "M5": ("text_temporal",),
}
EVIDENCE_SOURCE_NAMES = (
    "image_morphology",
    "text_support",
    "text_nonspecific_morphology",
    "bio_immune",
    "bio_function",
    "text_opposition",
    "text_temporal",
)
TRAINABLE_MODULES = {
    "mechanism_projectors": "mechanism_projectors",
    "trajectory_coordinate_heads": "trajectory_coordinate_heads",
    "mechanism_fallbacks": "mechanism_fallbacks",
    "anchor_center": "anchor_center",
    "anchor_direction": "anchor_direction",
}


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported C17 checkpoint payload: {path}")
    return state


def prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    selected = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        raise KeyError(f"No C17 state found for prefix {prefix}")
    return selected


def masked_arithmetic_mean(
    values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valid.bool().to(values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1)
    pooled = (values * weights).sum(dim=1) / count.clamp_min(1.0)
    return pooled, count.squeeze(-1) > 0.0


class FrozenC17EvidenceRepresentation(nn.Module):
    """C17 encoders and pre-propagation evidence projectors only."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c35"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(
            int(model_cfg["text_vocab_size"]), hidden_dim, dropout
        )
        self.bio_encoder = BioEncoder(int(model_cfg["bio_dim"]), hidden_dim, dropout)
        self.image_projector = ImageMorphologyEvidenceProjector(
            hidden_dim, dropout, num_heads=int(model_cfg["mea_num_heads"])
        )
        self.text_projector = TextEvidenceRoleProjector(hidden_dim, dropout)
        self.bio_projector = BioEvidenceProjector(hidden_dim, dropout)

        checkpoint = Path(
            str(phase_cfg["c17_checkpoint"]).replace("{seed}", str(seed))
        )
        state = checkpoint_state(checkpoint)
        self.image_encoder.load_state_dict(
            prefixed_state(state, "base_model.image_encoder."), strict=True
        )
        self.text_encoder.load_state_dict(
            prefixed_state(state, "base_model.text_encoder."), strict=True
        )
        self.bio_encoder.load_state_dict(
            prefixed_state(state, "base_model.bio_encoder."), strict=True
        )
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
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenC17EvidenceRepresentation":
        super().train(False)
        return self


class MechanismVisitProjector(nn.Module):
    def __init__(self, hidden_dim: int, mechanism_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mechanism_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class MechanismTrajectoryCoordinateHead(nn.Module):
    def __init__(self, trajectory_dim: int, coordinate_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(trajectory_dim),
            nn.Linear(trajectory_dim, coordinate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(coordinate_hidden_dim, 1),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(trajectory).squeeze(-1))


class C35MTSAModel(nn.Module):
    """Frozen C17 evidence, independent mechanism trajectories, and state anchors."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c35"])
        hidden_dim = int(model_cfg["hidden_dim"])
        mechanism_hidden_dim = int(phase_cfg["mechanism_hidden_dim"])
        coordinate_dim = int(phase_cfg["state_coordinate_dim"])
        dropout = float(model_cfg["dropout"])
        if coordinate_dim != len(MECHANISM_NAMES):
            raise ValueError("C35 state coordinate dimension must equal five mechanisms")
        self.seed = int(seed)
        self.hidden_dim = hidden_dim
        self.mechanism_hidden_dim = mechanism_hidden_dim
        self.frozen_sources = FrozenC17EvidenceRepresentation(config, seed)
        self.mechanism_projectors = nn.ModuleDict(
            {
                name: MechanismVisitProjector(hidden_dim, mechanism_hidden_dim, dropout)
                for name in MECHANISM_NAMES
            }
        )
        self.mechanism_fallbacks = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(hidden_dim) * 0.02)
                for name in MECHANISM_NAMES
            }
        )
        trajectory_dim = mechanism_hidden_dim * len(TRAJECTORY_COMPONENTS)
        self.trajectory_coordinate_heads = nn.ModuleDict(
            {
                name: MechanismTrajectoryCoordinateHead(
                    trajectory_dim,
                    int(phase_cfg["coordinate_hidden_dim"]),
                    dropout,
                )
                for name in MECHANISM_NAMES
            }
        )
        self.anchor_center = nn.Parameter(torch.zeros(coordinate_dim))
        self.anchor_direction = nn.Parameter(torch.randn(coordinate_dim) * 0.02)
        self.anchor_temperature = float(phase_cfg["anchor_temperature"])
        self._initial_trainable_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        self.train(False)

    def train(self, mode: bool = True) -> "C35MTSAModel":
        super().train(mode)
        self.frozen_sources.eval()
        self.mechanism_projectors.train(mode)
        self.trajectory_coordinate_heads.train(mode)
        return self

    @staticmethod
    def is_trainable_parameter(name: str) -> bool:
        return parameter_category(name) is not None

    def _frozen_evidence_nodes(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        input_ids = batch["report_input_ids"].flatten(0, 1)
        attention_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        with torch.no_grad():
            image_tokens, _ = self.frozen_sources.image_encoder(images, image_mask)
            text_tokens, _ = self.frozen_sources.text_encoder(input_ids, attention_mask)
            bio_tokens, _, _, _ = self.frozen_sources.bio_encoder(
                bio_values, bio_missing, bio_abnormal
            )
            image = self.frozen_sources.image_projector(image_tokens, image_mask)
            text = self.frozen_sources.text_projector(
                text_tokens, attention_mask, text_masks
            )
            bio = self.frozen_sources.bio_projector(bio_tokens, bio_missing)

            image_morphology, image_available = masked_arithmetic_mean(
                image["nodes"], image["valid"]
            )
            text_support = text["nodes"][:, 0]
            text_nonspecific = text["nodes"][:, 3]
            text_opposition = text["nodes"][:, 1]
            text_temporal = text["nodes"][:, 4]
            text_support_valid = text["guidance_present"][:, 0]
            text_nonspecific_valid = text["guidance_present"][:, 3]
            text_opposition_valid = text["guidance_present"][:, 1]
            text_temporal_valid = text["temporal_available"]

            m1_values = torch.stack(
                [image_morphology, text_support, text_nonspecific], dim=1
            )
            m1_valid = torch.stack(
                [image_available, text_support_valid, text_nonspecific_valid], dim=1
            )
            m1, m1_available = masked_arithmetic_mean(m1_values, m1_valid)
            source_states = torch.stack(
                [
                    m1,
                    bio["nodes"][:, 1],
                    bio["nodes"][:, 2],
                    text_opposition,
                    text_temporal,
                ],
                dim=1,
            )
            source_valid = torch.stack(
                [
                    m1_available,
                    bio["valid"][:, 1],
                    bio["valid"][:, 2],
                    text_opposition_valid,
                    text_temporal_valid,
                ],
                dim=1,
            )
        return (
            source_states.view(batch_size, visits, len(MECHANISM_NAMES), self.hidden_dim),
            source_valid.view(batch_size, visits, len(MECHANISM_NAMES))
            & batch["visit_mask"].unsqueeze(-1).bool(),
            {
                "image_morphology": image_available.view(batch_size, visits),
                "text_support": text_support_valid.view(batch_size, visits),
                "text_nonspecific_morphology": text_nonspecific_valid.view(
                    batch_size, visits
                ),
                "bio_immune": bio["valid"][:, 1].view(batch_size, visits),
                "bio_function": bio["valid"][:, 2].view(batch_size, visits),
                "text_opposition": text_opposition_valid.view(batch_size, visits),
                "text_temporal": text_temporal_valid.view(batch_size, visits),
            },
        )

    def _select_mechanism_inputs(
        self, source_states: torch.Tensor, source_valid: torch.Tensor
    ) -> torch.Tensor:
        fallback = torch.stack(
            [self.mechanism_fallbacks[name] for name in MECHANISM_NAMES], dim=0
        ).view(1, 1, len(MECHANISM_NAMES), self.hidden_dim)
        return torch.where(source_valid.unsqueeze(-1), source_states, fallback)

    def _trajectory(self, visit_states: torch.Tensor, visit_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, visits, mechanisms, hidden = visit_states.shape
        valid_visits = visit_mask.bool()
        counts = valid_visits.sum(dim=1)
        latest_index = (counts - 1).clamp_min(0).long()
        batch_index = torch.arange(batch_size, device=visit_states.device)
        latest = visit_states[batch_index, latest_index]

        history_mask = valid_visits.clone()
        history_mask[batch_index, latest_index] = False
        positions = torch.arange(
            visits, device=visit_states.device, dtype=visit_states.dtype
        )
        recency_weights = torch.log2(positions + 2.0).view(1, visits, 1, 1)
        history_weights = recency_weights * history_mask.to(visit_states.dtype).view(
            batch_size, visits, 1, 1
        )
        denominator = history_weights.sum(dim=1)
        history = (visit_states * history_weights).sum(dim=1) / denominator.clamp_min(1.0)
        has_history = history_mask.any(dim=1).view(batch_size, 1, 1)
        history = torch.where(has_history, history, latest)
        delta = latest - history
        trajectory = torch.cat([latest, history, delta], dim=-1)
        return {
            "latest_mechanism_state": latest,
            "history_mechanism_state": history,
            "mechanism_state_delta": delta,
            "mechanism_trajectory": trajectory,
        }

    def _anchor_outputs(self, coordinate: torch.Tensor) -> Dict[str, torch.Tensor]:
        direction = torch.nn.functional.normalize(self.anchor_direction, dim=0, eps=1e-8)
        non_ht = self.anchor_center - direction
        ht = self.anchor_center + direction
        d_non_ht = (coordinate - non_ht.unsqueeze(0)).pow(2).sum(dim=-1)
        d_ht = (coordinate - ht.unsqueeze(0)).pow(2).sum(dim=-1)
        anchor_distance = torch.linalg.vector_norm(ht - non_ht)
        state_margin = (d_non_ht - d_ht) / self.anchor_temperature
        anchor_cosine = torch.nn.functional.cosine_similarity(
            coordinate, direction.unsqueeze(0).expand_as(coordinate), dim=-1, eps=1e-8
        )
        return {
            "anchor_center": self.anchor_center,
            "anchor_direction": direction,
            "anchor_non_ht": non_ht,
            "anchor_ht": ht,
            "anchor_distance": anchor_distance,
            "d_non_ht": d_non_ht,
            "d_ht": d_ht,
            "state_margin": state_margin,
            "anchor_cosine": anchor_cosine,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_states, source_valid, source_evidence = self._frozen_evidence_nodes(batch)
        selected = self._select_mechanism_inputs(source_states, source_valid)
        projected = torch.stack(
            [
                self.mechanism_projectors[name](selected[:, :, index])
                for index, name in enumerate(MECHANISM_NAMES)
            ],
            dim=2,
        )
        trajectory_outputs = self._trajectory(projected, batch["visit_mask"])
        trajectory = trajectory_outputs["mechanism_trajectory"]
        coordinates = torch.stack(
            [
                self.trajectory_coordinate_heads[name](trajectory[:, index])
                for index, name in enumerate(MECHANISM_NAMES)
            ],
            dim=1,
        )
        anchor_outputs = self._anchor_outputs(coordinates)
        logit = anchor_outputs["state_margin"]
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "mechanism_source_states": source_states,
            "mechanism_source_valid": source_valid,
            "mechanism_source_evidence": torch.stack(
                [source_evidence[name] for name in EVIDENCE_SOURCE_NAMES], dim=-1
            ),
            "mechanism_visit_state": projected,
            "mechanism_visit_valid": batch["visit_mask"].unsqueeze(-1).expand_as(source_valid),
            "patient_coordinate": coordinates,
            **trajectory_outputs,
            **anchor_outputs,
        }

    def parameter_drift_rows(self) -> list[Dict[str, Any]]:
        current = dict(self.named_parameters())
        rows: list[Dict[str, Any]] = []
        for name, baseline in self._initial_trainable_state.items():
            category = parameter_category(name)
            if category is None:
                raise RuntimeError(f"Unknown C35 trainable parameter: {name}")
            value = current[name].detach().cpu()
            denominator = max(float(torch.linalg.vector_norm(baseline)), 1e-8)
            relative = float(torch.linalg.vector_norm(value - baseline)) / denominator
            rows.append(
                {
                    "seed": self.seed,
                    "category": category,
                    "module_name": TRAINABLE_MODULES[category],
                    "parameter_name": name,
                    "parameter_count": int(value.numel()),
                    "relative_parameter_drift": relative,
                    "finite": bool(np.isfinite(relative)),
                }
            )
        return rows


def parameter_category(name: str) -> str | None:
    if name.startswith("mechanism_projectors."):
        return "mechanism_projectors"
    if name.startswith("trajectory_coordinate_heads."):
        return "trajectory_coordinate_heads"
    if name.startswith("mechanism_fallbacks."):
        return "mechanism_fallbacks"
    if name == "anchor_center":
        return "anchor_center"
    if name == "anchor_direction":
        return "anchor_direction"
    return None


def named_trainable_parameters(
    model: C35MTSAModel,
) -> Iterable[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def trainable_parameter_count(model: C35MTSAModel) -> int:
    return int(sum(parameter.numel() for _, parameter in named_trainable_parameters(model)))


def parameter_audit(model: C35MTSAModel) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        category = parameter_category(name)
        rows.append(
            {
                "seed": model.seed,
                "category": category or "frozen_c17_representation",
                "module_name": TRAINABLE_MODULES[category] if category else name.rsplit(".", 1)[0],
                "parameter_name": name,
                "parameter_count": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows
