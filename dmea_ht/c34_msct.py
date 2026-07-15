from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS
from dmea_ht.models import BioEncoder, ImageEncoder, TextEncoder


SOURCE_NAMES = (
    "image",
    "text_support",
    "text_opposition",
    "bio_immune",
    "bio_function",
)
BIO_IMMUNE_INDICES = (2, 5)
BIO_FUNCTION_INDICES = (3, 4, 6)
TRAJECTORY_NAMES = (
    "latest_state",
    "history_state",
    "state_delta",
    "history_dispersion",
    "latest_source_disagreement",
)

TRAINABLE_MODULES = {
    "source_heads": "source_heads",
    "fallback_tokens": "fallback_tokens",
    "empty_visit_state": "empty_visit_logit",
    "trajectory_norm": "trajectory_norm",
    "classifier": "classifier",
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
        raise KeyError(f"No C17 encoder state found for prefix {prefix}")
    return selected


def masked_arithmetic_mean(
    values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valid.bool().to(values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1)
    pooled = (values * weights).sum(dim=1) / count.clamp_min(1.0)
    return pooled, count.squeeze(-1) > 0.0


class EvidenceStateCoordinateHead(nn.Module):
    """One independent low-capacity head for one evidence source."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class C34MSCTModel(nn.Module):
    """Frozen modality encoders followed by a five-coordinate state trajectory."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c34"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.seed = int(seed)
        self.hidden_dim = hidden_dim

        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(
            int(model_cfg["text_vocab_size"]), hidden_dim, dropout
        )
        self.bio_encoder = BioEncoder(int(model_cfg["bio_dim"]), hidden_dim, dropout)

        checkpoint_path = Path(
            str(phase_cfg["encoder_checkpoint"]).replace("{seed}", str(seed))
        )
        state = checkpoint_state(checkpoint_path)
        self.image_encoder.load_state_dict(
            prefixed_state(state, "base_model.image_encoder."), strict=True
        )
        self.text_encoder.load_state_dict(
            prefixed_state(state, "base_model.text_encoder."), strict=True
        )
        self.bio_encoder.load_state_dict(
            prefixed_state(state, "base_model.bio_encoder."), strict=True
        )
        for encoder in (self.image_encoder, self.text_encoder, self.bio_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            encoder.eval()

        self.source_heads = nn.ModuleDict(
            {
                name: EvidenceStateCoordinateHead(hidden_dim, dropout)
                for name in SOURCE_NAMES
            }
        )
        self.fallback_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(hidden_dim) * 0.02)
                for name in SOURCE_NAMES
            }
        )
        self.empty_visit_logit = nn.Parameter(torch.tensor(0.02))
        self.trajectory_norm = nn.LayerNorm(5)
        self.classifier = nn.Linear(5, 1)
        self._initial_trainable_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        self.train(False)

    def train(self, mode: bool = True) -> "C34MSCTModel":
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.bio_encoder.eval()
        self.source_heads.train(mode)
        self.trajectory_norm.train(mode)
        self.classifier.train(mode)
        return self

    @staticmethod
    def is_trainable_parameter(name: str) -> bool:
        return parameter_category(name) is not None

    def _encode_frozen_sources(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        input_ids = batch["report_input_ids"].flatten(0, 1)
        attention_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        text_masks = {
            key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS
        }

        with torch.no_grad():
            image_tokens, _ = self.image_encoder(images, image_mask)
            text_tokens, _ = self.text_encoder(input_ids, attention_mask)
            bio_tokens, _, _, _ = self.bio_encoder(
                bio_values, bio_missing, bio_abnormal
            )

            image_feature, image_valid = masked_arithmetic_mean(
                image_tokens, image_mask
            )
            support_mask = torch.maximum(
                text_masks["text_support_mask"],
                text_masks["text_diagnostic_hint_mask"],
            )
            text_support_feature, text_support_valid = masked_arithmetic_mean(
                text_tokens, attention_mask.bool() & support_mask.bool()
            )
            text_opposition_feature, text_opposition_valid = masked_arithmetic_mean(
                text_tokens,
                attention_mask.bool() & text_masks["text_opposition_mask"].bool(),
            )

            bio_observed = ~bio_missing.bool()
            immune_index = torch.tensor(
                BIO_IMMUNE_INDICES, device=bio_tokens.device, dtype=torch.long
            )
            function_index = torch.tensor(
                BIO_FUNCTION_INDICES, device=bio_tokens.device, dtype=torch.long
            )
            immune_feature, immune_valid = masked_arithmetic_mean(
                bio_tokens.index_select(1, immune_index),
                bio_observed.index_select(1, immune_index),
            )
            function_feature, function_valid = masked_arithmetic_mean(
                bio_tokens.index_select(1, function_index),
                bio_observed.index_select(1, function_index),
            )

        features = (
            image_feature,
            text_support_feature,
            text_opposition_feature,
            immune_feature,
            function_feature,
        )
        valid = (
            image_valid,
            text_support_valid,
            text_opposition_valid,
            immune_valid,
            function_valid,
        )
        feature_states = []
        state_logits = []
        for name, feature, available in zip(SOURCE_NAMES, features, valid):
            fallback = self.fallback_tokens[name].view(1, -1).expand_as(feature)
            selected = torch.where(available.unsqueeze(-1), feature, fallback)
            logit = self.source_heads[name](selected)
            state_logits.append(logit)
            feature_states.append(torch.sigmoid(logit))

        state_logits_tensor = torch.stack(state_logits, dim=1).view(
            batch_size, visits, len(SOURCE_NAMES)
        )
        source_states = torch.stack(feature_states, dim=1).view(
            batch_size, visits, len(SOURCE_NAMES)
        )
        source_evidence_valid = torch.stack(valid, dim=1).view(
            batch_size, visits, len(SOURCE_NAMES)
        )
        source_evidence_valid = source_evidence_valid & batch["visit_mask"].unsqueeze(-1).bool()
        source_used = batch["visit_mask"].unsqueeze(-1).expand_as(source_evidence_valid)
        return (
            state_logits_tensor,
            source_states,
            source_evidence_valid,
            source_used,
            batch["visit_mask"],
        )

    def _trajectory(
        self,
        source_states: torch.Tensor,
        source_evidence_valid: torch.Tensor,
        source_used: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source_weights = source_used.to(source_states.dtype)
        source_count = source_weights.sum(dim=-1)
        source_mean = (
            (source_states * source_weights).sum(dim=-1)
            / source_count.clamp_min(1.0)
        )
        empty_visit_state = torch.sigmoid(self.empty_visit_logit)
        visit_state = torch.where(
            source_evidence_valid.any(dim=-1),
            source_mean,
            empty_visit_state.expand_as(source_mean),
        )
        observed_weights = source_evidence_valid.to(source_states.dtype)
        observed_count = observed_weights.sum(dim=-1)
        observed_mean = (
            (source_states * observed_weights).sum(dim=-1)
            / observed_count.clamp_min(1.0)
        )
        source_variance = (
            (source_states - observed_mean.unsqueeze(-1)).pow(2) * observed_weights
        ).sum(dim=-1) / observed_count.clamp_min(1.0)
        source_disagreement = torch.sqrt(source_variance.clamp_min(0.0))
        source_disagreement = torch.where(
            observed_count >= 2.0,
            source_disagreement,
            torch.zeros_like(source_disagreement),
        )

        counts = visit_mask.bool().sum(dim=1)
        latest_index = (counts - 1).clamp_min(0).long()
        batch_index = torch.arange(
            visit_mask.shape[0], device=visit_mask.device, dtype=torch.long
        )
        latest_state = visit_state[batch_index, latest_index]
        latest_disagreement = source_disagreement[batch_index, latest_index]

        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=source_states.dtype
        )
        fixed_weights = torch.pow(source_states.new_tensor(2.0), positions).view(1, -1)
        history_mask = visit_mask.bool().clone()
        history_mask[batch_index, latest_index] = False
        history_weights = fixed_weights * history_mask.to(source_states.dtype)
        history_count = history_mask.sum(dim=1)
        history_denom = history_weights.sum(dim=1)
        weighted_history = (visit_state * history_weights).sum(dim=1)
        history_state = weighted_history / history_denom.clamp_min(1.0)
        history_state = torch.where(
            history_count > 0,
            history_state,
            latest_state,
        )
        history_variance = (
            (visit_state - history_state.unsqueeze(-1)).pow(2) * history_weights
        ).sum(dim=1) / history_denom.clamp_min(1.0)
        history_dispersion = torch.sqrt(history_variance.clamp_min(0.0))
        history_dispersion = torch.where(
            history_count >= 2,
            history_dispersion,
            torch.zeros_like(history_dispersion),
        )
        state_delta = latest_state - history_state
        trajectory = torch.stack(
            [
                latest_state,
                history_state,
                state_delta,
                history_dispersion,
                latest_disagreement,
            ],
            dim=-1,
        )
        return {
            "visit_state": visit_state,
            "source_disagreement": source_disagreement,
            "latest_state": latest_state,
            "history_state": history_state,
            "state_delta": state_delta,
            "history_dispersion": history_dispersion,
            "latest_source_disagreement": latest_disagreement,
            "trajectory": trajectory,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        (
            source_logits,
            source_states,
            source_evidence_valid,
            source_used,
            visit_mask,
        ) = self._encode_frozen_sources(batch)
        trajectory_outputs = self._trajectory(
            source_states, source_evidence_valid, source_used, visit_mask
        )
        normalized = self.trajectory_norm(trajectory_outputs["trajectory"])
        logit = self.classifier(normalized).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "source_logits": source_logits,
            "source_states": source_states,
            "source_valid": source_used,
            "source_evidence_valid": source_evidence_valid,
            **trajectory_outputs,
        }

    def parameter_drift_rows(self) -> list[Dict[str, Any]]:
        current = dict(self.named_parameters())
        rows: list[Dict[str, Any]] = []
        for name, baseline in self._initial_trainable_state.items():
            category = parameter_category(name)
            if category is None:
                raise RuntimeError(f"Unknown C34 trainable parameter: {name}")
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
    if name.startswith("source_heads."):
        return "source_heads"
    if name.startswith("fallback_tokens."):
        return "fallback_tokens"
    if name == "empty_visit_logit":
        return "empty_visit_state"
    if name.startswith("trajectory_norm."):
        return "trajectory_norm"
    if name.startswith("classifier."):
        return "classifier"
    return None


def named_trainable_parameters(
    model: C34MSCTModel,
) -> Iterable[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def trainable_parameter_count(model: C34MSCTModel) -> int:
    return int(sum(parameter.numel() for _, parameter in named_trainable_parameters(model)))


def parameter_audit(model: C34MSCTModel) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        category = parameter_category(name)
        rows.append(
            {
                "seed": model.seed,
                "category": category or "frozen",
                "module_name": TRAINABLE_MODULES[category] if category else name.rsplit(".", 1)[0],
                "parameter_name": name,
                "parameter_count": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows
