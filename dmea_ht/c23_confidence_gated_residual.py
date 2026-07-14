from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c17_residual import C17ResidualModel


def _resolve_seed_path(value: str | Path, seed: int) -> Path:
    return Path(str(value).replace("{seed}", str(seed))).expanduser()


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"C17 checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint payload at {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


def confidence_gate(logit: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return a deterministic, non-learnable low-confidence gate."""
    return torch.exp(-logit.detach().abs() / float(temperature))


class ConfidenceGatedLocalResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.mlp[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, mechanism_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(mechanism_state).squeeze(-1)


class C23ConfidenceGatedResidualModel(nn.Module):
    """Frozen C17 final predictor with one confidence-gated local residual."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c23_cfg = dict(config.get("c23", {}))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))

        reference_config = copy.deepcopy(config)
        reference_config["phase"] = "c17"
        reference_config.pop("c23", None)
        self.frozen_c17 = C17ResidualModel(reference_config, seed)
        c17_path = _resolve_seed_path(c23_cfg["c17_checkpoint"], seed)
        self.frozen_c17.load_state_dict(_checkpoint_state(c17_path), strict=True)
        for parameter in self.frozen_c17.parameters():
            parameter.requires_grad = False
        self.frozen_c17.eval()

        self.residual_head = ConfidenceGatedLocalResidualHead(
            hidden_dim=hidden_dim,
            dropout=float(model_cfg.get("dropout", 0.15)),
        )
        self.temperature = float(c23_cfg.get("temperature", 1.0))
        self.residual_max = float(c23_cfg.get("residual_max", 0.15))
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C23ConfidenceGatedResidualModel":
        super().train(mode)
        self.frozen_c17.eval()
        self.residual_head.train(mode)
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            reference = self.frozen_c17(batch)
            frozen_logit = reference["logit"].detach()
            mechanism_state = reference["mea_mechanism_state"].detach()
        gate = confidence_gate(frozen_logit, self.temperature)
        raw_delta = self.residual_head(mechanism_state)
        local_direction = torch.tanh(raw_delta)
        delta = self.residual_max * gate.detach() * local_direction
        final_logit = frozen_logit + delta
        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "frozen_c17_logit": frozen_logit,
            "frozen_c17_prob": torch.sigmoid(frozen_logit),
            "confidence_gate": gate,
            "mechanism_representation_norm": mechanism_state.norm(dim=-1),
            "raw_delta_c23": raw_delta,
            "local_direction": local_direction,
            "delta_c23": delta,
        }


def _masked_or_graph_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any().item()):
        return values[mask].mean()
    return values.sum() * 0.0


def c23_loss_terms(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    criterion = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"], reduction="none")
    sample_weight = batch.get("sample_weight")
    classification = (criterion * sample_weight).mean() if sample_weight is not None else criterion.mean()
    delta = outputs["delta_c23"]
    labels = batch["label"]
    positive = labels > 0.5
    negative = ~positive
    residual = delta.square().mean()
    positive_preserve = _masked_or_graph_zero(F.relu(-delta - 0.02), positive)
    negative_preserve = _masked_or_graph_zero(F.relu(delta - 0.02), negative)
    high_confidence = outputs["frozen_c17_logit"].detach().abs() >= 2.0
    high_confidence_preserve = _masked_or_graph_zero(delta.abs(), high_confidence)
    total = (
        classification
        + float(loss_cfg.get("lambda_residual", 0.001)) * residual
        + float(loss_cfg.get("lambda_positive_preserve", 0.02)) * positive_preserve
        + float(loss_cfg.get("lambda_negative_preserve", 0.02)) * negative_preserve
        + float(loss_cfg.get("lambda_high_confidence", 0.01)) * high_confidence_preserve
    )
    return {
        "total": total,
        "classification": classification,
        "residual": residual,
        "positive_preserve": positive_preserve,
        "negative_preserve": negative_preserve,
        "high_confidence_preserve": high_confidence_preserve,
    }
