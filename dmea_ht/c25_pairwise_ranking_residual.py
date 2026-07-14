from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c17_residual import C17ResidualModel
from dmea_ht.c23_confidence_gated_residual import ConfidenceGatedLocalResidualHead, confidence_gate


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


class C25PairwiseRankingResidualModel(nn.Module):
    """Frozen C17 predictor with the unchanged C23 local residual architecture."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c25_cfg = dict(config.get("c25", {}))
        reference_config = copy.deepcopy(config)
        reference_config["phase"] = "c17"
        reference_config.pop("c25", None)
        self.frozen_c17 = C17ResidualModel(reference_config, seed)
        checkpoint = _resolve_seed_path(c25_cfg["c17_checkpoint"], seed)
        self.frozen_c17.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        for parameter in self.frozen_c17.parameters():
            parameter.requires_grad = False
        self.frozen_c17.eval()

        self.residual_head = ConfidenceGatedLocalResidualHead(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.15)),
        )
        self.temperature = float(c25_cfg.get("temperature", 1.0))
        self.residual_max = float(c25_cfg.get("residual_max", 0.15))
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C25PairwiseRankingResidualModel":
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
        delta = self.residual_max * gate.detach() * torch.tanh(raw_delta)
        final_logit = frozen_logit + delta
        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "frozen_c17_logit": frozen_logit,
            "frozen_c17_prob": torch.sigmoid(frozen_logit),
            "confidence_gate": gate,
            "mechanism_representation_norm": mechanism_state.norm(dim=-1),
            "raw_delta_c25": raw_delta,
            "delta_c25": delta,
        }


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, int]:
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0, 0
    margins = (positive[:, None] - negative[None, :]) / float(temperature)
    return F.softplus(-margins).mean(), int(margins.numel())


def correct_case_preserve_loss(
    delta: torch.Tensor, frozen_logit: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    positive = labels > 0.5
    correct = (positive & (frozen_logit.detach() >= 0)) | ((~positive) & (frozen_logit.detach() < 0))
    signed_delta = torch.where(positive, delta, -delta)
    if bool(correct.any().item()):
        return F.relu(-signed_delta[correct]).mean()
    return delta.sum() * 0.0


def c25_loss_terms(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Mapping[str, Any]
) -> Dict[str, torch.Tensor | int]:
    labels = batch["label"]
    delta = outputs["delta_c25"]
    rank, pair_count = pairwise_rank_loss(
        outputs["logit"], labels, temperature=float(loss_cfg.get("rank_temperature", 1.0))
    )
    correct_preserve = correct_case_preserve_loss(delta, outputs["frozen_c17_logit"], labels)
    center = delta.mean().square()
    magnitude = delta.square().mean()
    total = (
        rank
        + float(loss_cfg.get("lambda_correct_preserve", 0.02)) * correct_preserve
        + float(loss_cfg.get("lambda_center", 0.01)) * center
        + float(loss_cfg.get("lambda_magnitude", 0.001)) * magnitude
    )
    return {
        "total": total,
        "pairwise_rank": rank,
        "correct_preserve": correct_preserve,
        "center": center,
        "magnitude": magnitude,
        "pair_count": pair_count,
    }
