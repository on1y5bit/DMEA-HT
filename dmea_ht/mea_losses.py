from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.sum() * 0.0
    margins = positives.unsqueeze(1) - negatives.unsqueeze(0)
    return F.softplus(-margins).mean()


def state_margin_loss(state_margin: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sign = labels.to(state_margin.dtype) * 2.0 - 1.0
    return F.softplus(-sign * state_margin).mean()


def mea_loss_weights_for_epoch(loss_cfg: Dict[str, Any], epoch: int) -> Dict[str, float]:
    warmup_epochs = int(loss_cfg.get("mea_warmup_epochs", 3))
    ramp_epochs = max(int(loss_cfg.get("mea_ramp_epochs", 5)), 1)
    if epoch <= warmup_epochs:
        scale = 0.0
    else:
        scale = min(max((epoch - warmup_epochs) / ramp_epochs, 0.0), 1.0)
    weights = {"mea_auxiliary_scale": scale}
    for suffix, config_key in (
        ("state", "state_margin_weight"),
        ("mech", "mechanism_alignment_weight"),
        ("role", "role_separation_weight"),
        ("rank", "pairwise_ranking_weight"),
    ):
        weights[f"effective_lambda_{suffix}"] = scale * float(loss_cfg.get(config_key, 0.0))
    return weights
