from __future__ import annotations

import torch
import torch.nn.functional as F


def confidence_weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Binary BCE that ignores unknown labels and weights by confidence."""
    labels = labels.to(device=logits.device)
    confidence = confidence.to(device=logits.device, dtype=logits.dtype)
    mask = (labels != -1) & (confidence > 0)
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float(), reduction="none")
    weights = confidence[mask].float()
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)
