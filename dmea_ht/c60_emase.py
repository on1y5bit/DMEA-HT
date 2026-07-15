from __future__ import annotations

from typing import Any, Dict

from torch import nn

from dmea_ht.c59_pmese import C59PMESEModel, HEAD_PREFIXES


class C60EMASEModel(C59PMESEModel):
    """C59 patient evidence-set model trained and selected with one EMA state."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        translated = dict(config)
        translated["c59"] = dict(config["c60"])
        super().__init__(translated, seed)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
