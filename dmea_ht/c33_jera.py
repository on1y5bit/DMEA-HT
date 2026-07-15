from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import nn

from dmea_ht.c32_vpa import (
    C32VPAModel,
    PROJECTOR_DESTINATIONS,
    PROJECTOR_MODULES,
)


READOUT_MODULES = {
    "patient_projection": "c27.core.patient_projection",
    "classifier": "c27.core.classifier",
}

TRAINABLE_MODULES = {**PROJECTOR_MODULES, **READOUT_MODULES}


def parameter_category(name: str) -> str | None:
    for category, prefix in TRAINABLE_MODULES.items():
        if name.startswith(f"{prefix}."):
            return category
    return None


class C33JERAModel(C32VPAModel):
    """C27 with joint adaptation of existing evidence and patient readout modules."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        for module in (self.c27.core.patient_projection, self.c27.core.classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self._initial_trainable_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if self.is_trainable_parameter(name)
        }
        self.train(False)

    @staticmethod
    def is_trainable_parameter(name: str) -> bool:
        return parameter_category(name) is not None

    def train(self, mode: bool = True) -> "C33JERAModel":
        super().train(mode)
        self.c27.core.patient_projection.train(mode)
        self.c27.core.classifier.train(mode)
        return self

    def parameter_drift_rows(self) -> list[Dict[str, Any]]:
        current = dict(self.named_parameters())
        rows: list[Dict[str, Any]] = []
        for name, baseline in self._initial_trainable_state.items():
            category = parameter_category(name)
            if category is None:
                raise RuntimeError(f"Unknown C33 trainable parameter: {name}")
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
                    "destination_evidence_role": (
                        PROJECTOR_DESTINATIONS[category]
                        if category in PROJECTOR_DESTINATIONS
                        else "patient-level projection and final classification readout"
                    ),
                    "relative_parameter_drift": relative,
                    "finite": bool(np.isfinite(relative)),
                }
            )
        return rows


def named_trainable_parameters(
    model: C33JERAModel,
) -> Iterable[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def parameter_audit(model: C33JERAModel) -> list[Dict[str, Any]]:
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
                "destination_evidence_role": (
                    PROJECTOR_DESTINATIONS[category]
                    if category in PROJECTOR_DESTINATIONS
                    else (
                        "patient-level projection and final classification readout"
                        if category
                        else "frozen"
                    )
                ),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows
