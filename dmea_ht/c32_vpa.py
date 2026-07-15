from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from dmea_ht.c27_vtme import (
    C27VTMEModel,
    MECHANISM_NAMES,
    masked_mean,
)
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


PROJECTOR_MODULES = {
    "image": "c27.frozen_sources.image_projector",
    "text": "c27.frozen_sources.text_projector",
    "bio": "c27.frozen_sources.bio_projector",
}

PROJECTOR_DESTINATIONS = {
    "image": "image encoder output -> image morphology evidence",
    "text": "text encoder output -> support/nonspecific/opposition/temporal evidence",
    "bio": "bio encoder tokens -> immune/function/other evidence",
}


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported C32 checkpoint payload: {path}")
    return payload


class C32VPAModel(nn.Module):
    """C27 with only its existing pre-propagation evidence projectors adapted."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        self.seed = int(seed)
        self.c27 = C27VTMEModel(config, seed)
        phase_key = getattr(self, "checkpoint_config_key", "c32")
        checkpoint_path = Path(
            str(config[phase_key]["c27_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = load_checkpoint(checkpoint_path)
        if int(payload.get("seed", -1)) != self.seed:
            raise RuntimeError(f"C32 C27 checkpoint seed mismatch for seed {seed}")
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise RuntimeError(f"C32 C27 checkpoint has no model state: {checkpoint_path}")
        self.c27.load_state_dict(state, strict=True)
        for parameter in self.c27.parameters():
            parameter.requires_grad_(False)
        for module in self.projector_modules().values():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.c27_checkpoint = str(checkpoint_path)
        self._initial_projector_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.named_parameters()
            if self.is_projector_parameter(name)
        }
        self.train(False)

    @staticmethod
    def is_projector_parameter(name: str) -> bool:
        return any(name.startswith(f"{prefix}.") for prefix in PROJECTOR_MODULES.values())

    def projector_modules(self) -> Dict[str, nn.Module]:
        sources = self.c27.frozen_sources
        return {
            "image": sources.image_projector,
            "text": sources.text_projector,
            "bio": sources.bio_projector,
        }

    def train(self, mode: bool = True) -> "C32VPAModel":
        super().train(mode)
        self.c27.eval()
        for module in self.projector_modules().values():
            module.train(mode)
        return self

    def _projected_visit_sources(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
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
            image_tokens, _ = self.c27.frozen_sources.image_encoder(images, image_mask)
            text_tokens, _ = self.c27.frozen_sources.text_encoder(input_ids, attention_mask)
            bio_tokens, _, _, _ = self.c27.frozen_sources.bio_encoder(
                bio_values, bio_missing, bio_abnormal
            )
        image = self.c27.frozen_sources.image_projector(image_tokens.detach(), image_mask)
        text = self.c27.frozen_sources.text_projector(
            text_tokens.detach(), attention_mask, text_masks
        )
        bio = self.c27.frozen_sources.bio_projector(bio_tokens.detach(), bio_missing)

        image_available = image["valid"].any(dim=-1)
        image_morphology = masked_mean(image["nodes"], image["valid"], dim=1)
        text_available = batch["visit_text_valid"].flatten(0, 1)
        text_morphology = text["nodes"][:, (0, 3)].mean(dim=1)
        m1_valid = torch.stack([image_available, text_available], dim=1)
        m1 = masked_mean(
            torch.stack([image_morphology, text_morphology], dim=1), m1_valid, dim=1
        )
        source_states = torch.stack(
            [
                m1,
                bio["nodes"][:, 1],
                bio["nodes"][:, 2],
                text["nodes"][:, 1],
                text["nodes"][:, (2, 4)].mean(dim=1),
            ],
            dim=1,
        )
        source_valid = torch.stack(
            [
                m1_valid.any(dim=1),
                bio["valid"][:, 1],
                bio["valid"][:, 2],
                text_available,
                text_available,
            ],
            dim=1,
        )
        source_states = source_states * source_valid.unsqueeze(-1).to(source_states.dtype)

        with torch.no_grad():
            fallback_values = batch["fallback_bio_values"]
            fallback_missing = batch["fallback_bio_missing_mask"]
            _, fallback_global, _, _ = self.c27.frozen_sources.bio_encoder(
                fallback_values,
                fallback_missing,
                torch.zeros_like(fallback_values),
            )
            fallback_global = fallback_global * batch["fallback_bio_valid"].unsqueeze(
                -1
            ).to(fallback_global.dtype)
        return {
            "source_states": source_states.view(
                batch_size, visits, len(MECHANISM_NAMES), -1
            ),
            "source_valid": source_valid.view(
                batch_size, visits, len(MECHANISM_NAMES)
            ),
            "fallback_bio_context": fallback_global,
            "morphology_alignment_valid": (image_available & text_available).view(
                batch_size, visits
            ),
            "image_morphology": image_morphology.view(batch_size, visits, -1),
            "text_morphology": text_morphology.view(batch_size, visits, -1),
            "projectors": {"image": image, "text": text, "bio": bio},
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sources = self._projected_visit_sources(batch)
        outputs = self.c27.core(
            sources["source_states"],
            sources["source_valid"],
            batch["visit_mask"],
            sources["fallback_bio_context"],
        )
        outputs.update(
            self.c27._alignment_summaries(
                sources["image_morphology"],
                sources["text_morphology"],
                sources["morphology_alignment_valid"],
                batch["visit_mask"],
            )
        )
        outputs["mechanism_source_valid"] = sources["source_valid"]
        return outputs

    @staticmethod
    def _patient_projector_mean(
        nodes: torch.Tensor,
        valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, visits = visit_mask.shape
        node_view = nodes.view(batch_size, visits, nodes.shape[1], nodes.shape[2])
        valid_view = valid.view(batch_size, visits, valid.shape[1])
        valid_view = valid_view & visit_mask.unsqueeze(-1)
        weights = valid_view.to(node_view.dtype).unsqueeze(-1)
        mean = (node_view * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
        return mean, valid_view.any(dim=(1, 2))

    def projector_patient_summaries(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        projected = self._projected_visit_sources(batch)["projectors"]
        result: Dict[str, torch.Tensor] = {}
        for modality, values in projected.items():
            state, valid = self._patient_projector_mean(
                values["nodes"], values["valid"], batch["visit_mask"]
            )
            result[f"{modality}_state"] = state
            result[f"{modality}_valid"] = valid
        return result

    def projector_drift_rows(self) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        current = dict(self.named_parameters())
        for name, baseline in self._initial_projector_state.items():
            modality = next(
                key for key, prefix in PROJECTOR_MODULES.items() if name.startswith(prefix)
            )
            value = current[name].detach().cpu()
            denominator = max(float(torch.linalg.vector_norm(baseline)), 1e-8)
            relative = float(torch.linalg.vector_norm(value - baseline)) / denominator
            rows.append(
                {
                    "seed": self.seed,
                    "modality": modality,
                    "module_name": PROJECTOR_MODULES[modality],
                    "parameter_name": name,
                    "parameter_count": int(value.numel()),
                    "destination_evidence_role": PROJECTOR_DESTINATIONS[modality],
                    "relative_parameter_drift": relative,
                    "finite": bool(np.isfinite(relative)),
                }
            )
        return rows


def projector_parameter_audit(model: C32VPAModel) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        modality = next(
            (
                key
                for key, prefix in PROJECTOR_MODULES.items()
                if name.startswith(f"{prefix}.")
            ),
            None,
        )
        rows.append(
            {
                "seed": model.seed,
                "module_name": PROJECTOR_MODULES[modality] if modality else name.rsplit(".", 1)[0],
                "parameter_name": name,
                "parameter_count": int(parameter.numel()),
                "destination_evidence_role": (
                    PROJECTOR_DESTINATIONS[modality] if modality else "frozen"
                ),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows


def named_trainable_parameters(model: C32VPAModel) -> Iterable[tuple[str, nn.Parameter]]:
    return (
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
