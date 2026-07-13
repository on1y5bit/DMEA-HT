from __future__ import annotations

from itertools import combinations
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


MODALITIES = ("img", "txt", "bio")


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


class DSSAAlignment(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        shared_dim: int | None = None,
        temperature: float = 0.10,
        anchor_temperature: float = 0.10,
        margin_proto: float = 0.0,
        specific_variance_gamma: float = 0.50,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        shared_dim = int(shared_dim or hidden_dim)
        if residual_scale < 0.0 or residual_scale > 0.15:
            raise ValueError("DSSA residual_scale must be within [0.0, 0.15]")
        if temperature <= 0.0 or anchor_temperature <= 0.0:
            raise ValueError("DSSA temperatures must be positive")

        self.shared_dim = shared_dim
        self.temperature = float(temperature)
        self.anchor_temperature = float(anchor_temperature)
        self.margin_proto = float(margin_proto)
        self.specific_variance_gamma = float(specific_variance_gamma)
        self.residual_scale = float(residual_scale)

        self.shared_projectors = nn.ModuleDict(
            {name: ProjectionHead(hidden_dim, shared_dim, dropout) for name in MODALITIES}
        )
        self.specific_projectors = nn.ModuleDict(
            {name: ProjectionHead(hidden_dim, shared_dim, dropout) for name in MODALITIES}
        )
        self.shared_score = nn.Linear(shared_dim, 1)
        self.specific_residual_projectors = nn.ModuleDict(
            {name: nn.Linear(shared_dim, shared_dim) for name in MODALITIES}
        )
        self.specific_gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(shared_dim * 3, shared_dim),
                    nn.GELU(),
                    nn.Linear(shared_dim, shared_dim),
                )
                for name in MODALITIES
            }
        )
        self.prototypes = nn.Parameter(torch.randn(2, shared_dim) * 0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim * 3 + 1),
            nn.Linear(shared_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_specific_residuals()

    def _initialize_specific_residuals(self) -> None:
        for projector in self.specific_residual_projectors.values():
            nn.init.normal_(projector.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(projector.bias)
        for gate in self.specific_gates.values():
            final = gate[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.constant_(final.bias, -2.0)

    @staticmethod
    def modality_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_available = batch["image_mask"].sum(dim=1) > 0
        text_available = batch["report_attention_mask"].sum(dim=1) > 2
        bio_available = (batch["bio_missing_mask"] < 0.5).any(dim=1)
        return torch.stack([image_available, text_available, bio_available], dim=1)

    def _shared_consistency_loss(self, shared: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        losses = []
        for left, right in combinations(range(len(MODALITIES)), 2):
            pair_available = available[:, left] & available[:, right]
            pair_loss = 1.0 - (shared[:, left] * shared[:, right]).sum(dim=-1)
            if bool(pair_available.any().item()):
                losses.append(pair_loss[pair_available])
        return torch.cat(losses).mean() if losses else shared.sum() * 0.0

    def _orthogonality_loss(
        self,
        shared: torch.Tensor,
        specific: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        specific_normalized = F.normalize(specific, dim=-1, eps=1e-8)
        cosine_squared = ((shared * specific_normalized).sum(dim=-1)) ** 2
        return masked_mean(cosine_squared, available)

    def _specific_variance_loss(self, specific: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        losses = []
        for index in range(len(MODALITIES)):
            selected = specific[available[:, index], index]
            if selected.shape[0] < 2:
                continue
            feature_std = selected.std(dim=0, unbiased=False)
            losses.append(F.relu(self.specific_variance_gamma - feature_std).mean())
        return torch.stack(losses).mean() if losses else specific.sum() * 0.0

    def forward(
        self,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        bio_feature: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        features = {"img": image_feature, "txt": text_feature, "bio": bio_feature}
        available = self.modality_mask(batch)
        available_float = available.unsqueeze(-1).to(image_feature.dtype)

        shared_by_modality = {
            name: F.normalize(self.shared_projectors[name](features[name]), dim=-1, eps=1e-8)
            for name in MODALITIES
        }
        specific_by_modality = {
            name: self.specific_projectors[name](features[name]) for name in MODALITIES
        }
        shared = torch.stack([shared_by_modality[name] for name in MODALITIES], dim=1) * available_float
        specific = torch.stack([specific_by_modality[name] for name in MODALITIES], dim=1) * available_float

        raw_attention = self.shared_score(shared).squeeze(-1).masked_fill(~available, -1e4)
        shared_attention = torch.softmax(raw_attention, dim=1) * available.to(raw_attention.dtype)
        shared_attention = shared_attention / shared_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        patient_shared = (shared_attention.unsqueeze(-1) * shared).sum(dim=1)

        prototypes = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        modality_prototype_logits = torch.einsum("bmh,kh->bmk", shared, prototypes) / self.temperature
        patient_similarities = F.cosine_similarity(
            patient_shared.unsqueeze(1), prototypes.unsqueeze(0), dim=-1, eps=1e-8
        )
        disease_margin = patient_similarities[:, 1] - patient_similarities[:, 0]
        disease_weights = torch.softmax(patient_similarities / self.anchor_temperature, dim=-1)
        disease_anchor = disease_weights @ prototypes

        residual_parts = []
        gate_means = []
        for index, name in enumerate(MODALITIES):
            gate_input = torch.cat([shared[:, index], specific[:, index], disease_anchor], dim=-1)
            gate = torch.sigmoid(self.specific_gates[name](gate_input)) * available_float[:, index]
            residual = self.specific_residual_projectors[name](specific[:, index])
            residual_parts.append(shared_attention[:, index : index + 1] * gate * residual)
            gate_means.append(gate.mean(dim=-1))
        specific_residual = torch.stack(residual_parts, dim=1).sum(dim=1)
        specific_gate = torch.stack(gate_means, dim=1)

        patient_shared_norm = patient_shared.norm(dim=-1)
        specific_residual_norm = specific_residual.norm(dim=-1)
        residual_shared_ratio = (
            self.residual_scale * specific_residual_norm / patient_shared_norm.clamp_min(1e-8)
        )
        final_representation = torch.cat(
            [
                patient_shared,
                disease_anchor,
                self.residual_scale * specific_residual,
                disease_margin.unsqueeze(-1),
            ],
            dim=-1,
        )
        logit = self.classifier(final_representation).squeeze(-1)

        prototype_cosine = (prototypes[0] * prototypes[1]).sum()
        outputs: Dict[str, torch.Tensor] = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "dssa_prototype_logits": modality_prototype_logits,
            "dssa_available_mask": available,
            "dssa_shared_consistency_loss": self._shared_consistency_loss(shared, available),
            "dssa_shared_specific_orth_loss": self._orthogonality_loss(shared, specific, available),
            "dssa_specific_variance_loss": self._specific_variance_loss(specific, available),
            "dssa_prototype_separation_loss": F.relu(prototype_cosine - self.margin_proto),
            "prototype_cosine": prototype_cosine,
            "prototype_distance": 1.0 - prototype_cosine,
            "prototype_similarity_non_ht": patient_similarities[:, 0],
            "prototype_similarity_ht": patient_similarities[:, 1],
            "disease_margin": disease_margin,
            "patient_shared_norm": patient_shared_norm,
            "specific_residual_norm": specific_residual_norm,
            "specific_residual_shared_ratio": residual_shared_ratio,
            "soft_disease_anchor_norm": disease_anchor.norm(dim=-1),
            "dssa_shared_representations": shared,
            "dssa_specific_representations": specific,
        }
        for index, name in enumerate(MODALITIES):
            outputs[f"modality_available_{name}"] = available[:, index].to(logit.dtype)
            outputs[f"shared_attention_{name}"] = shared_attention[:, index]
            outputs[f"specific_gate_{name}"] = specific_gate[:, index]
            outputs[f"shared_{name}_norm"] = shared[:, index].norm(dim=-1)
            outputs[f"specific_{name}_norm"] = specific[:, index].norm(dim=-1)
            outputs[f"shared_specific_cosine_{name}"] = (
                shared[:, index] * F.normalize(specific[:, index], dim=-1, eps=1e-8)
            ).sum(dim=-1)
        for left, right in combinations(range(len(MODALITIES)), 2):
            left_name = MODALITIES[left]
            right_name = MODALITIES[right]
            pair_available = available[:, left] & available[:, right]
            pair_cosine = (shared[:, left] * shared[:, right]).sum(dim=-1)
            outputs[f"shared_pair_available_{left_name}_{right_name}"] = pair_available.to(logit.dtype)
            outputs[f"shared_cosine_{left_name}_{right_name}"] = pair_cosine * pair_available.to(logit.dtype)
        return outputs
