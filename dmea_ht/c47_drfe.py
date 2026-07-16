from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c41_melr import FrozenC17ModalitySources, _masked_mean
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


HEAD_PREFIXES = (
    "stream_encoder.",
    "patient_readout.",
    "classifier.",
)


class C47DRFEModel(nn.Module):
    """Shared fixed-state readout over raw and aligned C17 evidence streams."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        c47_cfg = dict(config["c47"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.hidden_dim = hidden_dim
        self.seed = int(seed)
        self.end_to_end = bool(config.get("end_to_end", False))
        self.sources = FrozenC17ModalitySources(config, seed, trainable=self.end_to_end)
        self.stream_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        expected_order = ["raw_image", "raw_text", "raw_bio", "aligned_image", "aligned_text", "aligned_bio"]
        if list(c47_cfg["stream_order"]) != expected_order:
            raise RuntimeError("C47 stream order is fixed to raw then aligned image/text/bio")

    def train(self, mode: bool = True) -> "C47DRFEModel":
        nn.Module.train(self, mode)
        self.sources.train(mode if self.end_to_end else False)
        return self

    def _source_states(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        report_ids = batch["report_input_ids"].flatten(0, 1)
        report_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        image_tokens, image_global = self.sources.image_encoder(images, image_mask)
        text_tokens, text_global = self.sources.text_encoder(report_ids, report_mask)
        bio_tokens, bio_global, _, _ = self.sources.bio_encoder(bio_values, bio_missing, bio_abnormal)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        image = self.sources.image_projector(image_tokens, image_mask)
        text = self.sources.text_projector(text_tokens, report_mask, text_masks)
        bio = self.sources.bio_projector(bio_tokens, bio_missing)
        image_nodes = image["nodes"].reshape(batch_size, visits, 5, self.hidden_dim)
        text_nodes = text["nodes"].reshape(batch_size, visits, 6, self.hidden_dim)
        bio_nodes = bio["nodes"].reshape(batch_size, visits, 3, self.hidden_dim)
        image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
        text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
        bio_valid = bio["valid"].reshape(batch_size, visits, 3).bool()
        image_state, image_available = _masked_mean(image_nodes, image_valid, dim=2)
        text_state, text_available = _masked_mean(text_nodes, text_valid, dim=2)
        bio_state, bio_available = _masked_mean(bio_nodes, bio_valid, dim=2)
        image_global = image_global.view(batch_size, visits, self.hidden_dim)
        text_global = text_global.view(batch_size, visits, self.hidden_dim)
        bio_global = bio_global.view(batch_size, visits, self.hidden_dim)
        visit_mask = batch["visit_mask"].bool()
        raw_valid = torch.stack(
            [image_mask.any(dim=1), report_mask.any(dim=1), (~bio_missing.bool()).any(dim=1)], dim=1
        ).view(batch_size, visits, 3)
        aligned_valid = torch.stack([image_available, text_available, bio_available], dim=2)
        states = torch.stack(
            [image_global, text_global, bio_global, image_state, text_state, bio_state], dim=2
        )
        valid = torch.cat([raw_valid, aligned_valid], dim=2) & visit_mask.unsqueeze(-1)
        return {"states": states, "valid": valid, "visit_mask": visit_mask}

    @staticmethod
    def _fixed_trajectory_statistics(
        states: torch.Tensor,
        valid: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(states.dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(states.dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(states.dtype)
        latest_effective = latest_weights.unsqueeze(-1) * valid.to(states.dtype)
        latest_denominator = latest_effective.sum(dim=1)
        latest = (states * latest_effective.unsqueeze(-1)).sum(dim=1)
        latest = latest / latest_denominator.clamp_min(1.0).unsqueeze(-1)
        latest_valid = latest_denominator > 0.0
        history_effective = history_weights.unsqueeze(-1) * valid.to(states.dtype)
        history_denominator = history_effective.sum(dim=1)
        history = (states * history_effective.unsqueeze(-1)).sum(dim=1)
        history = history / history_denominator.clamp_min(1.0).unsqueeze(-1)
        history_valid = history_denominator > 0.0
        delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(states.dtype)
        centered = (states - history.unsqueeze(1)).pow(2)
        variance = (centered * history_effective.unsqueeze(-1)).sum(dim=1)
        variance = variance / history_denominator.clamp_min(1.0).unsqueeze(-1)
        dispersion = variance.clamp_min(1e-8).sqrt()
        dispersion = dispersion * history_valid.unsqueeze(-1).to(states.dtype)
        summary = torch.cat([latest, history, delta, dispersion], dim=-1)
        available = latest_valid | history_valid
        return summary, available, latest_mask, history_mask, latest_weights, history_weights

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        stream_tokens = self.stream_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(stream_tokens.dtype)
        stream_tokens = stream_tokens * weights.unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        consensus = (stream_tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        discordance = (
            (stream_tokens - consensus.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        patient_input = torch.cat([stream_tokens.flatten(start_dim=1), consensus, discordance], dim=-1)
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack(
            [consensus, discordance, consensus * discordance, stream_tokens.mean(dim=1)], dim=1
        )
        evidence_valid = available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(stream_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = source["states"][:, :, 5]
        bio_valid = source["valid"][:, :, 5]
        bio_weights = bio_valid.to(bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = (bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": consensus,
            "bio_state": bio_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": available,
            "stream_tokens": stream_tokens,
            "stream_valid": available,
            "consensus_state": consensus,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
