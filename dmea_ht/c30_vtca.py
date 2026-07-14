from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, masked_mean
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise TypeError(f"Unsupported C27 checkpoint payload: {path}")
    return payload


class VisitTextContextAdapter(nn.Module):
    """Shared masked local-context correction for one visit at a time."""

    def __init__(self, hidden_dim: int, dropout: float, max_delta: float = 0.10) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.depthwise_k3 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
        )
        self.depthwise_k7 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False
        )
        self.pointwise = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.max_delta = float(max_delta)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, text_tokens: torch.Tensor, text_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if text_tokens.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("VTCA expects token states [N,L,D] and a validity mask [N,L]")
        mask = text_attention_mask.to(text_tokens.dtype).unsqueeze(-1)
        normalized = self.input_norm(text_tokens) * mask
        channels = normalized.transpose(1, 2)
        channel_mask = mask.transpose(1, 2)
        c3 = self.depthwise_k3(channels) * channel_mask
        c7 = self.depthwise_k7(channels) * channel_mask
        fused = self.pointwise(torch.cat([c3, c7], dim=1)) * channel_mask
        raw_delta = self.output(self.dropout(self.activation(fused.transpose(1, 2))))
        token_delta = self.max_delta * torch.tanh(raw_delta) * mask
        return {
            "adapted_tokens": text_tokens + token_delta,
            "token_delta": token_delta,
            "raw_token_delta": raw_delta,
        }


class C30VTCAModel(nn.Module):
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c30"])
        self.c27 = C27VTMEModel(config, seed)
        checkpoint_path = Path(str(phase_cfg["c27_checkpoint"]).replace("{seed}", str(seed)))
        payload = load_checkpoint(checkpoint_path)
        if int(payload.get("seed", -1)) != int(seed):
            raise RuntimeError(f"C30 C27 checkpoint seed mismatch for seed {seed}")
        self.c27.load_state_dict(payload["model"], strict=True)
        for parameter in self.c27.parameters():
            parameter.requires_grad_(False)
        self.c27.eval()
        self.adapter = VisitTextContextAdapter(
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            max_delta=float(phase_cfg["adapter_max_delta"]),
        )
        self.seed = int(seed)
        self.c27_checkpoint = str(checkpoint_path)

    def train(self, mode: bool = True) -> "C30VTCAModel":
        super().train(mode)
        self.c27.eval()
        self.adapter.train(mode)
        return self

    @staticmethod
    def _patient_adapter_diagnostics(
        text_tokens: torch.Tensor,
        adapted_tokens: torch.Tensor,
        token_delta: torch.Tensor,
        attention_mask: torch.Tensor,
        visit_mask: torch.Tensor,
        text_visit_valid: torch.Tensor,
        text_nodes_before: torch.Tensor,
        text_nodes_after: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, visits = visit_mask.shape
        length, hidden = attention_mask.shape[-1], text_tokens.shape[-1]
        mask = attention_mask.bool().view(batch, visits, length)
        element_mask = mask.unsqueeze(-1).to(token_delta.dtype)
        delta = token_delta.view(batch, visits, length, hidden)
        before = text_tokens.view(batch, visits, length, hidden)
        after = adapted_tokens.view(batch, visits, length, hidden)
        valid_elements = element_mask.sum(dim=(1, 2, 3)) * hidden
        absolute = delta.abs()
        mean = (absolute * element_mask).sum(dim=(1, 2, 3)) / valid_elements.clamp_min(1.0)
        second = ((absolute * absolute) * element_mask).sum(dim=(1, 2, 3)) / valid_elements.clamp_min(1.0)
        std = (second - mean * mean).clamp_min(0.0).sqrt()
        near_bound = ((absolute >= 0.095).to(absolute.dtype) * element_mask).sum(
            dim=(1, 2, 3)
        ) / valid_elements.clamp_min(1.0)
        real_max = (absolute * element_mask).amax(dim=(1, 2, 3))
        padding_mask = (~mask).unsqueeze(-1).to(token_delta.dtype)
        padding_max = (absolute * padding_mask).amax(dim=(1, 2, 3))

        visit_denominator = mask.sum(dim=-1).to(token_delta.dtype) * hidden
        visit_delta_mean = (absolute * element_mask).sum(dim=(2, 3)) / visit_denominator.clamp_min(1.0)
        token_cosine = F.cosine_similarity(before, after, dim=-1)
        visit_token_cosine = (token_cosine * mask.to(token_cosine.dtype)).sum(dim=-1) / mask.sum(
            dim=-1
        ).clamp_min(1)
        before_norm = before.norm(dim=-1)
        after_norm = after.norm(dim=-1)
        visit_before_norm = (before_norm * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        visit_after_norm = (after_norm * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        evidence_cosine = F.cosine_similarity(
            text_nodes_before.flatten(start_dim=2), text_nodes_after.flatten(start_dim=2), dim=-1
        )
        valid_visits = (visit_mask & text_visit_valid).to(token_delta.dtype)
        text_visit_count = valid_visits.sum(dim=1)
        valid_visit_count = text_visit_count.clamp_min(1.0)
        patient_token_cosine = (visit_token_cosine * valid_visits).sum(dim=1) / valid_visit_count
        patient_before_norm = (visit_before_norm * valid_visits).sum(dim=1) / valid_visit_count
        patient_after_norm = (visit_after_norm * valid_visits).sum(dim=1) / valid_visit_count
        patient_evidence_cosine = (evidence_cosine * valid_visits).sum(dim=1) / valid_visit_count
        no_text = text_visit_count == 0
        patient_token_cosine = torch.where(no_text, torch.ones_like(patient_token_cosine), patient_token_cosine)
        patient_evidence_cosine = torch.where(
            no_text, torch.ones_like(patient_evidence_cosine), patient_evidence_cosine
        )

        counts = visit_mask.sum(dim=1)
        latest_index = (counts - 1).clamp_min(0)
        row_index = torch.arange(batch, device=visit_mask.device)
        latest_delta = visit_delta_mean[row_index, latest_index]
        history_mask = visit_mask.clone()
        history_mask[row_index, latest_index] = False
        history_count = history_mask.sum(dim=1)
        history_delta = (visit_delta_mean * history_mask.to(visit_delta_mean.dtype)).sum(dim=1) / history_count.clamp_min(
            1
        ).to(visit_delta_mean.dtype)
        history_delta = torch.where(history_count > 0, history_delta, torch.zeros_like(history_delta))
        return {
            "adapter_delta_abs_mean": mean,
            "adapter_delta_abs_std": std,
            "adapter_delta_abs_max": real_max,
            "adapter_near_bound_fraction": near_bound,
            "padding_delta_abs_max": padding_max,
            "latest_visit_adapter_delta_abs": latest_delta,
            "history_visit_adapter_delta_abs": history_delta,
            "text_token_norm_before_mean": patient_before_norm,
            "text_token_norm_after_mean": patient_after_norm,
            "text_token_cosine_before_after": patient_token_cosine,
            "text_evidence_state_cosine_before_after": patient_evidence_cosine,
        }

    def _adapted_visit_sources(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            image = self.c27.frozen_sources.image_projector(image_tokens, image_mask)
            bio = self.c27.frozen_sources.bio_projector(bio_tokens, bio_missing)
            text_before = self.c27.frozen_sources.text_projector(
                text_tokens, attention_mask, text_masks
            )
        adapted = self.adapter(text_tokens.detach(), attention_mask)
        text_after = self.c27.frozen_sources.text_projector(
            adapted["adapted_tokens"], attention_mask, text_masks
        )

        image_available = image["valid"].any(dim=-1)
        image_morphology = masked_mean(image["nodes"], image["valid"], dim=1)
        text_available = batch["visit_text_valid"].flatten(0, 1)
        text_morphology = text_after["nodes"][:, (0, 3)].mean(dim=1)
        m1_sources = torch.stack([image_morphology, text_morphology], dim=1)
        m1_valid = torch.stack([image_available, text_available], dim=1)
        m1 = masked_mean(m1_sources, m1_valid, dim=1)
        m2 = bio["nodes"][:, 1]
        m3 = bio["nodes"][:, 2]
        m4 = text_after["nodes"][:, 1]
        m5 = text_after["nodes"][:, (2, 4)].mean(dim=1)
        source_states = torch.stack([m1, m2, m3, m4, m5], dim=1)
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
                fallback_values, fallback_missing, torch.zeros_like(fallback_values)
            )
            fallback_global = fallback_global * batch["fallback_bio_valid"].unsqueeze(-1).to(
                fallback_global.dtype
            )
        diagnostics = self._patient_adapter_diagnostics(
            text_tokens,
            adapted["adapted_tokens"],
            adapted["token_delta"],
            attention_mask,
            batch["visit_mask"],
            batch["visit_text_valid"],
            text_before["nodes"].view(batch_size, visits, text_before["nodes"].shape[1], -1),
            text_after["nodes"].view(batch_size, visits, text_after["nodes"].shape[1], -1),
        )
        return {
            "source_states": source_states.view(batch_size, visits, len(MECHANISM_NAMES), -1),
            "source_valid": source_valid.view(batch_size, visits, len(MECHANISM_NAMES)),
            "fallback_bio_context": fallback_global,
            "morphology_alignment_valid": (image_available & text_available).view(batch_size, visits),
            "image_morphology": image_morphology.view(batch_size, visits, -1),
            "text_morphology": text_morphology.view(batch_size, visits, -1),
            **diagnostics,
        }

    def forward(
        self, batch: Dict[str, torch.Tensor], use_vtca: bool = True
    ) -> Dict[str, torch.Tensor]:
        if not use_vtca:
            with torch.no_grad():
                return self.c27(batch)
        sources = self._adapted_visit_sources(batch)
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
        for key in (
            "adapter_delta_abs_mean",
            "adapter_delta_abs_std",
            "adapter_delta_abs_max",
            "adapter_near_bound_fraction",
            "padding_delta_abs_max",
            "latest_visit_adapter_delta_abs",
            "history_visit_adapter_delta_abs",
            "text_token_norm_before_mean",
            "text_token_norm_after_mean",
            "text_token_cosine_before_after",
            "text_evidence_state_cosine_before_after",
        ):
            outputs[key] = sources[key]
        return outputs


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
