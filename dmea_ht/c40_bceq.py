from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import (
    ImageMorphologyEvidenceProjector,
    TEXT_MASK_KEYS,
    TextEvidenceRoleProjector,
)
from dmea_ht.models import ImageEncoder, TextEncoder


EVIDENCE_NAMES = ("latest_image", "history_image", "latest_text", "history_text")
HEAD_PREFIXES = (
    "evidence_token_fusions.",
    "bio_state_encoder.",
    "bio_query.",
    "evidence_attention.",
    "patient_readout.",
    "classifier.",
)


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return state


def _prefixed_state(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    selected = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not selected:
        raise KeyError(f"No state found for checkpoint prefix {prefix}")
    return selected


class FrozenC17ImageTextSources(nn.Module):
    """Frozen C17 image/text encoders and pre-propagation evidence projectors."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(int(model_cfg["text_vocab_size"]), hidden_dim, dropout)
        self.image_projector = ImageMorphologyEvidenceProjector(
            hidden_dim, dropout, num_heads=int(model_cfg["mea_num_heads"])
        )
        self.text_projector = TextEvidenceRoleProjector(hidden_dim, dropout)

        checkpoint = Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        state = _checkpoint_state(checkpoint)
        self.image_encoder.load_state_dict(
            _prefixed_state(state, "base_model.image_encoder."), strict=True
        )
        self.text_encoder.load_state_dict(
            _prefixed_state(state, "base_model.text_encoder."), strict=True
        )
        self.image_projector.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment.image."), strict=True
        )
        self.text_projector.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment.text."), strict=True
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenC17ImageTextSources":
        super().train(False)
        return self


class C40BCEQModel(nn.Module):
    """Continuous biochemical query over a fixed patient evidence memory."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        bio_dim = int(model_cfg["bio_dim"])
        self.hidden_dim = hidden_dim
        self.bio_dim = bio_dim
        self.seed = int(seed)
        self.sources = FrozenC17ImageTextSources(config, seed)

        self.evidence_token_fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in EVIDENCE_NAMES
            ]
        )
        self.bio_state_encoder = nn.Sequential(
            nn.LayerNorm(bio_dim * 3),
            nn.Linear(bio_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.bio_query = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.evidence_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=int(model_cfg["mea_num_heads"]),
            dropout=dropout,
            batch_first=True,
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> "C40BCEQModel":
        super().train(mode)
        self.sources.eval()
        return self

    def _frozen_evidence(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        report_ids = batch["report_input_ids"].flatten(0, 1)
        report_mask = batch["report_attention_mask"].flatten(0, 1)
        image_tokens, _ = self.sources.image_encoder(images, image_mask)
        text_tokens, _ = self.sources.text_encoder(report_ids, report_mask)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        image = self.sources.image_projector(image_tokens, image_mask)
        text = self.sources.text_projector(text_tokens, report_mask, text_masks)
        image_nodes = image["nodes"].reshape(batch_size, visits, 5, self.hidden_dim)
        image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
        text_nodes = text["nodes"].reshape(batch_size, visits, 6, self.hidden_dim)
        text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
        image_state = image_nodes.mean(dim=2)
        image_available = image_valid.any(dim=-1)
        text_state = text_nodes.mean(dim=2)
        text_available = text_valid.any(dim=-1)
        visit_mask = batch["visit_mask"].bool()
        return {
            "image_state": image_state,
            "image_valid": image_available & visit_mask,
            "text_state": text_state,
            "text_valid": text_available & visit_mask,
        }

    @staticmethod
    def _partition_weights(
        visit_mask: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
        latest_mask = visit_mask & positions.eq(latest_index)
        history_mask = visit_mask & ~latest_mask
        age = (latest_index - positions).clamp_min(0).to(dtype)
        kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(dtype)
        history_weights = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        latest_weights = latest_mask.to(dtype)
        return latest_mask, history_mask, latest_weights, history_weights

    @staticmethod
    def _pool(
        states: torch.Tensor,
        valid: torch.Tensor,
        bucket_mask: torch.Tensor,
        bucket_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective = bucket_weights * bucket_mask.to(bucket_weights.dtype) * valid.to(bucket_weights.dtype)
        denominator = effective.sum(dim=1, keepdim=True)
        pooled = (states * effective.unsqueeze(-1)).sum(dim=1) / denominator.clamp_min(1e-8)
        return pooled, denominator.squeeze(-1) > 0.0

    @staticmethod
    def _pool_bio(
        values: torch.Tensor,
        missing: torch.Tensor,
        bucket_mask: torch.Tensor,
        bucket_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = (~missing.bool()).to(values.dtype)
        effective = bucket_weights.unsqueeze(-1) * bucket_mask.unsqueeze(-1).to(values.dtype) * observed
        denominator = effective.sum(dim=1)
        pooled = (values * effective).sum(dim=1) / denominator.clamp_min(1e-8)
        return pooled, denominator > 0.0

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            evidence = self._frozen_evidence(batch)
        visit_mask = batch["visit_mask"].bool()
        latest_mask, history_mask, latest_weights, history_weights = self._partition_weights(
            visit_mask, batch["bio_values"].dtype
        )
        latest_image, latest_image_valid = self._pool(
            evidence["image_state"], evidence["image_valid"], latest_mask, latest_weights
        )
        history_image, history_image_valid = self._pool(
            evidence["image_state"], evidence["image_valid"], history_mask, history_weights
        )
        latest_text, latest_text_valid = self._pool(
            evidence["text_state"], evidence["text_valid"], latest_mask, latest_weights
        )
        history_text, history_text_valid = self._pool(
            evidence["text_state"], evidence["text_valid"], history_mask, history_weights
        )
        raw_bio = batch["bio_values"]
        bio_missing = batch["bio_missing_mask"]
        latest_bio, latest_bio_valid = self._pool_bio(raw_bio, bio_missing, latest_mask, latest_weights)
        history_bio, history_bio_valid = self._pool_bio(raw_bio, bio_missing, history_mask, history_weights)
        bio_input = torch.cat([latest_bio, history_bio, latest_bio - history_bio], dim=-1)
        bio_state = self.bio_state_encoder(bio_input)
        bio_state = bio_state * (latest_bio_valid | history_bio_valid).any(dim=-1, keepdim=True).to(bio_state.dtype)

        raw_tokens = torch.stack(
            [latest_image, history_image, latest_text, history_text], dim=1
        )
        raw_valid = torch.stack(
            [latest_image_valid, history_image_valid, latest_text_valid, history_text_valid], dim=1
        )
        evidence_tokens = []
        for index, fusion in enumerate(self.evidence_token_fusions):
            token = fusion(raw_tokens[:, index])
            evidence_tokens.append(token * raw_valid[:, index].unsqueeze(-1).to(token.dtype))
        evidence_tokens_tensor = torch.stack(evidence_tokens, dim=1)
        attention_mask = ~raw_valid
        all_missing = attention_mask.all(dim=1)
        if bool(all_missing.any().item()):
            attention_mask[all_missing, 0] = False
        query = self.bio_query(bio_state).unsqueeze(1)
        attended, attention = self.evidence_attention(
            query,
            evidence_tokens_tensor,
            evidence_tokens_tensor,
            key_padding_mask=attention_mask,
            need_weights=True,
        )
        attended_evidence = attended.squeeze(1)
        patient_input = torch.cat(
            [bio_state, attended_evidence, bio_state * attended_evidence], dim=-1
        )
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "bio_state": bio_state,
            "attended_evidence": attended_evidence,
            "evidence_tokens": evidence_tokens_tensor,
            "evidence_valid": raw_valid,
            "attention": attention.squeeze(1),
            "latest_bio_valid": latest_bio_valid,
            "history_bio_valid": history_bio_valid,
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
