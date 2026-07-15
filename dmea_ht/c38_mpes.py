from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import (
    MechanismEvidenceAlignment,
    TEXT_MASK_KEYS,
)
from dmea_ht.models import BioEncoder, ImageEncoder, TextEncoder


MECHANISM_NAMES = ("M1", "M2", "M3", "M4", "M5")
HEAD_PREFIXES = (
    "pool_norm.",
    "pool_queries",
    "empty_tokens",
    "mechanism_fusion.",
    "patient_query",
    "patient_attention.",
    "patient_norm.",
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


class FrozenC17PatientSources(nn.Module):
    """C17 encoders and mechanism evidence representation, used as frozen evidence."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(int(model_cfg["text_vocab_size"]), hidden_dim, dropout)
        self.bio_encoder = BioEncoder(int(model_cfg["bio_dim"]), hidden_dim, dropout)
        self.mechanism_evidence_alignment = MechanismEvidenceAlignment(
            hidden_dim,
            dropout,
            num_heads=int(model_cfg["mea_num_heads"]),
        )

        checkpoint = Path(
            str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed))
        )
        state = _checkpoint_state(checkpoint)
        self.image_encoder.load_state_dict(
            _prefixed_state(state, "base_model.image_encoder."), strict=True
        )
        self.text_encoder.load_state_dict(
            _prefixed_state(state, "base_model.text_encoder."), strict=True
        )
        self.bio_encoder.load_state_dict(
            _prefixed_state(state, "base_model.bio_encoder."), strict=True
        )
        self.mechanism_evidence_alignment.load_state_dict(
            _prefixed_state(state, "mechanism_evidence_alignment."), strict=True
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenC17PatientSources":
        super().train(False)
        return self


class C38MPESModel(nn.Module):
    """Patient-level mechanism-conditioned evidence-set model.

    Visits are used only to form a latest evidence bucket and a historical
    evidence set. The model never learns a visit score or consumes visit-count
    or raw structure fields.
    """

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.hidden_dim = hidden_dim
        self.seed = int(seed)
        self.sources = FrozenC17PatientSources(config, seed)

        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.pool_queries = nn.Parameter(torch.randn(2, len(MECHANISM_NAMES), hidden_dim) * 0.02)
        self.empty_tokens = nn.Parameter(torch.randn(2, len(MECHANISM_NAMES), hidden_dim) * 0.02)
        self.mechanism_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.patient_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=int(model_cfg["mea_num_heads"]),
            dropout=dropout,
            batch_first=True,
        )
        self.patient_norm = nn.LayerNorm(hidden_dim)
        classifier_input_dim = hidden_dim * (len(MECHANISM_NAMES) + 1) + len(MECHANISM_NAMES) * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> "C38MPESModel":
        super().train(mode)
        self.sources.eval()
        return self

    def _source_evidence(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, visits = batch["visit_mask"].shape
        image_mask = batch["image_mask"].flatten(0, 1)
        image_tokens, _ = self.sources.image_encoder(
            batch["images"].flatten(0, 1), image_mask
        )
        text_attention_mask = batch["report_attention_mask"].flatten(0, 1)
        text_tokens, _ = self.sources.text_encoder(
            batch["report_input_ids"].flatten(0, 1), text_attention_mask
        )
        bio_tokens, _, _, _ = self.sources.bio_encoder(
            batch["bio_values"].flatten(0, 1),
            batch["bio_missing_mask"].flatten(0, 1),
            batch["bio_abnormal_flags"].flatten(0, 1),
        )
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
        evidence = self.sources.mechanism_evidence_alignment(
            image_tokens=image_tokens,
            image_mask=image_mask,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            bio_tokens=bio_tokens,
            bio_missing_mask=batch["bio_missing_mask"].flatten(0, 1),
            text_masks=text_masks,
        )
        nodes = evidence["mea_mechanism_nodes"].reshape(
            batch_size, visits, len(MECHANISM_NAMES), self.hidden_dim
        )
        valid = evidence["mea_mechanism_valid"].reshape(
            batch_size, visits, len(MECHANISM_NAMES)
        ).bool()
        visit_mask = batch["visit_mask"].bool()
        return {
            "nodes": nodes,
            "valid": valid & visit_mask.unsqueeze(-1),
            "visit_mask": visit_mask,
        }

    def _pool_bucket(
        self,
        nodes: torch.Tensor,
        valid: torch.Tensor,
        bucket_mask: torch.Tensor,
        bucket_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bucket_valid = valid & bucket_mask.unsqueeze(-1)
        has_evidence = bucket_valid.any(dim=1)
        normalized_nodes = self.pool_norm(nodes)
        query = self.pool_queries[bucket_index]
        scores = torch.einsum("bvkh,kh->bvk", normalized_nodes, query)
        safe_valid = bucket_valid.clone()
        if safe_valid.shape[1] == 0:
            raise RuntimeError("C38 requires at least one visit slot")
        fallback_positions = torch.zeros_like(safe_valid)
        fallback_positions[:, 0, :] = ~has_evidence
        safe_valid = safe_valid | fallback_positions
        masked_scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=1) * safe_valid.to(scores.dtype)
        pooled = torch.einsum("bvk,bvkh->bkh", weights, nodes)
        fallback = self.empty_tokens[bucket_index].unsqueeze(0)
        pooled = torch.where(has_evidence.unsqueeze(-1), pooled, fallback)
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=1)
        return pooled, has_evidence, weights, entropy

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_evidence(batch)
        visit_mask = source["visit_mask"]
        counts = visit_mask.sum(dim=1)
        positions = torch.arange(
            visit_mask.shape[1], device=visit_mask.device, dtype=counts.dtype
        ).unsqueeze(0)
        latest_mask = visit_mask & positions.eq((counts.unsqueeze(-1) - 1).clamp_min(0))
        history_mask = visit_mask & ~latest_mask

        latest, latest_valid, latest_weights, latest_entropy = self._pool_bucket(
            source["nodes"], source["valid"], latest_mask, 0
        )
        history, history_valid, history_weights, history_entropy = self._pool_bucket(
            source["nodes"], source["valid"], history_mask, 1
        )
        pair_features = torch.cat(
            [latest, history, torch.abs(latest - history), latest * history], dim=-1
        )
        mechanism_states = self.mechanism_fusion(pair_features)
        no_mechanism_evidence = ~(latest_valid | history_valid)
        attention_mask = no_mechanism_evidence.clone()
        all_missing = attention_mask.all(dim=1)
        if bool(all_missing.any().item()):
            attention_mask[all_missing, 0] = False
        query = self.patient_query.expand(mechanism_states.shape[0], -1, -1)
        attended, patient_attention = self.patient_attention(
            query,
            mechanism_states,
            mechanism_states,
            key_padding_mask=attention_mask,
            need_weights=True,
        )
        patient_state = self.patient_norm(attended.squeeze(1))
        validity = torch.cat(
            [latest_valid.to(patient_state.dtype), history_valid.to(patient_state.dtype)], dim=-1
        )
        classifier_input = torch.cat(
            [patient_state, mechanism_states.flatten(start_dim=1), validity], dim=-1
        )
        logit = self.classifier(classifier_input).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "mechanism_states": mechanism_states,
            "source_valid": source["valid"],
            "latest_valid": latest_valid,
            "history_valid": history_valid,
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_entropy": latest_entropy,
            "history_entropy": history_entropy,
            "patient_attention": patient_attention.squeeze(1),
            "latest_mask": latest_mask,
            "history_mask": history_mask,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
