#!/usr/bin/env python3
"""Shared read-only C21-A graph tracing and inference-ablation helpers."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c17_residual import C17ResidualModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402


SEEDS = (0, 42, 3407)
NODE_NAMES = ("M1_morphology", "M2_immune", "M3_function", "M4_opposition", "M5_temporal")
RELATION_EDGES = (
    "image_morphology_to_M1_morphology",
    "text_morphology_to_M1_morphology",
    "bio_immune_to_M2_immune",
    "bio_function_to_M3_function",
    "text_opposition_to_M4_opposition",
    "text_temporal_to_M5_temporal",
    "text_global_to_final_mechanism",
    "bio_other_to_final_mechanism",
)
ATTENTION_EDGES = tuple(f"{node}_to_final_mechanism" for node in NODE_NAMES)
ALL_EDGES = RELATION_EDGES[:6] + ATTENTION_EDGES + RELATION_EDGES[6:]
RELATION_MODULES = {
    "image_morphology_to_M1_morphology": "image_morphology",
    "text_morphology_to_M1_morphology": "text_morphology",
    "bio_immune_to_M2_immune": "bio_immune",
    "bio_function_to_M3_function": "bio_function",
    "text_opposition_to_M4_opposition": "text_opposition",
    "text_temporal_to_M5_temporal": "text_temporal",
    "text_global_to_final_mechanism": "text_global",
    "bio_other_to_final_mechanism": "bio_other",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"unsupported checkpoint payload: {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def flatten_tensor(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 1:
        array = array[:, None]
    return array.reshape(array.shape[0], -1).astype(np.float32, copy=False)


def masked_mean(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(nodes.dtype).unsqueeze(-1)
    return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def load_rows(config: Dict[str, Any], manifest: str | None = None, data_root: str | None = None) -> List[Dict[str, Any]]:
    if manifest:
        config["project"]["manifest"] = manifest
    if data_root:
        config["project"]["data_root"] = data_root
    rows = read_manifest(config["project"]["manifest"])
    if not all(str(row.get("split", "")).strip() for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split
    return rows


def build_loader(config: Mapping[str, Any], rows: List[Dict[str, Any]], split: str, batch_size: int, num_workers: int) -> DataLoader:
    project = config["project"]
    model_cfg = config["model"]
    dataset = PatientHTDataset(
        rows=rows,
        data_root=project["data_root"],
        split=split,
        max_images=int(model_cfg.get("max_images_per_patient", 4)),
        image_size=int(model_cfg.get("image_size", 224)),
        text_max_length=int(model_cfg.get("text_max_length", 256)),
        text_vocab_size=int(model_cfg.get("text_vocab_size", 50000)),
        bio_dim=int(model_cfg.get("bio_dim", 32)),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_patient_batch,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(config: Dict[str, Any], run_dir: Path, seed: int, device: torch.device) -> C17ResidualModel:
    model = C17ResidualModel(config, seed).to(device)
    checkpoint = run_dir / "checkpoints" / f"seed_{seed}_best.pt"
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.eval()
    return model


def _edge_record(
    source: torch.Tensor,
    transformed: torch.Tensor,
    gate: torch.Tensor | None,
    effective: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor | None]:
    raw_message = transformed
    return {
        "source_representation": source,
        "transformed_source": transformed,
        "raw_message": raw_message,
        "message_norm": raw_message.norm(dim=-1, keepdim=True),
        "edge_weight": weight,
        "edge_gate": gate,
        "effective_message": effective,
    }


def _apply_zero(edge: str, value: torch.Tensor, zero_edges: set[str]) -> torch.Tensor:
    return torch.zeros_like(value) if edge in zero_edges else value


def _aggregate_without_modulation(
    nodes: torch.Tensor,
    valid: torch.Tensor,
    role_probs: torch.Tensor,
    modality_slices: Sequence[slice],
) -> Dict[str, torch.Tensor]:
    """Inference-only intervention: remove learned reliability/conflict modulation."""
    reliability = valid.to(nodes.dtype)
    role_weights = reliability.unsqueeze(-1) * role_probs
    states: List[torch.Tensor] = []
    strengths: List[torch.Tensor] = []
    for role_index in range(3):
        weights = role_weights[:, :, role_index]
        states.append(torch.einsum("bn,bnh->bh", weights, nodes) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        strengths.append(weights.sum(dim=-1) / valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0))
    support, opposition, uncertainty = states
    conflict = torch.zeros_like(support)
    conflict_score = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
    modality_raw = []
    for item in modality_slices:
        modality_valid = valid[:, item]
        modality_raw.append(reliability[:, item].sum(dim=-1) / modality_valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0))
    modality_weights = torch.stack(modality_raw, dim=-1)
    modality_weights = modality_weights / modality_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1)
    role_entropy = (role_entropy * valid.to(nodes.dtype)).sum(dim=-1) / valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0)
    return {
        "support": support,
        "opposition": opposition,
        "uncertainty": uncertainty,
        "conflict": conflict,
        "conflict_score": conflict_score,
        "strengths": torch.stack(strengths, dim=-1),
        "reliability": reliability,
        "modality_weights": modality_weights,
        "role_entropy": role_entropy,
    }


def mechanism_forward_trace(
    mea: torch.nn.Module,
    encoded: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    zero_edges: Iterable[str] = (),
    node_bypass: str | None = None,
    skip_modulation: bool = False,
) -> Dict[str, Any]:
    """Mirror the real MEA graph and optionally apply inference-only interventions."""
    zero_edges = set(zero_edges)
    image = mea.image(encoded["image_tokens"], batch["image_mask"])
    text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
    text = mea.text(encoded["text_tokens"], batch["report_attention_mask"], text_masks)
    bio = mea.bio(encoded["bio_tokens"], batch["bio_missing_mask"])
    nodes = torch.cat([image["nodes"], text["nodes"], bio["nodes"]], dim=1)
    valid = torch.cat([image["valid"], text["valid"], bio["valid"]], dim=1)
    role_logits, role_probs = mea.role_scorer(nodes)
    mechanisms = mea.mechanisms
    relation = mechanisms.relations

    image_available = image["valid"].any(dim=-1)
    image_source = masked_mean(image["nodes"], image["valid"])
    image_transformed = relation["image_morphology"](image_source)
    image_gate = image_available.to(image_transformed.dtype).unsqueeze(-1)
    image_effective = _apply_zero(
        "image_morphology_to_M1_morphology", image_transformed * image_gate, zero_edges
    )

    text_morph_valid = torch.stack([text["valid"][:, 0], text["valid"][:, 3]], dim=1)
    text_morph_source = masked_mean(torch.stack([text["nodes"][:, 0], text["nodes"][:, 3]], dim=1), text_morph_valid)
    text_morph_available = text_morph_valid.any(dim=-1)
    text_transformed = relation["text_morphology"](
        text_morph_source
    )
    text_gate = text_morph_available.to(text_transformed.dtype).unsqueeze(-1)
    text_effective = _apply_zero(
        "text_morphology_to_M1_morphology", text_transformed * text_gate, zero_edges
    )
    m1_valid = image_available | text_morph_available

    source_by_edge: Dict[str, Dict[str, torch.Tensor | None]] = {
        "image_morphology_to_M1_morphology": _edge_record(image_source, image_transformed, image_gate, image_effective),
        "text_morphology_to_M1_morphology": _edge_record(text_morph_source, text_transformed, text_gate, text_effective),
    }

    raw_sources = {
        "bio_immune_to_M2_immune": bio["nodes"][:, 1],
        "bio_function_to_M3_function": bio["nodes"][:, 2],
        "text_opposition_to_M4_opposition": text["nodes"][:, 1],
        "text_temporal_to_M5_temporal": text["nodes"][:, 4],
    }
    raw_valid = {
        "bio_immune_to_M2_immune": bio["valid"][:, 1],
        "bio_function_to_M3_function": bio["valid"][:, 2],
        "text_opposition_to_M4_opposition": text["valid"][:, 1],
        "text_temporal_to_M5_temporal": text["valid"][:, 4],
    }
    for edge, source in raw_sources.items():
        module_name = RELATION_MODULES[edge]
        transformed = relation[module_name](source)
        gate = raw_valid[edge].to(transformed.dtype).unsqueeze(-1)
        effective = _apply_zero(edge, transformed * gate, zero_edges)
        source_by_edge[edge] = _edge_record(source, transformed, gate, effective)

    node_pre = {
        "M1_morphology": image_source + text_morph_source,
        "M2_immune": raw_sources["bio_immune_to_M2_immune"],
        "M3_function": raw_sources["bio_function_to_M3_function"],
        "M4_opposition": raw_sources["text_opposition_to_M4_opposition"],
        "M5_temporal": raw_sources["text_temporal_to_M5_temporal"],
    }
    node_message = {
        "M1_morphology": image_effective + text_effective,
        "M2_immune": source_by_edge["bio_immune_to_M2_immune"]["effective_message"],
        "M3_function": source_by_edge["bio_function_to_M3_function"]["effective_message"],
        "M4_opposition": source_by_edge["text_opposition_to_M4_opposition"]["effective_message"],
        "M5_temporal": source_by_edge["text_temporal_to_M5_temporal"]["effective_message"],
    }
    node_valid = {
        "M1_morphology": m1_valid,
        "M2_immune": bio["valid"][:, 1],
        "M3_function": bio["valid"][:, 2],
        "M4_opposition": text["valid"][:, 1],
        "M5_temporal": text["valid"][:, 4],
    }
    states_list: List[torch.Tensor] = []
    node_trace: Dict[str, Dict[str, torch.Tensor]] = {}
    for index, node in enumerate(NODE_NAMES):
        aggregate = node_message[node]
        valid_node = node_valid[node]
        normalized = mechanisms.norms[index](aggregate) * valid_node.unsqueeze(-1).to(aggregate.dtype)
        if node == "M1_morphology":
            incoming_count = (
                image_available.to(aggregate.dtype) + text_morph_available.to(aggregate.dtype)
            ).unsqueeze(-1)
        else:
            incoming_count = valid_node.to(aggregate.dtype).unsqueeze(-1)
        node_trace[node] = {
            "node_pre": node_pre[node],
            "message_aggregate": aggregate,
            "incoming_message_count": incoming_count,
            "incoming_message_sum": aggregate,
            "incoming_message_mean": aggregate / incoming_count.clamp_min(1.0),
            "incoming_message_norm": aggregate.norm(dim=-1, keepdim=True),
            "node_after_update_before_norm": aggregate,
            "node_after_norm": normalized,
            "node_valid": valid_node.to(aggregate.dtype).unsqueeze(-1),
        }
        states_list.append(normalized)
    states = torch.stack(states_list, dim=1)

    if node_bypass in NODE_NAMES:
        bypass_index = NODE_NAMES.index(node_bypass)
        states[:, bypass_index] = node_pre[node_bypass]
        node_trace[node_bypass]["node_after_norm"] = node_pre[node_bypass]
        node_trace[node_bypass]["node_bypassed"] = torch.ones(
            (states.shape[0], 1), device=states.device, dtype=states.dtype
        )

    valid_mechanisms = torch.stack([node_valid[node] for node in NODE_NAMES], dim=1)
    safe_valid = valid_mechanisms.clone()
    empty = ~safe_valid.any(dim=-1)
    if bool(empty.any().item()):
        safe_valid[empty, 0] = True
    query = mechanisms.disease_query.expand(states.shape[0], -1, -1)
    attention_edges_zeroed = any(f"{node}_to_final_mechanism" in zero_edges for node in NODE_NAMES)
    attention_values = states.clone() if attention_edges_zeroed else states
    if attention_edges_zeroed:
        for index, node in enumerate(NODE_NAMES):
            if f"{node}_to_final_mechanism" in zero_edges:
                attention_values[:, index] = 0.0
    disease_attended, attention = mechanisms.disease_attn(
        query, states, attention_values, key_padding_mask=~safe_valid, need_weights=True
    )
    attention = attention.squeeze(1)
    text_global_source = text["nodes"][:, 5]
    text_global_transformed = relation["text_global"](text_global_source)
    text_global_gate = text["valid"][:, 5].to(text_global_transformed.dtype).unsqueeze(-1)
    text_global_effective = _apply_zero(
        "text_global_to_final_mechanism", text_global_transformed * text_global_gate, zero_edges
    )
    bio_other_source = bio["nodes"][:, 0]
    bio_other_transformed = relation["bio_other"](bio_other_source)
    bio_other_gate = bio["valid"][:, 0].to(bio_other_transformed.dtype).unsqueeze(-1)
    bio_other_effective = _apply_zero(
        "bio_other_to_final_mechanism", bio_other_transformed * bio_other_gate, zero_edges
    )
    source_by_edge["text_global_to_final_mechanism"] = _edge_record(
        text_global_source, text_global_transformed, text_global_gate, text_global_effective
    )
    source_by_edge["bio_other_to_final_mechanism"] = _edge_record(
        bio_other_source, bio_other_transformed, bio_other_gate, bio_other_effective
    )
    for index, node in enumerate(NODE_NAMES):
        edge = f"{node}_to_final_mechanism"
        state = states[:, index]
        weight = attention[:, index].unsqueeze(-1)
        gate = safe_valid[:, index].to(state.dtype).unsqueeze(-1)
        effective = state * weight
        if edge in zero_edges:
            effective = torch.zeros_like(effective)
        source_by_edge[edge] = _edge_record(state, state, gate, effective, weight=weight)
    context_effective = text_global_effective + bio_other_effective
    disease_pre_norm = disease_attended.squeeze(1) + context_effective
    disease_state = mechanisms.disease_norm(disease_pre_norm)

    if skip_modulation:
        aggregate = _aggregate_without_modulation(nodes, valid, role_probs, (slice(0, 5), slice(5, 11), slice(11, 14)))
    else:
        aggregate = mea.aggregator(nodes, valid, role_probs, (slice(0, 5), slice(5, 11), slice(11, 14)))
    head = mea.head(disease_state, aggregate)
    support_opposition_cosine = F.cosine_similarity(aggregate["support"], aggregate["opposition"], dim=-1)
    mea_outputs: Dict[str, torch.Tensor] = {
        **head,
        "mea_mechanism_state": disease_state,
        "mea_mechanism_nodes": states,
        "mea_mechanism_valid": valid_mechanisms,
        "mea_support_state": aggregate["support"],
        "mea_opposition_state": aggregate["opposition"],
        "mea_uncertainty_state": aggregate["uncertainty"],
        "mea_conflict_state": aggregate["conflict"],
        "mea_strengths": aggregate["strengths"],
        "conflict_suppression": (1.0 - aggregate["conflict_score"]).clamp(0.0, 1.0),
        "mea_mechanism_alignment_loss": disease_state.sum() * 0.0,
        "mea_role_separation_loss": support_opposition_cosine.square().mean(),
        "patient_support_strength": head["q_support"],
        "patient_opposition_strength": head["q_opposition"],
        "patient_uncertainty_strength": aggregate["strengths"][:, 2],
        "patient_conflict_score": aggregate["conflict_score"],
        "mea_role_logits": role_logits,
        "mea_role_probs": role_probs,
        "mea_node_valid": valid,
    }
    trace: Dict[str, Any] = {
        "nodes": node_trace,
        "edges": source_by_edge,
        "tensors": {
            "mechanism_state": disease_state,
            "mechanism_attention": attention,
            "mechanism_pre": states,
            "mechanism_message_aggregate": disease_pre_norm,
            "mechanism_after_norm": disease_state,
            "role_logits": role_logits,
            "role_probs": role_probs,
            "aggregate_support": aggregate["support"],
            "aggregate_opposition": aggregate["opposition"],
            "aggregate_uncertainty": aggregate["uncertainty"],
            "aggregate_conflict": aggregate["conflict"],
            "aggregate_reliability": aggregate["reliability"],
            "aggregate_conflict_score": aggregate["conflict_score"],
            "aggregate_modality_weights": aggregate["modality_weights"],
            "aggregate_strengths": aggregate["strengths"],
            "base_role_nodes": nodes,
            "base_role_valid": valid,
        },
    }
    return {"outputs": mea_outputs, "trace": trace, "encoded": encoded, "base_outputs": None}


def c17_forward_from_encoded(
    model: C17ResidualModel,
    batch: Dict[str, Any],
    encoded: Dict[str, torch.Tensor],
    base_outputs: Dict[str, torch.Tensor],
    zero_edges: Iterable[str] = (),
    node_bypass: str | None = None,
    skip_modulation: bool = False,
) -> Dict[str, Any]:
    mechanism = mechanism_forward_trace(
        model.mechanism_evidence_alignment,
        encoded,
        batch,
        zero_edges=zero_edges,
        node_bypass=node_bypass,
        skip_modulation=skip_modulation,
    )
    mea_outputs = mechanism["outputs"]
    correction_features = model._mechanism_correction_features(mea_outputs)
    residual = model.residual_head(correction_features)
    final_logit = base_outputs["logit"] + residual["delta_logit"]
    mechanism["base_outputs"] = base_outputs
    mechanism["outputs"] = {
        **mea_outputs,
        "logit": final_logit,
        "prob": torch.sigmoid(final_logit),
        "base_logit": base_outputs["logit"],
        "base_prob": torch.sigmoid(base_outputs["logit"]),
        "raw_delta": residual["raw_delta"],
        "delta_logit": residual["delta_logit"],
    }
    return mechanism


def c17_forward_variant(
    model: C17ResidualModel,
    batch: Dict[str, Any],
    zero_edges: Iterable[str] = (),
    node_bypass: str | None = None,
    skip_modulation: bool = False,
) -> Dict[str, Any]:
    with torch.no_grad():
        encoded = model.base_model.encode_modalities(batch)
        base_outputs = model.base_model.forward_from_encoded(batch, encoded)
        return c17_forward_from_encoded(
            model,
            batch,
            encoded,
            base_outputs,
            zero_edges=zero_edges,
            node_bypass=node_bypass,
            skip_modulation=skip_modulation,
        )


def read_validation_predictions(path: Path) -> Dict[str, Dict[str, float]]:
    import pandas as pd

    frame = pd.read_csv(path)
    probability_column = next(column for column in ("prob", "final_prob", "prediction", "y_prob") if column in frame.columns)
    return {
        str(row["patient_id"]): {"label": float(row["label"]), "prob": float(row[probability_column])}
        for _, row in frame.iterrows()
    }


def load_trace_npz(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load a C21-A trace archive keyed by seed without object deserialization."""
    traces: Dict[int, Dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            prefix, separator, suffix = key.partition("__")
            if not separator or not prefix.startswith("seed_"):
                continue
            seed = int(prefix[len("seed_") :])
            traces.setdefault(seed, {"tensors": {}, "shortcuts": {}})
            if suffix in {"patient_id", "labels"}:
                traces[seed][suffix] = archive[key]
            elif suffix.startswith("shortcut__"):
                traces[seed]["shortcuts"][suffix[len("shortcut__") :]] = archive[key]
            else:
                traces[seed]["tensors"][suffix] = archive[key]
    for seed, item in traces.items():
        if "patient_id" not in item or "labels" not in item:
            raise ValueError(f"Trace archive is missing identifiers for seed {seed}: {path}")
    return traces


def finite_matrix(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def common_index(left_ids: Sequence[str], right_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    right = {str(value): index for index, value in enumerate(right_ids)}
    left_indices: List[int] = []
    right_indices: List[int] = []
    for index, value in enumerate(left_ids):
        key = str(value)
        if key in right:
            left_indices.append(index)
            right_indices.append(right[key])
    if not left_indices:
        raise RuntimeError("no common patient IDs while aligning C21-A traces")
    return np.asarray(left_indices, dtype=np.int64), np.asarray(right_indices, dtype=np.int64)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            ranks[inverse == group] = float(np.mean(ranks[inverse == group]))
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def pairwise_distances(value: np.ndarray) -> np.ndarray:
    matrix = finite_matrix(value)
    gram = matrix @ matrix.T
    squared = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(squared)


def _knn_indices(value: np.ndarray, k: int = 10) -> np.ndarray:
    distances = pairwise_distances(value)
    if distances.shape[0] == 0:
        return np.empty((0, 0), dtype=np.int64)
    distances[np.arange(distances.shape[0]), np.arange(distances.shape[0])] = np.inf
    effective_k = min(int(k), max(distances.shape[0] - 1, 0))
    if effective_k <= 0:
        return np.empty((distances.shape[0], 0), dtype=np.int64)
    return np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]


def _knn_jaccard(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    left_neighbors = _knn_indices(left, k=k)
    right_neighbors = _knn_indices(right, k=k)
    values: List[float] = []
    for left_row, right_row in zip(left_neighbors, right_neighbors):
        left_set = set(int(item) for item in left_row)
        right_set = set(int(item) for item in right_row)
        union = left_set | right_set
        values.append(float(len(left_set & right_set) / len(union)) if union else float("nan"))
    return float(np.nanmean(values)) if values and np.isfinite(values).any() else float("nan")


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_matrix = finite_matrix(left)
    right_matrix = finite_matrix(right)
    left_matrix = left_matrix - left_matrix.mean(axis=0, keepdims=True)
    right_matrix = right_matrix - right_matrix.mean(axis=0, keepdims=True)
    left_gram = left_matrix @ left_matrix.T
    right_gram = right_matrix @ right_matrix.T
    left_gram -= left_gram.mean(axis=0, keepdims=True)
    left_gram -= left_gram.mean(axis=1, keepdims=True)
    right_gram -= right_gram.mean(axis=0, keepdims=True)
    right_gram -= right_gram.mean(axis=1, keepdims=True)
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(np.sqrt(np.sum(left_gram * left_gram) * np.sum(right_gram * right_gram)))
    return numerator / denominator if denominator > 1e-12 else float("nan")


def procrustes_metrics(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    left_matrix = finite_matrix(left)
    right_matrix = finite_matrix(right)
    if left_matrix.shape[0] != right_matrix.shape[0] or left_matrix.shape[0] < 2:
        return float("nan"), float("nan")
    left_centered = left_matrix - left_matrix.mean(axis=0, keepdims=True)
    right_centered = right_matrix - right_matrix.mean(axis=0, keepdims=True)
    cross = left_centered.T @ right_centered
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    aligned = left_centered @ rotation
    left_norm = np.linalg.norm(aligned, axis=1)
    right_norm = np.linalg.norm(right_centered, axis=1)
    cosine = np.divide(
        np.sum(aligned * right_centered, axis=1),
        np.maximum(left_norm * right_norm, 1e-12),
    )
    return float(np.nanmean(cosine)), float(np.sqrt(np.mean((aligned - right_centered) ** 2)))


def representation_metrics(left: np.ndarray, right: np.ndarray, k: int = 10) -> Dict[str, float]:
    left_matrix = finite_matrix(left)
    right_matrix = finite_matrix(right)
    if left_matrix.shape[0] != right_matrix.shape[0]:
        raise ValueError(f"sample count mismatch: {left_matrix.shape} versus {right_matrix.shape}")
    if left_matrix.shape[0] >= 2:
        distance_left = pairwise_distances(left_matrix)
        distance_right = pairwise_distances(right_matrix)
        triangle = np.triu_indices(left_matrix.shape[0], k=1)
        distance_spearman = _spearman(distance_left[triangle], distance_right[triangle])
    else:
        distance_spearman = float("nan")
    procrustes_cosine, procrustes_rmse = procrustes_metrics(left_matrix, right_matrix)
    return {
        "linear_cka": linear_cka(left_matrix, right_matrix),
        "distance_spearman": distance_spearman,
        "knn_jaccard": _knn_jaccard(left_matrix, right_matrix, k=k),
        "procrustes_cosine": procrustes_cosine,
        "procrustes_rmse": procrustes_rmse,
        "n": int(left_matrix.shape[0]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
