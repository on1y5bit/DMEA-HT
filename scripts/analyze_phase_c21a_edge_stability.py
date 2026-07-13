#!/usr/bin/env python3
"""Measure cross-seed edge stability, message deformation, and aggregator stability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from phase_c21a_common import (
    ALL_EDGES,
    ATTENTION_EDGES,
    common_index,
    finite_matrix,
    load_trace_npz,
    representation_metrics,
    resolve_path,
    write_rows,
)


EDGE_FIELDS = (
    "source_representation",
    "transformed_source",
    "raw_message",
    "message_norm",
    "edge_gate",
    "effective_message",
)
WEIGHT_FIELDS = ("edge_weight",)
AGGREGATOR_KEYS = (
    "tensor__role_logits",
    "tensor__role_probs",
    "tensor__aggregate_support",
    "tensor__aggregate_opposition",
    "tensor__aggregate_uncertainty",
    "tensor__aggregate_conflict",
    "tensor__aggregate_reliability",
    "tensor__aggregate_conflict_score",
    "tensor__aggregate_modality_weights",
    "tensor__aggregate_strengths",
)
SEED_PAIRS = ((0, 42), (0, 3407), (42, 3407))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--output", default="analysis_reports/phase_c21a_dema/c21a_edge_stability.csv")
    parser.add_argument("--weight-output", default="analysis_reports/phase_c21a_dema/c21a_edge_weight_consistency.csv")
    parser.add_argument("--aggregator-output", default="analysis_reports/phase_c21a_dema/c21a_conflict_reliability_stability.csv")
    parser.add_argument("--knn-k", type=int, default=10)
    return parser.parse_args()


def aligned_values(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_index, right_index = common_index(left["patient_id"], right["patient_id"])
    left_values = finite_matrix(left["tensors"][key])[left_index]
    right_values = finite_matrix(right["tensors"][key])[right_index]
    labels_left = np.asarray(left["labels"])[left_index].astype(np.int64)
    labels_right = np.asarray(right["labels"])[right_index].astype(np.int64)
    if not np.array_equal(labels_left, labels_right):
        raise RuntimeError(f"label mismatch while aligning {key}")
    return left_values, right_values, labels_left


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
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


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(rankdata(left), rankdata(right))


def add_cross_row(
    rows: List[Dict[str, Any]],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key: str,
    entity: str,
    field: str,
    split: str,
    seed_left: int,
    seed_right: int,
    k: int,
) -> None:
    values_left, values_right, labels = aligned_values(left, right, key)
    metrics = representation_metrics(values_left, values_right, k=k)
    source_key = f"edge__{entity}__source_representation"
    effective_key = f"edge__{entity}__effective_message"
    source_norm = np.linalg.norm(finite_matrix(left["tensors"].get(source_key, values_left)), axis=1)
    effective_norm = np.linalg.norm(finite_matrix(left["tensors"].get(effective_key, values_left)), axis=1)
    ratio = np.divide(effective_norm, np.maximum(source_norm, 1e-12))
    rows.append(
        {
            "comparison": "cross_seed",
            "split": split,
            "seed_left": seed_left,
            "seed_right": seed_right,
            "edge": entity,
            "field": field,
            "key": key,
            "label_positive_count": int(labels.sum()),
            "label_negative_count": int((labels == 0).sum()),
            "mean_source_norm": float(np.mean(source_norm)),
            "mean_effective_norm": float(np.mean(effective_norm)),
            "mean_effective_to_source_norm_ratio": float(np.mean(ratio)),
            "effective_collapse_fraction": float(np.mean(ratio < 0.10)),
            **metrics,
        }
    )


def main() -> None:
    args = parse_args()
    trace_dir = resolve_path(args.trace_dir)
    outputs = {
        "train": load_trace_npz(trace_dir / "c21a_trace_train.npz"),
        "val": load_trace_npz(trace_dir / "c21a_trace_val.npz"),
    }
    edge_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    aggregator_rows: List[Dict[str, Any]] = []

    for split, traces in outputs.items():
        for seed_left, seed_right in SEED_PAIRS:
            left = traces[seed_left]
            right = traces[seed_right]
            for edge in ALL_EDGES:
                for field in EDGE_FIELDS:
                    key = f"edge__{edge}__{field}"
                    if key not in left["tensors"] or key not in right["tensors"]:
                        continue
                    add_cross_row(edge_rows, left, right, key, edge, field, split, seed_left, seed_right, args.knn_k)
                weight_key = f"edge__{edge}__edge_weight"
                if edge not in ATTENTION_EDGES:
                    weight_rows.append(
                        {
                            "comparison": "cross_seed",
                            "split": split,
                            "seed_left": seed_left,
                            "seed_right": seed_right,
                            "edge": edge,
                            "available": False,
                            "reason": "unavailable: relation/context edge has no independent learned scalar weight",
                        }
                    )
                elif weight_key in left["tensors"] and weight_key in right["tensors"]:
                    left_values, right_values, _labels = aligned_values(left, right, weight_key)
                    left_flat = left_values.reshape(-1)
                    right_flat = right_values.reshape(-1)
                    weight_rows.append(
                        {
                            "comparison": "cross_seed",
                            "split": split,
                            "seed_left": seed_left,
                            "seed_right": seed_right,
                            "edge": edge,
                            "available": True,
                            "reason": "",
                            "pearson": pearson(left_flat, right_flat),
                            "spearman": spearman(left_flat, right_flat),
                            "mean_abs_difference": float(np.mean(np.abs(left_flat - right_flat))),
                            "sign_agreement": float(np.mean(np.sign(left_flat) == np.sign(right_flat))),
                            "left_mean": float(np.mean(left_flat)),
                            "right_mean": float(np.mean(right_flat)),
                            "left_saturation_fraction": float(np.mean((left_flat < 0.05) | (left_flat > 0.95))),
                            "right_saturation_fraction": float(np.mean((right_flat < 0.05) | (right_flat > 0.95))),
                        }
                    )
            for key in AGGREGATOR_KEYS:
                if key not in left["tensors"] or key not in right["tensors"]:
                    continue
                values_left, values_right, labels = aligned_values(left, right, key)
                metrics = representation_metrics(values_left, values_right, k=args.knn_k)
                aggregator_rows.append(
                    {
                        "comparison": "cross_seed",
                        "split": split,
                        "seed_left": seed_left,
                        "seed_right": seed_right,
                        "tensor": key,
                        "label_positive_count": int(labels.sum()),
                        "label_negative_count": int((labels == 0).sum()),
                        **metrics,
                    }
                )

        for seed, trace in traces.items():
            for edge in ALL_EDGES:
                source_key = f"edge__{edge}__source_representation"
                transformed_key = f"edge__{edge}__transformed_source"
                effective_key = f"edge__{edge}__effective_message"
                if source_key not in trace["tensors"] or effective_key not in trace["tensors"]:
                    continue
                for stage_left, stage_right, key_left, key_right in (
                    ("source_representation", "transformed_source", source_key, transformed_key),
                    ("source_representation", "effective_message", source_key, effective_key),
                    ("transformed_source", "effective_message", transformed_key, effective_key),
                ):
                    if key_left not in trace["tensors"] or key_right not in trace["tensors"]:
                        continue
                    values_left = finite_matrix(trace["tensors"][key_left])
                    values_right = finite_matrix(trace["tensors"][key_right])
                    metrics = representation_metrics(values_left, values_right, k=args.knn_k)
                    labels = np.asarray(trace["labels"]).astype(np.int64)
                    edge_rows.append(
                        {
                            "comparison": "within_seed_stage",
                            "split": split,
                            "seed_left": seed,
                            "seed_right": seed,
                            "edge": edge,
                            "field": f"{stage_left}_to_{stage_right}",
                            "key": f"{key_left}|{key_right}",
                            "label_positive_count": int(labels.sum()),
                            "label_negative_count": int((labels == 0).sum()),
                            "mean_source_norm": float(np.mean(np.linalg.norm(values_left, axis=1))),
                            "mean_effective_norm": float(np.mean(np.linalg.norm(values_right, axis=1))),
                            "mean_effective_to_source_norm_ratio": float(
                                np.mean(np.linalg.norm(values_right, axis=1) / np.maximum(np.linalg.norm(values_left, axis=1), 1e-12))
                            ),
                            "effective_collapse_fraction": float(
                                np.mean(np.linalg.norm(values_right, axis=1) < 0.10 * np.maximum(np.linalg.norm(values_left, axis=1), 1e-12))
                            ),
                            **metrics,
                        }
                    )

    write_rows(resolve_path(args.output), edge_rows)
    write_rows(resolve_path(args.weight_output), weight_rows)
    write_rows(resolve_path(args.aggregator_output), aggregator_rows)
    print(
        f"wrote {len(edge_rows)} edge rows, {len(weight_rows)} weight rows, "
        f"and {len(aggregator_rows)} aggregator rows"
    )


if __name__ == "__main__":
    main()
