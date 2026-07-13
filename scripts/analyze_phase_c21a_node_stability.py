#!/usr/bin/env python3
"""Measure cross-seed node stability and within-seed propagation deformation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase_c21a_common import (
    NODE_NAMES,
    common_index,
    finite_matrix,
    load_trace_npz,
    representation_metrics,
    resolve_path,
    write_rows,
)


NODE_STAGES = (
    "node_pre",
    "message_aggregate",
    "incoming_message_mean",
    "node_after_update_before_norm",
    "node_after_norm",
)
MECHANISM_STAGES = (
    "tensor__mechanism_pre",
    "tensor__mechanism_message_aggregate",
    "tensor__mechanism_after_norm",
)
SEED_PAIRS = ((0, 42), (0, 3407), (42, 3407))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--output", default="analysis_reports/phase_c21a_dema/c21a_node_stability_by_stage.csv")
    parser.add_argument("--summary-output", default="analysis_reports/phase_c21a_dema/c21a_node_stability_summary.csv")
    parser.add_argument("--knn-k", type=int, default=10)
    return parser.parse_args()


def aligned_values(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_index, right_index = common_index(left["patient_id"], right["patient_id"])
    left_values = finite_matrix(left["tensors"][key])[left_index]
    right_values = finite_matrix(right["tensors"][key])[right_index]
    left_labels = np.asarray(left["labels"])[left_index].astype(np.int64)
    right_labels = np.asarray(right["labels"])[right_index].astype(np.int64)
    if not np.array_equal(left_labels, right_labels):
        raise RuntimeError(f"label mismatch while aligning {key}")
    return left_values, right_values, left_labels


def add_metric_row(
    rows: List[Dict[str, Any]],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key: str,
    entity: str,
    stage: str,
    split: str,
    comparison: str,
    seed_left: int,
    seed_right: int,
    k: int,
) -> None:
    values_left, values_right, labels = aligned_values(left, right, key)
    metrics = representation_metrics(values_left, values_right, k=k)
    rows.append(
        {
            "comparison": comparison,
            "split": split,
            "seed_left": seed_left,
            "seed_right": seed_right,
            "entity": entity,
            "stage": stage,
            "stage_left": stage if comparison == "cross_seed" else key,
            "stage_right": stage if comparison == "cross_seed" else key,
            "key": key,
            "label_positive_count": int(labels.sum()),
            "label_negative_count": int((labels == 0).sum()),
            **metrics,
        }
    )


def main() -> None:
    args = parse_args()
    trace_dir = resolve_path(args.trace_dir)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    traces = {
        split: load_trace_npz(trace_dir / f"c21a_trace_{split}.npz")
        for split in ("train", "val")
    }
    rows: List[Dict[str, Any]] = []

    for split, split_traces in traces.items():
        for seed_left, seed_right in SEED_PAIRS:
            left = split_traces[seed_left]
            right = split_traces[seed_right]
            for node in NODE_NAMES:
                for stage in NODE_STAGES:
                    key = f"node__{node}__{stage}"
                    add_metric_row(
                        rows,
                        left,
                        right,
                        key,
                        node,
                        stage,
                        split,
                        "cross_seed",
                        seed_left,
                        seed_right,
                        args.knn_k,
                    )
            for key in MECHANISM_STAGES:
                if key not in left["tensors"] or key not in right["tensors"]:
                    continue
                add_metric_row(
                    rows,
                    left,
                    right,
                    key,
                    "final_mechanism",
                    key[len("tensor__mechanism_") :],
                    split,
                    "cross_seed",
                    seed_left,
                    seed_right,
                    args.knn_k,
                )

        for seed, trace in split_traces.items():
            for node in NODE_NAMES:
                stage_pairs = (
                    ("node_pre", "message_aggregate"),
                    ("node_pre", "node_after_update_before_norm"),
                    ("node_pre", "node_after_norm"),
                    ("message_aggregate", "node_after_norm"),
                )
                for stage_left, stage_right in stage_pairs:
                    key_left = f"node__{node}__{stage_left}"
                    key_right = f"node__{node}__{stage_right}"
                    left_values = finite_matrix(trace["tensors"][key_left])
                    right_values = finite_matrix(trace["tensors"][key_right])
                    metrics = representation_metrics(left_values, right_values, k=args.knn_k)
                    labels = np.asarray(trace["labels"]).astype(np.int64)
                    rows.append(
                        {
                            "comparison": "within_seed_stage",
                            "split": split,
                            "seed_left": seed,
                            "seed_right": seed,
                            "entity": node,
                            "stage": f"{stage_left}_to_{stage_right}",
                            "stage_left": stage_left,
                            "stage_right": stage_right,
                            "key": f"{key_left}|{key_right}",
                            "label_positive_count": int(labels.sum()),
                            "label_negative_count": int((labels == 0).sum()),
                            **metrics,
                        }
                    )
            for key_left, key_right in (
                ("tensor__mechanism_pre", "tensor__mechanism_message_aggregate"),
                ("tensor__mechanism_pre", "tensor__mechanism_after_norm"),
                ("tensor__mechanism_message_aggregate", "tensor__mechanism_after_norm"),
            ):
                left_values = finite_matrix(trace["tensors"][key_left])
                right_values = finite_matrix(trace["tensors"][key_right])
                metrics = representation_metrics(left_values, right_values, k=args.knn_k)
                labels = np.asarray(trace["labels"]).astype(np.int64)
                rows.append(
                    {
                        "comparison": "within_seed_stage",
                        "split": split,
                        "seed_left": seed,
                        "seed_right": seed,
                        "entity": "final_mechanism",
                        "stage": f"{key_left}|{key_right}",
                        "stage_left": key_left,
                        "stage_right": key_right,
                        "key": f"{key_left}|{key_right}",
                        "label_positive_count": int(labels.sum()),
                        "label_negative_count": int((labels == 0).sum()),
                        **metrics,
                    }
                )

    write_rows(output_path, rows)
    summary_rows: List[Dict[str, Any]] = []
    cross_rows = [row for row in rows if row["comparison"] == "cross_seed"]
    groups = sorted({(row["split"], row["entity"], row["stage"]) for row in cross_rows})
    for split, entity, stage in groups:
        subset = [row for row in cross_rows if (row["split"], row["entity"], row["stage"]) == (split, entity, stage)]
        summary_rows.append(
            {
                "split": split,
                "entity": entity,
                "stage": stage,
                "seed_pair_count": len(subset),
                "linear_cka_mean": float(np.nanmean([row["linear_cka"] for row in subset])),
                "distance_spearman_mean": float(np.nanmean([row["distance_spearman"] for row in subset])),
                "knn_jaccard_mean": float(np.nanmean([row["knn_jaccard"] for row in subset])),
                "procrustes_cosine_mean": float(np.nanmean([row["procrustes_cosine"] for row in subset])),
                "procrustes_rmse_mean": float(np.nanmean([row["procrustes_rmse"] for row in subset])),
            }
        )
    write_rows(summary_path, summary_rows)
    print(f"wrote {len(rows)} node stability rows to {output_path}")


if __name__ == "__main__":
    main()
