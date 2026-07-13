#!/usr/bin/env python3
"""Score C21-A mechanism-propagation responsibility without fitting a model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from phase_c21a_common import load_trace_npz, resolve_path, write_rows


SEEDS = (0, 42, 3407)
SEED_PAIRS = ((0, 42), (0, 3407), (42, 3407))
SCORE_WEIGHTS = {
    "cka": 0.30,
    "distance": 0.25,
    "knn": 0.20,
    "ablation_inconsistency": 0.15,
    "saturation_or_collapse": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--output", default="analysis_reports/phase_c21a_dema/c21a_instability_responsibility_scores.csv")
    parser.add_argument("--summary-output", default="analysis_reports/phase_c21a_dema/c21a_score_summary.json")
    parser.add_argument("--reproduction", default="analysis_reports/phase_c21a_dema/c21a_reproduction_check_by_seed.csv")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def mean_finite(values: Iterable[float], default: float = float("nan")) -> float:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(array)) if array.size else default


def clipped_drop(values: Iterable[float]) -> float:
    return mean_finite(np.clip(1.0 - np.asarray(list(values), dtype=np.float64), 0.0, 1.0))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2:
        return float("nan")
    if np.allclose(left, right, rtol=0.0, atol=1e-12):
        return 1.0
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


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (positive.size * negative.size))


def minmax(values: Sequence[float]) -> Dict[int, float]:
    finite = [(index, value) for index, value in enumerate(values) if np.isfinite(value)]
    if not finite:
        return {index: 0.0 for index in range(len(values))}
    low = min(value for _, value in finite)
    high = max(value for _, value in finite)
    if high - low <= 1e-12:
        return {index: (0.0 if not np.isfinite(value) else 0.0) for index, value in enumerate(values)}
    return {
        index: float((value - low) / (high - low)) if np.isfinite(value) else 0.0
        for index, value in enumerate(values)
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def group_metric_rows(rows: Sequence[Mapping[str, str]], predicate: Any) -> Dict[str, List[Mapping[str, str]]]:
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            groups[str(row["entity"])].append(row)
    return groups


def build_propagation_records(trace_dir: Path) -> List[Dict[str, Any]]:
    node_rows = read_csv(trace_dir / "c21a_node_stability_by_stage.csv")
    edge_rows = read_csv(trace_dir / "c21a_edge_stability.csv")
    aggregator_rows = read_csv(trace_dir / "c21a_conflict_reliability_stability.csv")
    records: List[Dict[str, Any]] = []

    node_groups = group_metric_rows(
        node_rows,
        lambda row: row["comparison"] == "within_seed_stage"
        and row["split"] == "val"
        and row["stage_left"] == "node_pre"
        and row["stage_right"] == "node_after_norm",
    )
    for entity, subset in node_groups.items():
        records.append(
            {
                "entity": entity,
                "category": "node",
                "scope": "mechanism_node",
                "evidence_rows": len(subset),
                "raw_cka_drop": clipped_drop(to_float(row.get("linear_cka")) for row in subset),
                "raw_distance_drop": clipped_drop(to_float(row.get("distance_spearman")) for row in subset),
                "raw_knn_drop": clipped_drop(to_float(row.get("knn_jaccard")) for row in subset),
                "raw_saturation_or_collapse": clipped_drop(to_float(row.get("procrustes_cosine")) for row in subset),
            }
        )

    final_groups = group_metric_rows(
        node_rows,
        lambda row: row["comparison"] == "within_seed_stage"
        and row["split"] == "val"
        and row["entity"] == "final_mechanism"
        and row["stage_right"] == "tensor__mechanism_after_norm",
    )
    for entity, subset in final_groups.items():
        records.append(
            {
                "entity": entity,
                "category": "final_mechanism",
                "scope": "mechanism_final_state",
                "evidence_rows": len(subset),
                "raw_cka_drop": clipped_drop(to_float(row.get("linear_cka")) for row in subset),
                "raw_distance_drop": clipped_drop(to_float(row.get("distance_spearman")) for row in subset),
                "raw_knn_drop": clipped_drop(to_float(row.get("knn_jaccard")) for row in subset),
                "raw_saturation_or_collapse": clipped_drop(to_float(row.get("procrustes_cosine")) for row in subset),
            }
        )

    edge_groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in edge_rows:
        if (
            row["comparison"] == "within_seed_stage"
            and row["split"] == "val"
            and row["field"] == "source_representation_to_effective_message"
        ):
            edge_groups[str(row["edge"])].append(row)
    for entity, subset in edge_groups.items():
        ratio = mean_finite(to_float(row.get("mean_effective_to_source_norm_ratio")) for row in subset)
        collapse = mean_finite(to_float(row.get("effective_collapse_fraction")) for row in subset)
        saturation = mean_finite([collapse, np.clip(1.0 - ratio, 0.0, 1.0)])
        records.append(
            {
                "entity": entity,
                "category": "edge",
                "scope": "mechanism_edge",
                "evidence_rows": len(subset),
                "raw_cka_drop": clipped_drop(to_float(row.get("linear_cka")) for row in subset),
                "raw_distance_drop": clipped_drop(to_float(row.get("distance_spearman")) for row in subset),
                "raw_knn_drop": clipped_drop(to_float(row.get("knn_jaccard")) for row in subset),
                "raw_saturation_or_collapse": saturation,
            }
        )

    role_rows = [
        row
        for row in aggregator_rows
        if row["comparison"] == "cross_seed"
        and row["split"] == "val"
        and row["tensor"] in {"tensor__role_logits", "tensor__role_probs"}
    ]
    if role_rows:
        records.append(
            {
                "entity": "role_scoring",
                "category": "role_scoring",
                "scope": "evidence_role_scorer",
                "evidence_rows": len(role_rows),
                "raw_cka_drop": clipped_drop(to_float(row.get("linear_cka")) for row in role_rows),
                "raw_distance_drop": clipped_drop(to_float(row.get("distance_spearman")) for row in role_rows),
                "raw_knn_drop": clipped_drop(to_float(row.get("knn_jaccard")) for row in role_rows),
                "raw_saturation_or_collapse": 0.0,
            }
        )
    aggregation_rows = [
        row
        for row in aggregator_rows
        if row["comparison"] == "cross_seed"
        and row["split"] == "val"
        and row["tensor"].startswith("tensor__aggregate_")
    ]
    if aggregation_rows:
        records.append(
            {
                "entity": "aggregation",
                "category": "aggregation",
                "scope": "evidence_conflict_aggregator",
                "evidence_rows": len(aggregation_rows),
                "raw_cka_drop": clipped_drop(to_float(row.get("linear_cka")) for row in aggregation_rows),
                "raw_distance_drop": clipped_drop(to_float(row.get("distance_spearman")) for row in aggregation_rows),
                "raw_knn_drop": clipped_drop(to_float(row.get("knn_jaccard")) for row in aggregation_rows),
                "raw_saturation_or_collapse": 0.0,
            }
        )
    return records


def load_ablation_stats(trace_dir: Path) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for filename, field in (
        ("c21a_edge_ablation_patient.csv", "edge"),
        ("c21a_node_bypass_patient.csv", "intervention"),
    ):
        rows = read_csv(trace_dir / filename)
        grouped: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        abs_deltas: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            entity = str(row[field])
            seed = int(row["seed"])
            patient_id = str(row["patient_id"])
            delta = to_float(row.get("delta_prob_ablated_minus_baseline", row.get("delta_prob_intervention_minus_baseline")))
            grouped[entity][seed][patient_id] = delta
            if np.isfinite(delta):
                abs_deltas[entity].append(abs(delta))
        for entity, by_seed in grouped.items():
            pair_correlations: List[float] = []
            direction_scores: List[float] = []
            for seed_left, seed_right in SEED_PAIRS:
                common = sorted(set(by_seed[seed_left]) & set(by_seed[seed_right]))
                if not common:
                    continue
                left = np.asarray([by_seed[seed_left][patient_id] for patient_id in common], dtype=np.float64)
                right = np.asarray([by_seed[seed_right][patient_id] for patient_id in common], dtype=np.float64)
                finite = np.isfinite(left) & np.isfinite(right)
                left = left[finite]
                right = right[finite]
                if left.size:
                    pair_correlations.append(spearman(left, right))
                    direction_scores.append(float(np.mean(np.sign(left) == np.sign(right))))
            stats[entity] = {
                "ablation_supported": len(pair_correlations) >= 2,
                "ablation_seed_spearman": mean_finite(pair_correlations),
                "ablation_direction_consistency": mean_finite(direction_scores),
                "mean_abs_delta_prob": mean_finite(abs_deltas[entity], default=0.0),
            }
    return stats


def add_shortcut_audit(trace_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    traces = load_trace_npz(trace_dir / "c21a_trace_val.npz")
    for seed, trace in traces.items():
        labels = np.asarray(trace["labels"]).astype(np.int64)
        for field in ("selected_n_visits", "used_images", "raw_n_visits", "raw_n_images"):
            values = trace.get("shortcuts", {}).get(field)
            if values is None:
                continue
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if numeric.size != labels.size or not np.isfinite(numeric).all():
                continue
            raw_auc = auc_score(labels, numeric)
            rows.append(
                {
                    "seed": seed,
                    "split": "val",
                    "shortcut": field,
                    "raw_auc": raw_auc,
                    "orientation_invariant_auc": max(raw_auc, 1.0 - raw_auc),
                    "audit_only": True,
                    "model_or_probe_input": False,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    trace_dir = resolve_path(args.trace_dir)
    records = build_propagation_records(trace_dir)
    ablation = load_ablation_stats(trace_dir)
    for record in records:
        evidence = ablation.get(record["entity"], {})
        record.update(
            {
                "ablation_supported": bool(evidence.get("ablation_supported", False)),
                "raw_ablation_inconsistency": (
                    float(np.clip(1.0 - max(to_float(evidence.get("ablation_seed_spearman"), 0.0), 0.0), 0.0, 1.0))
                    if evidence.get("ablation_seed_spearman") is not None and np.isfinite(to_float(evidence.get("ablation_seed_spearman")))
                    else float("nan")
                ),
                "ablation_seed_spearman": to_float(evidence.get("ablation_seed_spearman")),
                "ablation_direction_consistency": to_float(evidence.get("ablation_direction_consistency")),
                "mean_abs_delta_prob": to_float(evidence.get("mean_abs_delta_prob"), 0.0),
            }
        )

    cka_components = minmax([record["raw_cka_drop"] for record in records])
    distance_components = minmax([record["raw_distance_drop"] for record in records])
    knn_components = minmax([record["raw_knn_drop"] for record in records])
    inconsistency_components = minmax([record["raw_ablation_inconsistency"] for record in records])
    saturation_components = minmax([record["raw_saturation_or_collapse"] for record in records])
    for index, record in enumerate(records):
        record["component_cka"] = cka_components[index]
        record["component_distance"] = distance_components[index]
        record["component_knn"] = knn_components[index]
        record["component_ablation_inconsistency"] = inconsistency_components[index]
        record["component_saturation_or_collapse"] = saturation_components[index]
        record["responsibility_score"] = (
            SCORE_WEIGHTS["cka"] * record["component_cka"]
            + SCORE_WEIGHTS["distance"] * record["component_distance"]
            + SCORE_WEIGHTS["knn"] * record["component_knn"]
            + SCORE_WEIGHTS["ablation_inconsistency"] * record["component_ablation_inconsistency"]
            + SCORE_WEIGHTS["saturation_or_collapse"] * record["component_saturation_or_collapse"]
        )

    records.sort(key=lambda row: float(row["responsibility_score"]), reverse=True)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    write_rows(resolve_path(args.output), records)

    reproduction_rows = read_csv(resolve_path(args.reproduction))
    reproduction_pass = bool(reproduction_rows) and all(row.get("pass", "").lower() == "true" for row in reproduction_rows)
    supported = [record for record in records if record["ablation_supported"]]
    top = supported[0] if supported else (records[0] if records else None)
    second = supported[1] if len(supported) > 1 else None
    total_score = float(sum(max(float(record["responsibility_score"]), 0.0) for record in supported))
    top_score = float(top["responsibility_score"]) if top else float("nan")
    second_score = float(second["responsibility_score"]) if second else 0.0
    top_share = top_score / total_score if top and total_score > 1e-12 else 0.0
    margin = top_score - second_score if top else float("nan")
    seed_consistency = to_float(top.get("ablation_seed_spearman")) if top else float("nan")
    direction_consistency = to_float(top.get("ablation_direction_consistency")) if top else float("nan")
    localized = bool(
        reproduction_pass
        and top is not None
        and top["ablation_supported"]
        and top_score >= 0.55
        and top_share >= 0.08
        and margin >= 0.05
        and np.isfinite(seed_consistency)
        and seed_consistency >= 0.50
        and np.isfinite(direction_consistency)
        and direction_consistency >= 0.60
    )
    if not reproduction_pass:
        route = "C21A_REPRODUCTION_GATE_FAIL"
    elif localized and top["category"] == "edge":
        route = "C21A_LOCALIZED_EDGE_RESPONSIBILITY"
    elif localized and top["category"] == "node":
        route = "C21A_LOCALIZED_NODE_RESPONSIBILITY"
    elif localized and top["category"] in {"aggregation", "role_scoring", "final_mechanism"}:
        route = "C21A_LOCALIZED_DOWNSTREAM_RESPONSIBILITY"
    else:
        route = "C21A_DIFFUSE_MECHANISM_PROPAGATION_INSTABILITY"

    shortcut_rows = add_shortcut_audit(trace_dir)
    write_rows(trace_dir / "c21a_shortcut_exclusion_audit.csv", shortcut_rows)
    summary = {
        "phase": "C21-A",
        "route": route,
        "reproduction_pass": reproduction_pass,
        "localized_reproducible": localized,
        "top_responsibility": top,
        "second_responsibility": second,
        "top_score": top_score,
        "second_score": second_score,
        "top_share_among_supported_entities": top_share,
        "top_margin": margin,
        "top_ablation_seed_spearman": seed_consistency,
        "top_ablation_direction_consistency": direction_consistency,
        "score_formula": "0.30*CKA_drop + 0.25*distance_drop + 0.20*kNN_drop + 0.15*cross_seed_ablation_inconsistency + 0.10*saturation_or_collapse; each raw component min-max normalized across candidate entities",
        "localization_gate": {
            "reproduction_pass": reproduction_pass,
            "top_score_at_least": 0.55,
            "top_share_at_least": 0.08,
            "top_margin_at_least": 0.05,
            "top_ablation_seed_spearman_at_least": 0.50,
            "top_ablation_direction_consistency_at_least": 0.60,
        },
        "shortcut_audit": "validation-only orientation-invariant diagnostics; not model or probe inputs",
        "shortcut_rows": shortcut_rows,
        "c22_design_authorized": localized,
        "training_authorized": False,
        "test_data_read": False,
        "new_branch_or_worktree": False,
    }
    resolve_path(args.summary_output).write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"route": route, "localized_reproducible": localized, "top": top}, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
