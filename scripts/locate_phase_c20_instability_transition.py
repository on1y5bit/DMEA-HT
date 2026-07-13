#!/usr/bin/env python3
"""Locate the earliest cross-seed identifiability loss in the C20 layer order."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


LAYER_ORDER = (
    "raw_image_global",
    "raw_text_global",
    "raw_bio_global",
    "raw_patient_anchor",
    "evidence_image_morphology",
    "evidence_text_support",
    "evidence_text_opposition",
    "evidence_text_uncertainty",
    "evidence_text_temporal",
    "evidence_bio_immune_observed",
    "evidence_bio_function_observed",
    "evidence_role_pooled",
    "mechanism_morphology_node",
    "mechanism_immune_node",
    "mechanism_function_node",
    "mechanism_opposition_node",
    "mechanism_temporal_node",
    "mechanism_nodes_all",
    "mechanism_final_representation",
    "evidence_role_logits_per_evidence",
    "evidence_role_probabilities_per_evidence",
    "aggregate_support",
    "aggregate_opposition",
    "aggregate_uncertainty",
    "aggregate_conflict",
    "scalar_support_strength",
    "scalar_opposition_strength",
    "scalar_uncertainty_strength",
    "scalar_conflict_score",
    "scalar_temporal_conflict_score",
    "scalar_morphology_alignment_cosine",
    "scalar_base_logit",
    "scalar_residual_logit",
    "scalar_final_logit",
    "scalar_final_prob",
)
STAGE_LAYERS = OrderedDict(
    [
        ("raw_modality_encoders", ("raw_image_global", "raw_text_global", "raw_bio_global", "raw_patient_anchor")),
        ("evidence_role_pooling", ("evidence_image_morphology", "evidence_text_support", "evidence_text_opposition", "evidence_text_uncertainty", "evidence_text_temporal", "evidence_bio_immune_observed", "evidence_bio_function_observed", "evidence_role_pooled")),
        ("mechanism_propagation", ("mechanism_morphology_node", "mechanism_immune_node", "mechanism_function_node", "mechanism_opposition_node", "mechanism_temporal_node", "mechanism_nodes_all", "mechanism_final_representation")),
        ("role_scoring", ("evidence_role_logits_per_evidence", "evidence_role_probabilities_per_evidence")),
        ("mechanism_aggregation", ("aggregate_support", "aggregate_opposition", "aggregate_uncertainty", "aggregate_conflict")),
        ("scalar_compression", ("scalar_support_strength", "scalar_opposition_strength", "scalar_uncertainty_strength", "scalar_conflict_score", "scalar_temporal_conflict_score", "scalar_morphology_alignment_cosine")),
        ("c17_residual", ("scalar_base_logit", "scalar_residual_logit")),
        ("final_prediction", ("scalar_final_logit", "scalar_final_prob")),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c20_dema")
    return parser.parse_args()


def path_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def minimum(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(array.min()) if array.size else float("nan")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_by_layer(rows: Sequence[Mapping[str, str]], metric: str, split: str = "val", group: str | None = None) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    for row in rows:
        if split and row.get("split") not in (split, ""):
            continue
        if group is not None and row.get("group") != group:
            continue
        layer = row.get("layer", "")
        value = as_float(row.get(metric))
        if layer and math.isfinite(value):
            result.setdefault(layer, []).append(value)
    return result


def main() -> None:
    args = parse_args()
    output_dir = path_from_root(args.output_dir)
    cka_rows = read_csv(output_dir / "c20_linear_cka_by_layer.csv")
    distance_rows = read_csv(output_dir / "c20_distance_spearman_by_layer.csv")
    knn_rows = read_csv(output_dir / "c20_knn_overlap_by_layer.csv")
    procrustes_rows = read_csv(output_dir / "c20_procrustes_generalization_by_layer.csv")
    probe_rows = read_csv(output_dir / "c20_layer_probe_summary.csv")
    if not cka_rows or not distance_rows or not knn_rows or not probe_rows:
        raise RuntimeError("C20 transition analysis requires identifiability and probe outputs")

    cka = metric_by_layer(cka_rows, "linear_cka")
    distance = metric_by_layer(distance_rows, "distance_spearman")
    knn_all = metric_by_layer(knn_rows, "mean_jaccard", group="all")
    knn_positive = metric_by_layer(knn_rows, "mean_jaccard", group="positive")
    knn_negative = metric_by_layer(knn_rows, "mean_jaccard", group="negative")
    knn_hard = metric_by_layer(knn_rows, "mean_jaccard", group="hard")
    knn_non_hard = metric_by_layer(knn_rows, "mean_jaccard", group="non_hard")
    probe = {row.get("layer", ""): row for row in probe_rows}
    procrustes = metric_by_layer(procrustes_rows, "validation_cosine_mean", split="")

    summaries: Dict[str, Dict[str, Any]] = {}
    for layer in LAYER_ORDER:
        if layer not in cka:
            continue
        probe_row = probe.get(layer, {})
        summaries[layer] = {
            "layer": layer,
            "mean_cka": mean(cka.get(layer, [])),
            "min_cka": minimum(cka.get(layer, [])),
            "mean_distance_spearman": mean(distance.get(layer, [])),
            "min_distance_spearman": minimum(distance.get(layer, [])),
            "mean_knn_jaccard": mean(knn_all.get(layer, [])),
            "mean_positive_knn_jaccard": mean(knn_positive.get(layer, [])),
            "mean_negative_knn_jaccard": mean(knn_negative.get(layer, [])),
            "mean_hard_knn_jaccard": mean(knn_hard.get(layer, [])),
            "mean_non_hard_knn_jaccard": mean(knn_non_hard.get(layer, [])),
            "procrustes_validation_cosine": mean(procrustes.get(layer, [])),
            "mean_validation_probe_auc": as_float(probe_row.get("mean_validation_probe_auc")),
            "seeds_ge_0_83": as_float(probe_row.get("seeds_ge_0_83")),
            "random_label_stable_signal": str(probe_row.get("random_label_stable_signal", "False")).lower() == "true",
        }
        item = summaries[layer]
        item.update(
            {
                "cka_mean_pass": bool(item["mean_cka"] >= 0.70),
                "distance_mean_pass": bool(item["mean_distance_spearman"] >= 0.60),
                "knn_mean_pass": bool(item["mean_knn_jaccard"] >= 0.50),
                "cka_min_pass": bool(item["min_cka"] >= 0.55),
                "distance_min_pass": bool(item["min_distance_spearman"] >= 0.45),
                "probe_mean_pass": bool(item["mean_validation_probe_auc"] >= 0.8396),
                "probe_seed_count_pass": bool(item["seeds_ge_0_83"] >= 2),
                "random_label_pass": not item["random_label_stable_signal"],
                "positive_knn_pass": bool(item["mean_positive_knn_jaccard"] >= 0.40),
                "negative_knn_pass": bool(item["mean_negative_knn_jaccard"] >= 0.40),
                "hard_knn_pass": bool(
                    math.isfinite(item["mean_hard_knn_jaccard"])
                    and math.isfinite(item["mean_non_hard_knn_jaccard"])
                    and item["mean_hard_knn_jaccard"] >= item["mean_non_hard_knn_jaccard"] * 0.60
                ),
            }
        )
        gate_keys = (
            "cka_mean_pass", "distance_mean_pass", "knn_mean_pass", "cka_min_pass", "distance_min_pass",
            "probe_mean_pass", "probe_seed_count_pass", "random_label_pass", "positive_knn_pass", "negative_knn_pass", "hard_knn_pass",
        )
        item["stable_layer_gate_pass"] = all(item[key] for key in gate_keys)

    stable_rows = list(summaries.values())
    write_rows(output_dir / "c20_stable_layer_gate.csv", stable_rows)

    stage_rows: List[Dict[str, Any]] = []
    stage_summary: Dict[str, Dict[str, float]] = {}
    for stage, stage_layers in STAGE_LAYERS.items():
        present = [summaries[layer] for layer in stage_layers if layer in summaries]
        if not present:
            continue
        metrics = {
            "mean_cka": mean(item["mean_cka"] for item in present),
            "mean_distance_spearman": mean(item["mean_distance_spearman"] for item in present),
            "mean_knn_jaccard": mean(item["mean_knn_jaccard"] for item in present),
            "mean_probe_auc": mean(item["mean_validation_probe_auc"] for item in present),
        }
        stage_summary[stage] = metrics
        stage_rows.append({"transition": stage, "transition_type": "stage_summary", "layers": ";".join(item["layer"] for item in present), **metrics})

    stage_names = list(stage_summary)
    for previous, current in zip(stage_names, stage_names[1:]):
        previous_metrics = stage_summary[previous]
        current_metrics = stage_summary[current]
        stage_rows.append(
            {
                "transition": f"{previous}_to_{current}",
                "transition_type": "adjacent_drop",
                "layers": "",
                "cka_drop": previous_metrics["mean_cka"] - current_metrics["mean_cka"],
                "distance_spearman_drop": previous_metrics["mean_distance_spearman"] - current_metrics["mean_distance_spearman"],
                "knn_overlap_drop": previous_metrics["mean_knn_jaccard"] - current_metrics["mean_knn_jaccard"],
                "probe_auc_change": current_metrics["mean_probe_auc"] - previous_metrics["mean_probe_auc"],
            }
        )
    write_rows(output_dir / "c20_instability_transition_table.csv", stage_rows)

    raw_names = [name for name in ("raw_image_global", "raw_text_global", "raw_bio_global", "raw_patient_anchor") if name in summaries]
    raw_pass = bool(raw_names) and all(
        summaries[name]["mean_cka"] >= 0.70
        and summaries[name]["mean_distance_spearman"] >= 0.60
        and summaries[name]["mean_knn_jaccard"] >= 0.50
        for name in raw_names
    )
    evidence_names = [name for name in STAGE_LAYERS["evidence_role_pooling"] if name in summaries]
    mechanism_names = [name for name in STAGE_LAYERS["mechanism_propagation"] if name in summaries]
    evidence_pass = bool(evidence_names) and mean(summaries[name]["mean_cka"] for name in evidence_names) >= 0.70 and mean(summaries[name]["mean_distance_spearman"] for name in evidence_names) >= 0.60
    mechanism_pass = bool(mechanism_names) and mean(summaries[name]["mean_cka"] for name in mechanism_names) >= 0.70 and mean(summaries[name]["mean_distance_spearman"] for name in mechanism_names) >= 0.60
    candidate_layers = {
        name
        for name in summaries
        if name.startswith("evidence_") or name.startswith("mechanism_") or name.startswith("aggregate_")
    }
    stable_layers = [item["layer"] for item in stable_rows if item["stable_layer_gate_pass"] and item["layer"] in candidate_layers]
    stable_scalar_layers = [item["layer"] for item in stable_rows if item["stable_layer_gate_pass"] and item["layer"].startswith("scalar_")]
    any_probe = any(item["probe_mean_pass"] and item["probe_seed_count_pass"] for item in stable_rows)
    if stable_layers:
        earliest = stable_layers[0]
        if not stable_scalar_layers and (earliest.startswith("mechanism_") or earliest.startswith("aggregate_")):
            decision = "C20_STABLE_MECHANISM_UNSTABLE_SCALAR_COMPRESSION"
        elif not stable_scalar_layers and earliest.startswith("evidence_"):
            decision = "C20_STABLE_ROLE_UNSTABLE_SCALAR_COMPRESSION"
        else:
            decision = "C20_STABLE_IDENTIFIABLE_CANDIDATE_LAYER"
    elif not raw_pass:
        decision = "C20_INSTABILITY_FROM_MODALITY_ENCODERS"
        earliest = next((name for name in raw_names if not summaries[name]["stable_layer_gate_pass"]), raw_names[0] if raw_names else "raw_modality_encoders")
    elif not evidence_pass:
        decision = "C20_INSTABILITY_FROM_EVIDENCE_POOLING"
        earliest = "evidence_role_pooling"
    elif not mechanism_pass:
        decision = "C20_INSTABILITY_FROM_MECHANISM_PROPAGATION"
        earliest = "mechanism_propagation"
    elif any_probe:
        decision = "C20_PREDICTIVE_BUT_NON_IDENTIFIABLE"
        earliest = "cross_seed_identifiability_gate"
    else:
        decision = "C20_NO_STABLE_MECHANISM_LAYER"
        earliest = "mechanism_propagation"

    result = {
        "decision": decision,
        "earliest_unstable_layer_or_stage": earliest,
        "stable_layers": stable_layers,
        "c21_authorized": bool(stable_layers),
        "strict_best": "DEMA_C17_POSITIVE_PRESERVATION",
    }
    (output_dir / "c20_transition_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
