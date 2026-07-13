#!/usr/bin/env python3
"""Measure cross-seed identifiability of C17 internal representations."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_c20_common import (  # noqa: E402
    EXPECTED_SEEDS,
    SEED_PAIRS,
    align_pair,
    common_index,
    finite_matrix,
    jaccard_by_patient,
    linear_cka,
    load_representation_npz,
    mean_or_nan,
    normalized_cosine,
    pairwise_distances,
    pearson,
    rankdata,
    spearman,
    upper_triangle,
    auc_score,
)


ORDERED_LAYERS = (
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
    "evidence_role_logits_per_evidence",
    "evidence_role_probabilities_per_evidence",
    "mechanism_morphology_node",
    "mechanism_immune_node",
    "mechanism_function_node",
    "mechanism_opposition_node",
    "mechanism_temporal_node",
    "mechanism_nodes_all",
    "mechanism_final_representation",
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
SCALAR_LAYERS = tuple(name for name in ORDERED_LAYERS if name.startswith("scalar_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c20_dema")
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--c18-prediction-dir", default="runs/dema_ht_c18_directional_multiseed/predictions")
    parser.add_argument("--c18-hardrank-prediction-dir", default="runs/dema_ht_c18_directional_hardrank_multiseed/predictions")
    parser.add_argument("--knn-k", type=int, default=10)
    return parser.parse_args()


def path_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value_or_nan(value: float) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def read_predictions(directory: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    result: Dict[int, Dict[str, Dict[str, Any]]] = {}
    if not directory.exists():
        return result
    for seed in EXPECTED_SEEDS:
        path = directory / f"val_predictions_seed_{seed}.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            pid = str(row.get("patient_id", ""))
            if not pid:
                continue
            label = int(float(row.get("label", 0)))
            probability_key = next((key for key in ("prob", "final_prob", "prediction", "y_prob") if key in row), None)
            probability = float(row[probability_key]) if probability_key else float("nan")
            prediction_key = next((key for key in ("pred", "y_pred", "prediction_label", "hard_prediction") if key in row), None)
            prediction = int(float(row[prediction_key])) if prediction_key else int(probability >= 0.5)
            by_id[pid] = {"label": label, "prob": probability, "prediction": prediction}
        result[seed] = by_id
    return result


def c17_group_masks(val: Mapping[int, Mapping[str, Mapping[str, Any]]], patient_ids: Sequence[str], labels: np.ndarray) -> Dict[str, np.ndarray]:
    by_patient: Dict[str, List[Tuple[int, int, int]]] = {}
    for seed in EXPECTED_SEEDS:
        for patient_id, row in val.get(seed, {}).items():
            by_patient.setdefault(patient_id, []).append((seed, int(row["label"]), int(row["prediction"])))
    hard: List[bool] = []
    for patient_id, label in zip(patient_ids, labels):
        records = by_patient.get(str(patient_id), [])
        predictions = [item[2] for item in records]
        hard.append(bool(predictions) and (len(set(predictions)) > 1 or any(prediction != int(label) for prediction in predictions)))
    return {
        "positive": labels == 1,
        "negative": labels == 0,
        "hard": np.asarray(hard, dtype=bool),
        "non_hard": ~np.asarray(hard, dtype=bool),
    }


def c18_transition_groups(
    c17: Mapping[int, Mapping[str, Mapping[str, Any]]],
    c18_dirs: Sequence[Path],
    patient_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    c17_by_id: Dict[str, List[Tuple[int, int, int]]] = {}
    for seed in EXPECTED_SEEDS:
        for pid, row in c17.get(seed, {}).items():
            c17_by_id.setdefault(pid, []).append((seed, int(row["label"]), int(row["prediction"])))
    c18_by_id: Dict[str, List[Tuple[int, int, int]]] = {}
    for directory in c18_dirs:
        rows = read_predictions(directory)
        for seed in EXPECTED_SEEDS:
            for pid, row in rows.get(seed, {}).items():
                c18_by_id.setdefault(pid, []).append((seed, int(row["label"]), int(row["prediction"])))
    repaired: List[bool] = []
    introduced: List[bool] = []
    for patient_id in patient_ids:
        base_records = c17_by_id.get(str(patient_id), [])
        new_records = c18_by_id.get(str(patient_id), [])
        base_wrong = any(label != prediction for _seed, label, prediction in base_records)
        base_correct = bool(base_records) and all(label == prediction for _seed, label, prediction in base_records)
        new_correct = bool(new_records) and any(label == prediction for _seed, label, prediction in new_records)
        new_wrong = any(label != prediction for _seed, label, prediction in new_records)
        repaired.append(base_wrong and new_correct)
        introduced.append(base_correct and new_wrong)
    return {"c18_repaired": np.asarray(repaired, dtype=bool), "c18_introduced": np.asarray(introduced, dtype=bool)}


def fit_orthogonal_map(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = finite_matrix(source)
    target = finite_matrix(target)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    cross = centered_source.T @ centered_target
    left, _singular, right_transpose = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_transpose
    return source_mean, target_mean, rotation


def procrustes_metrics(source: np.ndarray, target: np.ndarray, source_mean: np.ndarray, target_mean: np.ndarray, rotation: np.ndarray) -> Dict[str, float]:
    mapped = (finite_matrix(source) - source_mean) @ rotation + target_mean
    target = finite_matrix(target)
    error = mapped - target
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "cosine_mean": float(np.nanmean(normalized_cosine(mapped, target))),
    }


def scalar_consistency(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    left = finite_matrix(left)[:, 0]
    right = finite_matrix(right)[:, 0]
    comparisons = np.sign(left[:, None] - left[None, :]) == np.sign(right[:, None] - right[None, :])
    upper = np.triu(comparisons, k=1)
    total = int(np.triu(np.ones_like(comparisons, dtype=bool), k=1).sum())
    left_direction = float(np.mean(left[labels == 1]) - np.mean(left[labels == 0])) if np.any(labels == 1) and np.any(labels == 0) else float("nan")
    right_direction = float(np.mean(right[labels == 1]) - np.mean(right[labels == 0])) if np.any(labels == 1) and np.any(labels == 0) else float("nan")
    return {
        "spearman": value_or_nan(spearman(left, right)),
        "pearson": value_or_nan(pearson(left, right)),
        "rank_consistency": float(upper.sum() / total) if total else float("nan"),
        "sign_consistency": float(np.mean(np.sign(left) == np.sign(right))),
        "label_direction_left": left_direction,
        "label_direction_right": right_direction,
        "label_direction_consistency": float(np.sign(left_direction) == np.sign(right_direction)) if math.isfinite(left_direction) and math.isfinite(right_direction) else float("nan"),
    }


def audit_shortcuts(data_by_split: Mapping[str, Mapping[int, Mapping[str, Any]]], output_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for split, data in data_by_split.items():
        for seed in EXPECTED_SEEDS:
            payload = data[seed]
            labels = np.asarray(payload["labels"], dtype=np.int64)
            for field, values in payload.get("shortcuts", {}).items():
                if field == "patient_id_encoding":
                    rows.append({
                        "split": split,
                        "seed": seed,
                        "field": field,
                        "label_auc_orientation_invariant": float("nan"),
                        "used_in_probe": False,
                        "used_in_representation": False,
                        "audit_only": True,
                        "note": "patient IDs used only for alignment; numeric encoding forbidden",
                    })
                    continue
                numeric: np.ndarray | None = None
                try:
                    numeric = np.asarray(values, dtype=np.float64)
                    if not np.isfinite(numeric).any():
                        numeric = None
                except (TypeError, ValueError):
                    numeric = None
                if numeric is None:
                    categories = {str(value): index for index, value in enumerate(sorted(set(str(item) for item in values)))}
                    numeric = np.asarray([categories[str(value)] for value in values], dtype=np.float64)
                raw_auc = auc_score(labels, numeric)
                invariant_auc = max(raw_auc, 1.0 - raw_auc) if math.isfinite(raw_auc) else float("nan")
                rows.append({
                    "split": split,
                    "seed": seed,
                    "field": field,
                    "label_auc_orientation_invariant": invariant_auc,
                    "used_in_probe": False,
                    "used_in_representation": False,
                    "audit_only": True,
                    "note": "shortcut audit only; excluded from all C20 representations and probes",
                })
    write_rows(output_dir / "c20_shortcut_exclusion_audit.csv", rows)


def main() -> None:
    args = parse_args()
    output_dir = path_from_root(args.output_dir)
    train = load_representation_npz(output_dir / "c20_internal_representations_train.npz")
    val = load_representation_npz(output_dir / "c20_internal_representations_val.npz")
    data_by_split = {"train": train, "val": val}
    audit_shortcuts(data_by_split, output_dir)
    c17_predictions = read_predictions(path_from_root(args.c17_prediction_dir))
    transition_groups = c18_transition_groups(
        c17_predictions,
        [path_from_root(args.c18_prediction_dir), path_from_root(args.c18_hardrank_prediction_dir)],
        val[0]["patient_id"],
    )
    group_masks = c17_group_masks(c17_predictions, val[0]["patient_id"], val[0]["labels"])
    group_masks.update(transition_groups)
    group_masks["all"] = np.ones(len(val[0]["patient_id"]), dtype=bool)
    group_masks["hard"] = group_masks["hard"] | group_masks["c18_repaired"] | group_masks["c18_introduced"]
    group_masks["non_hard"] = ~group_masks["hard"]

    common_layers = set(train[0]["layers"])
    for seed in EXPECTED_SEEDS:
        common_layers &= set(train[seed]["layers"]) & set(val[seed]["layers"])
    layers = [layer for layer in ORDERED_LAYERS if layer in common_layers]
    if not layers:
        raise RuntimeError("no common C20 layers found across all seeds")

    cka_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    knn_rows: List[Dict[str, Any]] = []
    procrustes_records: List[Dict[str, Any]] = []
    scalar_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    for split_name, data in data_by_split.items():
        for pair_left, pair_right in SEED_PAIRS:
            left_payload = data[pair_left]
            right_payload = data[pair_right]
            for layer in layers:
                left, right, labels, patient_ids = align_pair(left_payload, right_payload, layer)
                cka = linear_cka(left, right)
                left_dist = upper_triangle(pairwise_distances(left))
                right_dist = upper_triangle(pairwise_distances(right))
                distance = spearman(left_dist, right_dist)
                cka_rows.append({"split": split_name, "pair": f"{pair_left}_vs_{pair_right}", "layer": layer, "linear_cka": value_or_nan(cka), "n": len(patient_ids)})
                distance_rows.append({"split": split_name, "pair": f"{pair_left}_vs_{pair_right}", "layer": layer, "distance_spearman": value_or_nan(distance), "n": len(patient_ids)})
                jaccard = jaccard_by_patient(left, right, k=args.knn_k)
                if split_name == "val":
                    masks = {"all": np.ones(len(patient_ids), dtype=bool)}
                    masks.update(
                        {
                            group: np.asarray(
                                [
                                    bool(group_masks[group][np.where(val[0]["patient_id"] == patient_id)[0][0]])
                                    for patient_id in patient_ids
                                ],
                                dtype=bool,
                            )
                            for group in group_masks
                        }
                    )
                else:
                    masks = {"all": np.ones(len(labels), dtype=bool), "positive": labels == 1, "negative": labels == 0, "hard": np.zeros(len(labels), dtype=bool), "non_hard": np.ones(len(labels), dtype=bool)}
                for group, mask in masks.items():
                    selected = jaccard[mask]
                    if selected.size == 0:
                        continue
                    knn_rows.append({
                        "split": split_name,
                        "pair": f"{pair_left}_vs_{pair_right}",
                        "layer": layer,
                        "group": group,
                        "k": args.knn_k,
                        "n": int(selected.size),
                        "mean_jaccard": mean_or_nan(selected),
                        "median_jaccard": float(np.nanmedian(selected)),
                    })

                if split_name == "val":
                    for group in ("all", "positive", "negative", "hard", "non_hard", "c18_repaired", "c18_introduced"):
                        mask = group_masks[group]
                        selected = jaccard[mask]
                        if selected.size:
                            group_rows.append({
                                "pair": f"{pair_left}_vs_{pair_right}",
                                "layer": layer,
                                "group": group,
                                "n": int(selected.size),
                                "mean_knn_jaccard": mean_or_nan(selected),
                                "median_knn_jaccard": float(np.nanmedian(selected)),
                            })
                if split_name == "val" and layer in SCALAR_LAYERS:
                    scalar_rows.append({
                        "split": split_name,
                        "pair": f"{pair_left}_vs_{pair_right}",
                        "layer": layer,
                        "n": len(patient_ids),
                        **scalar_consistency(left, right, labels),
                    })

            for layer in layers:
                train_left, train_right, _train_labels, _train_ids = align_pair(train[pair_left], train[pair_right], layer)
                val_left, val_right, _val_labels, _val_ids = align_pair(val[pair_left], val[pair_right], layer)
                source_mean, target_mean, rotation = fit_orthogonal_map(train_right, train_left)
                train_metrics = procrustes_metrics(train_right, train_left, source_mean, target_mean, rotation)
                val_metrics = procrustes_metrics(val_right, val_left, source_mean, target_mean, rotation)
                procrustes_records.append({
                    "pair": f"{pair_left}_vs_{pair_right}",
                    "layer": layer,
                    "train_fit_rmse": train_metrics["rmse"],
                    "validation_rmse": val_metrics["rmse"],
                    "validation_cosine_mean": val_metrics["cosine_mean"],
                    "train_n": len(train_left),
                    "validation_n": len(val_left),
                })

    write_rows(output_dir / "c20_linear_cka_by_layer.csv", cka_rows)
    write_rows(output_dir / "c20_distance_spearman_by_layer.csv", distance_rows)
    write_rows(output_dir / "c20_knn_overlap_by_layer.csv", knn_rows)
    write_rows(output_dir / "c20_procrustes_generalization_by_layer.csv", procrustes_records)
    write_rows(output_dir / "c20_scalar_consistency_by_layer.csv", scalar_rows)
    write_rows(output_dir / "c20_group_stability_analysis.csv", group_rows)
    print(f"C20 identifiability analysis complete: {len(layers)} layers")


if __name__ == "__main__":
    main()
