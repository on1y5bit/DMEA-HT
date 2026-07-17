#!/usr/bin/env python3
"""Reproduce and attribute C64 OOF variance without training or Test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_reporting as c64  # noqa: E402
from scripts import c65a_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65a.yaml")
    return parser.parse_args()


def read_assignments(config: Mapping[str, Any]) -> Dict[str, int]:
    path = common.c64_cv_dir(config) / "fold_assignments.json"
    if not path.exists():
        raise FileNotFoundError(f"C64 fold assignments missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(patient_id): int(fold) for fold, patient_ids in payload.items() for patient_id in patient_ids}


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"Prediction probability column missing: {list(frame.columns)}")


def logit_column(frame: pd.DataFrame) -> str | None:
    for name in ("logit", "final_logit"):
        if name in frame.columns:
            return name
    return None


def value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return np.nan


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float((left & right).sum() / max(int(union.sum()), 1))


def reproduce_oof(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    report = common.c64_report_dir(config)
    fold_metrics = pd.read_csv(report / "c64_cv_metrics_by_fold.csv")
    fold_predictions = pd.read_csv(report / "c64_oof_predictions.csv", dtype={"patient_id": str})
    recorded_oof = pd.read_csv(report / "c64_oof_metrics_by_seed.csv")
    assignments = read_assignments(config)
    development = common.development_rows(config)
    expected_labels = {str(row["patient_id"]): int(row["label"]) for row in development}
    expected_ids = set(expected_labels)
    required_metric_rows = {(fold, seed) for fold in range(common.FOLD_COUNT) for seed in common.SEEDS}
    actual_metric_rows = {
        (int(row.fold), int(row.seed)) for row in fold_metrics.itertuples(index=False)
    }
    if actual_metric_rows != required_metric_rows:
        raise RuntimeError(f"C64 fold metric matrix mismatch: {actual_metric_rows}")
    actual_prediction_rows = {
        (int(row.fold), int(row.seed)) for row in fold_predictions.itertuples(index=False)
    }
    if actual_prediction_rows != required_metric_rows:
        raise RuntimeError(f"C64 fold prediction matrix mismatch: {actual_prediction_rows}")

    reproduction_rows = []
    all_fold_rows = []
    for seed in common.SEEDS:
        frame = fold_predictions[fold_predictions["seed"].astype(int) == seed].copy()
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frame = frame.sort_values("patient_id").reset_index(drop=True)
        ids_ok = set(frame["patient_id"]) == expected_ids and not frame["patient_id"].duplicated().any()
        labels_ok = ids_ok and all(expected_labels[patient_id] == int(label) for patient_id, label in zip(frame["patient_id"], frame["label"]))
        fold_ok = ids_ok and all(assignments[patient_id] == int(fold) for patient_id, fold in zip(frame["patient_id"], frame["fold"]))
        probability = frame[probability_column(frame)].to_numpy(dtype=float)
        label_array = frame["label"].to_numpy(dtype=int)
        recomputed_oof = c64.auc(label_array, probability)
        recorded_row = recorded_oof[recorded_oof["seed"].astype(int) == seed]
        if len(recorded_row) != 1:
            raise RuntimeError(f"C64 recorded OOF row missing for seed {seed}")
        recorded_auc = float(recorded_row.iloc[0]["OOF_AUC"])
        fold_deltas = []
        for fold in range(common.FOLD_COUNT):
            fold_frame = frame[frame["fold"].astype(int) == fold]
            fold_auc = c64.auc(
                fold_frame["label"].to_numpy(dtype=int),
                fold_frame[probability_column(fold_frame)].to_numpy(dtype=float),
            )
            metric_row = fold_metrics[(fold_metrics["fold"].astype(int) == fold) & (fold_metrics["seed"].astype(int) == seed)]
            recorded_fold_auc = float(metric_row.iloc[0]["AUC"])
            fold_deltas.append(abs(fold_auc - recorded_fold_auc))
            all_fold_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "AUC_recomputed": fold_auc,
                    "AUC_recorded": recorded_fold_auc,
                    "AUC_abs_delta": abs(fold_auc - recorded_fold_auc),
                    "patient_count": int(len(fold_frame)),
                }
            )
        row_pass = bool(
            ids_ok
            and labels_ok
            and fold_ok
            and len(frame) == common.DEVELOPMENT_COUNT
            and np.isfinite(probability).all()
            and abs(recomputed_oof - recorded_auc) <= 1e-10
            and max(fold_deltas) <= 1e-10
        )
        reproduction_rows.append(
            {
                "seed": seed,
                "patient_count": int(len(frame)),
                "unique_patient_count": int(frame["patient_id"].nunique()),
                "patient_id_set_pass": ids_ok,
                "label_alignment_pass": labels_ok,
                "fold_assignment_pass": fold_ok,
                "finite_prediction_pass": bool(np.isfinite(probability).all()),
                "OOF_AUC_recomputed": recomputed_oof,
                "OOF_AUC_recorded": recorded_auc,
                "OOF_AUC_abs_delta": abs(recomputed_oof - recorded_auc),
                "max_fold_AUC_abs_delta": max(fold_deltas),
                "reproduction_pass": row_pass,
            }
        )

    reproduction_frame = pd.DataFrame(reproduction_rows).sort_values("seed")
    fold_detail = pd.DataFrame(all_fold_rows).sort_values(["fold", "seed"])
    summary = {
        "status": "C65A_OOF_REPRODUCTION_PASS" if bool(reproduction_frame["reproduction_pass"].all()) else "C65A_OOF_REPRODUCTION_FAIL",
        "reproduction_pass": bool(reproduction_frame["reproduction_pass"].all()),
        "fold_integrity_pass": bool(fold_detail["patient_count"].eq(fold_detail.groupby("fold")["patient_count"].transform("first")).all()),
        "test_loaded": False,
        "development_patient_count": common.DEVELOPMENT_COUNT,
        "seeds": list(common.SEEDS),
    }
    return reproduction_frame, fold_metrics, summary


def build_epoch_analysis(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in fold_metrics.sort_values(["fold", "seed"]).to_dict(orient="records"):
        rows.append(
            {
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "AUC": float(row["AUC"]),
                "selected_epoch": int(value(row, "best_epoch", "selected_epoch")),
                "validation_bce": float(value(row, "bce_loss", "BCE", "validation_bce")),
                "sensitivity": float(value(row, "Sensitivity", "sensitivity")),
                "specificity": float(value(row, "Specificity", "specificity")),
                "balanced_accuracy": float(value(row, "Balanced_ACC", "balanced_accuracy")),
                "positive_negative_gap": float(value(row, "positive_negative_gap", "positive-negative_gap")),
                "prediction_std": float(value(row, "prediction_std")),
                "pairwise_inversions": float(value(row, "pairwise_inversion_count", "pairwise_inversions")),
            }
        )
    return pd.DataFrame(rows)


def build_auc_matrix(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = fold_metrics.pivot(index="fold", columns="seed", values="AUC").reset_index()
    frame.columns = ["fold" if str(column) == "fold" else f"seed_{int(column)}_AUC" for column in frame.columns]
    auc_columns = [f"seed_{seed}_AUC" for seed in common.SEEDS]
    frame["fold_mean_AUC"] = frame[auc_columns].mean(axis=1)
    frame["fold_std_AUC"] = frame[auc_columns].std(axis=1, ddof=1)
    frame["fold_range_AUC"] = frame[auc_columns].max(axis=1) - frame[auc_columns].min(axis=1)
    return frame.sort_values("fold").reset_index(drop=True)


def build_variance_decomposition(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    values = fold_metrics.pivot(index="fold", columns="seed", values="AUC").loc[list(range(common.FOLD_COUNT)), list(common.SEEDS)].to_numpy(dtype=float)
    grand = float(values.mean())
    fold_means = values.mean(axis=1)
    seed_means = values.mean(axis=0)
    total_ss = float(np.square(values - grand).sum())
    fold_ss = float(len(common.SEEDS) * np.square(fold_means - grand).sum())
    seed_ss = float(common.FOLD_COUNT * np.square(seed_means - grand).sum())
    interaction = values - fold_means[:, None] - seed_means[None, :] + grand
    interaction_ss = float(np.square(interaction).sum())
    components = {
        "total": total_ss,
        "fold_main": fold_ss,
        "seed_main": seed_ss,
        "seed_x_fold_interaction_residual": interaction_ss,
    }
    rows = []
    for name, ss in components.items():
        rows.append(
            {
                "component": name,
                "sum_squared_deviation": ss,
                "variance_fraction_of_total": ss / total_ss if total_ss > 0.0 else 0.0,
                "grand_mean_AUC": grand,
                "n_observations": int(values.size),
            }
        )
    rows.append(
        {
            "component": "seed_plus_interaction",
            "sum_squared_deviation": seed_ss + interaction_ss,
            "variance_fraction_of_total": (seed_ss + interaction_ss) / total_ss if total_ss > 0.0 else 0.0,
            "grand_mean_AUC": grand,
            "n_observations": int(values.size),
        }
    )
    return pd.DataFrame(rows)


def build_cross_seed_stability(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    path = common.c64_report_dir(config) / "c64_oof_predictions.csv"
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frames = {}
    for seed in common.SEEDS:
        seed_frame = frame[frame["seed"].astype(int) == seed].copy().sort_values("patient_id").reset_index(drop=True)
        frames[seed] = seed_frame
    rows = []
    for index, left_seed in enumerate(common.SEEDS):
        for right_seed in common.SEEDS[index + 1 :]:
            left = frames[left_seed]
            right = frames[right_seed]
            if not np.array_equal(left["patient_id"].to_numpy(dtype=str), right["patient_id"].to_numpy(dtype=str)):
                raise RuntimeError("C65 cross-seed patient ordering mismatch")
            if not np.array_equal(left["label"].to_numpy(dtype=int), right["label"].to_numpy(dtype=int)):
                raise RuntimeError("C65 cross-seed labels mismatch")
            labels = left["label"].to_numpy(dtype=int)
            left_prob = left[probability_column(left)].to_numpy(dtype=float)
            right_prob = right[probability_column(right)].to_numpy(dtype=float)
            left_logit_name = logit_column(left)
            right_logit_name = logit_column(right)
            left_logit = left[left_logit_name].to_numpy(dtype=float) if left_logit_name else np.log(np.clip(left_prob, 1e-7, 1 - 1e-7) / np.clip(1 - left_prob, 1e-7, 1.0))
            right_logit = right[right_logit_name].to_numpy(dtype=float) if right_logit_name else np.log(np.clip(right_prob, 1e-7, 1 - 1e-7) / np.clip(1 - right_prob, 1e-7, 1.0))
            left_errors = (left_prob >= 0.5) != labels
            right_errors = (right_prob >= 0.5) != labels
            left_inversions = c64.inversion_vector(labels, left_prob)
            right_inversions = c64.inversion_vector(labels, right_prob)
            rows.append(
                {
                    "seed_a": left_seed,
                    "seed_b": right_seed,
                    "patient_count": int(len(left)),
                    "probability_spearman": common.safe_spearman(left_prob, right_prob),
                    "logit_spearman": common.safe_spearman(left_logit, right_logit),
                    "error_count_a": int(left_errors.sum()),
                    "error_count_b": int(right_errors.sum()),
                    "error_overlap_count": int((left_errors & right_errors).sum()),
                    "error_jaccard": jaccard(left_errors, right_errors),
                    "inversion_count_a": int(left_inversions.sum()),
                    "inversion_count_b": int(right_inversions.sum()),
                    "inversion_overlap_count": int((left_inversions & right_inversions).sum()),
                    "inversion_jaccard": jaccard(left_inversions, right_inversions),
                }
            )
    stability = pd.DataFrame(rows)
    summary = {
        "mean_probability_spearman": float(stability["probability_spearman"].mean()),
        "mean_logit_spearman": float(stability["logit_spearman"].mean()),
        "mean_error_jaccard": float(stability["error_jaccard"].mean()),
        "mean_inversion_jaccard": float(stability["inversion_jaccard"].mean()),
        "test_loaded": False,
    }
    return stability, summary


def main() -> None:
    args = parse_args()
    config = common.load_c65a_config(args.config)
    output = common.report_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    reproduction, fold_metrics, reproduction_summary = reproduce_oof(config)
    epoch_analysis = build_epoch_analysis(fold_metrics)
    auc_matrix = build_auc_matrix(fold_metrics)
    variance = build_variance_decomposition(fold_metrics)
    cross_seed, cross_summary = build_cross_seed_stability(config)
    reproduction.to_csv(output / "c65a_oof_reproduction.csv", index=False)
    auc_matrix.to_csv(output / "c65a_fold_seed_auc_matrix.csv", index=False)
    variance.to_csv(output / "c65a_variance_decomposition.csv", index=False)
    epoch_analysis.to_csv(output / "c65a_selected_epoch_analysis.csv", index=False)
    cross_seed.to_csv(output / "c65a_cross_seed_prediction_stability.csv", index=False)
    common.write_json(output / "c65a_c64_variance_summary.json", {**reproduction_summary, **cross_summary, "test_loaded": False})
    print(json.dumps({"status": reproduction_summary["status"], **cross_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
