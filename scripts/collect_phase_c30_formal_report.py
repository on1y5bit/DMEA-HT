#!/usr/bin/env python3
"""Consolidate C30-VTCA outputs and freeze the validation-only route decision."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SEEDS = (0, 42, 3407)
MECHANISMS = ("M1", "M2", "M3", "M4", "M5")
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits", "raw_n_images")
TEXT_GROUPS = (
    ("morphology_visible", "group_morphology_visible"),
    ("diffuse_ht_like_visible", "group_diffuse_ht_like_visible"),
    ("opposition_normal_visible", "group_opposition_normal_visible"),
    ("uncertainty_visible", "group_uncertainty_visible"),
    ("latest_history_mixed", "group_latest_history_mixed"),
    ("latest_positive_like_history_negative_like", "group_latest_positive_history_negative"),
    ("latest_negative_like_history_positive_like", "group_latest_negative_history_positive"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/dema_ht_c30_vtca_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--c27-run-dir", default="runs/dema_ht_c27_vtme_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c30_dema")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path, dtype={"patient_id": str})
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = (
            int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else seed_from_name(path)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "pred_prob", "prediction", "y_prob"):
        if name in frame:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    labels_array = np.asarray(list(labels), dtype=int)
    probability_array = np.asarray(list(probabilities), dtype=float)
    if len(np.unique(labels_array)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_array, probability_array))


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if array.size else float("nan")


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.size < 2 or np.nanstd(left_array) == 0 or np.nanstd(right_array) == 0:
        return 0.0
    value = spearmanr(left_array, right_array, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else 0.0


def inversion_count(labels: np.ndarray, probabilities: np.ndarray) -> int:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def shortcut_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float:
    present = [field for field in fields if field in frame]
    if not present or frame["label"].nunique() < 2:
        return float("nan")
    matrix = pd.DataFrame(index=frame.index)
    for field in present:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if not values.dropna().empty else 0.0)
    folds = min(5, int(frame["label"].value_counts().min()))
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        matrix.to_numpy(),
        frame["label"].astype(int).to_numpy(),
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return auc(frame["label"], probabilities)


def transition_columns(
    frame: pd.DataFrame, baseline_name: str, baseline_prob: str
) -> pd.DataFrame:
    result = frame[
        ["seed", "patient_id", "label", baseline_prob, "c30_prob", "temporal_group"]
    ].copy()
    result = result.rename(columns={baseline_prob: f"{baseline_name}_prob"})
    labels = result["label"].astype(int).to_numpy()
    baseline_prediction = result[f"{baseline_name}_prob"].to_numpy(dtype=float) >= 0.5
    c30_prediction = result["c30_prob"].to_numpy(dtype=float) >= 0.5
    positive = labels == 1
    negative = labels == 0
    result[f"{baseline_name}_class"] = baseline_prediction.astype(int)
    result["c30_class"] = c30_prediction.astype(int)
    result[f"{baseline_name}_tp_to_c30_fn"] = (
        positive & baseline_prediction & ~c30_prediction
    ).astype(int)
    result[f"{baseline_name}_fn_to_c30_tp"] = (
        positive & ~baseline_prediction & c30_prediction
    ).astype(int)
    result[f"{baseline_name}_tn_to_c30_fp"] = (
        negative & ~baseline_prediction & c30_prediction
    ).astype(int)
    result[f"{baseline_name}_fp_to_c30_tn"] = (
        negative & baseline_prediction & ~c30_prediction
    ).astype(int)
    result["probability_difference"] = result["c30_prob"] - result[f"{baseline_name}_prob"]
    return result


def pairwise_table(frame: pd.DataFrame) -> pd.DataFrame:
    positives = frame[frame["label"].astype(int) == 1].sort_values("patient_id")
    negatives = frame[frame["label"].astype(int) == 0].sort_values("patient_id")
    seed = int(frame["seed"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            values: Dict[str, Any] = {
                "seed": seed,
                "positive_patient_id": positive["patient_id"],
                "negative_patient_id": negative["patient_id"],
            }
            inversions: Dict[str, bool] = {}
            for model in ("c17", "c27", "c30"):
                positive_score = float(positive[f"{model}_prob"])
                negative_score = float(negative[f"{model}_prob"])
                margin = positive_score - negative_score
                inversion = margin < 0
                values.update(
                    {
                        f"{model}_positive_score": positive_score,
                        f"{model}_negative_score": negative_score,
                        f"{model}_margin": margin,
                        f"{model}_inversion": int(inversion),
                    }
                )
                inversions[model] = inversion
            values.update(
                {
                    "c27_to_c30_repaired": int(inversions["c27"] and not inversions["c30"]),
                    "c27_to_c30_introduced": int(not inversions["c27"] and inversions["c30"]),
                    "c17_to_c30_repaired": int(inversions["c17"] and not inversions["c30"]),
                    "c17_to_c30_introduced": int(not inversions["c17"] and inversions["c30"]),
                }
            )
            rows.append(values)
    return pd.DataFrame(rows)


def group_audit(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = int(frame["seed"].iloc[0])
    for group, column in TEXT_GROUPS:
        subset = frame[frame[column].astype(bool)]
        if subset.empty:
            rows.append({"seed": seed, "group": group, "patient_count": 0})
            continue
        labels = subset["label"].to_numpy(dtype=int)
        c27_prob = subset["c27_prob"].to_numpy(dtype=float)
        c30_prob = subset["c30_prob"].to_numpy(dtype=float)
        positive = labels == 1
        negative = labels == 0
        c27_pred = c27_prob >= 0.5
        c30_pred = c30_prob >= 0.5
        rows.append(
            {
                "seed": seed,
                "group": group,
                "patient_count": int(len(subset)),
                "positive_count": int(positive.sum()),
                "negative_count": int(negative.sum()),
                "c27_auc": auc(labels, c27_prob),
                "c30_auc": auc(labels, c30_prob),
                "positive_probability_change": safe_mean((c30_prob - c27_prob)[positive]),
                "negative_probability_change": safe_mean((c30_prob - c27_prob)[negative]),
                "adapter_magnitude": float(subset["adapter_delta_abs_mean"].mean()),
                "c27_tp_to_c30_fn": int((positive & c27_pred & ~c30_pred).sum()),
                "c27_fn_to_c30_tp": int((positive & ~c27_pred & c30_pred).sum()),
                "c27_tn_to_c30_fp": int((negative & ~c27_pred & c30_pred).sum()),
                "c27_fp_to_c30_tn": int((negative & c27_pred & ~c30_pred).sum()),
                "c27_inversions": inversion_count(labels, c27_prob),
                "c30_inversions": inversion_count(labels, c30_prob),
                "inversion_change": inversion_count(labels, c30_prob)
                - inversion_count(labels, c27_prob),
            }
        )
    return rows


def median_validation_seed(comparison: pd.DataFrame) -> int:
    values = sorted(comparison["c30_auc"].astype(float).tolist())
    target = values[len(values) // 2]
    candidates = comparison[
        np.isclose(comparison["c30_auc"].astype(float), target, rtol=0.0, atol=1e-12)
    ]
    return int(candidates["seed"].astype(int).min())


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    c17_run = resolve_path(args.c17_run_dir)
    c27_run = resolve_path(args.c27_run_dir)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    c30 = read_predictions(run_dir, "val")
    c27 = read_predictions(c27_run, "val")
    c17 = read_predictions(c17_run, "val")
    for name, frame in (("C30", c30), ("C27", c27), ("C17", c17)):
        if frame.empty or set(frame["seed"].astype(int)) != set(SEEDS):
            raise RuntimeError(f"Complete {name} validation predictions are required")
    c30["c30_prob"] = c30[probability_column(c30)].astype(float)
    c30["c30_logit"] = c30["final_logit"].astype(float)
    c30["c30_class"] = (c30["c30_prob"] >= 0.5).astype(int)
    c17 = c17.rename(columns={probability_column(c17): "c17_prob"})
    c17["c17_class"] = (c17["c17_prob"].astype(float) >= 0.5).astype(int)
    c27 = c27.rename(columns={probability_column(c27): "c27_prob"})
    c27["c27_class"] = (c27["c27_prob"].astype(float) >= 0.5).astype(int)
    official_columns = [
        "patient_id",
        "seed",
        "label",
        "c27_prob",
        "c27_class",
        *[f"temporal_weight_latest_{name}" for name in MECHANISMS],
        *[f"conflict_{name}" for name in MECHANISMS],
    ]
    diagnostics = c30.merge(
        c17[["patient_id", "seed", "label", "c17_prob", "c17_class"]],
        on=["patient_id", "seed"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_c17"),
    ).merge(
        c27[official_columns],
        on=["patient_id", "seed"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_c27"),
    )
    if diagnostics[["c17_prob", "c27_prob"]].isna().any().any():
        raise RuntimeError("C17/C27/C30 validation patient alignment failed")
    if not diagnostics["label"].astype(int).eq(diagnostics["label_c17"].astype(int)).all():
        raise RuntimeError("C17/C30 labels differ")
    if not diagnostics["label"].astype(int).eq(diagnostics["label_c27"].astype(int)).all():
        raise RuntimeError("C27/C30 labels differ")
    diagnostics = diagnostics.drop(columns=["label_c17", "label_c27"])
    for name in MECHANISMS:
        diagnostics = diagnostics.rename(
            columns={
                f"temporal_weight_latest_{name}_c27": f"official_c27_temporal_weight_latest_{name}",
                f"conflict_{name}_c27": f"official_c27_conflict_{name}",
            }
        )
    diagnostics = diagnostics.sort_values(["seed", "patient_id"]).reset_index(drop=True)
    diagnostics.to_csv(output / "c30_patient_diagnostics_val.csv", index=False)

    patient_audit_columns = [
        "seed",
        "patient_id",
        "label",
        "c17_prob",
        "c27_prob",
        "c30_prob",
        "c17_class",
        "c27_class",
        "c30_class",
        "visit_count_audit_only",
        "reconstructable_visit_count_audit_only",
        "mean_adapter_delta_abs",
        "max_adapter_delta_abs",
        "latest_visit_adapter_delta_abs",
        "history_visit_adapter_delta_abs",
        "text_token_cosine_before_after",
        "text_evidence_state_cosine_before_after",
        *[column for _, column in TEXT_GROUPS],
        *[f"official_c27_temporal_weight_latest_{name}" for name in MECHANISMS],
        *[f"official_c27_conflict_{name}" for name in MECHANISMS],
        "final_logit",
        "final_prob",
        "predicted_class",
    ]
    diagnostics[patient_audit_columns].to_csv(
        output / "c30_text_adapter_patient_audit.csv", index=False
    )

    c17_transition = transition_columns(diagnostics, "c17", "c17_prob")
    c27_transition = transition_columns(diagnostics, "c27", "c27_prob")
    c17_transition.to_csv(output / "c30_c17_transition_audit.csv", index=False)
    c27_transition.to_csv(output / "c30_c27_transition_audit.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    pairwise_frames: List[pd.DataFrame] = []
    inversion_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        frame = diagnostics[diagnostics["seed"].astype(int) == seed].copy()
        if len(frame) != 94 or frame["label"].astype(int).value_counts().to_dict() != {0: 47, 1: 47}:
            raise RuntimeError(f"C30 validation split contract failed for seed {seed}")
        labels = frame["label"].to_numpy(dtype=int)
        positive = labels == 1
        negative = labels == 0
        c17_prob = frame["c17_prob"].to_numpy(dtype=float)
        c27_prob = frame["c27_prob"].to_numpy(dtype=float)
        c30_prob = frame["c30_prob"].to_numpy(dtype=float)
        c17_pred = c17_prob >= 0.5
        c27_pred = c27_prob >= 0.5
        c30_pred = c30_prob >= 0.5
        c17_auc = auc(labels, c17_prob)
        c27_auc = auc(labels, c27_prob)
        c30_auc = auc(labels, c30_prob)
        comparison_rows.append(
            {
                "seed": seed,
                "c17_auc": c17_auc,
                "c27_auc": c27_auc,
                "c30_auc": c30_auc,
                "c30_minus_c17_auc": c30_auc - c17_auc,
                "c30_minus_c27_auc": c30_auc - c27_auc,
            }
        )
        c27_material = positive & (
            (c17_pred & ~c27_pred) | ((c27_prob - c17_prob) <= -0.05)
        )
        c30_material = positive & (
            (c17_pred & ~c30_pred) | ((c30_prob - c17_prob) <= -0.05)
        )
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c30_fn": int((positive & c17_pred & ~c30_pred).sum()),
                "c17_fn_to_c30_tp": int((positive & ~c17_pred & c30_pred).sum()),
                "c17_tn_to_c30_fp": int((negative & ~c17_pred & c30_pred).sum()),
                "c17_fp_to_c30_tn": int((negative & c17_pred & ~c30_pred).sum()),
                "c27_tp_to_c30_fn": int((positive & c27_pred & ~c30_pred).sum()),
                "c27_fn_to_c30_tp": int((positive & ~c27_pred & c30_pred).sum()),
                "c27_tn_to_c30_fp": int((negative & ~c27_pred & c30_pred).sum()),
                "c27_fp_to_c30_tn": int((negative & c27_pred & ~c30_pred).sum()),
                "c17_sensitivity": float((positive & c17_pred).sum() / positive.sum()),
                "c27_sensitivity": float((positive & c27_pred).sum() / positive.sum()),
                "c30_sensitivity": float((positive & c30_pred).sum() / positive.sum()),
                "sensitivity_difference_vs_c17": float(
                    (positive & c30_pred).sum() / positive.sum()
                    - (positive & c17_pred).sum() / positive.sum()
                ),
                "c17_positive_probability_mean": float(c17_prob[positive].mean()),
                "c27_positive_probability_mean": float(c27_prob[positive].mean()),
                "c30_positive_probability_mean": float(c30_prob[positive].mean()),
                "positive_probability_difference_vs_c17": float(
                    c30_prob[positive].mean() - c17_prob[positive].mean()
                ),
                "positive_probability_difference_vs_c27": float(
                    c30_prob[positive].mean() - c27_prob[positive].mean()
                ),
                "c27_material_positive_damage_count": int(c27_material.sum()),
                "c30_material_positive_damage_count": int(c30_material.sum()),
            }
        )
        pairwise = pairwise_table(frame)
        if len(pairwise) != 2209:
            raise RuntimeError(f"C30 pairwise count failed for seed {seed}: {len(pairwise)}")
        pairwise_frames.append(pairwise)
        inversion_rows.append(
            {
                "seed": seed,
                "total_pairs": int(len(pairwise)),
                "c17_inversions": int(pairwise["c17_inversion"].sum()),
                "c27_inversions": int(pairwise["c27_inversion"].sum()),
                "c30_inversions": int(pairwise["c30_inversion"].sum()),
                "c30_minus_c17_inversions": int(
                    pairwise["c30_inversion"].sum() - pairwise["c17_inversion"].sum()
                ),
                "c30_minus_c27_inversions": int(
                    pairwise["c30_inversion"].sum() - pairwise["c27_inversion"].sum()
                ),
                "c27_to_c30_repaired": int(pairwise["c27_to_c30_repaired"].sum()),
                "c27_to_c30_introduced": int(pairwise["c27_to_c30_introduced"].sum()),
                "c17_to_c30_repaired": int(pairwise["c17_to_c30_repaired"].sum()),
                "c17_to_c30_introduced": int(pairwise["c17_to_c30_introduced"].sum()),
            }
        )
        adapter_values = frame["adapter_delta_abs_mean"].to_numpy(dtype=float)
        health_rows.append(
            {
                "seed": seed,
                "adapter_delta_abs_mean": float(adapter_values.mean()),
                "adapter_delta_abs_std": safe_std(adapter_values),
                "adapter_delta_abs_max": float(frame["adapter_delta_abs_max"].max()),
                "adapter_near_bound_fraction": float(frame["adapter_near_bound_fraction"].mean()),
                "text_token_norm_before_mean": float(frame["text_token_norm_before_mean"].mean()),
                "text_token_norm_after_mean": float(frame["text_token_norm_after_mean"].mean()),
                "text_token_cosine_before_after": float(
                    frame["text_token_cosine_before_after"].mean()
                ),
                "text_evidence_state_cosine_before_after": float(
                    frame["text_evidence_state_cosine_before_after"].mean()
                ),
                "padding_delta_abs_max": float(frame["padding_delta_abs_max"].max()),
                "latest_visit_adapter_delta_abs": float(
                    frame["latest_visit_adapter_delta_abs"].mean()
                ),
                "history_visit_adapter_delta_abs": float(
                    frame["history_visit_adapter_delta_abs"].mean()
                ),
                "adapter_output_nonzero": bool(float(frame["adapter_delta_abs_max"].max()) > 0.0),
                "adapter_variance_nonzero": bool(safe_std(adapter_values) > 0.0),
                "all_finite": bool(
                    np.isfinite(
                        frame[
                            [
                                "adapter_delta_abs_mean",
                                "adapter_delta_abs_max",
                                "adapter_near_bound_fraction",
                                "text_token_cosine_before_after",
                                "final_prob",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
            }
        )
        group_rows.extend(group_audit(frame))
        shortcut_row: Dict[str, Any] = {
            "seed": seed,
            "selected_structure_shortcut_auc": shortcut_auc(frame, SELECTED_SHORTCUT_FIELDS),
        }
        selected_correlations: List[float] = []
        for field in SELECTED_SHORTCUT_FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
            correlation = safe_spearman(frame["c30_prob"].to_numpy(dtype=float), values)
            shortcut_row[f"c30_prob_spearman_{field}"] = correlation
            selected_correlations.append(abs(correlation))
        shortcut_row["max_abs_c30_prediction_selected_structure_spearman"] = max(
            selected_correlations
        )
        for field in RAW_SHORTCUT_FIELDS:
            raw = pd.DataFrame(
                {"label": frame["label"], "value": pd.to_numeric(frame[field], errors="coerce")}
            ).dropna()
            raw_auc = auc(raw["label"], raw["value"])
            shortcut_row[f"{field}_orientation_invariant_label_auc_warning"] = max(
                raw_auc, 1.0 - raw_auc
            )
        shortcut_rows.append(shortcut_row)

    comparison = pd.DataFrame(comparison_rows)
    positive_audit = pd.DataFrame(positive_rows)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    inversions = pd.DataFrame(inversion_rows)
    health = pd.DataFrame(health_rows)
    evidence_groups = pd.DataFrame(group_rows)
    shortcuts = pd.DataFrame(shortcut_rows)
    comparison.to_csv(output / "c30_c17_c27_auc_comparison.csv", index=False)
    positive_audit.to_csv(output / "c30_positive_preservation_audit.csv", index=False)
    pairwise.to_csv(output / "c30_pairwise_ranking_val.csv", index=False)
    inversions.to_csv(output / "c30_pairwise_inversion_summary.csv", index=False)
    health.to_csv(output / "c30_adapter_health.csv", index=False)
    evidence_groups.to_csv(output / "c30_text_evidence_group_audit.csv", index=False)
    shortcuts.to_csv(output / "c30_shortcut_audit.csv", index=False)

    for source, target in (
        ("metrics_by_epoch.csv", "c30_metrics_by_epoch.csv"),
        ("metrics_by_seed.csv", "c30_metrics_by_seed.csv"),
        ("metrics_summary.csv", "c30_metrics_summary.csv"),
    ):
        pd.read_csv(run_dir / "reports" / source).to_csv(output / target, index=False)

    c17_auc = comparison["c17_auc"].to_numpy(dtype=float)
    c27_auc = comparison["c27_auc"].to_numpy(dtype=float)
    c30_auc = comparison["c30_auc"].to_numpy(dtype=float)
    c30_c17 = c30_auc - c17_auc
    c30_c27 = c30_auc - c27_auc
    c17_auc_gate = bool(
        c30_auc.mean() >= 0.8746242644
        and int((c30_c17 > 0).sum()) >= 2
        and (c30_c17 >= -0.008).all()
        and safe_std(c30_auc) <= 0.025
    )
    aggregate_c27_damage = int(positive_audit["c27_material_positive_damage_count"].sum())
    aggregate_c30_damage = int(positive_audit["c30_material_positive_damage_count"].sum())
    if aggregate_c27_damage > 0:
        material_damage_reduction = (
            aggregate_c27_damage - aggregate_c30_damage
        ) / aggregate_c27_damage
    else:
        material_damage_reduction = 0.0 if aggregate_c30_damage == 0 else -1.0
    ranking_nonworse_mean = bool(
        inversions["c30_inversions"].mean() <= inversions["c27_inversions"].mean()
    )
    c27_route_a = bool(c30_auc.mean() >= c27_auc.mean() + 0.003)
    c27_route_b = bool(
        c30_auc.mean() >= c27_auc.mean() - 0.003
        and material_damage_reduction >= 0.20
        and ranking_nonworse_mean
    )
    auc_gate = c17_auc_gate and (c27_route_a or c27_route_b)
    positive_gate = bool(
        int(positive_audit["c17_tp_to_c30_fn"].sum())
        <= int(positive_audit["c17_fn_to_c30_tp"].sum())
        and (positive_audit["sensitivity_difference_vs_c17"] >= -0.05).all()
        and (positive_audit["positive_probability_difference_vs_c17"] >= -0.03).all()
    )
    inversion_gate = bool(
        inversions["c30_inversions"].mean() <= inversions["c27_inversions"].mean() + 3
        and int(inversions["c27_to_c30_repaired"].sum())
        >= int(inversions["c27_to_c30_introduced"].sum())
        and (inversions["c30_minus_c27_inversions"] <= 10).all()
    )
    adapter_health_gate = bool(
        health["all_finite"].astype(bool).all()
        and health["adapter_output_nonzero"].astype(bool).all()
        and health["adapter_variance_nonzero"].astype(bool).all()
        and (health["padding_delta_abs_max"] == 0.0).all()
        and (health["adapter_near_bound_fraction"] <= 0.20).all()
        and (health["text_token_cosine_before_after"] >= 0.90).all()
    )
    shortcut_auc_max = float(shortcuts["selected_structure_shortcut_auc"].max())
    shortcut_corr_max = float(
        shortcuts["max_abs_c30_prediction_selected_structure_spearman"].max()
    )
    shortcut_gate = bool(
        np.isfinite(shortcut_auc_max)
        and shortcut_auc_max <= 0.55
        and shortcut_corr_max <= 0.35
    )
    training_valid = bool(
        np.isfinite(c30_auc).all()
        and np.isfinite(diagnostics["c30_prob"].to_numpy(dtype=float)).all()
        and all(diagnostics.groupby("seed")["c30_prob"].nunique() > 1)
    )
    gate_payload = json.loads(
        (output / "c30_static_synthetic_gate.json").read_text(encoding="utf-8")
    )
    path_gate = bool(
        gate_payload.get("pass", False)
        and gate_payload.get("decision") == "C30_VTCA_DIRECT_MULTI_SEED_AUTHORIZED"
    )
    initial_equivalence_gate = bool(
        all(
            item.get("pass", False)
            for item in gate_payload.get("checks", [])
            if item.get("name")
            in ("26_initial_train_logits_equal_c27", "27_initial_validation_logits_equal_c27")
        )
    )
    runtime = json.loads((run_dir / "reports" / "run_config.json").read_text(encoding="utf-8"))
    trainable_counts = [
        int(runtime["seed_runtime"][str(seed)]["trainable_parameter_count"]) for seed in SEEDS
    ]
    capacity_gate = max(trainable_counts) <= 1_000_000

    if not capacity_gate:
        decision = "DEMA_C30_CAPACITY_CONTRACT_FAIL"
    elif not initial_equivalence_gate:
        decision = "DEMA_C30_INITIAL_EQUIVALENCE_FAIL"
    elif not path_gate:
        decision = "DEMA_C30_PATH_GATE_FAIL"
    elif not training_valid:
        decision = "DEMA_C30_TRAINING_INVALID"
    elif not positive_gate:
        decision = "DEMA_C30_POSITIVE_RECALL_DAMAGE"
    elif not inversion_gate:
        decision = "DEMA_C30_INVERSION_WORSENING"
    elif not adapter_health_gate:
        decision = "DEMA_C30_ADAPTER_SATURATION"
    elif not shortcut_gate:
        decision = "DEMA_C30_SHORTCUT_SAFETY_FAIL"
    elif auc_gate:
        decision = "PROMOTE_DEMA_C30_VTCA"
    elif (
        (c30_c17 < -0.008).any()
        or safe_std(c30_auc) > 0.025
        or c30_auc.mean() < 0.8746242644
    ):
        decision = "DEMA_C30_FORMAL_FAIL_KEEP_C17"
    else:
        decision = "DEMA_C30_NO_AUC_GAIN_KEEP_C17"

    promoted = decision == "PROMOTE_DEMA_C30_VTCA"
    representative_seed = median_validation_seed(comparison) if promoted else None
    deployment_checkpoint = (
        str((run_dir / "checkpoints" / f"seed_{representative_seed}_best.pt").resolve())
        if representative_seed is not None
        else None
    )
    validation_decision = {
        "phase": "C30-VTCA",
        "decision": decision,
        "c17_mean_auc": float(c17_auc.mean()),
        "c17_std_auc": safe_std(c17_auc),
        "c27_mean_auc": float(c27_auc.mean()),
        "c27_std_auc": safe_std(c27_auc),
        "c30_mean_auc": float(c30_auc.mean()),
        "c30_std_auc": safe_std(c30_auc),
        "c17_auc_gate": c17_auc_gate,
        "c27_route_a_auc_gain": c27_route_a,
        "c27_route_b_safety_near_tie": c27_route_b,
        "auc_gate": auc_gate,
        "positive_preservation_pass": positive_gate,
        "inversion_pass": inversion_gate,
        "adapter_health_pass": adapter_health_gate,
        "shortcut_pass": shortcut_gate,
        "capacity_contract_pass": capacity_gate,
        "initial_equivalence_pass": initial_equivalence_gate,
        "path_gate_pass": path_gate,
        "training_valid": training_valid,
        "c27_material_positive_damage_count": aggregate_c27_damage,
        "c30_material_positive_damage_count": aggregate_c30_damage,
        "material_positive_damage_reduction_fraction": material_damage_reduction,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "checkpoint_averaging_used": False,
        "validation_decision_frozen_before_test": True,
        "deployment_contract": "one_checkpoint_one_model_one_forward",
        "representative_median_validation_seed": representative_seed,
        "deployment_checkpoint": deployment_checkpoint,
        "keep_c17_strict_best": not promoted,
        "stop_c30_vtca_tuning": not promoted,
    }
    decision_path = output / "c30_validation_decision.json"
    if not args.validation_only and decision_path.exists():
        frozen = json.loads(decision_path.read_text(encoding="utf-8"))
        immutable = (
            "decision",
            "c30_mean_auc",
            "c30_std_auc",
            "representative_median_validation_seed",
            "deployment_checkpoint",
        )
        if any(frozen.get(key) != validation_decision.get(key) for key in immutable):
            raise RuntimeError("C30 reporting-only collection would alter the frozen validation decision")
    decision_path.write_text(json.dumps(validation_decision, indent=2) + "\n", encoding="utf-8")

    metrics_by_seed = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    test_metrics = metrics_by_seed[metrics_by_seed["split"].eq("test")]
    if args.validation_only:
        if not test_metrics.empty:
            raise RuntimeError("Validation decision must be frozen before test metrics exist")
        test_summary = "not run; validation decision frozen first"
        test_lines = ["- reporting-only test: not run before this validation decision"]
    else:
        if set(test_metrics["seed"].astype(int)) != set(SEEDS):
            raise RuntimeError("Complete reporting-only test metrics are required")
        test_summary = (
            f"{test_metrics['AUC'].mean():.10f} +/- {safe_std(test_metrics['AUC']):.10f}"
        )
        test_lines = [
            f"- test seed {int(row.seed)}: AUC `{row.AUC:.10f}`, sensitivity `{row.Sensitivity:.10f}`, "
            f"specificity `{row.Specificity:.10f}`, balanced accuracy `{row.Balanced_ACC:.10f}`"
            for row in test_metrics.sort_values("seed").itertuples()
        ]

    selected_epochs = {
        int(row.seed): int(row.best_epoch)
        for row in metrics_by_seed[metrics_by_seed["split"].eq("val")].itertuples()
    }
    common = [
        "- official model name remains `DEMA-HT`",
        "- C17 Positive Preservation remains the strict best before C30",
        "- C27 is the strongest unpromoted new-backbone signal before C30",
        "- C30 changes only the visit-text token representation before frozen text-evidence pooling",
        "- validation AUC is the only checkpoint and promotion metric; test is reporting-only",
        "- one checkpoint, one model, one forward; predictions and checkpoint weights are not combined",
        f"- selected epochs: `{selected_epochs}`",
        f"- C17 validation AUC mean/std: `{c17_auc.mean():.10f} +/- {safe_std(c17_auc):.10f}`",
        f"- C27 validation AUC mean/std: `{c27_auc.mean():.10f} +/- {safe_std(c27_auc):.10f}`",
        f"- C30 validation AUC mean/std: `{c30_auc.mean():.10f} +/- {safe_std(c30_auc):.10f}`",
        f"- C30 minus C17 per seed: `{c30_c17.tolist()}`; mean `{c30_auc.mean() - c17_auc.mean():+.10f}`",
        f"- C30 minus C27 per seed: `{c30_c27.tolist()}`; mean `{c30_auc.mean() - c27_auc.mean():+.10f}`",
        f"- positive preservation pass: `{positive_gate}`; C17 TP->FN/FN->TP aggregate "
        f"`{int(positive_audit['c17_tp_to_c30_fn'].sum())}/{int(positive_audit['c17_fn_to_c30_tp'].sum())}`",
        f"- material positive damage C27/C30: `{aggregate_c27_damage}/{aggregate_c30_damage}`; "
        f"reduction `{material_damage_reduction:.10f}`",
        f"- inversion pass: `{inversion_gate}`; C27->C30 repaired/introduced "
        f"`{int(inversions['c27_to_c30_repaired'].sum())}/{int(inversions['c27_to_c30_introduced'].sum())}`",
        f"- adapter health pass: `{adapter_health_gate}`; near-bound max "
        f"`{health['adapter_near_bound_fraction'].max():.10f}`; token cosine min "
        f"`{health['text_token_cosine_before_after'].min():.10f}`; padding delta max "
        f"`{health['padding_delta_abs_max'].max():.10f}`",
        f"- shortcut-only AUC/max prediction correlation: `{shortcut_auc_max:.10f}`/`{shortcut_corr_max:.10f}`; pass `{shortcut_gate}`",
        f"- reporting-only test AUC mean/std: `{test_summary}`",
        f"- decision: `{decision}`",
    ]
    contract_lines = [
        "# C30 Single-Model Deployment Contract",
        "",
        *common,
        "",
        "Each seed is an independent architecture-stability replicate. Deployment loads one validation-selected checkpoint and executes one model forward pass.",
    ]
    if promoted:
        contract_lines.extend(
            [
                f"- representative median-validation seed: `{representative_seed}`",
                f"- deployment checkpoint: `{deployment_checkpoint}`",
            ]
        )
    else:
        contract_lines.append("- C30 is not promoted; the C17 strict-best deployment remains unchanged.")
    (output / "c30_single_model_deployment_contract.md").write_text(
        "\n".join(contract_lines) + "\n", encoding="utf-8"
    )
    route_lines = ["# C30 Route Decision", "", *common]
    if not promoted:
        route_lines.extend(["- `KEEP_DEMA_C17_STRICT_BEST`", "- `STOP_C30_VTCA_TUNING`"])
    (output / "c30_route_decision.md").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )
    final_lines = [
        "# Phase C30 DEMA-HT Final Report",
        "",
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`",
        "- runtime: `/home/linruixin/chen/conda/envs/ma`",
        "- direct formal seeds: `[0, 42, 3407]`",
        "- no branch, worktree, project copy, smoke, pilot, variant, sweep, fallback, or distillation",
        *common,
        *test_lines,
        "",
        decision,
    ]
    if promoted:
        final_lines.extend(
            [
                f"Representative median-validation seed: {representative_seed}",
                f"Deployment checkpoint: {deployment_checkpoint}",
            ]
        )
    else:
        final_lines.extend(["KEEP_DEMA_C17_STRICT_BEST", "STOP_C30_VTCA_TUNING"])
    (output / "phase_c30_dema_final_report.md").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation_decision))
    if args.require_pass and not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
