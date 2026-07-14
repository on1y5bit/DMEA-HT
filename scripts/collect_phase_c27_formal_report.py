#!/usr/bin/env python3
"""Consolidate C27-VTME single-model outputs and freeze the formal decision."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c20_common import (  # noqa: E402
    jaccard_by_patient,
    linear_cka,
    pairwise_distances,
    spearman,
    upper_triangle,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/dema_ht_c27_vtme_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--c26sm-report-dir", default="analysis_reports/phase_c26sm_dema")
    parser.add_argument("--visit-design-dir", default="analysis_reports/phase_c27_visit_design")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c27_dema")
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
    frames = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path, dtype={"patient_id": str})
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else seed_from_name(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("prob", "final_prob", "pred_prob", "prediction", "y_prob"):
        if name in frame:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def auc(labels: Iterable[int], probs: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probs), dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    value = spearman(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    return float(value) if np.isfinite(value) else 0.0


def classification_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    labels = frame["label"].astype(int).to_numpy()
    probs = frame["final_prob"].to_numpy(dtype=float)
    predictions = probs >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & predictions).sum())
    fn = int((positive & ~predictions).sum())
    tn = int((negative & ~predictions).sum())
    fp = int((negative & predictions).sum())
    return {
        "patient_count": int(len(frame)),
        "validation_auc": auc(labels, probs),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "error_rate": float((predictions != labels).mean()),
    }


def shortcut_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    present = [field for field in fields if field in frame]
    if not present or frame["label"].nunique() < 2:
        return None
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


def pairwise_table(frame: pd.DataFrame) -> pd.DataFrame:
    positives = frame[frame["label"].astype(int) == 1].sort_values("patient_id")
    negatives = frame[frame["label"].astype(int) == 0].sort_values("patient_id")
    seed = int(frame["seed"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c17_margin = float(positive["c17_prob"]) - float(negative["c17_prob"])
            c27_margin = float(positive["final_prob"]) - float(negative["final_prob"])
            c17_inversion = c17_margin < 0
            c27_inversion = c27_margin < 0
            rows.append(
                {
                    "seed": seed,
                    "positive_patient_id": positive["patient_id"],
                    "negative_patient_id": negative["patient_id"],
                    "c17_positive_score": positive["c17_prob"],
                    "c17_negative_score": negative["c17_prob"],
                    "c17_margin": c17_margin,
                    "c17_inversion": int(c17_inversion),
                    "c27_positive_score": positive["final_prob"],
                    "c27_negative_score": negative["final_prob"],
                    "c27_margin": c27_margin,
                    "c27_inversion": int(c27_inversion),
                    "repaired": int(c17_inversion and not c27_inversion),
                    "introduced": int((not c17_inversion) and c27_inversion),
                    "positive_temporal_group": positive["temporal_group"],
                    "negative_temporal_group": negative["temporal_group"],
                }
            )
    return pd.DataFrame(rows)


def load_representation(run_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    path = run_dir / "representations" / f"val_patient_state_seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    required = {"patient_id", "label", "patient_state", "mechanism_states", "temporal_latest", "conflicts"}
    if not required <= set(result):
        raise RuntimeError(f"C27 representation fields missing for seed {seed}: {required - set(result)}")
    result["patient_id"] = result["patient_id"].astype(str)
    return result


def temporal_summary(frame: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    group_masks = (
        ("single_visit", frame["visit_count_audit_only"].astype(int) == 1),
        ("multi_visit", frame["visit_count_audit_only"].astype(int) > 1),
        ("latest_history_agreement", frame["temporal_group"].eq("latest_history_agreement")),
        ("latest_history_conflict", frame["temporal_group"].eq("latest_history_conflict")),
        (
            "latest_positive_like_history_negative_like",
            frame["temporal_group"].eq("latest_positive_like_history_negative_like"),
        ),
        (
            "latest_negative_like_history_positive_like",
            frame["temporal_group"].eq("latest_negative_like_history_positive_like"),
        ),
    )
    seed = int(frame["seed"].iloc[0])
    for group, mask in group_masks:
        subset = frame[mask]
        if subset.empty:
            rows.append({"row_type": "behavior_group", "seed": seed, "group": group, "patient_count": 0})
            continue
        values = classification_metrics(subset)
        values.update(
            {
                "row_type": "behavior_group",
                "seed": seed,
                "group": group,
                "mean_latest_temporal_weight": float(subset["mean_temporal_weight_latest"].mean()),
                "mean_conflict": float(subset[[f"conflict_{name}" for name in MECHANISMS]].to_numpy(dtype=float).mean()),
            }
        )
        rows.append(values)

    multi = frame[frame["visit_count_audit_only"].astype(int) > 1]
    conflicts = multi[[f"conflict_{name}" for name in MECHANISMS]].to_numpy(dtype=float)
    latest_columns = [f"temporal_weight_latest_{name}" for name in MECHANISMS]
    latest = multi[latest_columns].to_numpy(dtype=float)
    count_values = pd.to_numeric(multi["selected_n_visits"], errors="coerce").fillna(0.0).to_numpy()
    mean_latest = latest.mean(axis=1) if len(multi) else np.asarray([], dtype=float)
    rows.append(
        {
            "row_type": "health",
            "seed": seed,
            "group": "multi_visit",
            "patient_count": int(len(multi)),
            "fraction_latest_weight_above_0_90": float((latest > 0.90).mean()) if latest.size else 0.0,
            "mean_normalized_temporal_entropy": float(multi["mean_normalized_temporal_entropy"].mean()),
            "latest_weight_selected_n_visits_spearman": safe_spearman(mean_latest, count_values),
            "fraction_uniform_temporal_weight": float(multi["fraction_uniform_temporal_weight"].mean()),
            "temporal_weights_finite": bool(np.isfinite(latest).all()),
            "conflicts_finite": bool(np.isfinite(conflicts).all()),
            "conflicts_nonconstant": bool(conflicts.size and float(conflicts.std(ddof=1)) > 1e-6),
        }
    )
    for mechanism in MECHANISMS:
        values = pd.to_numeric(frame[f"conflict_{mechanism}"], errors="coerce").to_numpy(dtype=float)
        available = frame[f"history_available_{mechanism}"].astype(bool).to_numpy()
        conflict_rows.append(
            {
                "seed": seed,
                "mechanism": mechanism,
                "history_available_count": int(available.sum()),
                "mean_conflict": float(values.mean()),
                "std_conflict": safe_std(values),
                "min_conflict": float(values.min()),
                "max_conflict": float(values.max()),
                "finite": bool(np.isfinite(values).all()),
                "nonconstant": bool(float(values.std(ddof=1)) > 1e-6),
            }
        )
    return rows, conflict_rows


def median_validation_seed(comparison: pd.DataFrame) -> int:
    values = sorted(comparison["c27_auc"].astype(float).tolist())
    target = values[len(values) // 2]
    candidates = comparison[np.isclose(comparison["c27_auc"].astype(float), target, rtol=0.0, atol=1e-12)]
    return int(candidates["seed"].astype(int).min())


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    c17_run = resolve_path(args.c17_run_dir)
    c26sm_report = resolve_path(args.c26sm_report_dir)
    visit_design = resolve_path(args.visit_design_dir)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    c27 = read_predictions(run_dir, "val")
    c17 = read_predictions(c17_run, "val")
    if c27.empty or c17.empty or set(c27["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Complete C27 and C17 validation predictions are required")
    if "base_prob" not in c17:
        raise RuntimeError("C17 validation predictions must retain the C13 base probability")
    c17 = c17.rename(columns={probability_column(c17): "c17_prob"})
    diagnostics = c27.merge(
        c17[["patient_id", "seed", "label", "base_prob", "c17_prob"]],
        on=["patient_id", "seed"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_c17"),
    )
    if diagnostics["c17_prob"].isna().any() or not diagnostics["label"].astype(int).eq(
        diagnostics["label_c17"].astype(int)
    ).all():
        raise RuntimeError("C17/C27 validation patient alignment failed")
    diagnostics = diagnostics.sort_values(["seed", "patient_id"]).reset_index(drop=True)
    diagnostics.to_csv(output / "c27_patient_diagnostics_val.csv", index=False)

    transition_frames: List[pd.DataFrame] = []
    positive_rows: List[Dict[str, Any]] = []
    pair_frames: List[pd.DataFrame] = []
    inversion_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []

    c26_metrics = pd.read_csv(c26sm_report / "c26sm_metrics_by_seed.csv")
    c26_val = c26_metrics[c26_metrics["split"].eq("val")].set_index(c26_metrics[c26_metrics["split"].eq("val")]["seed"].astype(int))
    for seed in SEEDS:
        frame = diagnostics[diagnostics["seed"].astype(int) == seed].copy()
        if len(frame) != 94 or frame["label"].value_counts().to_dict() != {0: 47, 1: 47}:
            raise RuntimeError(f"C27 validation split contract failed for seed {seed}")
        labels = frame["label"].astype(int).to_numpy()
        c13_prob = frame["base_prob"].to_numpy(dtype=float)
        c17_prob = frame["c17_prob"].to_numpy(dtype=float)
        c27_prob = frame["final_prob"].to_numpy(dtype=float)
        c17_pred = c17_prob >= 0.5
        c27_pred = c27_prob >= 0.5
        positive = labels == 1
        negative = labels == 0
        c17_auc = auc(labels, c17_prob)
        c27_auc = auc(labels, c27_prob)
        comparison_rows.append(
            {
                "seed": seed,
                "c13_auc": auc(labels, c13_prob),
                "c17_auc": c17_auc,
                "c26sm_auc": float(c26_val.loc[seed, "AUC"]),
                "c27_auc": c27_auc,
                "c27_minus_c17_auc": c27_auc - c17_auc,
                "c27_minus_c26sm_auc": c27_auc - float(c26_val.loc[seed, "AUC"]),
            }
        )
        tp_to_fn = int((positive & c17_pred & ~c27_pred).sum())
        fn_to_tp = int((positive & ~c17_pred & c27_pred).sum())
        tn_to_fp = int((negative & ~c17_pred & c27_pred).sum())
        fp_to_tn = int((negative & c17_pred & ~c27_pred).sum())
        c17_sensitivity = float((positive & c17_pred).sum() / max(positive.sum(), 1))
        c27_sensitivity = float((positive & c27_pred).sum() / max(positive.sum(), 1))
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c27_fn": tp_to_fn,
                "c17_fn_to_c27_tp": fn_to_tp,
                "c17_tn_to_c27_fp": tn_to_fp,
                "c17_fp_to_c27_tn": fp_to_tn,
                "c17_sensitivity": c17_sensitivity,
                "c27_sensitivity": c27_sensitivity,
                "sensitivity_difference_vs_c17": c27_sensitivity - c17_sensitivity,
                "c17_positive_probability_mean": float(c17_prob[positive].mean()),
                "c27_positive_probability_mean": float(c27_prob[positive].mean()),
                "positive_probability_difference_vs_c17": float(c27_prob[positive].mean() - c17_prob[positive].mean()),
            }
        )
        transition = frame[["patient_id", "seed", "label", "base_prob", "c17_prob", "final_prob", "temporal_group"]].copy()
        transition["c17_prediction"] = c17_pred.astype(int)
        transition["c27_prediction"] = c27_pred.astype(int)
        transition["c17_tp_to_c27_fn"] = (positive & c17_pred & ~c27_pred).astype(int)
        transition["c17_fn_to_c27_tp"] = (positive & ~c17_pred & c27_pred).astype(int)
        transition["c17_tn_to_c27_fp"] = (negative & ~c17_pred & c27_pred).astype(int)
        transition["c17_fp_to_c27_tn"] = (negative & c17_pred & ~c27_pred).astype(int)
        transition_frames.append(transition)
        pairs = pairwise_table(frame)
        if len(pairs) != 2209:
            raise RuntimeError(f"C27 pairwise count failed for seed {seed}: {len(pairs)}")
        pair_frames.append(pairs)
        c17_inversions = int(pairs["c17_inversion"].sum())
        c27_inversions = int(pairs["c27_inversion"].sum())
        inversion_rows.append(
            {
                "seed": seed,
                "total_pairs": int(len(pairs)),
                "c17_inversions": c17_inversions,
                "c27_inversions": c27_inversions,
                "net_change": c27_inversions - c17_inversions,
                "repaired": int(pairs["repaired"].sum()),
                "introduced": int(pairs["introduced"].sum()),
            }
        )
        seed_temporal, seed_conflicts = temporal_summary(frame)
        temporal_rows.extend(seed_temporal)
        conflict_rows.extend(seed_conflicts)

        shortcut_row: Dict[str, Any] = {
            "seed": seed,
            "selected_structure_shortcut_auc": shortcut_auc(frame, SELECTED_SHORTCUT_FIELDS),
        }
        selected_correlations: List[float] = []
        for field in SELECTED_SHORTCUT_FIELDS:
            correlation = frame["final_prob"].corr(pd.to_numeric(frame[field], errors="coerce"), method="spearman")
            correlation = float(correlation) if pd.notna(correlation) else 0.0
            shortcut_row[f"final_prob_spearman_{field}"] = correlation
            selected_correlations.append(abs(correlation))
        shortcut_row["max_abs_final_prob_selected_structure_spearman"] = max(selected_correlations)
        for field in RAW_SHORTCUT_FIELDS:
            raw = pd.DataFrame(
                {"label": frame["label"], "value": pd.to_numeric(frame[field], errors="coerce")}
            ).dropna()
            raw_value = auc(raw["label"], raw["value"])
            shortcut_row[f"{field}_orientation_invariant_label_auc_warning"] = max(raw_value, 1.0 - raw_value)
        shortcut_rows.append(shortcut_row)

    comparison = pd.DataFrame(comparison_rows)
    positive_audit = pd.DataFrame(positive_rows)
    transitions = pd.concat(transition_frames, ignore_index=True)
    pairwise = pd.concat(pair_frames, ignore_index=True)
    inversions = pd.DataFrame(inversion_rows)
    temporal = pd.DataFrame(temporal_rows)
    conflicts = pd.DataFrame(conflict_rows)
    shortcuts = pd.DataFrame(shortcut_rows)

    transition_columns = [
        "patient_id", "seed", "label", "base_prob", "c17_prob", "final_prob", "temporal_group",
        "c17_prediction", "c27_prediction", "c17_tp_to_c27_fn", "c17_fn_to_c27_tp",
        "c17_tn_to_c27_fp", "c17_fp_to_c27_tn",
    ]
    transitions[transition_columns].to_csv(output / "c27_c17_transition_audit.csv", index=False)
    positive_audit.to_csv(output / "c27_positive_preservation_audit.csv", index=False)
    pairwise.to_csv(output / "c27_pairwise_ranking_val.csv", index=False)
    inversions.to_csv(output / "c27_pairwise_inversion_summary.csv", index=False)
    temporal.to_csv(output / "c27_temporal_weight_summary.csv", index=False)
    conflicts.to_csv(output / "c27_temporal_conflict_audit.csv", index=False)
    shortcuts.to_csv(output / "c27_shortcut_audit.csv", index=False)
    comparison.to_csv(output / "c27_c13_c17_c26sm_comparison.csv", index=False)

    temporal_columns = [
        "seed", "patient_id", "label", "visit_count_audit_only", "reconstructable_visit_count_audit_only",
        "visit_report_coverage_audit_only", "temporal_group", "mean_temporal_weight_latest",
        "mean_temporal_weight_history", "mean_temporal_weight_entropy", "mean_normalized_temporal_entropy",
        "fraction_latest_weight_above_0_90", "fraction_uniform_temporal_weight",
        *[f"temporal_weight_latest_{name}" for name in MECHANISMS],
        *[f"conflict_{name}" for name in MECHANISMS],
    ]
    diagnostics[temporal_columns].to_csv(output / "c27_temporal_weight_patient_audit.csv", index=False)
    alignment_columns = [
        "seed", "patient_id", "label", "same_visit_alignment_count", "cross_visit_alignment_pair_count",
        "same_visit_image_text_cosine", "cross_visit_image_text_cosine", "latest_same_visit_alignment",
        "history_same_visit_alignment",
    ]
    diagnostics[diagnostics["same_visit_alignment_count"].astype(int) > 0][alignment_columns].to_csv(
        output / "c27_same_visit_alignment_audit.csv", index=False
    )

    representations = {seed: load_representation(run_dir, seed) for seed in SEEDS}
    stability_rows: List[Dict[str, Any]] = []
    for left, right in ((0, 42), (0, 3407), (42, 3407)):
        left_rep = representations[left]
        right_rep = representations[right]
        if not np.array_equal(left_rep["patient_id"], right_rep["patient_id"]) or not np.array_equal(
            left_rep["label"], right_rep["label"]
        ):
            raise RuntimeError(f"C27 representation alignment failed for {left}/{right}")
        left_state = left_rep["patient_state"].astype(np.float64)
        right_state = right_rep["patient_state"].astype(np.float64)
        left_frame = diagnostics[diagnostics["seed"].astype(int) == left].sort_values("patient_id")
        right_frame = diagnostics[diagnostics["seed"].astype(int) == right].sort_values("patient_id")
        left_norms = np.linalg.norm(left_rep["mechanism_states"].astype(np.float64), axis=-1)
        right_norms = np.linalg.norm(right_rep["mechanism_states"].astype(np.float64), axis=-1)
        stability_rows.append(
            {
                "seed_left": left,
                "seed_right": right,
                "patient_state_linear_cka": linear_cka(left_state, right_state),
                "patient_state_distance_spearman": spearman(
                    upper_triangle(pairwise_distances(left_state)),
                    upper_triangle(pairwise_distances(right_state)),
                ),
                "patient_state_knn_jaccard": float(np.nanmean(jaccard_by_patient(left_state, right_state, k=10))),
                "final_probability_spearman": safe_spearman(left_frame["final_prob"], right_frame["final_prob"]),
                "mechanism_state_norm_spearman": safe_spearman(left_norms.reshape(-1), right_norms.reshape(-1)),
                "temporal_weight_spearman": safe_spearman(
                    left_rep["temporal_latest"].reshape(-1), right_rep["temporal_latest"].reshape(-1)
                ),
                "conflict_scalar_spearman": safe_spearman(
                    left_rep["conflicts"].reshape(-1), right_rep["conflicts"].reshape(-1)
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "c27_cross_seed_representation_stability.csv", index=False)

    for source, target in (
        ("metrics_by_epoch.csv", "c27_metrics_by_epoch.csv"),
        ("metrics_by_seed.csv", "c27_metrics_by_seed.csv"),
        ("metrics_summary.csv", "c27_metrics_summary.csv"),
    ):
        pd.read_csv(run_dir / "reports" / source).to_csv(output / target, index=False)

    c17_auc = comparison["c17_auc"].to_numpy(dtype=float)
    c27_auc = comparison["c27_auc"].to_numpy(dtype=float)
    auc_difference = c27_auc - c17_auc
    auc_gate = bool(
        c27_auc.mean() > c17_auc.mean()
        and int((auc_difference > 0).sum()) >= 2
        and (auc_difference >= -0.005).all()
        and safe_std(c27_auc) <= 0.02
    )
    positive_gate = bool(
        int(positive_audit["c17_tp_to_c27_fn"].sum()) <= int(positive_audit["c17_fn_to_c27_tp"].sum())
        and (positive_audit["sensitivity_difference_vs_c17"] >= -0.05).all()
        and (positive_audit["positive_probability_difference_vs_c17"] >= -0.03).all()
    )
    inversion_gate = bool(
        int((inversions["net_change"] < 0).sum()) >= 2
        and int(inversions["repaired"].sum()) > int(inversions["introduced"].sum())
        and (inversions["net_change"] <= 3).all()
    )
    health = temporal[temporal["row_type"].eq("health")]
    temporal_gate = bool(
        (health["fraction_latest_weight_above_0_90"] <= 0.80).all()
        and (health["mean_normalized_temporal_entropy"] >= 0.20).all()
        and (health["latest_weight_selected_n_visits_spearman"].abs() <= 0.30).all()
        and (health["fraction_uniform_temporal_weight"] < 1.0).all()
        and health["temporal_weights_finite"].astype(bool).all()
        and health["conflicts_finite"].astype(bool).all()
        and health["conflicts_nonconstant"].astype(bool).all()
    )
    stability_means = {
        column: float(stability[column].mean())
        for column in ("patient_state_linear_cka", "patient_state_distance_spearman", "patient_state_knn_jaccard")
    }
    stability_gate = bool(
        stability_means["patient_state_linear_cka"] >= 0.55
        and stability_means["patient_state_distance_spearman"] >= 0.55
        and stability_means["patient_state_knn_jaccard"] >= 0.40
    )
    shortcut_auc_max = float(pd.to_numeric(shortcuts["selected_structure_shortcut_auc"], errors="coerce").max())
    shortcut_corr_max = float(shortcuts["max_abs_final_prob_selected_structure_spearman"].max())
    shortcut_gate = bool(np.isfinite(shortcut_auc_max) and shortcut_auc_max <= 0.55 and shortcut_corr_max <= 0.35)
    training_valid = bool(
        np.isfinite(c27_auc).all()
        and all(diagnostics.groupby("seed")["final_prob"].nunique() > 1)
        and np.isfinite(diagnostics["final_prob"].to_numpy(dtype=float)).all()
    )
    visit_payload = json.loads((visit_design / "c27_visit_reconstruction_decision.json").read_text(encoding="utf-8"))
    visit_gate = visit_payload.get("decision") == "C27_VISIT_RECONSTRUCTION_PASS"
    static_payload = json.loads((output / "c27_static_synthetic_gate.json").read_text(encoding="utf-8"))
    path_gate = bool(static_payload.get("pass", False))
    runtime = json.loads((run_dir / "reports" / "run_config.json").read_text(encoding="utf-8"))
    trainable_counts = [int(values["trainable"]) for values in runtime["parameter_counts_by_seed"].values()]
    capacity_gate = max(trainable_counts) <= 1_000_000

    if not visit_gate:
        decision = "DEMA_C27_VISIT_RECONSTRUCTION_INSUFFICIENT"
    elif not capacity_gate:
        decision = "DEMA_C27_CAPACITY_CONTRACT_FAIL"
    elif not path_gate:
        decision = "DEMA_C27_PATH_GATE_FAIL"
    elif not training_valid:
        decision = "DEMA_C27_TRAINING_INVALID"
    elif not positive_gate:
        decision = "DEMA_C27_POSITIVE_RECALL_DAMAGE"
    elif not inversion_gate:
        decision = "DEMA_C27_INVERSION_WORSENING"
    elif not temporal_gate:
        decision = "DEMA_C27_TEMPORAL_COLLAPSE"
    elif not stability_gate:
        decision = "DEMA_C27_CROSS_SEED_INSTABILITY"
    elif not shortcut_gate:
        decision = "DEMA_C27_SHORTCUT_SAFETY_FAIL"
    elif (auc_difference < -0.005).any() or safe_std(c27_auc) > 0.02:
        decision = "DEMA_C27_FORMAL_FAIL_KEEP_C17"
    elif auc_gate:
        decision = "PROMOTE_DEMA_C27_VTME"
    else:
        decision = "DEMA_C27_NO_AUC_GAIN_KEEP_C17"

    promoted = decision == "PROMOTE_DEMA_C27_VTME"
    representative_seed = median_validation_seed(comparison) if promoted else None
    deployment_checkpoint = (
        str((run_dir / "checkpoints" / f"seed_{representative_seed}_best.pt").resolve())
        if representative_seed is not None
        else None
    )
    metrics_by_seed = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    test_metrics = metrics_by_seed[metrics_by_seed["split"].eq("test")]
    if args.validation_only:
        test_summary = "not run; validation decision frozen first"
    elif set(test_metrics["seed"].astype(int)) == set(SEEDS):
        test_summary = f"{test_metrics['AUC'].mean():.10f} +/- {safe_std(test_metrics['AUC']):.10f}"
    else:
        raise RuntimeError("Complete reporting-only test metrics are required for final collection")

    common = [
        "- C26-E status: `C26E_WITHDRAWN_BY_USER`; no ensemble artifact exists",
        "- C26-SM status: `STOP_C26SM_TUNING`",
        "- deployment contract: one checkpoint, one model, one forward",
        "- checkpoint selection and route promotion: validation AUC only; test reporting-only",
        f"- visit reconstruction: `{visit_payload['decision']}`; report coverage=`{visit_payload['visit_report_coverage']:.10f}`",
        f"- C17 validation AUC mean/std: `{c17_auc.mean():.10f} +/- {safe_std(c17_auc):.10f}`",
        f"- C26-SM validation AUC mean/std: `{comparison['c26sm_auc'].mean():.10f} +/- {safe_std(comparison['c26sm_auc']):.10f}`",
        f"- C27 validation AUC mean/std: `{c27_auc.mean():.10f} +/- {safe_std(c27_auc):.10f}`",
        f"- C27 minus C17 mean: `{c27_auc.mean() - c17_auc.mean():+.10f}`; AUC gate=`{auc_gate}`",
        f"- positive preservation: `{positive_gate}`; TP->FN/FN->TP=`{int(positive_audit['c17_tp_to_c27_fn'].sum())}/{int(positive_audit['c17_fn_to_c27_tp'].sum())}`",
        f"- inversion gate: `{inversion_gate}`; repaired/introduced=`{int(inversions['repaired'].sum())}/{int(inversions['introduced'].sum())}`",
        f"- temporal health: `{temporal_gate}`",
        f"- stability means: CKA=`{stability_means['patient_state_linear_cka']:.10f}`, distance Spearman=`{stability_means['patient_state_distance_spearman']:.10f}`, kNN Jaccard=`{stability_means['patient_state_knn_jaccard']:.10f}`; pass=`{stability_gate}`",
        f"- selected-structure shortcut-only AUC/max prediction correlation: `{shortcut_auc_max:.10f}`/`{shortcut_corr_max:.10f}`; pass=`{shortcut_gate}`",
        f"- reporting-only test AUC mean/std: `{test_summary}`",
        f"- decision: `{decision}`",
    ]
    (output / "c27_cross_seed_stability_report.md").write_text(
        "# C27 Cross-Seed Stability\n\n" + "\n".join(common) + "\n", encoding="utf-8"
    )
    (output / "c27_residual_or_posthoc_absence_audit.md").write_text(
        "# C27 Residual and Post-Hoc Absence Audit\n\n"
        "C27 produces its final logit directly from one patient-state projection and one classifier. "
        "It does not read C13/C17 logits, does not add a post-C17 correction, and does not load or combine multiple checkpoints.\n",
        encoding="utf-8",
    )
    (output / "c27_single_model_deployment_contract.md").write_text(
        "# C27 Single-Model Deployment Contract\n\n"
        "Each seed is an independent architecture-stability replicate. Inference loads exactly one checkpoint, "
        "one model, and performs one forward pass. No predictions or checkpoint weights are combined.\n\n"
        + "\n".join(common)
        + (f"\n- representative median-validation seed: `{representative_seed}`\n- deployment checkpoint: `{deployment_checkpoint}`\n" if promoted else "\n- C27 was not promoted; the C17 strict-best deployment remains unchanged.\n"),
        encoding="utf-8",
    )
    route_lines = ["# C27 Route Decision", "", *common]
    if not promoted:
        route_lines.extend(["- `KEEP_DEMA_C17_STRICT_BEST`", "- `STOP_C27_VTME_TUNING`"])
    (output / "c27_route_decision.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    (output / "phase_c27_dema_final_report.md").write_text(
        "# Phase C27 DEMA-HT Final Report\n\n"
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`\n"
        "- runtime: `/home/linruixin/chen/conda/envs/ma`\n"
        "- single-model route only; no ensemble or checkpoint averaging\n"
        + "\n".join(common)
        + "\n"
        + (f"\nPROMOTE_DEMA_C27_VTME\nRepresentative seed: {representative_seed}\n" if promoted else "\nKEEP_DEMA_C17_STRICT_BEST\nSTOP_C27_VTME_TUNING\n"),
        encoding="utf-8",
    )
    payload = {
        "phase": "C27-VTME",
        "decision": decision,
        "c17_mean_auc": float(c17_auc.mean()),
        "c27_mean_auc": float(c27_auc.mean()),
        "c27_std_auc": safe_std(c27_auc),
        "auc_gate": auc_gate,
        "positive_preservation_pass": positive_gate,
        "inversion_pass": inversion_gate,
        "temporal_health_pass": temporal_gate,
        "cross_seed_stability_pass": stability_gate,
        "shortcut_pass": shortcut_gate,
        "visit_reconstruction_pass": visit_gate,
        "capacity_contract_pass": capacity_gate,
        "path_gate_pass": path_gate,
        "training_valid": training_valid,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "checkpoint_averaging_used": False,
        "deployment_contract": "one_checkpoint_one_model_one_forward",
        "validation_decision_frozen_before_test": True,
        "representative_median_validation_seed": representative_seed,
        "deployment_checkpoint": deployment_checkpoint,
        "keep_c17_strict_best": not promoted,
        "stop_c27_vtme_tuning": not promoted,
    }
    (output / "c27_final_decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if args.require_pass and not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
