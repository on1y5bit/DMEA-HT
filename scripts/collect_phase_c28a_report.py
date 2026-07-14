#!/usr/bin/env python3
"""Collect C28-A validation-only temporal-attribution reports."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SEEDS = (0, 42, 3407)
VARIANTS = (
    "V0_official",
    "V1_uniform",
    "V2_recency_only",
    "V3_content_only",
    "V4_latest_only",
    "V5_history_mean_only",
)
MECHANISMS = ("M1", "M2", "M3", "M4", "M5")
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits_audit_only",
    "used_images_audit_only",
    "image_padding_count_audit_only",
    "has_bio_audit_only",
    "bio_missing_count_audit_only",
    "report_length_audit_only",
    "reconstructable_visit_count_audit_only",
    "visit_report_coverage_audit_only",
    "dated_bio_visit_count_audit_only",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits_audit_only", "raw_n_images_audit_only")
AUC_MINOR_LIMIT = 0.003
SENSITIVITY_MINOR_LIMIT = 0.03
SENSITIVITY_MATERIAL_DROP = 0.05
INVERSION_MINOR_LIMIT = 3
MEAN_AUC_MATERIAL_GAIN = 0.005
POSITIVE_DAMAGE_REDUCTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="analysis_reports/phase_c28a_dema")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c28a_dema")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def safe_auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return 0.0
    value = float(spearmanr(x[valid], y[valid]).statistic)
    return value if np.isfinite(value) else 0.0


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def as_bool(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes"}
    )


def classification_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    eligible = frame[as_bool(frame["available"])].copy()
    labels = eligible["label"].astype(int).to_numpy()
    probabilities = eligible["final_prob"].astype(float).to_numpy()
    predictions = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & predictions).sum())
    fn = int((positive & ~predictions).sum())
    tn = int((negative & ~predictions).sum())
    fp = int((negative & predictions).sum())
    c17_predictions = eligible["c17_predicted_class"].astype(int).to_numpy() == 1
    c17_tp_to_fn = int((positive & c17_predictions & ~predictions).sum())
    c17_fn_to_tp = int((positive & ~c17_predictions & predictions).sum())
    c17_tn_to_fp = int((negative & ~c17_predictions & predictions).sum())
    c17_fp_to_tn = int((negative & c17_predictions & ~predictions).sum())
    positive_delta = probabilities[positive] - eligible.loc[positive, "c17_prob"].astype(float).to_numpy()
    material_damage = positive & (
        (c17_predictions & ~predictions) | ((probabilities - eligible["c17_prob"].astype(float).to_numpy()) <= -0.05)
    )
    severe_damage = positive & ((probabilities - eligible["c17_prob"].astype(float).to_numpy()) <= -0.10)
    return {
        "patient_count": int(len(eligible)),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "AUC": safe_auc(labels, probabilities),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "Balanced_ACC": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "positive_probability_mean": float(probabilities[positive].mean()),
        "negative_probability_mean": float(probabilities[negative].mean()),
        "positive_negative_probability_gap": float(probabilities[positive].mean() - probabilities[negative].mean()),
        "mean_positive_probability_difference_vs_c17": float(positive_delta.mean()),
        "c17_tp_to_variant_fn": c17_tp_to_fn,
        "c17_fn_to_variant_tp": c17_fn_to_tp,
        "c17_tn_to_variant_fp": c17_tn_to_fp,
        "c17_fp_to_variant_tn": c17_fp_to_tn,
        "material_positive_damage_count": int(material_damage.sum()),
        "severe_positive_damage_count": int(severe_damage.sum()),
    }


def inversion_count(frame: pd.DataFrame) -> int:
    eligible = frame[as_bool(frame["available"])]
    positives = eligible.loc[eligible["label"].astype(int) == 1, "final_prob"].to_numpy(dtype=float)
    negatives = eligible.loc[eligible["label"].astype(int) == 0, "final_prob"].to_numpy(dtype=float)
    return int((positives[:, None] < negatives[None, :]).sum()) if positives.size and negatives.size else 0


def matched_official(predictions: pd.DataFrame, seed: int, variant_frame: pd.DataFrame) -> pd.DataFrame:
    ids = set(variant_frame.loc[as_bool(variant_frame["available"]), "patient_id"].astype(str))
    return predictions[
        (predictions["seed"].astype(int) == seed)
        & predictions["variant"].eq("V0_official")
        & predictions["patient_id"].astype(str).isin(ids)
    ].copy()


def collect_metrics(predictions: pd.DataFrame, pairwise: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            frame = predictions[
                (predictions["seed"].astype(int) == seed) & predictions["variant"].eq(variant)
            ]
            values = classification_metrics(frame)
            pairs = pairwise[
                (pairwise["seed"].astype(int) == seed)
                & pairwise["variant"].eq(variant)
                & as_bool(pairwise["available"])
            ]
            variant_inversions = int(pd.to_numeric(pairs["inversion"], errors="coerce").fillna(0).sum())
            official_inversions = int(pairs["official_inversion"].astype(int).sum())
            repaired = int(pd.to_numeric(pairs["repaired"], errors="coerce").fillna(0).sum())
            introduced = int(pd.to_numeric(pairs["introduced"], errors="coerce").fillna(0).sum())
            metric_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    **values,
                    "pairwise_eligible_count": int(len(pairs)),
                    "pairwise_inversion_count": variant_inversions,
                }
            )
            inversion_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "eligible_pairs": int(len(pairs)),
                    "official_inversions_on_eligible_pairs": official_inversions,
                    "variant_inversions": variant_inversions,
                    "net_inversion_change": variant_inversions - official_inversions,
                    "repaired": repaired,
                    "introduced": introduced,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    inversions = pd.DataFrame(inversion_rows)
    summary_rows: List[Dict[str, Any]] = []
    mean_columns = (
        "AUC",
        "Sensitivity",
        "Specificity",
        "Balanced_ACC",
        "positive_probability_mean",
        "negative_probability_mean",
        "positive_negative_probability_gap",
        "mean_positive_probability_difference_vs_c17",
        "pairwise_inversion_count",
    )
    sum_columns = (
        "c17_tp_to_variant_fn",
        "c17_fn_to_variant_tp",
        "c17_tn_to_variant_fp",
        "c17_fp_to_variant_tn",
        "material_positive_damage_count",
        "severe_positive_damage_count",
    )
    for variant, frame in metrics.groupby("variant", sort=False):
        row: Dict[str, Any] = {"variant": variant, "seed_count": len(frame)}
        for column in mean_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = safe_std(frame[column])
        for column in sum_columns:
            row[f"{column}_aggregate"] = int(frame[column].sum())
        row["patient_count_per_seed"] = "/".join(str(int(value)) for value in frame.sort_values("seed")["patient_count"])
        summary_rows.append(row)
    return metrics, pd.DataFrame(summary_rows), inversions


def visit_count_stratum(value: int) -> str:
    if value == 1:
        return "V=1"
    if value == 2:
        return "V=2"
    if value == 3:
        return "V=3"
    return "V>=4"


def normalization_audit(baseline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[int, int]]:
    frame = baseline.copy()
    frame["visit_count_stratum"] = frame["visit_count_audit_only"].astype(int).map(visit_count_stratum)
    count_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = frame[frame["seed"].astype(int) == seed]
        for stratum in ("V=1", "V=2", "V=3", "V>=4"):
            stratum_frame = seed_frame[seed_frame["visit_count_stratum"].eq(stratum)]
            for mechanism in ("ALL", *MECHANISMS):
                subset = stratum_frame if mechanism == "ALL" else stratum_frame[stratum_frame["mechanism"].eq(mechanism)]
                count_rows.append(
                    {
                        "seed": seed,
                        "visit_count_stratum": stratum,
                        "mechanism": mechanism,
                        "patient_count": int(subset["patient_id"].nunique()),
                        "patient_slot_count": int(len(subset)),
                        "actual_latest_weight_mean": float(subset["actual_latest_weight"].mean()) if len(subset) else float("nan"),
                        "baseline_latest_weight_mean": float(subset["baseline_latest_weight"].mean()) if len(subset) else float("nan"),
                        "latest_weight_excess_mean": float(subset["latest_weight_excess"].mean()) if len(subset) else float("nan"),
                        "latest_weight_ratio_mean": float(subset["latest_weight_ratio"].mean()) if len(subset) else float("nan"),
                        "latest_weight_log_ratio_mean": float(subset["latest_weight_log_ratio"].mean()) if len(subset) else float("nan"),
                    }
                )
    by_count = pd.DataFrame(count_rows)

    audit_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = frame[(frame["seed"].astype(int) == seed) & (frame["visit_count_audit_only"].astype(int) > 1)]
        patient_mean = seed_frame.groupby(["patient_id", "visit_count_audit_only"], as_index=False).agg(
            selected_n_visits_audit_only=("selected_n_visits_audit_only", "first"),
            actual_latest_weight=("actual_latest_weight", "mean"),
            latest_weight_excess=("latest_weight_excess", "mean"),
            latest_weight_log_ratio=("latest_weight_log_ratio", "mean"),
        )
        scopes = [("patient_mean_multi_visit", patient_mean)]
        scopes.extend(
            (f"{mechanism}_multi_visit", seed_frame[seed_frame["mechanism"].eq(mechanism)])
            for mechanism in MECHANISMS
        )
        for scope, subset in scopes:
            count_values = pd.to_numeric(subset["selected_n_visits_audit_only"], errors="coerce")
            audit_rows.append(
                {
                    "seed": seed,
                    "scope": scope,
                    "row_count": len(subset),
                    "raw_latest_weight_count_spearman": safe_spearman(subset["actual_latest_weight"], count_values),
                    "latest_weight_excess_count_spearman": safe_spearman(subset["latest_weight_excess"], count_values),
                    "latest_weight_log_ratio_count_spearman": safe_spearman(subset["latest_weight_log_ratio"], count_values),
                }
            )
    audit = pd.DataFrame(audit_rows)

    trend_by_seed: Dict[int, int] = {}
    for seed in SEEDS:
        ordered = by_count[
            (by_count["seed"].astype(int) == seed)
            & by_count["mechanism"].eq("ALL")
            & by_count["visit_count_stratum"].isin(("V=2", "V=3", "V>=4"))
        ].set_index("visit_count_stratum")
        if not all(name in ordered.index and int(ordered.loc[name, "patient_count"]) > 0 for name in ("V=2", "V=3", "V>=4")):
            trend_by_seed[seed] = 0
            continue
        values = [float(ordered.loc[name, "latest_weight_excess_mean"]) for name in ("V=2", "V=3", "V>=4")]
        differences = np.diff(values)
        trend_by_seed[seed] = 1 if np.all(differences > 0) else (-1 if np.all(differences < 0) else 0)

    main = audit[audit["scope"].eq("patient_mean_multi_visit")].sort_values("seed")
    raw_high = bool((main["raw_latest_weight_count_spearman"].abs() > 0.30).all())
    corrected_low = bool(
        (main["latest_weight_excess_count_spearman"].abs() <= 0.30).all()
        and (main["latest_weight_log_ratio_count_spearman"].abs() <= 0.30).all()
    )
    corrected_high_seeds = int(
        (
            main[["latest_weight_excess_count_spearman", "latest_weight_log_ratio_count_spearman"]]
            .abs()
            .max(axis=1)
            > 0.30
        ).sum()
    )
    nonzero_trends = [value for value in trend_by_seed.values() if value != 0]
    consistent_trend = len(nonzero_trends) >= 2 and len(set(nonzero_trends)) == 1
    if raw_high and corrected_low and not consistent_trend:
        decision = "C28A_ORIGINAL_TEMPORAL_COUNT_GATE_WAS_NORMALIZATION_ARTIFACT"
    elif corrected_high_seeds >= 2 and consistent_trend:
        decision = "C28A_CONTENT_SCORER_REMAINS_COUNT_ASSOCIATED"
    else:
        decision = "C28A_TEMPORAL_COUNT_ASSOCIATION_INCONCLUSIVE"
    return by_count, audit, decision, trend_by_seed


def collect_group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups = {
        "conflict_group": (
            "single_visit",
            "multi_visit_low_conflict",
            "multi_visit_medium_conflict",
            "multi_visit_high_conflict",
        ),
        "text_evidence_group": (
            "single_visit",
            "latest_positive_like_history_negative_like",
            "latest_negative_like_history_positive_like",
            "latest_history_mixed_or_uncertain",
        ),
    }
    for seed in SEEDS:
        seed_frame = predictions[predictions["seed"].astype(int) == seed]
        official = seed_frame[seed_frame["variant"].eq("V0_official")].set_index("patient_id")
        for variant in VARIANTS:
            variant_frame = seed_frame[seed_frame["variant"].eq(variant)]
            for family, names in groups.items():
                for name in names:
                    subset = variant_frame[
                        variant_frame[family].eq(name) & as_bool(variant_frame["available"])
                    ].copy()
                    if subset.empty:
                        rows.append(
                            {"seed": seed, "variant": variant, "group_family": family, "group": name, "patient_count": 0}
                        )
                        continue
                    values = classification_metrics(subset)
                    ids = subset["patient_id"].astype(str)
                    official_subset = official.loc[ids]
                    rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "group_family": family,
                            "group": name,
                            "patient_count": len(subset),
                            "positive_count": int((subset["label"].astype(int) == 1).sum()),
                            "negative_count": int((subset["label"].astype(int) == 0).sum()),
                            "validation_auc": values["AUC"],
                            "positive_recall": values["Sensitivity"],
                            "mean_probability_change_vs_official": float(
                                subset.set_index("patient_id")["final_prob"].astype(float).sub(
                                    official_subset["final_prob"].astype(float)
                                ).mean()
                            ),
                            "c17_tp_to_variant_fn": values["c17_tp_to_variant_fn"],
                            "c17_fn_to_variant_tp": values["c17_fn_to_variant_tp"],
                            "variant_inversions_within_group": inversion_count(subset),
                            "official_inversions_within_group": inversion_count(official_subset.reset_index()),
                            "strong_inference_allowed": bool(
                                len(subset) >= 10 and subset["label"].astype(int).nunique() == 2
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def shortcut_probe_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    matrix = pd.DataFrame(index=frame.index)
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if not values.dropna().empty else 0.0)
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        matrix.to_numpy(),
        frame["label"].astype(int).to_numpy(),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return safe_auc(frame["label"], probabilities)


def orientation_invariant_univariate_auc(frame: pd.DataFrame, field: str) -> float:
    values = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
    value = safe_auc(frame["label"], values)
    return max(value, 1.0 - value)


def collect_shortcut_audit(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = predictions[predictions["seed"].astype(int) == seed]
        official = seed_frame[seed_frame["variant"].eq("V0_official")].copy()
        selected_auc = shortcut_probe_auc(official, SELECTED_SHORTCUT_FIELDS)
        raw_warnings = {
            field: orientation_invariant_univariate_auc(official, field) for field in RAW_SHORTCUT_FIELDS
        }
        for variant in VARIANTS:
            subset = seed_frame[
                seed_frame["variant"].eq(variant) & as_bool(seed_frame["available"])
            ]
            correlations = {
                field: safe_spearman(subset["final_prob"], pd.to_numeric(subset[field], errors="coerce"))
                for field in SELECTED_SHORTCUT_FIELDS
            }
            max_correlation = max(abs(value) for value in correlations.values())
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "patient_count": len(subset),
                    "selected_structure_shortcut_auc": selected_auc,
                    "max_abs_prediction_selected_structure_spearman": max_correlation,
                    **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
                    **{f"{field}_orientation_invariant_label_auc_warning": value for field, value in raw_warnings.items()},
                }
            )
    audit = pd.DataFrame(rows)
    passed = bool(
        (audit["selected_structure_shortcut_auc"] <= 0.55).all()
        and (audit["max_abs_prediction_selected_structure_spearman"] <= 0.35).all()
    )
    return audit, passed


def materiality_classification(
    predictions: pd.DataFrame, inversions: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS[1:]:
            variant_frame = predictions[
                (predictions["seed"].astype(int) == seed) & predictions["variant"].eq(variant)
            ]
            variant_eligible = variant_frame[as_bool(variant_frame["available"])].copy()
            official = matched_official(predictions, seed, variant_frame)
            variant_values = classification_metrics(variant_eligible)
            official_values = classification_metrics(official)
            official_classes = official.set_index("patient_id")["predicted_class"].astype(int)
            variant_classes = variant_eligible.set_index("patient_id")["predicted_class"].astype(int)
            transition_count = int((official_classes.loc[variant_classes.index] != variant_classes).sum())
            inversion = inversions[
                (inversions["seed"].astype(int) == seed) & inversions["variant"].eq(variant)
            ].iloc[0]
            auc_change = float(variant_values["AUC"] - official_values["AUC"])
            sensitivity_change = float(variant_values["Sensitivity"] - official_values["Sensitivity"])
            inversion_change = int(inversion["net_inversion_change"])
            tp_fn_reduction = int(
                official_values["c17_tp_to_variant_fn"] - variant_values["c17_tp_to_variant_fn"]
            )
            minor = bool(
                abs(auc_change) < AUC_MINOR_LIMIT
                and abs(sensitivity_change) < SENSITIVITY_MINOR_LIMIT
                and transition_count < 2
                and abs(inversion_change) <= INVERSION_MINOR_LIMIT
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "comparator_patient_count": len(official),
                    "auc_change_vs_matched_official": auc_change,
                    "sensitivity_change_vs_matched_official": sensitivity_change,
                    "official_to_variant_threshold_transition_count": transition_count,
                    "net_inversion_change_vs_matched_official": inversion_change,
                    "official_c17_tp_to_fn": official_values["c17_tp_to_variant_fn"],
                    "variant_c17_tp_to_fn": variant_values["c17_tp_to_variant_fn"],
                    "c17_tp_to_fn_reduction": tp_fn_reduction,
                    "minor_variation": minor,
                    "materiality_label": "minor_variation" if minor else "material_change",
                    "directional_material_improvement": bool(
                        auc_change >= AUC_MINOR_LIMIT or tp_fn_reduction > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def damage_sets(predictions: pd.DataFrame, seed: int, variant: str) -> set[str]:
    frame = predictions[
        (predictions["seed"].astype(int) == seed)
        & predictions["variant"].eq(variant)
        & as_bool(predictions["available"])
        & (predictions["label"].astype(int) == 1)
    ]
    mask = (
        ((frame["c17_predicted_class"].astype(int) == 1) & (frame["predicted_class"].astype(int) == 0))
        | ((frame["final_prob"].astype(float) - frame["c17_prob"].astype(float)) <= -0.05)
    )
    return set(frame.loc[mask, "patient_id"].astype(str))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def attribution_decision(
    predictions: pd.DataFrame,
    materiality: pd.DataFrame,
    shortcut_pass: bool,
    reproduction_pass: bool,
) -> Tuple[pd.DataFrame, str, str, str, float]:
    summary_rows: List[Dict[str, Any]] = []
    generic_support: Dict[str, bool] = {}
    for variant in VARIANTS[1:]:
        frame = materiality[materiality["variant"].eq(variant)].sort_values("seed")
        mean_auc_gain = float(frame["auc_change_vs_matched_official"].mean())
        directional_seed_count = int(as_bool(frame["directional_material_improvement"]).sum())
        official_tp_fn = int(frame["official_c17_tp_to_fn"].sum())
        variant_tp_fn = int(frame["variant_c17_tp_to_fn"].sum())
        reduction_fraction = (
            (official_tp_fn - variant_tp_fn) / official_tp_fn if official_tp_fn > 0 else 0.0
        )
        mean_inversion_change = float(frame["net_inversion_change_vs_matched_official"].mean())
        minimum_sensitivity_change = float(frame["sensitivity_change_vs_matched_official"].min())
        material_threshold = bool(
            mean_auc_gain >= MEAN_AUC_MATERIAL_GAIN or reduction_fraction >= POSITIVE_DAMAGE_REDUCTION
        )
        supported = bool(
            directional_seed_count >= 2
            and material_threshold
            and mean_inversion_change <= 0.0
            and minimum_sensitivity_change >= -SENSITIVITY_MATERIAL_DROP
            and variant_tp_fn < official_tp_fn
        )
        generic_support[variant] = supported
        summary_rows.append(
            {
                "variant": variant,
                "same_direction_material_improvement_seed_count": directional_seed_count,
                "mean_auc_gain_vs_matched_official": mean_auc_gain,
                "aggregate_official_c17_tp_to_fn": official_tp_fn,
                "aggregate_variant_c17_tp_to_fn": variant_tp_fn,
                "positive_damage_reduction_fraction": reduction_fraction,
                "mean_inversion_change": mean_inversion_change,
                "minimum_sensitivity_change": minimum_sensitivity_change,
                "material_threshold_pass": material_threshold,
                "directional_safety_support": supported,
            }
        )

    overlap_values: List[float] = []
    for seed in SEEDS:
        sets = {variant: damage_sets(predictions, seed, variant) for variant in VARIANTS[1:]}
        eligible = {
            variant: set(
                predictions.loc[
                    (predictions["seed"].astype(int) == seed)
                    & predictions["variant"].eq(variant)
                    & as_bool(predictions["available"])
                    & (predictions["label"].astype(int) == 1),
                    "patient_id",
                ].astype(str)
            )
            for variant in VARIANTS[1:]
        }
        for left, right in itertools.combinations(VARIANTS[1:], 2):
            common = eligible[left] & eligible[right]
            overlap_values.append(jaccard(sets[left] & common, sets[right] & common))
    mean_damage_overlap = float(np.mean(overlap_values)) if overlap_values else 0.0

    learned_supported = generic_support["V1_uniform"] or generic_support["V2_recency_only"]
    fixed_prior_supported = bool(
        generic_support["V3_content_only"]
        and not generic_support["V2_recency_only"]
        and not generic_support["V4_latest_only"]
    )
    if learned_supported:
        primary = "C28A_LEARNED_TEMPORAL_SCORER_DAMAGE_SUPPORTED"
        chosen = "V2_recency_only" if generic_support["V2_recency_only"] else "V1_uniform"
        design = (
            "fixed recency-only temporal aggregator"
            if chosen == "V2_recency_only"
            else "fixed uniform temporal aggregator"
        )
    elif fixed_prior_supported:
        primary = "C28A_FIXED_RECENCY_PRIOR_DAMAGE_SUPPORTED"
        chosen = "V3_content_only"
        design = "content-only temporal scorer"
    elif not any(generic_support.values()) and mean_damage_overlap >= 0.75:
        primary = "C28A_TEMPORAL_AGGREGATION_NOT_PRIMARY"
        chosen = ""
        design = ""
    else:
        primary = "C28A_MIXED_OR_INCONCLUSIVE"
        chosen = ""
        design = ""

    authorization = bool(
        reproduction_pass
        and shortcut_pass
        and primary
        in {
            "C28A_LEARNED_TEMPORAL_SCORER_DAMAGE_SUPPORTED",
            "C28A_FIXED_RECENCY_PRIOR_DAMAGE_SUPPORTED",
        }
        and chosen
        and generic_support.get(chosen, False)
    )
    authorization_label = "C28B_SINGLE_MODEL_DESIGN_AUTHORIZED" if authorization else "C28B_NOT_AUTHORIZED"
    attribution = pd.DataFrame(summary_rows)
    attribution["mean_material_damage_set_jaccard_across_variants"] = mean_damage_overlap
    attribution["primary_attribution_label"] = primary
    attribution["c28b_authorization"] = authorization_label
    attribution["authorized_variant"] = chosen
    attribution["authorized_design"] = design
    return attribution, primary, authorization_label, design, mean_damage_overlap


def write_reproduction_report(reproduction: pd.DataFrame, output: Path) -> None:
    lines = [
        "# C28-A Reproduction Report",
        "",
        "The official frozen C27 validation forward was rerun independently for every seed before counterfactual analysis.",
        "",
    ]
    for row in reproduction.sort_values("seed").itertuples(index=False):
        lines.append(
            f"- seed {row.seed}: patients=`{row.patient_count}`; max logit/probability error="
            f"`{row.max_abs_logit_error:.12g}`/`{row.max_abs_probability_error:.12g}`; "
            f"latest/full-weight error=`{row.max_abs_latest_weight_error:.12g}`/"
            f"`{row.max_abs_full_temporal_weight_formula_error:.12g}`; threshold mismatches="
            f"`{row.threshold_prediction_mismatch_count}`; state unchanged=`{bool(row.checkpoint_state_unchanged)}`."
        )
    (output / "c28a_reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_normalization_report(
    audit: pd.DataFrame, decision: str, trend_by_seed: Mapping[int, int], output: Path
) -> None:
    lines = [
        "# C28-A Temporal Normalization Report",
        "",
        "The prior-only baseline is computed on the exact official temporal mask as "
        "`exp(log(2) * recency_t) / sum_j exp(log(2) * recency_j)`.",
        "",
    ]
    main = audit[audit["scope"].eq("patient_mean_multi_visit")].sort_values("seed")
    for row in main.itertuples(index=False):
        lines.append(
            f"- seed {row.seed}: raw/excess/log-ratio count Spearman=`{row.raw_latest_weight_count_spearman:.10f}`/"
            f"`{row.latest_weight_excess_count_spearman:.10f}`/`{row.latest_weight_log_ratio_count_spearman:.10f}`; "
            f"ordered multi-visit stratum trend=`{trend_by_seed[int(row.seed)]}`."
        )
    lines.extend(["", f"Normalization decision: `{decision}`"])
    (output / "c28a_temporal_normalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_positive_report(
    predictions: pd.DataFrame, response: pd.DataFrame, output: Path
) -> None:
    lines = ["# C28-A Positive-Damage Report", ""]
    for seed in SEEDS:
        official_set = damage_sets(predictions, seed, "V0_official")
        official = predictions[
            (predictions["seed"].astype(int) == seed)
            & predictions["variant"].eq("V0_official")
            & predictions["patient_id"].isin(official_set)
        ]
        severe = int(((official["final_prob"] - official["c17_prob"]) <= -0.10).sum())
        lines.append(f"- seed {seed}: material=`{len(official_set)}`; severe=`{severe}`.")
        seed_response = response[response["seed"].astype(int) == seed]
        for variant in VARIANTS[1:]:
            subset = seed_response[seed_response["variant"].eq(variant)]
            lines.append(
                f"  - {variant}: available=`{int(as_bool(subset['available']).sum())}`; "
                f"material rescues=`{int(as_bool(subset['rescued_official_material_damage']).sum())}`; "
                f"TP-to-FN rescues=`{int(as_bool(subset['rescued_official_c17_tp_to_fn']).sum())}`."
            )
    lines.extend(
        [
            "",
            "Only material probability loss or threshold transitions are interpreted; smaller patient-level changes are retained as descriptive evidence.",
        ]
    )
    (output / "c28a_positive_damage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_reports(
    metrics: pd.DataFrame,
    primary: str,
    normalization: str,
    authorization: str,
    design: str,
    shortcut_pass: bool,
    overlap: float,
    output: Path,
) -> None:
    lines = [
        "# C28-A Route Decision",
        "",
        "- analysis scope: frozen C27 validation checkpoints, seeds `[0, 42, 3407]`",
        "- counterfactual status: diagnostic only; no checkpoint or model promotion",
        f"- material-damage overlap across temporal variants: `{overlap:.10f}`",
        f"- selected-structure shortcut safety: `{shortcut_pass}`",
        "",
        "## Validation AUC By Variant",
        "",
    ]
    for variant in VARIANTS:
        frame = metrics[metrics["variant"].eq(variant)].sort_values("seed")
        values = ", ".join(f"seed {int(row.seed)} `{row.AUC:.10f}`" for row in frame.itertuples(index=False))
        lines.append(f"- {variant}: {values}; mean `{frame['AUC'].mean():.10f}`.")
    lines.extend(
        [
            "",
            f"Primary attribution: `{primary}`",
            f"Normalization: `{normalization}`",
            f"C28-B authorization: `{authorization}`",
        ]
    )
    if design:
        lines.append(f"Only authorized architecture change: `{design}`")
    else:
        lines.extend(["`KEEP_DEMA_C17_STRICT_BEST`", "`STOP_VTME_TEMPORAL_TUNING`"])
    route_text = "\n".join(lines) + "\n"
    (output / "c28a_route_decision.md").write_text(route_text, encoding="utf-8")
    final_lines = [
        "# Phase C28-A DEMA-HT Final Report",
        "",
        "C28-A is a frozen, validation-only attribution audit. It performs no fitting, threshold adjustment, model combination, or checkpoint selection.",
        "",
        *lines[4:],
        "",
        "Current strict best: `DEMA_C17_POSITIVE_PRESERVATION`",
    ]
    (output / "phase_c28a_dema_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = resolve_path(args.input_dir)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = {
        "gate": source / "c28a_runtime_gate.json",
        "reproduction": source / "c28a_reproduction_by_seed.csv",
        "baseline": source / "c28a_temporal_baseline_by_patient_slot.csv",
        "predictions": source / "c28a_counterfactual_predictions_val.csv",
        "positive": source / "c28a_positive_damage_patients.csv",
        "response": source / "c28a_positive_damage_variant_response.csv",
        "pairwise": source / "c28a_pairwise_ranking_by_variant.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"C28A analysis artifacts missing: {missing}")
    gate = json.loads(required["gate"].read_text(encoding="utf-8"))
    if not gate.get("pass", False) or gate.get("status") != "C28A_ANALYSIS_AUTHORIZED":
        raise RuntimeError("C28A reproduction/runtime gate did not authorize collection")

    reproduction = pd.read_csv(required["reproduction"])
    baseline = pd.read_csv(required["baseline"], dtype={"patient_id": str})
    predictions = pd.read_csv(required["predictions"], dtype={"patient_id": str})
    positive = pd.read_csv(required["positive"], dtype={"patient_id": str})
    response = pd.read_csv(required["response"], dtype={"patient_id": str})
    pairwise = pd.read_csv(
        required["pairwise"], dtype={"positive_patient_id": str, "negative_patient_id": str}
    )
    if set(predictions["seed"].astype(int)) != set(SEEDS) or set(predictions["variant"]) != set(VARIANTS):
        raise RuntimeError("C28A seed/variant contract failed")
    expected_prediction_rows = len(SEEDS) * len(VARIANTS) * 94
    expected_pair_rows = len(SEEDS) * len(VARIANTS) * 2209
    if len(predictions) != expected_prediction_rows or len(pairwise) != expected_pair_rows:
        raise RuntimeError("C28A prediction/pairwise row contract failed")

    metrics, metrics_summary, inversions = collect_metrics(predictions, pairwise)
    metrics.to_csv(output / "c28a_counterfactual_metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(output / "c28a_counterfactual_metrics_summary.csv", index=False)
    inversions.to_csv(output / "c28a_pairwise_inversion_summary.csv", index=False)

    by_count, normalization, normalization_label, trends = normalization_audit(baseline)
    by_count.to_csv(output / "c28a_temporal_baseline_by_visit_count.csv", index=False)
    normalization.to_csv(output / "c28a_temporal_normalization_audit.csv", index=False)
    groups = collect_group_metrics(predictions)
    groups.to_csv(output / "c28a_temporal_group_metrics.csv", index=False)
    shortcut, shortcut_pass = collect_shortcut_audit(predictions)
    shortcut.to_csv(output / "c28a_shortcut_audit.csv", index=False)
    materiality = materiality_classification(predictions, inversions)
    materiality.to_csv(output / "c28a_materiality_classification.csv", index=False)
    attribution, primary, authorization, design, overlap = attribution_decision(
        predictions, materiality, shortcut_pass, bool(gate["pass"])
    )
    attribution.to_csv(output / "c28a_variant_attribution_summary.csv", index=False)

    write_reproduction_report(reproduction, output)
    write_normalization_report(normalization, normalization_label, trends, output)
    write_positive_report(predictions, response, output)
    write_route_reports(
        metrics,
        primary,
        normalization_label,
        authorization,
        design,
        shortcut_pass,
        overlap,
        output,
    )
    decision = {
        "phase": "C28-A",
        "primary_attribution_label": primary,
        "normalization_label": normalization_label,
        "c28b_authorization": authorization,
        "authorized_design": design or None,
        "current_strict_best": "DEMA_C17_POSITIVE_PRESERVATION",
        "validation_only": True,
        "parameter_updates": False,
        "threshold_tuning": False,
        "model_combination": False,
        "all_seeds_analyzed": list(SEEDS),
        "shortcut_pass": shortcut_pass,
        "mean_material_damage_set_jaccard_across_variants": overlap,
    }
    (output / "c28a_final_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
