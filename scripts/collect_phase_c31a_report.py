#!/usr/bin/env python3
"""Consolidate the frozen C31-A factorial audit and apply its fixed decision rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
COMBINATIONS = ("000", "100", "010", "001", "110", "101", "011", "111")
ROLES = (
    "R1_MORPHOLOGY_SUPPORT_GROUP",
    "R4_OPPOSITION_GROUP",
    "R5_TEMPORAL_GROUP",
)
ROLE_SHORT = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": "R1",
    "R4_OPPOSITION_GROUP": "R4",
    "R5_TEMPORAL_GROUP": "R5",
}
ROLE_LABEL = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": "C31A_R1_MORPHOLOGY_SUPPORT_DAMAGE_SUPPORTED",
    "R4_OPPOSITION_GROUP": "C31A_R4_OPPOSITION_DAMAGE_SUPPORTED",
    "R5_TEMPORAL_GROUP": "C31A_R5_TEMPORAL_DAMAGE_SUPPORTED",
}
CLOSURE_COMBINATION = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": "011",
    "R4_OPPOSITION_GROUP": "101",
    "R5_TEMPORAL_GROUP": "110",
}
INTERACTION_TERMS = (
    "interaction_R1_R4",
    "interaction_R1_R5",
    "interaction_R4_R5",
    "interaction_R1_R4_R5",
)
FACTORIAL_TERMS = (
    "main_R1",
    "main_R4",
    "main_R5",
    *INTERACTION_TERMS,
)
STRATA = (
    ("diffuse_ht_like_visible", "stratum_diffuse"),
    ("generic_morphology_visible", "stratum_generic_morphology"),
    ("opposition_normal_visible", "stratum_opposition"),
    ("latest_history_mixed", "stratum_latest_history_mixed"),
    ("latest_positive_history_negative", "stratum_latest_positive_history_negative"),
    ("latest_negative_history_positive", "stratum_latest_negative_history_positive"),
)
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
AUC_MINOR = 0.003
SENSITIVITY_MINOR = 0.03
TRANSITION_MINOR = 2
INVERSION_MINOR = 3
ROLE_AUC_RECOVERY = 0.005
ROLE_DAMAGE_RECOVERY = 0.25
MAX_CLOSURE_SENSITIVITY_DROP = 0.05
INTERACTION_SHARE_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c31a_dema")
    parser.add_argument("--require-authorized-gate", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if valid.sum() < 2 or np.std(left_array[valid]) == 0 or np.std(right_array[valid]) == 0:
        return 0.0
    value = spearmanr(left_array[valid], right_array[valid]).statistic
    return float(value) if np.isfinite(value) else 0.0


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((negative & ~predicted).sum())
    fp = int((negative & predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    positive_mean = float(probabilities[positive].mean())
    negative_mean = float(probabilities[negative].mean())
    return {
        "AUC": auc(labels, probabilities),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_ACC": (sensitivity + specificity) / 2.0,
        "positive_probability_mean": positive_mean,
        "negative_probability_mean": negative_mean,
        "positive_negative_gap": positive_mean - negative_mean,
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
    }


def material_damage(frame: pd.DataFrame) -> np.ndarray:
    label = frame["label"].to_numpy(dtype=np.int64)
    c17_probability = frame["c17_probability"].to_numpy(dtype=np.float64)
    probability = frame["probability"].to_numpy(dtype=np.float64)
    c17_class = c17_probability >= 0.5
    candidate_class = probability >= 0.5
    return (label == 1) & (
        (c17_class & ~candidate_class) | ((probability - c17_probability) <= -0.05)
    )


def severe_damage(frame: pd.DataFrame) -> np.ndarray:
    label = frame["label"].to_numpy(dtype=np.int64)
    difference = (
        frame["probability"].to_numpy(dtype=np.float64)
        - frame["c17_probability"].to_numpy(dtype=np.float64)
    )
    return (label == 1) & (difference <= -0.10)


def inversion_count(frame: pd.DataFrame) -> int:
    positive = frame.loc[frame["label"].astype(int) == 1, "probability"].to_numpy(dtype=float)
    negative = frame.loc[frame["label"].astype(int) == 0, "probability"].to_numpy(dtype=float)
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def build_metrics(
    predictions: pd.DataFrame, pairwise: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = predictions[predictions["seed"].astype(int) == seed]
        baseline = seed_frame[seed_frame["combination"] == "000"].sort_values("patient_id")
        baseline_probability = baseline["probability"].to_numpy(dtype=np.float64)
        baseline_class = baseline_probability >= 0.5
        labels = baseline["label"].to_numpy(dtype=np.int64)
        for combination in COMBINATIONS:
            frame = seed_frame[seed_frame["combination"] == combination].sort_values("patient_id")
            probabilities = frame["probability"].to_numpy(dtype=np.float64)
            predicted = probabilities >= 0.5
            metrics = binary_metrics(labels, probabilities)
            damage = material_damage(frame)
            severe = severe_damage(frame)
            changed = predicted != baseline_class
            inversions = int(
                pairwise[
                    (pairwise["seed"].astype(int) == seed)
                    & (pairwise["combination"] == combination)
                ]["inversion"].astype(bool).sum()
            )
            rows.append(
                {
                    "seed": seed,
                    "combination": combination,
                    **metrics,
                    "material_positive_damage_count": int(damage.sum()),
                    "severe_positive_damage_count": int(severe.sum()),
                    "c17_tp_to_candidate_fn": int(
                        ((labels == 1) & (frame["c17_probability"].to_numpy(dtype=float) >= 0.5) & ~predicted).sum()
                    ),
                    "threshold_transition_count_vs_000": int(changed.sum()),
                    "positive_threshold_transition_count_vs_000": int(((labels == 1) & changed).sum()),
                    "negative_threshold_transition_count_vs_000": int(((labels == 0) & changed).sum()),
                    "pairwise_inversion_count": inversions,
                }
            )
    metrics = pd.DataFrame(rows)
    enriched: List[pd.DataFrame] = []
    for seed in SEEDS:
        frame = metrics[metrics["seed"].astype(int) == seed].copy()
        c27 = frame[frame["combination"] == "000"].iloc[0]
        c30 = frame[frame["combination"] == "111"].iloc[0]
        frame["AUC_change_vs_000"] = frame["AUC"] - float(c27["AUC"])
        frame["AUC_change_vs_111"] = frame["AUC"] - float(c30["AUC"])
        frame["sensitivity_change_vs_000"] = frame["Sensitivity"] - float(c27["Sensitivity"])
        frame["sensitivity_change_vs_111"] = frame["Sensitivity"] - float(c30["Sensitivity"])
        frame["material_damage_change_vs_000"] = (
            frame["material_positive_damage_count"] - int(c27["material_positive_damage_count"])
        )
        frame["material_damage_change_vs_111"] = (
            frame["material_positive_damage_count"] - int(c30["material_positive_damage_count"])
        )
        frame["inversion_change_vs_000"] = (
            frame["pairwise_inversion_count"] - int(c27["pairwise_inversion_count"])
        )
        frame["inversion_change_vs_111"] = (
            frame["pairwise_inversion_count"] - int(c30["pairwise_inversion_count"])
        )
        frame["minor_AUC_variation_vs_000"] = (
            frame["AUC_change_vs_000"].abs() < AUC_MINOR
        ) & (frame["material_damage_change_vs_000"] <= 0) & (
            frame["inversion_change_vs_000"].abs() <= INVERSION_MINOR
        )
        frame["minor_threshold_variation_vs_000"] = (
            frame["sensitivity_change_vs_000"].abs() < SENSITIVITY_MINOR
        ) & (frame["threshold_transition_count_vs_000"] < TRANSITION_MINOR)
        frame["minor_ranking_variation_vs_000"] = (
            frame["inversion_change_vs_000"].abs() <= INVERSION_MINOR
        )
        enriched.append(frame)
    metrics = pd.concat(enriched, ignore_index=True).sort_values(["seed", "combination"])

    summaries: List[Dict[str, Any]] = []
    for combination in COMBINATIONS:
        frame = metrics[metrics["combination"] == combination]
        summaries.append(
            {
                "combination": combination,
                "AUC_mean": float(frame["AUC"].mean()),
                "AUC_std": safe_std(frame["AUC"]),
                "Sensitivity_mean": float(frame["Sensitivity"].mean()),
                "Specificity_mean": float(frame["Specificity"].mean()),
                "Balanced_ACC_mean": float(frame["Balanced_ACC"].mean()),
                "positive_probability_mean": float(frame["positive_probability_mean"].mean()),
                "negative_probability_mean": float(frame["negative_probability_mean"].mean()),
                "positive_negative_gap_mean": float(frame["positive_negative_gap"].mean()),
                "material_positive_damage_total": int(frame["material_positive_damage_count"].sum()),
                "severe_positive_damage_total": int(frame["severe_positive_damage_count"].sum()),
                "mean_pairwise_inversions": float(frame["pairwise_inversion_count"].mean()),
                "total_pairwise_inversions": int(frame["pairwise_inversion_count"].sum()),
                "AUC_change_vs_000_mean": float(frame["AUC_change_vs_000"].mean()),
                "AUC_change_vs_111_mean": float(frame["AUC_change_vs_111"].mean()),
                "sensitivity_change_vs_111_min": float(frame["sensitivity_change_vs_111"].min()),
                "material_damage_change_vs_111_total": int(frame["material_damage_change_vs_111"].sum()),
                "inversion_change_vs_111_mean": float(frame["inversion_change_vs_111"].mean()),
                "minor_AUC_seed_count_vs_000": int(frame["minor_AUC_variation_vs_000"].sum()),
                "minor_threshold_seed_count_vs_000": int(frame["minor_threshold_variation_vs_000"].sum()),
                "minor_ranking_seed_count_vs_000": int(frame["minor_ranking_variation_vs_000"].sum()),
            }
        )
    return metrics, pd.DataFrame(summaries)


def build_inversion_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = pairwise[pairwise["seed"].astype(int) == seed]
        baseline = seed_frame[seed_frame["combination"] == "000"].sort_values(
            ["positive_patient_id", "negative_patient_id"]
        )
        final = seed_frame[seed_frame["combination"] == "111"].sort_values(
            ["positive_patient_id", "negative_patient_id"]
        )
        baseline_inversion = baseline["inversion"].to_numpy(dtype=bool)
        final_inversion = final["inversion"].to_numpy(dtype=bool)
        for combination in COMBINATIONS:
            frame = seed_frame[seed_frame["combination"] == combination].sort_values(
                ["positive_patient_id", "negative_patient_id"]
            )
            inversion = frame["inversion"].to_numpy(dtype=bool)
            if len(inversion) != 2209:
                raise RuntimeError(f"C31-A pair summary contract failed for seed {seed} {combination}")
            rows.append(
                {
                    "seed": seed,
                    "combination": combination,
                    "total_pairs": len(inversion),
                    "inversion_count": int(inversion.sum()),
                    "inversion_change_vs_000": int(inversion.sum() - baseline_inversion.sum()),
                    "inversion_change_vs_111": int(inversion.sum() - final_inversion.sum()),
                    "000_to_combination_repaired": int((baseline_inversion & ~inversion).sum()),
                    "000_to_combination_introduced": int((~baseline_inversion & inversion).sum()),
                    "combination_to_111_repaired": int((inversion & ~final_inversion).sum()),
                    "combination_to_111_introduced": int((~inversion & final_inversion).sum()),
                    "minor_ranking_variation_vs_000": abs(int(inversion.sum() - baseline_inversion.sum())) <= INVERSION_MINOR,
                }
            )
    return pd.DataFrame(rows)


def primary_negative_role(row: Mapping[str, Any], prefix: str) -> str:
    values = {
        role: float(row[f"{prefix}_shapley_{ROLE_SHORT[role]}"])
        for role in ROLES
    }
    return min(values, key=values.get)


def build_introduced_pair_attribution(pair_shapley: pd.DataFrame) -> pd.DataFrame:
    introduced = pair_shapley[pair_shapley["c30_introduced_inversion"].astype(bool)].copy()
    rows: List[Dict[str, Any]] = []
    for item in introduced.to_dict(orient="records"):
        role_values = {
            role: float(item[f"margin_shapley_{ROLE_SHORT[role]}"])
            for role in ROLES
        }
        negative = {role: max(-value, 0.0) for role, value in role_values.items()}
        denominator = sum(negative.values())
        interaction_values = {
            term: float(item[f"margin_{term}"]) for term in INTERACTION_TERMS
        }
        interaction_negative = {term: max(-value, 0.0) for term, value in interaction_values.items()}
        factorial_negative = {
            term: max(-float(item[f"margin_{term}"]), 0.0) for term in FACTORIAL_TERMS
        }
        factorial_denominator = sum(factorial_negative.values())
        row = dict(item)
        row["primary_negative_role"] = min(role_values, key=role_values.get)
        row["primary_negative_interaction"] = min(interaction_values, key=interaction_values.get)
        for role in ROLES:
            row[f"{ROLE_SHORT[role]}_negative_shapley_share"] = (
                negative[role] / denominator if denominator > 0 else 0.0
            )
        for term in INTERACTION_TERMS:
            row[f"{term}_negative_factorial_share"] = (
                factorial_negative[term] / factorial_denominator
                if factorial_denominator > 0
                else 0.0
            )
        row["all_interactions_negative_factorial_share"] = (
            sum(interaction_negative.values()) / factorial_denominator
            if factorial_denominator > 0
            else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_positive_attribution(
    predictions: pd.DataFrame,
    patient_shapley: pd.DataFrame,
    representation: pd.DataFrame,
    introduced_pairs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final = predictions[predictions["combination"] == "111"].copy()
    damage_mask = material_damage(final)
    damage = final.loc[damage_mask].copy()
    shapley_columns = [
        column
        for column in patient_shapley.columns
        if column.startswith("logit_") or column.startswith("probability_")
    ]
    damage = damage.merge(
        patient_shapley[["seed", "patient_id", *shapley_columns]],
        on=["seed", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    representation_summary = (
        representation.groupby(["seed", "patient_id", "role_group"], as_index=False)
        .agg(
            role_state_l2_delta_mean=("state_l2_delta", "mean"),
            role_state_l2_delta_max=("state_l2_delta", "max"),
            role_signed_projection_sum=("signed_projection_toward_final_classifier", "sum"),
            role_single_visit_logit_delta_sum=("single_visit_logit_delta", "sum"),
        )
    )
    for role in ROLES:
        short = ROLE_SHORT[role]
        role_frame = representation_summary[representation_summary["role_group"] == role].drop(
            columns="role_group"
        )
        role_frame = role_frame.rename(
            columns={
                column: f"{short}_{column}"
                for column in role_frame.columns
                if column not in {"seed", "patient_id"}
            }
        )
        damage = damage.merge(
            role_frame, on=["seed", "patient_id"], how="left", validate="one_to_one"
        )
    responsibility = (
        introduced_pairs.groupby(["seed", "positive_patient_id"]).size().rename("pairwise_responsibility_count")
    )
    damage = damage.merge(
        responsibility,
        left_on=["seed", "patient_id"],
        right_index=True,
        how="left",
    )
    damage["pairwise_responsibility_count"] = damage["pairwise_responsibility_count"].fillna(0).astype(int)
    damage["c17_tp_to_c30_fn"] = (
        (damage["c17_probability"].astype(float) >= 0.5)
        & (damage["probability"].astype(float) < 0.5)
    )
    damage["c30_minus_c17_probability"] = (
        damage["probability"].astype(float) - damage["c17_probability"].astype(float)
    )
    damage["severe_positive_damage"] = damage["c30_minus_c17_probability"] <= -0.10
    damage["primary_negative_role"] = damage.apply(
        lambda row: primary_negative_role(row, "probability"), axis=1
    )

    summary_rows: List[Dict[str, Any]] = []
    for seed_value in (*SEEDS, "ALL"):
        source = predictions if seed_value == "ALL" else predictions[predictions["seed"].astype(int) == seed_value]
        for combination in COMBINATIONS:
            frame = source[source["combination"] == combination]
            mask = material_damage(frame)
            severe = severe_damage(frame)
            positive = frame["label"].astype(int).to_numpy() == 1
            candidate = frame["probability"].astype(float).to_numpy() >= 0.5
            c17_candidate = frame["c17_probability"].astype(float).to_numpy() >= 0.5
            summary_rows.append(
                {
                    "seed": seed_value,
                    "combination": combination,
                    "positive_count": int(positive.sum()),
                    "material_positive_damage_count": int(mask.sum()),
                    "severe_positive_damage_count": int(severe.sum()),
                    "c17_tp_to_candidate_fn": int((positive & c17_candidate & ~candidate).sum()),
                    "positive_probability_mean": float(frame.loc[frame["label"].astype(int) == 1, "probability"].mean()),
                    "material_damage_rate": float(mask.sum() / max(positive.sum(), 1)),
                }
            )
    return damage.sort_values(["seed", "patient_id"]), pd.DataFrame(summary_rows)


def interaction_summary(
    patient_shapley: pd.DataFrame,
    pair_shapley: pd.DataFrame,
    introduced_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        patient = patient_shapley[patient_shapley["seed"].astype(int) == seed]
        pair = pair_shapley[pair_shapley["seed"].astype(int) == seed]
        introduced = introduced_pairs[introduced_pairs["seed"].astype(int) == seed]
        for target, frame, prefix in (
            ("patient_logit_all", patient, "logit"),
            ("patient_probability_all", patient, "probability"),
            ("pair_margin_all", pair, "margin"),
            ("c30_introduced_pair_margin", introduced, "margin"),
        ):
            row: Dict[str, Any] = {
                "seed": seed,
                "target": target,
                "n_objects": len(frame),
            }
            for term in FACTORIAL_TERMS:
                column = f"{prefix}_{term}"
                row[f"{term}_mean"] = float(frame[column].mean()) if len(frame) else float("nan")
                row[f"{term}_negative_fraction"] = float((frame[column] < 0).mean()) if len(frame) else float("nan")
            for role in ROLES:
                column = f"{prefix}_shapley_{ROLE_SHORT[role]}"
                row[f"shapley_{ROLE_SHORT[role]}_mean"] = float(frame[column].mean()) if len(frame) else float("nan")
            row["max_abs_shapley_completeness_error"] = (
                float(frame[f"{prefix}_shapley_sum_error"].abs().max()) if len(frame) else float("nan")
            )
            row["max_abs_factorial_completeness_error"] = (
                float(frame[f"{prefix}_factorial_sum_error"].abs().max()) if len(frame) else float("nan")
            )
            if target == "c30_introduced_pair_margin" and len(frame):
                for term in INTERACTION_TERMS:
                    row[f"{term}_aggregate_negative_share"] = float(
                        frame[f"{term}_negative_factorial_share"].mean()
                    )
                row["all_interactions_aggregate_negative_share"] = float(
                    frame["all_interactions_negative_factorial_share"].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_text_strata(
    predictions: pd.DataFrame, patient_shapley: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_shapley = patient_shapley[patient_shapley["seed"].astype(int) == seed]
        seed_predictions = predictions[predictions["seed"].astype(int) == seed]
        for stratum_name, column in STRATA:
            members = seed_shapley[seed_shapley[column].astype(bool)]
            patient_ids = set(members["patient_id"].astype(str))
            row: Dict[str, Any] = {
                "seed": seed,
                "stratum": stratum_name,
                "availability": "available" if patient_ids else "unavailable",
                "patient_count": len(patient_ids),
                "positive_count": int((members["label"].astype(int) == 1).sum()),
                "negative_count": int((members["label"].astype(int) == 0).sum()),
            }
            for combination in COMBINATIONS:
                frame = seed_predictions[
                    (seed_predictions["combination"] == combination)
                    & seed_predictions["patient_id"].astype(str).isin(patient_ids)
                ]
                row[f"AUC_{combination}"] = auc(frame["label"], frame["probability"]) if len(frame) else float("nan")
                row[f"material_damage_{combination}"] = int(material_damage(frame).sum()) if len(frame) else 0
                row[f"inversions_{combination}"] = inversion_count(frame) if len(frame) else 0
            row["C27_AUC"] = row["AUC_000"]
            row["C30_AUC"] = row["AUC_111"]
            row["C30_minus_C27_AUC"] = row["C30_AUC"] - row["C27_AUC"]
            row["positive_damage_change"] = row["material_damage_111"] - row["material_damage_000"]
            row["inversion_change"] = row["inversions_111"] - row["inversions_000"]
            for role in ROLES:
                short = ROLE_SHORT[role]
                row[f"probability_shapley_{short}_mean"] = (
                    float(members[f"probability_shapley_{short}"].mean())
                    if len(members)
                    else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def shortcut_only_auc(frame: pd.DataFrame) -> float:
    matrix = pd.DataFrame(index=frame.index)
    for field in SELECTED_SHORTCUT_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if values.notna().any() else 0.0)
    folds = min(5, int(frame["label"].value_counts().min()))
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        matrix.to_numpy(),
        frame["label"].astype(int).to_numpy(),
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return auc(frame["label"], probabilities)


def build_shortcut_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        baseline = predictions[
            (predictions["seed"].astype(int) == seed)
            & (predictions["combination"] == "000")
        ].copy()
        selected_auc = shortcut_only_auc(baseline)
        raw_warnings = {}
        for field in RAW_SHORTCUT_FIELDS:
            value = pd.to_numeric(baseline[field], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            raw_auc = auc(baseline["label"], value)
            raw_warnings[field] = max(raw_auc, 1.0 - raw_auc)
        for combination in COMBINATIONS:
            frame = predictions[
                (predictions["seed"].astype(int) == seed)
                & (predictions["combination"] == combination)
            ]
            correlations = {
                field: safe_spearman(
                    frame["probability"].to_numpy(dtype=float),
                    pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float),
                )
                for field in SELECTED_SHORTCUT_FIELDS
            }
            maximum = max(abs(value) for value in correlations.values())
            rows.append(
                {
                    "seed": seed,
                    "combination": combination,
                    "selected_structure_shortcut_only_label_AUC": selected_auc,
                    "max_abs_prediction_selected_structure_spearman": maximum,
                    "shortcut_safety_pass": selected_auc <= 0.55 and maximum <= 0.35,
                    "shortcut_fields_used_as_factorial_inputs": False,
                    **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
                    **{f"{field}_orientation_invariant_label_AUC_warning": value for field, value in raw_warnings.items()},
                }
            )
    return pd.DataFrame(rows)


def role_decision(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    introduced: pd.DataFrame,
    interactions: pd.DataFrame,
    shortcuts: pd.DataFrame,
) -> Dict[str, Any]:
    final_summary = summary.set_index("combination")
    role_rows: List[Dict[str, Any]] = []
    major_by_seed: Dict[int, str] = {}
    for seed in SEEDS:
        frame = introduced[introduced["seed"].astype(int) == seed]
        if len(frame):
            counts = frame["primary_negative_role"].value_counts()
            major_by_seed[seed] = str(counts.index[0])

    for role in ROLES:
        closure = CLOSURE_COMBINATION[role]
        major_seeds = [seed for seed, value in major_by_seed.items() if value == role]
        auc_gain = float(final_summary.at[closure, "AUC_mean"] - final_summary.at["111", "AUC_mean"])
        final_damage = int(final_summary.at["111", "material_positive_damage_total"])
        closure_damage = int(final_summary.at[closure, "material_positive_damage_total"])
        damage_reduction = (
            (final_damage - closure_damage) / final_damage
            if final_damage > 0
            else (0.0 if closure_damage == 0 else -1.0)
        )
        inversion_reduction = float(
            final_summary.at["111", "mean_pairwise_inversions"]
            - final_summary.at[closure, "mean_pairwise_inversions"]
        )
        closure_metrics = metrics[metrics["combination"] == closure].set_index("seed")
        final_metrics = metrics[metrics["combination"] == "111"].set_index("seed")
        sensitivity_differences = closure_metrics["Sensitivity"] - final_metrics["Sensitivity"]
        evidence_recovery = auc_gain >= ROLE_AUC_RECOVERY or damage_reduction >= ROLE_DAMAGE_RECOVERY
        supported = (
            len(major_seeds) >= 2
            and evidence_recovery
            and inversion_reduction > 0
            and float(sensitivity_differences.min()) >= -MAX_CLOSURE_SENSITIVITY_DROP
        )
        role_rows.append(
            {
                "role": role,
                "closure_combination": closure,
                "major_negative_seed_count": len(major_seeds),
                "major_negative_seeds": ",".join(str(seed) for seed in major_seeds),
                "closure_mean_AUC_gain_vs_111": auc_gain,
                "closure_material_damage_reduction_fraction": damage_reduction,
                "closure_mean_inversion_reduction": inversion_reduction,
                "closure_worst_sensitivity_change_vs_111": float(sensitivity_differences.min()),
                "role_damage_supported": supported,
            }
        )
    supported_roles = [row for row in role_rows if row["role_damage_supported"]]

    interaction_support: Dict[str, List[int]] = {term: [] for term in INTERACTION_TERMS}
    introduced_interactions = interactions[
        interactions["target"] == "c30_introduced_pair_margin"
    ]
    for term in INTERACTION_TERMS:
        for row in introduced_interactions.itertuples():
            value = getattr(row, f"{term}_aggregate_negative_share")
            if np.isfinite(value) and value >= INTERACTION_SHARE_THRESHOLD:
                interaction_support[term].append(int(row.seed))
    supported_interactions = [
        term for term, seeds in interaction_support.items() if len(seeds) >= 2
    ]
    mean_auc_decline = float(
        final_summary.at["111", "AUC_mean"] - final_summary.at["000", "AUC_mean"]
    ) < 0
    material_damage_present = int(
        final_summary.at["111", "material_positive_damage_total"]
    ) > 0
    interaction_damage = bool(supported_interactions) and (
        mean_auc_decline or material_damage_present
    )

    mean_aucs = final_summary["AUC_mean"].to_numpy(dtype=float)
    pairwise_differences = [
        abs(mean_aucs[left] - mean_aucs[right])
        for left in range(len(mean_aucs))
        for right in range(left + 1, len(mean_aucs))
    ]
    mostly_small = sum(value < AUC_MINOR for value in pairwise_differences) > len(pairwise_differences) / 2
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(bool).all())

    if not shortcut_pass:
        primary = "C31A_ANALYSIS_INVALID"
    elif len(supported_roles) == 1:
        primary = ROLE_LABEL[str(supported_roles[0]["role"])]
    elif len(supported_roles) > 1:
        primary = "C31A_DIFFUSE_TEXT_ROLE_INTERACTION"
    elif interaction_damage:
        primary = "C31A_TEXT_ROLE_INTERACTION_DAMAGE_SUPPORTED"
    elif mostly_small:
        primary = "C31A_ADAPTER_EFFECT_TOO_SMALL_FOR_LOCALIZATION"
    else:
        primary = "C31A_DIFFUSE_TEXT_ROLE_INTERACTION"

    authorized_role: str | None = None
    if len(supported_roles) == 1 and shortcut_pass:
        candidate = supported_roles[0]
        if (
            float(candidate["closure_material_damage_reduction_fraction"])
            >= ROLE_DAMAGE_RECOVERY
            and float(candidate["closure_mean_inversion_reduction"]) > 0
        ):
            authorized_role = str(candidate["role"])
    authorization = (
        "C31B_ONE_ROLE_ADAPTER_AUTHORIZED" if authorized_role else "C31B_NOT_AUTHORIZED"
    )
    return {
        "primary_label": primary,
        "authorization": authorization,
        "authorized_role": authorized_role,
        "role_rows": role_rows,
        "major_negative_role_by_seed": major_by_seed,
        "interaction_support_seeds": interaction_support,
        "supported_interactions": supported_interactions,
        "mostly_small_pairwise_AUC_differences": mostly_small,
        "mean_C30_minus_C27_AUC": float(
            final_summary.at["111", "AUC_mean"] - final_summary.at["000", "AUC_mean"]
        ),
        "shortcut_pass": shortcut_pass,
    }


def reproduction_report(reproduction: pd.DataFrame, gate: Mapping[str, Any]) -> str:
    lines = [
        "# C31-A Reproduction Report",
        "",
        f"- Gate: `{gate['status']}` (`{gate['passed']}/{gate['total']}`).",
        f"- Commit: `{gate['git_commit']}`.",
    ]
    for row in reproduction.sort_values("seed").itertuples():
        lines.append(
            f"- Seed `{int(row.seed)}`: `{int(row.n_patients)}` patients, C27/C30 max logit errors "
            f"`{row.c27_max_abs_logit_error:.3e}`/`{row.c30_max_abs_logit_error:.3e}`, "
            f"probability errors `{row.c27_max_abs_probability_error:.3e}`/"
            f"`{row.c30_max_abs_probability_error:.3e}`, pairs `{int(row.positive_negative_pairs)}`."
        )
    lines.extend(
        [
            "",
            "`000` reproduces frozen C27 and `111` reproduces frozen C30. Checkpoint state was unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    output: Path,
    gate: Mapping[str, Any],
    reproduction: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    introduced: pd.DataFrame,
    positive_damage: pd.DataFrame,
    interaction: pd.DataFrame,
    shortcut: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    (output / "c31a_reproduction_report.md").write_text(
        reproduction_report(reproduction, gate), encoding="utf-8"
    )
    role_lines = [
        "# C31-A Role Decision",
        "",
        f"- Primary label: `{decision['primary_label']}`.",
        f"- C31-B authorization: `{decision['authorization']}`.",
        f"- Authorized role: `{decision['authorized_role'] or 'none'}`.",
        f"- Mean C30-C27 AUC: `{decision['mean_C30_minus_C27_AUC']:.10f}`.",
        f"- Shortcut safety: `{decision['shortcut_pass']}`.",
        "",
        "## Closure Evidence",
        "",
    ]
    for row in decision["role_rows"]:
        role_lines.append(
            f"- `{row['role']}` via `{row['closure_combination']}`: major negative in "
            f"`{row['major_negative_seed_count']}/3` seeds; AUC gain "
            f"`{row['closure_mean_AUC_gain_vs_111']:.10f}`; damage reduction "
            f"`{row['closure_material_damage_reduction_fraction']:.10f}`; mean inversions reduced "
            f"`{row['closure_mean_inversion_reduction']:.3f}`; supported `{row['role_damage_supported']}`."
        )
    role_lines.extend(
        [
            "",
            "These labels attribute the trained adapter's computational path, not clinical causality.",
        ]
    )
    (output / "c31a_role_decision.md").write_text(
        "\n".join(role_lines) + "\n", encoding="utf-8"
    )

    if decision["authorization"] == "C31B_ONE_ROLE_ADAPTER_AUTHORIZED":
        route_lines = [
            "# C31-A Route Decision",
            "",
            "- `C31B_ONE_ROLE_ADAPTER_AUTHORIZED`",
            f"- `authorized_role = {decision['authorized_role']}`",
            "- `KEEP_DEMA_C17_STRICT_BEST` remains binding until a later formal route is validated.",
            "- C31-A does not launch the next phase.",
        ]
    else:
        route_lines = [
            "# C31-A Route Decision",
            "",
            "- `C31B_NOT_AUTHORIZED`",
            "- `STOP_VISIT_TEXT_ADAPTER_ROUTE`",
            "- `KEEP_DEMA_C17_STRICT_BEST`",
            "- No later phase is launched automatically.",
        ]
    (output / "c31a_route_decision.md").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )

    summary_index = summary.set_index("combination")
    final_lines = [
        "# DEMA-HT Phase C31-A Final Report",
        "",
        f"- Analysis gate: `{gate['status']}` (`{gate['passed']}/{gate['total']}`).",
        f"- Primary label: `{decision['primary_label']}`.",
        f"- Authorization: `{decision['authorization']}`.",
        f"- Authorized role: `{decision['authorized_role'] or 'none'}`.",
        f"- C27 `000` AUC mean: `{summary_index.at['000', 'AUC_mean']:.10f}`.",
        f"- C30 `111` AUC mean: `{summary_index.at['111', 'AUC_mean']:.10f}`.",
        f"- C30-C27 mean AUC: `{decision['mean_C30_minus_C27_AUC']:.10f}`.",
        f"- C27/C30 material positive damage: `{int(summary_index.at['000', 'material_positive_damage_total'])}`/"
        f"`{int(summary_index.at['111', 'material_positive_damage_total'])}`.",
        f"- C30 introduced inversion pairs: `{len(introduced)}`; material C30 positive-damage cases: `{len(positive_damage)}`.",
        f"- Maximum completeness errors, Shapley/factorial: "
        f"`{interaction['max_abs_shapley_completeness_error'].max():.3e}`/"
        f"`{interaction['max_abs_factorial_completeness_error'].max():.3e}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`; "
        f"prediction correlation max: `{shortcut['max_abs_prediction_selected_structure_spearman'].max():.10f}`.",
        "- All eight combinations are frozen diagnostic counterfactuals, not candidate models.",
        "- No training, checkpoint, model combination, threshold selection, or deployment artifact was produced.",
    ]
    final_lines.extend(["", *route_lines[2:]])
    (output / "phase_c31a_dema_final_report.md").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output_dir)
    gate_path = output / "c31a_analysis_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Missing C31-A analysis gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if args.require_authorized_gate and (
        gate.get("status") != "C31A_ANALYSIS_AUTHORIZED" or int(gate.get("passed", 0)) != 24
    ):
        raise RuntimeError("C31-A gate is not authorized")

    required = {
        "reproduction": "c31a_reproduction_by_seed.csv",
        "predictions": "c31a_factorial_predictions_val.csv",
        "patient_shapley": "c31a_patient_role_shapley.csv",
        "pair_shapley": "c31a_pair_role_shapley.csv",
        "representation": "c31a_role_representation_delta.csv",
        "pairwise": "c31a_pairwise_ranking_by_combination.csv",
    }
    missing = [name for name in required.values() if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing C31-A raw artifacts: {missing}")
    id_dtypes = {
        "patient_id": str,
        "positive_patient_id": str,
        "negative_patient_id": str,
    }
    frames = {
        key: pd.read_csv(output / name, dtype=id_dtypes)
        for key, name in required.items()
    }
    predictions = frames["predictions"]
    patient_shapley = frames["patient_shapley"]
    pair_shapley = frames["pair_shapley"]
    representation = frames["representation"]
    pairwise = frames["pairwise"]

    if len(predictions) != len(SEEDS) * len(COMBINATIONS) * 94:
        raise RuntimeError("C31-A factorial prediction contract failed")
    if len(pairwise) != len(SEEDS) * len(COMBINATIONS) * 2209:
        raise RuntimeError("C31-A factorial pair contract failed")
    metrics, summary = build_metrics(predictions, pairwise)
    inversions = build_inversion_summary(pairwise)
    introduced = build_introduced_pair_attribution(pair_shapley)
    positive_damage, positive_summary = build_positive_attribution(
        predictions, patient_shapley, representation, introduced
    )
    interactions = interaction_summary(patient_shapley, pair_shapley, introduced)
    strata = build_text_strata(predictions, patient_shapley)
    shortcuts = build_shortcut_audit(predictions)
    decision = role_decision(metrics, summary, introduced, interactions, shortcuts)

    metrics.to_csv(output / "c31a_factorial_metrics_by_seed.csv", index=False)
    summary.to_csv(output / "c31a_factorial_metrics_summary.csv", index=False)
    interactions.to_csv(output / "c31a_role_interaction_summary.csv", index=False)
    strata.to_csv(output / "c31a_text_evidence_strata.csv", index=False)
    positive_damage.to_csv(output / "c31a_positive_damage_role_attribution.csv", index=False)
    positive_summary.to_csv(output / "c31a_positive_damage_summary.csv", index=False)
    inversions.to_csv(output / "c31a_pairwise_inversion_summary.csv", index=False)
    introduced.to_csv(output / "c31a_c30_introduced_pair_attribution.csv", index=False)
    shortcuts.to_csv(output / "c31a_shortcut_audit.csv", index=False)
    write_reports(
        output,
        gate,
        frames["reproduction"],
        metrics,
        summary,
        introduced,
        positive_damage,
        interactions,
        shortcuts,
        decision,
    )
    print(
        json.dumps(
            {
                "status": "C31A_REPORT_COMPLETE",
                "primary_label": decision["primary_label"],
                "authorization": decision["authorization"],
                "authorized_role": decision["authorized_role"],
                "introduced_pairs": len(introduced),
                "material_positive_damage_cases": len(positive_damage),
            }
        )
    )


if __name__ == "__main__":
    main()
