#!/usr/bin/env python3
"""Audit C19 polarity locking and residual safety using validation only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (0, 42, 3407)
SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
)
PATIENT_COLUMNS = (
    "route",
    "seed",
    "patient_id",
    "label",
    "frozen_c17_logit",
    "frozen_c17_prob",
    "support_strength",
    "opposition_strength",
    "uncertainty_strength",
    "conflict_score",
    "temporal_conflict_score",
    "morphology_alignment_cosine",
    "q_support",
    "q_opposition",
    "evidence_gap",
    "evidence_polarity",
    "evidence_confidence",
    "correction_magnitude",
    "effective_correction_magnitude",
    "delta_c19",
    "final_logit",
    "final_prob",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c19_dema")
    return parser.parse_args()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def std_or_zero(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def pairwise_auc(labels: pd.Series, scores: pd.Series) -> float:
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    positive = s[valid & (y == 1)]
    negative = s[valid & (y == 0)]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    margins = positive[:, None] - negative[None, :]
    return float((margins > 0.0).mean() + 0.5 * (margins == 0.0).mean())


def spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return math.nan
    value = pair["left"].rank().corr(pair["right"].rank())
    return float(value) if pd.notna(value) else math.nan


def linear_r2(target: pd.Series, feature: pd.Series) -> float:
    pair = pd.DataFrame({"target": target, "feature": feature}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["feature"].nunique() < 2 or pair["target"].nunique() < 2:
        return 0.0
    x = pair["feature"].to_numpy(dtype=float)
    y = pair["target"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficient, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficient
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 1e-12:
        return 0.0
    return float(1.0 - ((y - prediction) ** 2).sum() / total)


def read_validation(run_dir: Path, seed: int) -> pd.DataFrame:
    path = run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing validation prediction: {path}")
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["seed"] = seed
    return frame


def read_c17(c17_prediction_dir: Path, seed: int) -> pd.DataFrame:
    path = c17_prediction_dir / f"val_predictions_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing C17 validation prediction: {path}")
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def merge_c17_reference(frame: pd.DataFrame, c17_prediction_dir: Path, seed: int) -> pd.DataFrame:
    c17 = read_c17(c17_prediction_dir, seed)
    required = {"patient_id", "label", "final_logit", "final_prob"}
    missing = sorted(required - set(c17.columns))
    if missing:
        raise ValueError(f"C17 validation prediction missing columns for seed {seed}: {missing}")
    reference = c17[list(required)].rename(
        columns={
            "label": "c17_label",
            "final_logit": "c17_final_logit",
            "final_prob": "c17_final_prob",
        }
    )
    merged = frame.merge(reference, on="patient_id", how="inner")
    if len(merged) != len(frame):
        raise ValueError(f"C17/C19 validation patient mismatch for seed {seed}: {len(merged)} vs {len(frame)}")
    if not np.array_equal(numeric(merged, "label").to_numpy(), numeric(merged, "c17_label").to_numpy()):
        raise ValueError(f"C17/C19 validation labels differ for seed {seed}")
    merged["base_logit_equivalence_abs_error"] = (
        numeric(merged, "frozen_c17_logit") - numeric(merged, "c17_final_logit")
    ).abs()
    return merged


def pairwise_audit(frame: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    positives = frame[numeric(frame, "label") == 1]
    negatives = frame[numeric(frame, "label") == 0]
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c17_margin = float(positive["frozen_c17_logit"] - negative["frozen_c17_logit"])
            c19_margin = float(positive["final_logit"] - negative["final_logit"])
            base_inversion = c17_margin <= 0.0
            final_inversion = c19_margin <= 0.0
            rows.append(
                {
                    "seed": seed,
                    "positive_patient_id": str(positive["patient_id"]),
                    "negative_patient_id": str(negative["patient_id"]),
                    "c17_margin": c17_margin,
                    "c19_margin": c19_margin,
                    "base_inversion": int(base_inversion),
                    "final_inversion": int(final_inversion),
                    "repaired": int(base_inversion and not final_inversion),
                    "introduced": int((not base_inversion) and final_inversion),
                    "remained_inverted": int(base_inversion and final_inversion),
                    "remained_correct": int((not base_inversion) and (not final_inversion)),
                    "positive_delta_c19": float(positive["delta_c19"]),
                    "negative_delta_c19": float(negative["delta_c19"]),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        summary = {
            "seed": seed,
            "base_inversions": 0,
            "final_inversions": 0,
            "repaired_inversions": 0,
            "introduced_inversions": 0,
            "remained_inverted": 0,
            "remained_correct": 0,
        }
    else:
        summary = {
            "seed": seed,
            "base_inversions": int(pairs["base_inversion"].sum()),
            "final_inversions": int(pairs["final_inversion"].sum()),
            "repaired_inversions": int(pairs["repaired"].sum()),
            "introduced_inversions": int(pairs["introduced"].sum()),
            "remained_inverted": int(pairs["remained_inverted"].sum()),
            "remained_correct": int(pairs["remained_correct"].sum()),
        }
    return pairs, summary


def polarity_audit(frame: pd.DataFrame, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels = numeric(frame, "label")
    positive = labels == 1
    negative = labels == 0
    q_support = numeric(frame, "q_support")
    q_opposition = numeric(frame, "q_opposition")
    delta = numeric(frame, "delta_c19")
    gap = numeric(frame, "evidence_gap")
    conflict = numeric(frame, "conflict_score")
    uncertainty = numeric(frame, "uncertainty_strength")
    abs_delta = delta.abs()
    summary = {
        "seed": seed,
        "n_patients": int(len(frame)),
        "positive_support_dominant_rate": mean_or_nan((q_support[positive] > q_opposition[positive]).astype(float)),
        "negative_opposition_dominant_rate": mean_or_nan((q_opposition[negative] > q_support[negative]).astype(float)),
        "polarity_sign_match_rate": mean_or_nan((np.sign(gap) == np.sign(delta)).astype(float)),
        "support_strength_std": std_or_zero(q_support),
        "opposition_strength_std": std_or_zero(q_opposition),
        "evidence_gap_std": std_or_zero(gap),
        "evidence_polarity_std": std_or_zero(numeric(frame, "evidence_polarity")),
        "delta_std": std_or_zero(delta),
        "correction_magnitude_std": std_or_zero(numeric(frame, "correction_magnitude")),
        "magnitude_at_bound_rate": float((numeric(frame, "correction_magnitude") >= 0.20 - 1e-5).mean()),
        "magnitude_max": float(numeric(frame, "correction_magnitude").max()),
        "base_logit_equivalence_max_abs": float(numeric(frame, "base_logit_equivalence_abs_error").max()),
    }
    strata: List[Dict[str, Any]] = []
    for feature, values in (("conflict", conflict), ("uncertainty", uncertainty)):
        high = values >= 0.35
        low = values < 0.35
        for stratum, mask in (("high", high), ("low", low)):
            selected = abs_delta[mask].dropna()
            strata.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "stratum": stratum,
                    "n": int(selected.size),
                    "mean_abs_delta": float(selected.mean()) if selected.size else math.nan,
                    "mean_delta": float(delta[mask].mean()) if int(mask.sum()) else math.nan,
                }
            )
    conflict_high = next((row["mean_abs_delta"] for row in strata if row["feature"] == "conflict" and row["stratum"] == "high"), math.nan)
    conflict_low = next((row["mean_abs_delta"] for row in strata if row["feature"] == "conflict" and row["stratum"] == "low"), math.nan)
    uncertainty_high = next((row["mean_abs_delta"] for row in strata if row["feature"] == "uncertainty" and row["stratum"] == "high"), math.nan)
    uncertainty_low = next((row["mean_abs_delta"] for row in strata if row["feature"] == "uncertainty" and row["stratum"] == "low"), math.nan)
    summary.update(
        {
            "high_conflict_abs_delta": conflict_high,
            "low_conflict_abs_delta": conflict_low,
            "high_conflict_delta_smaller": bool(np.isfinite(conflict_high) and np.isfinite(conflict_low) and conflict_high < conflict_low),
            "high_uncertainty_abs_delta": uncertainty_high,
            "low_uncertainty_abs_delta": uncertainty_low,
            "high_uncertainty_delta_smaller": bool(np.isfinite(uncertainty_high) and np.isfinite(uncertainty_low) and uncertainty_high < uncertainty_low),
        }
    )
    return summary, strata


def preservation_audit(frame: pd.DataFrame, seed: int) -> Dict[str, Any]:
    labels = numeric(frame, "label")
    base_prob = numeric(frame, "frozen_c17_prob")
    final_prob = numeric(frame, "final_prob")
    base_pred = base_prob >= 0.5
    final_pred = final_prob >= 0.5
    positive = labels == 1
    negative = labels == 0
    delta = numeric(frame, "delta_c19")
    return {
        "seed": seed,
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "mean_positive_delta": float(delta[positive].mean()),
        "mean_negative_delta": float(delta[negative].mean()),
        "fraction_positive_delta_below_minus_0_05": float((delta[positive] < -0.05).mean()),
        "fraction_negative_delta_above_plus_0_05": float((delta[negative] > 0.05).mean()),
        "tp_to_fn": int((positive & base_pred & ~final_pred).sum()),
        "fn_to_tp": int((positive & ~base_pred & final_pred).sum()),
        "fp_to_tn": int((negative & base_pred & ~final_pred).sum()),
        "tn_to_fp": int((negative & ~base_pred & final_pred).sum()),
        "negative_probability_change": float(final_prob[negative].mean() - base_prob[negative].mean()),
    }


def shortcut_audit(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for field in SHORTCUT_FIELDS:
        values = numeric(frame, field)
        valid = values.notna() & np.isfinite(values) & numeric(frame, "label").notna()
        if int(valid.sum()) < 2:
            label_auc = 0.5
            prediction_spearman = math.nan
            r2 = 0.0
        else:
            label_auc = max(pairwise_auc(numeric(frame.loc[valid], "label"), values[valid]), 1.0 - pairwise_auc(numeric(frame.loc[valid], "label"), values[valid]))
            prediction_spearman = spearman(numeric(frame.loc[valid], "final_prob"), values[valid])
            r2 = linear_r2(numeric(frame.loc[valid], "final_prob"), values[valid])
        rows.append(
            {
                "seed": seed,
                "shortcut_field": field,
                "n": int(valid.sum()),
                "shortcut_label_auc": label_auc,
                "prediction_shortcut_spearman": prediction_spearman,
                "abs_prediction_shortcut_spearman": abs(prediction_spearman) if np.isfinite(prediction_spearman) else math.nan,
                "linear_r2": r2,
            }
        )
    return pd.DataFrame(rows)


def audit_run(run_dir: Path, c17_prediction_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for seed in EXPECTED_SEEDS:
        frame = read_validation(run_dir, seed)
        frame = merge_c17_reference(frame, c17_prediction_dir, seed)
        frame.insert(0, "route", "C19")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    metrics_path = run_dir / "reports" / "metrics_by_seed.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing C19 metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required = set(PATIENT_COLUMNS)
    missing_columns = sorted(required - set(predictions.columns))
    pair_frames: List[pd.DataFrame] = []
    pair_summaries: List[Dict[str, Any]] = []
    polarity_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    preservation_rows: List[Dict[str, Any]] = []
    shortcut_frames: List[pd.DataFrame] = []
    for seed in EXPECTED_SEEDS:
        frame = predictions[predictions["seed"] == seed].copy()
        pairs, pair_summary = pairwise_audit(frame, seed)
        pair_frames.append(pairs)
        pair_summaries.append(pair_summary)
        polarity, conflict = polarity_audit(frame, seed)
        polarity_rows.append(polarity)
        conflict_rows.extend(conflict)
        preservation_rows.append(preservation_audit(frame, seed))
        shortcut_frames.append(shortcut_audit(frame, seed))

    pair_frame = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    polarity_frame = pd.DataFrame(polarity_rows)
    conflict_frame = pd.DataFrame(conflict_rows)
    preservation_frame = pd.DataFrame(preservation_rows)
    shortcut_frame = pd.concat(shortcut_frames, ignore_index=True) if shortcut_frames else pd.DataFrame()
    patient_output_columns = [column for column in PATIENT_COLUMNS if column in predictions.columns]
    patient_output_columns.extend(column for column in SHORTCUT_FIELDS if column in predictions.columns)
    predictions.reindex(columns=patient_output_columns).to_csv(output_dir / "c19_patient_polarity_diagnostics_val.csv", index=False)
    polarity_frame.to_csv(output_dir / "c19_polarity_consistency_audit.csv", index=False)
    preservation_frame.to_csv(output_dir / "c19_positive_negative_preservation_audit.csv", index=False)
    pair_frame.to_csv(output_dir / "c19_pairwise_ranking_val.csv", index=False)
    pd.DataFrame(pair_summaries).to_csv(output_dir / "c19_pairwise_inversion_summary.csv", index=False)
    conflict_frame.to_csv(output_dir / "c19_conflict_suppression_audit.csv", index=False)
    shortcut_frame.to_csv(output_dir / "c19_shortcut_residual_audit.csv", index=False)

    val_metrics = metrics[metrics["split"] == "val"] if "split" in metrics.columns else pd.DataFrame()
    test_metrics = metrics[metrics["split"] == "test"] if "split" in metrics.columns else pd.DataFrame()
    summary = {
        "route": "C19",
        "run_dir": str(run_dir),
        "seeds": sorted(int(value) for value in predictions["seed"].unique()),
        "prediction_rows": int(len(predictions)),
        "missing_prediction_columns": missing_columns,
        "metrics_val_rows": int(len(val_metrics)),
        "metrics_test_rows": int(len(test_metrics)),
        "auc_only_metrics": "AUPRC" not in metrics.columns,
        "valid": sorted(int(value) for value in predictions["seed"].unique()) == list(EXPECTED_SEEDS)
        and not missing_columns
        and len(val_metrics) == 3,
        "mean_validation_auc": float(pd.to_numeric(val_metrics.get("AUC", pd.Series(dtype=float)), errors="coerce").mean()),
        "std_validation_auc": float(pd.to_numeric(val_metrics.get("AUC", pd.Series(dtype=float)), errors="coerce").std(ddof=1)),
        "auc_by_seed": {
            int(row["seed"]): float(row["AUC"])
            for _, row in val_metrics.iterrows()
            if pd.notna(row.get("seed")) and pd.notna(row.get("AUC"))
        },
        "mean_test_auc_reporting_only": float(pd.to_numeric(test_metrics.get("AUC", pd.Series(dtype=float)), errors="coerce").mean()) if not test_metrics.empty else math.nan,
        "polarity": polarity_frame.to_dict(orient="records"),
        "preservation": preservation_frame.to_dict(orient="records"),
        "pair_summary": pair_summaries,
        "conflict": conflict_frame.to_dict(orient="records"),
        "shortcut_max_auc": float(pd.to_numeric(shortcut_frame.get("shortcut_label_auc", pd.Series(dtype=float)), errors="coerce").max()),
        "max_abs_prediction_shortcut_spearman": float(pd.to_numeric(shortcut_frame.get("abs_prediction_shortcut_spearman", pd.Series(dtype=float)), errors="coerce").max()),
        "max_shortcut_linear_r2": float(pd.to_numeric(shortcut_frame.get("linear_r2", pd.Series(dtype=float)), errors="coerce").max()),
        "test_predictions_read": False,
    }
    (output_dir / "c19_route_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = audit_run(Path(args.run_dir), Path(args.c17_prediction_dir), Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
