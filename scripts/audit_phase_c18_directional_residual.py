#!/usr/bin/env python3
"""Run validation-only directional residual audits for one C18 formal route."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "base_logit",
    "base_prob",
    "support_delta",
    "opposition_delta",
    "support_gate",
    "opposition_gate",
    "conflict_suppression",
    "effective_support_delta",
    "effective_opposition_delta",
    "directional_delta",
    "final_logit",
    "final_prob",
    "patient_support_strength",
    "patient_opposition_strength",
    "patient_uncertainty_strength",
    "patient_conflict_score",
    "text_temporal_conflict_score",
    "morphology_alignment_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c18_dema")
    parser.add_argument("--route", required=True)
    return parser.parse_args()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def read_predictions(run_dir: Path, route: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob("val_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            digits = "".join(character for character in path.stem if character.isdigit())
            frame["seed"] = int(digits) if digits else 0
        frame["route"] = route
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_metrics(run_dir: Path, route: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).drop(columns=["AUPRC"], errors="ignore")
    frame["route"] = route
    return frame


def pairwise_audit(predictions: pd.DataFrame, route: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    pair_columns = [
        "route", "seed", "positive_patient_id", "negative_patient_id", "base_margin", "final_margin",
        "base_inversion", "final_inversion", "margin_delta",
    ]
    summary_columns = [
        "route", "seed", "base_inversions", "final_inversions", "repaired_inversions", "introduced_inversions",
        "net_inversion_change",
    ]
    for seed, frame in predictions.groupby("seed", sort=True):
        positives = frame[numeric(frame, "label") == 1]
        negatives = frame[numeric(frame, "label") == 0]
        for _, positive in positives.iterrows():
            for _, negative in negatives.iterrows():
                base_margin = float(positive["base_logit"] - negative["base_logit"])
                final_margin = float(positive["final_logit"] - negative["final_logit"])
                pair_rows.append(
                    {
                        "route": route,
                        "seed": int(seed),
                        "positive_patient_id": positive["patient_id"],
                        "negative_patient_id": negative["patient_id"],
                        "base_margin": base_margin,
                        "final_margin": final_margin,
                        "base_inversion": int(base_margin <= 0.0),
                        "final_inversion": int(final_margin <= 0.0),
                        "margin_delta": final_margin - base_margin,
                    }
                )
        seed_pairs = pd.DataFrame([row for row in pair_rows if row["seed"] == int(seed)])
        base_inv = int(seed_pairs["base_inversion"].sum()) if not seed_pairs.empty else 0
        final_inv = int(seed_pairs["final_inversion"].sum()) if not seed_pairs.empty else 0
        repaired = int(((seed_pairs["base_inversion"] == 1) & (seed_pairs["final_inversion"] == 0)).sum()) if not seed_pairs.empty else 0
        introduced = int(((seed_pairs["base_inversion"] == 0) & (seed_pairs["final_inversion"] == 1)).sum()) if not seed_pairs.empty else 0
        summary_rows.append(
            {
                "route": route,
                "seed": int(seed),
                "base_inversions": base_inv,
                "final_inversions": final_inv,
                "repaired_inversions": repaired,
                "introduced_inversions": introduced,
                "net_inversion_change": final_inv - base_inv,
            }
        )
    return pd.DataFrame(pair_rows, columns=pair_columns), pd.DataFrame(summary_rows, columns=summary_columns)


def positive_audit(predictions: pd.DataFrame, route: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, frame in predictions.groupby("seed", sort=True):
        labels = numeric(frame, "label")
        positive = frame[labels == 1]
        base_pred = numeric(positive, "base_prob") >= 0.5
        final_pred = numeric(positive, "final_prob") >= 0.5
        delta = numeric(positive, "directional_delta")
        negative = frame[labels == 0]
        rows.append(
            {
                "route": route,
                "seed": int(seed),
                "positive_count": int(len(positive)),
                "mean_positive_directional_delta": float(delta.mean()) if not delta.empty else 0.0,
                "fraction_positive_delta_below_minus_0_10": float((delta < -0.10).mean()) if not delta.empty else 0.0,
                "tp_to_fn": int((base_pred & ~final_pred).sum()),
                "fn_to_tp": int((~base_pred & final_pred).sum()),
                "mean_negative_directional_delta": float(numeric(negative, "directional_delta").mean()) if not negative.empty else 0.0,
                "negative_probability_change": float(numeric(negative, "final_prob").mean() - numeric(negative, "base_prob").mean()) if not negative.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def mechanism_audit(predictions: pd.DataFrame, route: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    features = {
        "support": "patient_support_strength",
        "opposition": "patient_opposition_strength",
        "uncertainty": "patient_uncertainty_strength",
        "conflict": "patient_conflict_score",
        "temporal_conflict": "text_temporal_conflict_score",
        "morphology_alignment": "morphology_alignment_cosine",
    }
    for seed, frame in predictions.groupby("seed", sort=True):
        delta = numeric(frame, "directional_delta")
        for feature, column in features.items():
            values = numeric(frame, column)
            valid = values.notna() & delta.notna()
            if int(valid.sum()) < 2:
                continue
            threshold = float(values[valid].median())
            for stratum, mask in (("high", valid & (values >= threshold)), ("low", valid & (values < threshold))):
                selected = delta[mask].to_numpy(dtype=float)
                rows.append(
                    {
                        "route": route,
                        "seed": int(seed),
                        "feature": feature,
                        "stratum": stratum,
                        "n": int(mask.sum()),
                        "threshold": threshold,
                        "mean_directional_delta": float(selected.mean()) if selected.size else 0.0,
                        "mean_effective_support_delta": float(numeric(frame.loc[mask], "effective_support_delta").mean()) if selected.size else 0.0,
                        "mean_effective_opposition_delta": float(numeric(frame.loc[mask], "effective_opposition_delta").mean()) if selected.size else 0.0,
                        "fraction_directional_delta_negative": float((selected < 0).mean()) if selected.size else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def gate_health(predictions: pd.DataFrame, route: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, frame in predictions.groupby("seed", sort=True):
        support_delta = numeric(frame, "support_delta")
        opposition_delta = numeric(frame, "opposition_delta")
        support_gate = numeric(frame, "support_gate")
        opposition_gate = numeric(frame, "opposition_gate")
        suppression = numeric(frame, "conflict_suppression")
        directional = numeric(frame, "directional_delta")
        rows.append(
            {
                "route": route,
                "seed": int(seed),
                "finite": bool(np.isfinite(pd.concat([support_delta, opposition_delta, support_gate, opposition_gate, suppression, directional])).all()),
                "support_delta_std": float(support_delta.std(ddof=1)) if len(support_delta.dropna()) > 1 else 0.0,
                "opposition_delta_std": float(opposition_delta.std(ddof=1)) if len(opposition_delta.dropna()) > 1 else 0.0,
                "mean_support_gate": float(support_gate.mean()),
                "mean_opposition_gate": float(opposition_gate.mean()),
                "mean_conflict_suppression": float(suppression.mean()),
                "support_upper_fraction": float((support_delta >= 0.49999).mean()),
                "opposition_upper_fraction": float((opposition_delta >= 0.49999).mean()),
                "directional_std": float(directional.std(ddof=1)) if len(directional.dropna()) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def shortcut_audit(predictions: pd.DataFrame, route: str) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows: List[Dict[str, Any]] = []
    labels = numeric(predictions, "label")
    final_prob = numeric(predictions, "final_prob")
    for seed, frame in predictions.groupby("seed", sort=True):
        y = numeric(frame, "label")
        p = numeric(frame, "final_prob")
        for field in SHORTCUT_FIELDS:
            values = numeric(frame, field)
            valid = y.notna() & p.notna() & values.notna() & np.isfinite(values)
            if int(valid.sum()) < 2 or len(np.unique(y[valid])) < 2 or values[valid].nunique() < 2:
                auc = 0.5
            else:
                auc = float(max(roc_auc_score(y[valid], values[valid]), 1.0 - roc_auc_score(y[valid], values[valid])))
            rows.append({
                "route": route,
                "seed": int(seed),
                "shortcut_field": field,
                "n": int(valid.sum()),
                "shortcut_label_auc": auc,
            })
    return pd.DataFrame(rows)


def audit_run(run_dir: Path, output_dir: Path, route: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(run_dir, route)
    metrics = read_metrics(run_dir, route)
    required = set(PATIENT_COLUMNS)
    missing_columns = sorted(required - set(predictions.columns)) if not predictions.empty else sorted(required)
    pair_frame, pair_summary = pairwise_audit(predictions, route) if not predictions.empty else (pd.DataFrame(), pd.DataFrame())
    positive = positive_audit(predictions, route) if not predictions.empty else pd.DataFrame()
    mechanism = mechanism_audit(predictions, route) if not predictions.empty else pd.DataFrame()
    health = gate_health(predictions, route) if not predictions.empty else pd.DataFrame()
    shortcut = shortcut_audit(predictions, route) if not predictions.empty else pd.DataFrame()
    predictions.reindex(columns=[column for column in PATIENT_COLUMNS if column in predictions.columns] + [column for column in SHORTCUT_FIELDS if column in predictions.columns]).to_csv(output_dir / "c18_patient_directional_diagnostics_val.csv", index=False)
    positive.to_csv(output_dir / "c18_positive_preservation_audit.csv", index=False)
    pair_frame.to_csv(output_dir / "c18_pairwise_ranking_val.csv", index=False)
    pair_summary.to_csv(output_dir / "c18_pairwise_inversion_summary.csv", index=False)
    mechanism.to_csv(output_dir / "c18_directional_mechanism_audit.csv", index=False)
    health.to_csv(output_dir / "c18_gate_health_audit.csv", index=False)
    shortcut.to_csv(output_dir / "c18_shortcut_residual_audit.csv", index=False)
    val_metrics = metrics[metrics["split"] == "val"] if "split" in metrics.columns else pd.DataFrame()
    summary = {
        "route": route,
        "run_dir": str(run_dir),
        "seeds": sorted(int(value) for value in predictions.get("seed", pd.Series(dtype=int)).dropna().unique()),
        "prediction_rows": int(len(predictions)),
        "missing_prediction_columns": missing_columns,
        "metrics_val_rows": int(len(val_metrics)),
        "valid": sorted(int(value) for value in predictions.get("seed", pd.Series(dtype=int)).dropna().unique()) == list(EXPECTED_SEEDS) and not missing_columns and len(val_metrics) == 3,
        "mean_validation_auc": float(pd.to_numeric(val_metrics.get("AUC", pd.Series(dtype=float)), errors="coerce").mean()) if not val_metrics.empty else math.nan,
        "mean_test_auc_reporting_only": float(pd.to_numeric(metrics.loc[metrics["split"] == "test", "AUC"], errors="coerce").mean()) if "split" in metrics.columns and (metrics["split"] == "test").any() else math.nan,
        "pair_summary": pair_summary.to_dict(orient="records"),
        "positive": positive.to_dict(orient="records"),
        "health": health.to_dict(orient="records"),
        "shortcut_max_auc": float(pd.to_numeric(shortcut.get("shortcut_label_auc", pd.Series(dtype=float)), errors="coerce").max()) if not shortcut.empty else math.nan,
    }
    (output_dir / "c18_route_audit_summary.json").write_text(json_dumps(summary), encoding="utf-8")
    return summary


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n"


def main() -> None:
    args = parse_args()
    summary = audit_run(Path(args.run_dir), Path(args.output_dir), args.route)
    print(summary)
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
