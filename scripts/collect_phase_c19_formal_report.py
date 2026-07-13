#!/usr/bin/env python3
"""Collect C19 validation audits and apply the validation-AUC-only decision gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from audit_phase_c19_polarity_locked_residual import audit_run


EXPECTED_SEEDS = (0, 42, 3407)
C17_SEED_AUC = {0: 0.8700769579, 42: 0.8768673608, 3407: 0.8619284744}
C17_MEAN_AUC = 0.8696242644


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/dema_ht_c19_polarity_locked_multiseed")
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c19_dema")
    parser.add_argument("--require-formal-pass", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).drop(columns=["AUPRC"], errors="ignore")


def route_summary(summary: Dict[str, Any], metrics: pd.DataFrame) -> Dict[str, Any]:
    val = metrics[metrics["split"] == "val"].copy() if "split" in metrics.columns else pd.DataFrame()
    val["seed"] = pd.to_numeric(val.get("seed", pd.Series(dtype=float)), errors="coerce")
    auc_by_seed = {
        int(row["seed"]): float(row["AUC"])
        for _, row in val.iterrows()
        if pd.notna(row.get("seed")) and pd.notna(row.get("AUC"))
    }
    auc_values = np.asarray([auc_by_seed.get(seed, math.nan) for seed in EXPECTED_SEEDS], dtype=float)
    improvements = {seed: auc_by_seed.get(seed, math.nan) - C17_SEED_AUC[seed] for seed in EXPECTED_SEEDS}
    polarity = pd.DataFrame(summary.get("polarity", []))
    preservation = pd.DataFrame(summary.get("preservation", []))
    pairs = pd.DataFrame(summary.get("pair_summary", []))
    conflict = pd.DataFrame(summary.get("conflict", []))
    finite_auc = auc_values[np.isfinite(auc_values)]
    positive_mean = pd.to_numeric(preservation.get("mean_positive_delta", pd.Series(dtype=float)), errors="coerce")
    negative_mean = pd.to_numeric(preservation.get("mean_negative_delta", pd.Series(dtype=float)), errors="coerce")
    positive_tail = pd.to_numeric(
        preservation.get("fraction_positive_delta_below_minus_0_05", pd.Series(dtype=float)), errors="coerce"
    )
    negative_tail = pd.to_numeric(
        preservation.get("fraction_negative_delta_above_plus_0_05", pd.Series(dtype=float)), errors="coerce"
    )
    pair_reduced = (
        pd.to_numeric(pairs.get("final_inversions", pd.Series(dtype=float)), errors="coerce")
        < pd.to_numeric(pairs.get("base_inversions", pd.Series(dtype=float)), errors="coerce")
    )
    checks = {
        "training_valid": bool(summary.get("valid", False))
        and sorted(auc_by_seed) == list(EXPECTED_SEEDS)
        and bool(summary.get("auc_only_metrics", False)),
        "mean_validation_auc_above_c17": bool(finite_auc.size == 3 and float(finite_auc.mean()) > C17_MEAN_AUC),
        "at_least_two_seed_auc_improvements": sum(
            bool(np.isfinite(value) and value > 0.0) for value in improvements.values()
        )
        >= 2,
        "no_seed_drop_over_0_005": bool(
            all(np.isfinite(value) and value >= -0.005 for value in improvements.values())
        ),
        "validation_auc_std_le_0_02": bool(finite_auc.size == 3 and float(finite_auc.std(ddof=1)) <= 0.02),
        "positive_support_dominant_ge_0_70": bool(
            not polarity.empty
            and pd.to_numeric(polarity["positive_support_dominant_rate"], errors="coerce").min() >= 0.70
        ),
        "negative_opposition_dominant_ge_0_70": bool(
            not polarity.empty
            and pd.to_numeric(polarity["negative_opposition_dominant_rate"], errors="coerce").min() >= 0.70
        ),
        "polarity_not_collapsed": bool(
            not polarity.empty
            and pd.to_numeric(polarity["evidence_gap_std"], errors="coerce").min() > 1e-8
            and pd.to_numeric(polarity["evidence_polarity_std"], errors="coerce").min() > 1e-8
            and pd.to_numeric(polarity["polarity_sign_match_rate"], errors="coerce").min() >= 0.99
        ),
        "positive_preservation": bool(
            not preservation.empty
            and positive_mean.min() >= -0.005
            and positive_tail.max() <= 0.10
            and pd.to_numeric(preservation["tp_to_fn"], errors="coerce").sum()
            <= pd.to_numeric(preservation["fn_to_tp"], errors="coerce").sum()
        ),
        "negative_preservation": bool(
            not preservation.empty
            and negative_mean.max() <= 0.005
            and negative_tail.max() <= 0.10
        ),
        "at_least_two_seed_inversion_reduction": int(pair_reduced.sum()) >= 2,
        "repaired_greater_than_introduced": bool(
            not pairs.empty
            and pd.to_numeric(pairs["repaired_inversions"], errors="coerce").sum()
            > pd.to_numeric(pairs["introduced_inversions"], errors="coerce").sum()
        ),
        "conflict_suppression": bool(
            not polarity.empty
            and polarity["high_conflict_delta_smaller"].astype(bool).all()
            and polarity["high_uncertainty_delta_smaller"].astype(bool).all()
        ),
        "magnitude_not_saturated": bool(
            not polarity.empty
            and pd.to_numeric(polarity["magnitude_at_bound_rate"], errors="coerce").max() <= 0.10
            and pd.to_numeric(polarity["correction_magnitude_std"], errors="coerce").min() > 1e-8
        ),
        "shortcut_safety": bool(
            np.isfinite(float(summary.get("shortcut_max_auc", math.nan)))
            and float(summary.get("shortcut_max_auc", math.nan)) <= 0.55
            and float(summary.get("max_abs_prediction_shortcut_spearman", math.nan)) <= 0.55
            and float(summary.get("max_shortcut_linear_r2", math.nan)) <= 0.30
        ),
        "frozen_c17_equivalence": bool(
            not polarity.empty
            and pd.to_numeric(polarity["base_logit_equivalence_max_abs"], errors="coerce").max() <= 1e-8
        ),
        "test_is_reporting_only": bool(summary.get("test_predictions_read") is False),
    }
    return {
        "route": "C19",
        "mean_validation_auc": float(finite_auc.mean()) if finite_auc.size else math.nan,
        "std_validation_auc": float(finite_auc.std(ddof=1)) if finite_auc.size > 1 else math.nan,
        "auc_by_seed": auc_by_seed,
        "seed_improvements": improvements,
        "mean_test_auc_reporting_only": float(summary.get("mean_test_auc_reporting_only", math.nan)),
        "positive_support_dominant_by_seed": polarity.set_index("seed")["positive_support_dominant_rate"].to_dict() if not polarity.empty else {},
        "negative_opposition_dominant_by_seed": polarity.set_index("seed")["negative_opposition_dominant_rate"].to_dict() if not polarity.empty else {},
        "positive_preservation": preservation.to_dict(orient="records"),
        "negative_preservation": preservation.to_dict(orient="records"),
        "pair_summary": pairs.to_dict(orient="records"),
        "repaired_inversions": int(pd.to_numeric(pairs.get("repaired_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pairs.empty else 0,
        "introduced_inversions": int(pd.to_numeric(pairs.get("introduced_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pairs.empty else 0,
        "inversion_reduction_seed_count": int(pair_reduced.sum()),
        "conflict": conflict.to_dict(orient="records"),
        "shortcut_max_auc": float(summary.get("shortcut_max_auc", math.nan)),
        "max_abs_prediction_shortcut_spearman": float(summary.get("max_abs_prediction_shortcut_spearman", math.nan)),
        "max_shortcut_linear_r2": float(summary.get("max_shortcut_linear_r2", math.nan)),
        "checks": checks,
        "safe": all(checks.values()),
    }


def decision_for(result: Dict[str, Any]) -> tuple[str, List[str]]:
    checks = result["checks"]
    failures: List[str] = []
    if not checks["training_valid"]:
        failures.append("DEMA_C19_TRAINING_INVALID")
    if (
        not checks["polarity_not_collapsed"]
        or not checks["positive_support_dominant_ge_0_70"]
        or not checks["negative_opposition_dominant_ge_0_70"]
    ):
        failures.append("DEMA_C19_POLARITY_COLLAPSE")
    if not checks["positive_preservation"]:
        failures.append("DEMA_C19_POSITIVE_SUPPRESSION")
    if not checks["negative_preservation"]:
        failures.append("DEMA_C19_NEGATIVE_INFLATION")
    if not checks["conflict_suppression"]:
        failures.append("DEMA_C19_CONFLICT_SUPPRESSION_FAIL")
    if not checks["magnitude_not_saturated"]:
        failures.append("DEMA_C19_POLARITY_COLLAPSE")
    failures = list(dict.fromkeys(failures))
    if result["safe"]:
        return "PROMOTE_DEMA_C19_POLARITY_LOCKED", failures
    if result["checks"].get("mean_validation_auc_above_c17", False) and not result["checks"].get("validation_auc_std_le_0_02", False):
        return "DEMA_C19_SMALL_GAIN_NOT_STABLE", failures
    return (failures[0] if failures else "DEMA_C19_FORMAL_FAIL_KEEP_C17"), failures


def report_lines(gate: Dict[str, Any]) -> List[str]:
    result = gate["route"]
    lines = [
        "# DEMA-HT Phase C19 Final Report",
        "",
        "- Official model name: DEMA-HT.",
        "- Frozen base: promoted C17 Positive Preservation.",
        "- Evidence polarity determines correction direction; the trainable head learns magnitude only.",
        "- Validation AUC is the only selection metric.",
        "- Test predictions are reporting-only.",
        "- No smoke and no seed-0-only pilot were used.",
        "",
        f"## Decision: {gate['decision']}",
        "",
        f"- Current strict best: {gate['current_strict_best']}.",
        f"- C17 reference mean validation AUC: {C17_MEAN_AUC:.10f}.",
        f"- C19 mean validation AUC: {result['mean_validation_auc']:.10f}.",
        f"- C19 validation AUC std: {result['std_validation_auc']:.10f}.",
        f"- Validation AUC 0.90 reached: {str(gate['auc_0_90_reached'])}.",
        "",
        "## Validation AUC By Seed",
        "",
        "| Seed | C17 AUC | C19 AUC | Difference |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in EXPECTED_SEEDS:
        auc = result["auc_by_seed"].get(seed, math.nan)
        lines.append(f"| {seed} | {C17_SEED_AUC[seed]:.10f} | {auc:.10f} | {auc - C17_SEED_AUC[seed]:+.10f} |")
    lines.extend(
        [
            "",
            "## Evidence And Safety",
            "",
            f"- Positive support-dominant rate by seed: {result['positive_support_dominant_by_seed']}.",
            f"- Negative opposition-dominant rate by seed: {result['negative_opposition_dominant_by_seed']}.",
            f"- Repaired inversions: {result['repaired_inversions']}.",
            f"- Introduced inversions: {result['introduced_inversions']}.",
            f"- Seeds with fewer inversions: {result['inversion_reduction_seed_count']}/3.",
            f"- Shortcut-only maximum label AUC: {result['shortcut_max_auc']:.10f}.",
            f"- Maximum absolute prediction-shortcut Spearman: {result['max_abs_prediction_shortcut_spearman']:.10f}.",
            f"- Reporting-only test AUC mean: {result['mean_test_auc_reporting_only']:.10f}.",
            "",
            "## Gate Details (JSON)",
            "",
            json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True),
            "",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_run(run_dir, Path(args.c17_prediction_dir), output_dir)
    metrics = read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    epoch_metrics = read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    metrics.to_csv(output_dir / "c19_metrics_by_seed.csv", index=False)
    epoch_metrics.to_csv(output_dir / "c19_metrics_by_epoch.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        subset = metrics[metrics["split"] == split] if "split" in metrics.columns else pd.DataFrame()
        row: Dict[str, Any] = {"split": split}
        for key in (
            "AUC",
            "ACC",
            "Sensitivity",
            "Specificity",
            "Balanced_ACC",
            "positive_prob_mean",
            "negative_prob_mean",
            "mean_delta_c19",
            "mean_positive_delta_c19",
            "mean_negative_delta_c19",
        ):
            if key in subset.columns:
                values = pd.to_numeric(subset[key], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size:
                    row[f"{key}_mean"] = float(values.mean())
                    row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "c19_metrics_summary.csv", index=False)

    result = route_summary(summary, metrics)
    decision, failures = decision_for(result)
    pretraining_gate_path = output_dir / "c19_pretraining_gate.json"
    pretraining_gate = json.loads(pretraining_gate_path.read_text(encoding="utf-8")) if pretraining_gate_path.exists() else {}
    gate = {
        "decision": decision,
        "selected_route": "C19" if decision == "PROMOTE_DEMA_C19_POLARITY_LOCKED" else "C17",
        "current_strict_best": "DEMA-HT C19 Polarity Locked" if decision == "PROMOTE_DEMA_C19_POLARITY_LOCKED" else "DEMA-HT C17 Positive Preservation",
        "c17_mean_validation_auc": C17_MEAN_AUC,
        "route": result,
        "failure_labels": failures,
        "c19a_decision": pretraining_gate.get("c19a_decision"),
        "static_synthetic_gate": pretraining_gate.get("static_synthetic_gate"),
        "no_smoke": True,
        "no_seed0_pilot": True,
        "direct_seeds": list(EXPECTED_SEEDS),
        "test_reporting_only": True,
        "auc_0_90_reached": bool(np.isfinite(result["mean_validation_auc"]) and result["mean_validation_auc"] >= 0.90),
    }
    (output_dir / "c19_formal_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    lines = report_lines(gate)
    (output_dir / "c19_seed_stability_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "c19_model_comparison_report.md").write_text("\n".join(lines[:30]) + "\n", encoding="utf-8")
    (output_dir / "phase_c19_dema_final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True))
    if args.require_formal_pass and decision != "PROMOTE_DEMA_C19_POLARITY_LOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
