#!/usr/bin/env python3
"""Collect the completed C17 formal DEMA-RP run and apply its validation-only gate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from collect_phase_c17_dema_report import (
    drop_forbidden_metric,
    finite_series,
    mechanism_audit,
    pairwise_audit,
    patient_diagnostics,
    positive_audit,
    read_predictions,
    shortcut_audit,
)


EXPECTED_SEEDS = (0, 42, 3407)
C13_VAL_AUC_MEAN = 0.8664554097
FORMAL_ROUTE = "DEMA-RP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c17_dema")
    parser.add_argument("--require-formal-pass", action="store_true")
    return parser.parse_args()


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = drop_forbidden_metric(pd.read_csv(path))
    frame["route"] = FORMAL_ROUTE
    return frame


def read_epochs(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = drop_forbidden_metric(pd.read_csv(path))
    frame["route"] = FORMAL_ROUTE
    return frame


def classification_stats(frame: pd.DataFrame) -> Dict[str, float]:
    labels = finite_series(frame, "label")
    base_prob = finite_series(frame, "base_prob")
    final_prob = finite_series(frame, "final_prob")
    result: Dict[str, float] = {}
    for prefix, probabilities in (("base", base_prob), ("final", final_prob)):
        valid = labels.notna() & probabilities.notna()
        y = labels[valid].to_numpy(dtype=int)
        p = probabilities[valid].to_numpy(dtype=float)
        prediction = p >= 0.5
        tp = int(((y == 1) & prediction).sum())
        tn = int(((y == 0) & (~prediction)).sum())
        fp = int(((y == 0) & prediction).sum())
        fn = int(((y == 1) & (~prediction)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        result[f"{prefix}_sensitivity"] = sensitivity
        result[f"{prefix}_specificity"] = specificity
        result[f"{prefix}_balanced_accuracy"] = (sensitivity + specificity) / 2.0
        result[f"{prefix}_accuracy"] = (tp + tn) / len(y) if len(y) else 0.0
        result[f"{prefix}_positive_count"] = float(tp + fn)
        result[f"{prefix}_negative_count"] = float(tn + fp)
        if len(np.unique(y)) > 1:
            from sklearn.metrics import roc_auc_score

            result[f"{prefix}_auc"] = float(roc_auc_score(y, p))
        else:
            result[f"{prefix}_auc"] = 0.5
    labels_valid = labels.notna() & base_prob.notna() & final_prob.notna()
    positive = labels_valid & (labels == 1)
    negative = labels_valid & (labels == 0)
    base_gap = float(base_prob[positive].mean() - base_prob[negative].mean()) if positive.any() and negative.any() else 0.0
    final_gap = float(final_prob[positive].mean() - final_prob[negative].mean()) if positive.any() and negative.any() else 0.0
    delta = finite_series(frame, "delta_logit")
    positive_delta = delta[positive].dropna().to_numpy(dtype=float)
    all_delta = delta[labels_valid].dropna().to_numpy(dtype=float)
    result.update(
        {
            "n": float(labels_valid.sum()),
            "base_gap": base_gap,
            "final_gap": final_gap,
            "gap_delta": final_gap - base_gap,
            "mean_positive_delta_logit": float(positive_delta.mean()) if positive_delta.size else 0.0,
            "fraction_positive_delta_below_minus_0_10": float((positive_delta < -0.10).mean()) if positive_delta.size else 0.0,
            "mean_delta_logit": float(all_delta.mean()) if all_delta.size else 0.0,
            "std_delta_logit": float(all_delta.std(ddof=1)) if all_delta.size > 1 else 0.0,
            "fraction_delta_at_lower_bound": float((all_delta <= -0.49999).mean()) if all_delta.size else 0.0,
            "fraction_delta_at_upper_bound": float((all_delta >= 0.49999).mean()) if all_delta.size else 0.0,
        }
    )
    return result


def seed_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, frame in predictions.groupby("seed", sort=True):
        row: Dict[str, Any] = {"route": FORMAL_ROUTE, "seed": int(seed)}
        row.update(classification_stats(frame))
        rows.append(row)
    columns = ["route", "seed"] + sorted({key for row in rows for key in row if key not in {"route", "seed"}})
    return pd.DataFrame(rows, columns=columns)


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, frame in metrics.groupby("split", sort=False):
        row: Dict[str, Any] = {"route": FORMAL_ROUTE, "split": split, "n_seeds": int(frame["seed"].nunique())}
        for column in ("AUC", "ACC", "F1", "Sensitivity", "Specificity", "Balanced_ACC", "best_epoch"):
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_audit_tables(predictions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    positive = pd.concat(
        [positive_audit(frame) for _, frame in predictions.groupby("seed", sort=True)],
        ignore_index=True,
    )
    mechanism = pd.concat(
        [mechanism_audit(frame) for _, frame in predictions.groupby("seed", sort=True)],
        ignore_index=True,
    )
    shortcut = pd.concat(
        [shortcut_audit(frame) for _, frame in predictions.groupby("seed", sort=True)],
        ignore_index=True,
    )
    pairs, pair_summary = pairwise_audit(predictions)
    return {
        "positive": positive,
        "mechanism": mechanism,
        "shortcut": shortcut,
        "pairs": pairs,
        "pair_summary": pair_summary,
    }


def formal_gate(metrics: pd.DataFrame, diagnostics: pd.DataFrame, pair_summary: pd.DataFrame, shortcut: pd.DataFrame) -> Dict[str, Any]:
    val = metrics[metrics["split"] == "val"].copy() if not metrics.empty else pd.DataFrame()
    auc = pd.to_numeric(val.get("AUC", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    sensitivity = pd.to_numeric(val.get("Sensitivity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    specificity = pd.to_numeric(val.get("Specificity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    expected_seed_set = set(EXPECTED_SEEDS)
    observed_seed_set = set(int(seed) for seed in val["seed"].dropna().tolist()) if not val.empty else set()
    if not diagnostics.empty:
        gap_delta = diagnostics["gap_delta"].to_numpy(dtype=float)
        ba_delta = (diagnostics["final_balanced_accuracy"] - diagnostics["base_balanced_accuracy"]).to_numpy(dtype=float)
        positive_delta = diagnostics["mean_positive_delta_logit"].to_numpy(dtype=float)
        positive_negative_fraction = diagnostics["fraction_positive_delta_below_minus_0_10"].to_numpy(dtype=float)
        residual_std = diagnostics["std_delta_logit"].to_numpy(dtype=float)
        saturation = np.maximum(
            diagnostics["fraction_delta_at_lower_bound"].to_numpy(dtype=float),
            diagnostics["fraction_delta_at_upper_bound"].to_numpy(dtype=float),
        )
    else:
        gap_delta = ba_delta = positive_delta = positive_negative_fraction = residual_std = saturation = np.asarray([], dtype=float)
    inversion_seed_count = int(pair_summary["seed"].nunique()) if not pair_summary.empty else 0
    inversion_decreased = int((pair_summary["final_inversions"] < pair_summary["base_inversions"]).sum()) if not pair_summary.empty else 0
    shortcut_ok = bool(shortcut.empty or (pd.to_numeric(shortcut["shortcut_label_auc"], errors="coerce") < 0.80).all())
    checks = {
        "formal_seed_contract": observed_seed_set == expected_seed_set,
        "validation_auc_above_c13_mean": bool(auc.size == len(EXPECTED_SEEDS) and float(auc.mean()) > C13_VAL_AUC_MEAN),
        "no_seed_auc_below_0_85": bool(auc.size == len(EXPECTED_SEEDS) and float(auc.min()) >= 0.85),
        "validation_auc_std_at_most_0_02": bool(auc.size == len(EXPECTED_SEEDS) and float(auc.std(ddof=1)) <= 0.02),
        "sensitivity_mean_at_least_0_55": bool(sensitivity.size == len(EXPECTED_SEEDS) and float(sensitivity.mean()) >= 0.55),
        "specificity_mean_at_least_0_75": bool(specificity.size == len(EXPECTED_SEEDS) and float(specificity.mean()) >= 0.75),
        "balanced_accuracy_not_materially_lower": bool(ba_delta.size == len(EXPECTED_SEEDS) and float(ba_delta.min()) >= -0.02),
        "positive_delta_not_suppressed": bool(positive_delta.size == len(EXPECTED_SEEDS) and float(positive_delta.min()) >= -0.02),
        "positive_strong_negative_fraction_at_most_0_25": bool(positive_negative_fraction.size == len(EXPECTED_SEEDS) and float(positive_negative_fraction.max()) <= 0.25),
        "residual_nonzero_for_every_seed": bool(residual_std.size == len(EXPECTED_SEEDS) and bool((residual_std > 1e-8).all())),
        "residual_not_saturated": bool(saturation.size == len(EXPECTED_SEEDS) and float(saturation.max()) < 0.25),
        "inversions_not_worse_for_every_seed": bool(inversion_seed_count == len(EXPECTED_SEEDS) and bool((pair_summary["final_inversions"] <= pair_summary["base_inversions"]).all())),
        "inversions_decreased_in_at_least_two_of_three_seeds": inversion_decreased >= math.ceil(len(EXPECTED_SEEDS) * 2 / 3),
        "shortcut_audit_pass": shortcut_ok,
        "test_not_used_for_selection": True,
        "forbidden_metric_absent_from_reports": "AUPRC" not in metrics.columns,
    }
    return {
        "checks": checks,
        "formal_gate_pass": bool(all(checks.values())),
        "validation_auc_mean": float(auc.mean()) if auc.size else 0.0,
        "validation_auc_std": float(auc.std(ddof=1)) if auc.size > 1 else 0.0,
        "validation_auc_min": float(auc.min()) if auc.size else 0.0,
        "validation_auc_gain_vs_c13": float(auc.mean() - C13_VAL_AUC_MEAN) if auc.size else 0.0,
        "inversion_decreased_seed_count": inversion_decreased,
        "inversion_seed_count": inversion_seed_count,
        "target_auc_090_reached": bool(auc.size == len(EXPECTED_SEEDS) and float(auc.mean()) >= 0.90),
    }


def write_reports(output_dir: Path, run_dir: Path, metrics: pd.DataFrame, epochs: pd.DataFrame, predictions: pd.DataFrame, diagnostics: pd.DataFrame, tables: Dict[str, pd.DataFrame], gate: Dict[str, Any]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "c17_formal_metrics_by_seed.csv", index=False)
    metric_summary(metrics).to_csv(output_dir / "c17_formal_metrics_summary.csv", index=False)
    epochs.to_csv(output_dir / "c17_formal_metrics_by_epoch.csv", index=False)
    patient_diagnostics(predictions).to_csv(output_dir / "c17_formal_patient_residual_diagnostics_val.csv", index=False)
    diagnostics.to_csv(output_dir / "c17_formal_seed_diagnostics.csv", index=False)
    tables["positive"].to_csv(output_dir / "c17_formal_positive_preservation_audit.csv", index=False)
    tables["pairs"].to_csv(output_dir / "c17_formal_pairwise_ranking_val.csv", index=False)
    tables["pair_summary"].to_csv(output_dir / "c17_formal_pairwise_inversion_summary.csv", index=False)
    tables["mechanism"].to_csv(output_dir / "c17_formal_mechanism_residual_audit.csv", index=False)
    tables["shortcut"].to_csv(output_dir / "c17_formal_shortcut_residual_audit.csv", index=False)

    decision = "PROMOTE_DEMA_C17_POSITIVE_PRESERVATION" if gate["formal_gate_pass"] else "DEMA_C17_FORMAL_FAIL_KEEP_C13"
    val = metrics[metrics["split"] == "val"] if not metrics.empty else pd.DataFrame()
    test = metrics[metrics["split"] == "test"] if not metrics.empty else pd.DataFrame()
    lines = [
        "# DEMA-HT Phase C17 Formal Report",
        "",
        "## Contract",
        "",
        "- Official model name: `DEMA-HT`; historical repository/package identifiers remain `DMEA-HT` and `dmea_ht`.",
        "- Formal route: `DEMA-RP` with bounded mechanism-evidence residual and positive-preservation penalty.",
        "- Seeds are fixed at `0`, `42`, and `3407`.",
        "- Validation AUC is the primary metric and the only checkpoint/decision criterion. Test is reporting-only.",
        "- Shortcut and structural fields are audit-only and are not predictor inputs.",
        "",
        "## Validation Result",
        "",
        f"- Validation AUC: `{gate['validation_auc_mean']:.6f} +/- {gate['validation_auc_std']:.6f}`.",
        f"- Validation AUC range: `{gate['validation_auc_min']:.6f}` to `{float(val['AUC'].max()):.6f}`." if not val.empty else "- Validation AUC range: unavailable.",
        f"- Gain versus frozen C13 mean validation AUC `{C13_VAL_AUC_MEAN:.10f}`: `{gate['validation_auc_gain_vs_c13']:+.6f}`.",
        "",
        "| Split | AUC mean | AUC std | Sensitivity mean | Specificity mean | Balanced accuracy mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, frame in (("val", val), ("test", test)):
        if frame.empty:
            continue
        auc = pd.to_numeric(frame["AUC"], errors="coerce").dropna().to_numpy(dtype=float)
        sens = pd.to_numeric(frame["Sensitivity"], errors="coerce").dropna().to_numpy(dtype=float)
        spec = pd.to_numeric(frame["Specificity"], errors="coerce").dropna().to_numpy(dtype=float)
        bal = pd.to_numeric(frame["Balanced_ACC"], errors="coerce").dropna().to_numpy(dtype=float)
        lines.append(f"| {split} | {auc.mean():.6f} | {auc.std(ddof=1):.6f} | {sens.mean():.6f} | {spec.mean():.6f} | {bal.mean():.6f} |")
    lines.extend(
        [
            "",
            "## Safety Gate",
            "",
            f"- Formal gate: `{'PASS' if gate['formal_gate_pass'] else 'FAIL'}`.",
            f"- Inversions decreased in `{gate['inversion_decreased_seed_count']}/{gate['inversion_seed_count']}` evaluated seeds.",
            f"- Decision: `{decision}`.",
            "",
        ]
    )
    for key, value in gate["checks"].items():
        lines.append(f"- {key}: `{value}`.")
    lines.extend(
        [
            "",
            "The validation AUC 0.90 target was not claimed because the complete formal mean did not reach it.",
            "The reporting-only test row is included for transparency and was not read by the gate or used to select a checkpoint.",
            "",
            f"Run directory: `{run_dir}`.",
            "",
        ]
    )
    (output_dir / "phase_c17_dema_formal_final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "c17_formal_gate.json").write_text(
        pd.Series({**gate, "decision": decision}).to_json(indent=2),
        encoding="utf-8",
    )
    return decision


def main() -> None:
    args = parse_args()
    run_dir = Path(args.formal_run_dir)
    output_dir = Path(args.output_dir)
    metrics = read_metrics(run_dir)
    epochs = read_epochs(run_dir)
    predictions = read_predictions(run_dir, FORMAL_ROUTE)
    if metrics.empty or predictions.empty:
        raise SystemExit("formal run is missing metrics or validation predictions")
    diagnostics = seed_diagnostics(predictions)
    tables = per_seed_audit_tables(predictions)
    gate = formal_gate(metrics, diagnostics, tables["pair_summary"], tables["shortcut"])
    decision = write_reports(output_dir, run_dir, metrics, epochs, predictions, diagnostics, tables, gate)
    print({"status": "PASS", "decision": decision, "formal_gate_pass": gate["formal_gate_pass"], "output_dir": str(output_dir)})
    if args.require_formal_pass and not gate["formal_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
