from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def read_text(path: Path) -> str:
    if not path.exists():
        return f"Missing report: {path.name}"
    return path.read_text(encoding="utf-8").strip()


def recommendation(seed_summary: pd.DataFrame, delta: pd.DataFrame, residual: pd.DataFrame, loss: pd.DataFrame) -> tuple[str, list[str]]:
    good = seed_summary[seed_summary["seed_group"] == "good"]
    bad = seed_summary[seed_summary["seed_group"] == "bad"]
    reasons: list[str] = []
    good_auc = float(good["val_auc"].mean())
    bad_auc = float(bad["val_auc"].mean())
    good_gap = float(good["pos_neg_pred_gap"].mean())
    bad_gap = float(bad["pos_neg_pred_gap"].mean())
    bad_residual = residual[residual["seed_group"] == "bad"]["abs_spearman"].dropna()
    max_bad_residual = float(bad_residual.max()) if not bad_residual.empty else 0.0
    bad_delta = delta[delta["seed_group"] == "bad"]
    bad_pos = bad_delta[bad_delta["label"].astype(int) == 1]["abs_error_delta"].mean()
    bad_neg = bad_delta[bad_delta["label"].astype(int) == 0]["abs_error_delta"].mean()

    if bad_auc < good_auc - 0.02 and bad_gap < good_gap:
        reasons.append("bad seeds have materially lower validation AUC and compressed positive-negative prediction separation")
    if max_bad_residual > 0.35:
        reasons.append("bad seeds show high shortcut residual association")
        return "E. Insufficient evidence; do not train", reasons
    if pd.notna(bad_pos) and pd.notna(bad_neg):
        if bad_pos > 0 and bad_neg <= 0:
            reasons.append("C1 bad seeds mainly harm positive cases relative to MVP")
        elif bad_neg > 0 and bad_pos <= 0:
            reasons.append("C1 bad seeds mainly harm negative cases relative to MVP")
        elif bad_pos > 0 and bad_neg > 0:
            reasons.append("C1 bad seeds harm both positive and negative cases relative to MVP")
    reasons.append("per-epoch loss curves are unavailable, so the next phase should improve optimization/logging before changing architecture")
    return "A. Optimization stabilization first", reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C5 diagnostics into a final report.")
    parser.add_argument("--phase-c5-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_dir = Path(args.phase_c5_dir)
    seed_summary = pd.read_csv(phase_dir / "c1_seed_failure_summary.csv")
    delta = pd.read_csv(phase_dir / "c1_vs_mvp_patient_delta_val.csv")
    residual = pd.read_csv(phase_dir / "c1_seed_shortcut_residual.csv")
    loss = pd.read_csv(phase_dir / "c1_loss_dynamics_by_seed.csv")
    rec, reasons = recommendation(seed_summary, delta, residual, loss)
    good = seed_summary[seed_summary["seed_group"] == "good"]
    bad = seed_summary[seed_summary["seed_group"] == "bad"]

    lines = [
        "# Phase C5 Final Report",
        "",
        "## Executive Summary",
        "",
        "Phase C5 is diagnostic only. It does not promote a model and does not launch training.",
        f"Recommended next phase direction: `{rec}`.",
        "",
        "Main reasons:",
        "",
    ]
    lines.extend(f"- {reason}." for reason in reasons)
    lines.extend(
        [
            "",
            "## Current Model Status",
            "",
            "C1 text morphology only remains unstable after Phase C4 and should not be treated as a stable main model.",
            "",
            "## Good Vs Bad Seed Comparison",
            "",
            f"Good seed mean validation AUC: {fmt(good['val_auc'].mean())}.",
            f"Bad seed mean validation AUC: {fmt(bad['val_auc'].mean())}.",
            f"Good seed mean pos-neg prediction gap: {fmt(good['pos_neg_pred_gap'].mean())}.",
            f"Bad seed mean pos-neg prediction gap: {fmt(bad['pos_neg_pred_gap'].mean())}.",
            "",
            "## C1 Vs MVP Patient-Delta Findings",
            "",
            read_text(phase_dir / "c1_vs_mvp_patient_delta_report.md"),
            "",
            "## Evidence-Label Dependence Findings",
            "",
            read_text(phase_dir / "c1_vs_mvp_stratified_delta_report.md"),
            "",
            "## Prediction Distribution Findings",
            "",
            read_text(phase_dir / "c1_prediction_distribution_report.md"),
            "",
            "## Loss Dynamics Findings",
            "",
            read_text(phase_dir / "c1_loss_dynamics_report.md"),
            "",
            "## Shortcut Residual Findings",
            "",
            read_text(phase_dir / "c1_seed_shortcut_residual_report.md"),
            "",
            "## Most Likely Failure Cause Ranking",
            "",
            "1. Optimization variance / checkpoint instability, with missing per-epoch curves preventing a sharper diagnosis.",
            "2. Threshold and sensitivity/specificity imbalance in selected bad seeds.",
            "3. Evidence-label dependence, to be inspected through harmed strata rather than assumed.",
            "4. Residual shortcut coupling, if per-seed shortcut residuals exceed prior C4/C3 levels.",
            "",
            "## Recommendation For Next Phase",
            "",
            f"`{rec}`.",
            "",
            "Do not start a new architecture phase. If training is approved later, first add better training-curve logging and run a small stabilization pilot.",
        ]
    )
    (phase_dir / "phase_c5_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {phase_dir / 'phase_c5_final_report.md'}")


if __name__ == "__main__":
    main()
