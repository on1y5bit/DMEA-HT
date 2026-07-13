#!/usr/bin/env python3
"""Collect the C17 epoch comparison and conservative mechanism diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DIAGNOSIS_LABELS = (
    "AUXILIARY_OBJECTIVE_POSITIVE_SUPPRESSION_CONFIRMED",
    "MECHANISM_AGGREGATION_BIAS_CONFIRMED",
    "CALIBRATION_SHIFT_WITH_RANKING_GAIN",
    "MIXED_C17_DIAGNOSIS",
    "INSUFFICIENT_EPOCH_DIAGNOSTICS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-run-dir", required=True)
    parser.add_argument("--rank-run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c17_dema")
    return parser.parse_args()


def read_epochs(run_dir: Path, route: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["route"] = route
    return frame


def number(row: pd.Series, key: str, default: float = float("nan")) -> float:
    value = pd.to_numeric(row.get(key, default), errors="coerce")
    return float(value) if np.isfinite(value) else default


def selected(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty or "val_auc" not in frame:
        return None
    scores = pd.to_numeric(frame["val_auc"], errors="coerce")
    if not bool(scores.notna().any()):
        return None
    return frame.loc[scores.idxmax()]


def change_after_epoch_three(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    early = pd.to_numeric(frame.loc[pd.to_numeric(frame["epoch"], errors="coerce") <= 3, column], errors="coerce").dropna()
    later = pd.to_numeric(frame.loc[pd.to_numeric(frame["epoch"], errors="coerce") > 3, column], errors="coerce").dropna()
    if early.empty or later.empty:
        return float("nan")
    return float(later.mean() - early.mean())


def diagnose(core: pd.DataFrame, rank: pd.DataFrame) -> str:
    combined = pd.concat([core, rank], ignore_index=True)
    if combined.empty or "epoch" not in combined or len(combined) < 2:
        return "INSUFFICIENT_EPOCH_DIAGNOSTICS"
    positive_shift = change_after_epoch_three(combined, "mean_positive_delta_logit")
    support_shift = change_after_epoch_three(combined, "support_strength")
    opposition_shift = change_after_epoch_three(combined, "opposition_strength")
    conflict_shift = change_after_epoch_three(combined, "conflict_score")
    auc_shift = change_after_epoch_three(combined, "val_auc")
    suppression = np.isfinite(positive_shift) and positive_shift < -0.02
    mechanism_bias = (
        np.isfinite(support_shift)
        and np.isfinite(opposition_shift)
        and (support_shift < -0.01 or opposition_shift > 0.01 or conflict_shift > 0.01)
    )
    ranking_gain = np.isfinite(auc_shift) and auc_shift > 0.002
    if suppression and mechanism_bias:
        return "MIXED_C17_DIAGNOSIS"
    if suppression:
        return "AUXILIARY_OBJECTIVE_POSITIVE_SUPPRESSION_CONFIRMED"
    if mechanism_bias:
        return "MECHANISM_AGGREGATION_BIAS_CONFIRMED"
    if ranking_gain:
        return "CALIBRATION_SHIFT_WITH_RANKING_GAIN"
    return "INSUFFICIENT_EPOCH_DIAGNOSTICS"


def comparison_table(frames: List[pd.DataFrame]) -> str:
    lines = [
        "| Route | Seed | Selected epoch | Validation AUC | Sensitivity | Specificity | Balanced accuracy | Positive-negative gap | Mean positive delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for frame in frames:
        row = selected(frame)
        if row is None:
            lines.append(f"| {frame.get('route', pd.Series(['unknown'])).iloc[0] if not frame.empty else 'unknown'} | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {route} | {seed:.0f} | {epoch:.0f} | {auc:.6f} | {sens:.6f} | {spec:.6f} | {bal:.6f} | {gap:.6f} | {delta:.6f} |".format(
                route=row.get("route", "unknown"),
                seed=number(row, "seed", 0.0),
                epoch=number(row, "epoch", 0.0),
                auc=number(row, "val_auc", 0.0),
                sens=number(row, "val_sensitivity", 0.0),
                spec=number(row, "val_specificity", 0.0),
                bal=number(row, "val_balanced_accuracy", 0.0),
                gap=number(row, "val_positive_negative_gap", 0.0),
                delta=number(row, "mean_positive_delta_logit", 0.0),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = read_epochs(Path(args.core_run_dir), "DEMA-R")
    rank = read_epochs(Path(args.rank_run_dir), "DEMA-RP")
    diagnosis = diagnose(core, rank)
    comparison = comparison_table([core, rank])

    comparison_text = "\n".join(
        [
            "# DEMA-HT C17 Selected Epoch Comparison",
            "",
            "Model name: `DEMA-HT`. Historical repository/package identifiers remain `DMEA-HT` and `dmea_ht`.",
            "",
            "Checkpoint selection uses validation AUC only. Test is not used in this comparison.",
            "",
            comparison,
            "",
        ]
    )
    (output_dir / "c17_selected_epoch_comparison.md").write_text(comparison_text, encoding="utf-8")

    combined = pd.concat([core, rank], ignore_index=True)
    positive_shift = change_after_epoch_three(combined, "mean_positive_delta_logit")
    support_shift = change_after_epoch_three(combined, "support_strength")
    opposition_shift = change_after_epoch_three(combined, "opposition_strength")
    conflict_shift = change_after_epoch_three(combined, "conflict_score")
    auc_shift = change_after_epoch_three(combined, "val_auc")
    diagnosis_text = "\n".join(
        [
            "# DEMA-HT C17 Training Diagnosis",
            "",
            f"- Diagnosis label: `{diagnosis}`",
            "- Selection scope: validation only.",
            "- C17 ranking loss: disabled by contract.",
            "",
            "## Epoch Questions",
            "",
            f"1. Post-epoch-3 mean positive residual change: `{positive_shift:.6f}`.",
            f"2. Post-epoch-3 support change: `{support_shift:.6f}`; opposition change: `{opposition_shift:.6f}`.",
            f"3. Post-epoch-3 conflict-score change: `{conflict_shift:.6f}`.",
            f"4. Post-epoch-3 validation AUC change: `{auc_shift:.6f}`.",
            "5. C17 does not activate a ranking objective, so no ranking-loss effect is attributed.",
            "6. Route-level patient identity comparison is deferred to the residual preservation audit.",
            "7. The residual audit stratifies support, opposition, uncertainty, conflict, temporal conflict, and morphology alignment.",
            "",
            "The diagnosis is conservative when the epoch record is incomplete. It does not authorize formal training by itself.",
            "",
        ]
    )
    (output_dir / "c17_diagnosis_report.md").write_text(diagnosis_text, encoding="utf-8")
    print({"status": "PASS" if diagnosis != "INSUFFICIENT_EPOCH_DIAGNOSTICS" else "INCOMPLETE", "diagnosis": diagnosis})


if __name__ == "__main__":
    main()
