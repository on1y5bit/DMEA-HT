#!/usr/bin/env python3
"""Audit C16-MEA prediction and alignment health without selecting on test."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REQUIRED_DIAGNOSTICS = (
    "patient_support_strength",
    "patient_opposition_strength",
    "patient_uncertainty_strength",
    "patient_conflict_score",
    "image_support_score",
    "image_opposition_score",
    "image_uncertainty_score",
    "text_support_score",
    "text_opposition_score",
    "text_uncertainty_score",
    "text_temporal_conflict_score",
    "text_temporal_available",
    "text_latest_support_score",
    "text_latest_opposition_score",
    "text_latest_available",
    "text_history_support_score",
    "text_history_opposition_score",
    "text_history_available",
    "bio_support_score",
    "bio_opposition_score",
    "bio_uncertainty_score",
    "image_evidence_weight",
    "text_evidence_weight",
    "bio_evidence_weight",
    "morphology_alignment_cosine",
    "morphology_alignment_available",
    "mechanism_state_norm",
    "evidence_role_entropy",
    "evidence_role_prob_sum_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--synthetic-report")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_predictions(run_dir: Path, split: str = "val") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = seed_from_path(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)


def finite_frame(frame: pd.DataFrame, columns: List[str]) -> bool:
    if frame.empty:
        return False
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(run_dir, "val")
    epoch_path = run_dir / "reports" / "metrics_by_epoch.csv"
    seed_path = run_dir / "reports" / "metrics_by_seed.csv"
    epochs = pd.read_csv(epoch_path) if epoch_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    check("validation_predictions_present", not predictions.empty, len(predictions))
    check("epoch_metrics_present", not epochs.empty, len(epochs))
    check("seed_metrics_present", not metrics.empty, len(metrics))
    missing = [column for column in REQUIRED_DIAGNOSTICS if column not in predictions.columns]
    check("required_patient_diagnostics_present", not missing, missing)

    if not predictions.empty and not missing:
        finite_columns = ["label", "pred_prob", "logit", *REQUIRED_DIAGNOSTICS]
        check("all_prediction_diagnostics_finite", finite_frame(predictions, finite_columns), finite_columns)
        pred_std = float(numeric(predictions, "pred_prob").std(ddof=1))
        check("predictions_not_constant", np.isfinite(pred_std) and pred_std > 1e-6, pred_std)
        prob_sum_error = float(numeric(predictions, "evidence_role_prob_sum_error").max())
        check("role_probabilities_normalized", np.isfinite(prob_sum_error) and prob_sum_error <= 1e-5, prob_sum_error)

        conflict = numeric(predictions, "patient_conflict_score")
        conflict_saturation = float(((conflict <= 0.01) | (conflict >= 0.99)).mean())
        check("conflict_not_globally_saturated", conflict_saturation < 0.95, conflict_saturation)

        modality_columns = ["image_evidence_weight", "text_evidence_weight", "bio_evidence_weight"]
        modality = predictions[modality_columns].apply(pd.to_numeric, errors="coerce")
        modality_sum_error = float((modality.sum(axis=1) - 1.0).abs().max())
        modality_means = modality.mean().to_dict()
        global_saturation = float(modality.max(axis=1).gt(0.95).mean())
        check("modality_weights_normalized", modality_sum_error <= 1e-5, modality_sum_error)
        check("no_global_modality_saturation", max(modality_means.values()) < 0.95 and global_saturation < 0.95, {"means": modality_means, "rate": global_saturation})

        mechanism_norm = numeric(predictions, "mechanism_state_norm")
        check("mechanism_norms_bounded", bool(np.isfinite(mechanism_norm).all()) and float(mechanism_norm.max()) < 100.0, {"mean": float(mechanism_norm.mean()), "max": float(mechanism_norm.max())})
        role_columns = {
            "support": ["image_support_score", "text_support_score", "bio_support_score"],
            "opposition": ["image_opposition_score", "text_opposition_score", "bio_opposition_score"],
            "uncertainty": ["image_uncertainty_score", "text_uncertainty_score", "bio_uncertainty_score"],
        }
        role_means = {
            role: float(predictions[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).mean())
            for role, columns in role_columns.items()
        }
        role_total = max(sum(role_means.values()), 1e-8)
        max_role_fraction = max(role_means.values()) / role_total
        entropy_mean = float(numeric(predictions, "evidence_role_entropy").mean())
        check("evidence_roles_not_collapsed", max_role_fraction < 0.95 and entropy_mean > 0.05, {"means": role_means, "max_fraction": max_role_fraction, "entropy": entropy_mean})
    else:
        for name in (
            "all_prediction_diagnostics_finite",
            "predictions_not_constant",
            "role_probabilities_normalized",
            "conflict_not_globally_saturated",
            "modality_weights_normalized",
            "no_global_modality_saturation",
            "mechanism_norms_bounded",
            "evidence_roles_not_collapsed",
        ):
            check(name, False, "predictions or diagnostics unavailable")

    if not epochs.empty:
        loss_columns = [column for column in epochs.columns if column.endswith("_loss") or column.startswith("effective_lambda_")]
        check("epoch_losses_and_weights_finite", finite_frame(epochs, loss_columns), loss_columns)
        selected_count = int(pd.to_numeric(epochs.get("selected_by_val_auc", 0), errors="coerce").fillna(0).sum())
        check("one_selected_epoch_per_seed", selected_count == int(epochs["seed"].nunique()), {"selected": selected_count, "seeds": int(epochs["seed"].nunique())})
    else:
        check("epoch_losses_and_weights_finite", False, "missing metrics_by_epoch.csv")
        check("one_selected_epoch_per_seed", False, "missing metrics_by_epoch.csv")

    if args.synthetic_report:
        synthetic_path = Path(args.synthetic_report)
        synthetic = json.loads(synthetic_path.read_text(encoding="utf-8")) if synthetic_path.exists() else {}
        check("synthetic_gate_passed", synthetic.get("status") == "PASS", synthetic.get("status", "missing"))

    status = "PASS" if checks and all(item["pass"] for item in checks) else "FAIL"
    summary = {
        "route": args.route,
        "run_dir": str(run_dir),
        "selection_scope": "validation_only",
        "test_used_for_selection": False,
        "status": status,
        "passed": sum(int(item["pass"]) for item in checks),
        "total": len(checks),
        "checks": checks,
    }
    (output_dir / "c16_mea_alignment_health.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(checks).to_csv(output_dir / "c16_mea_alignment_health.csv", index=False)
    report_lines = [
        f"# C16-MEA Alignment Health: {args.route}",
        "",
        f"- Status: `{status}`",
        f"- Checks: `{summary['passed']}/{summary['total']}`",
        "- Selection scope: validation only; test is not read by this audit.",
        "",
        "| Check | Pass | Detail |",
        "| --- | --- | --- |",
    ]
    report_lines.extend(
        f"| {item['check']} | {item['pass']} | {str(item['detail']).replace('|', '/')} |" for item in checks
    )
    (output_dir / "c16_mea_alignment_health.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_pass and status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
