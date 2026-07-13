#!/usr/bin/env python3
"""Audit one completed C17 validation-only run without selecting on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def finite(frame: pd.DataFrame, columns: List[str]) -> bool:
    if frame.empty or any(column not in frame.columns for column in columns):
        return False
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    prediction_paths = sorted((run_dir / "predictions").glob("val_predictions_seed_*.csv"))
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True) if prediction_paths else pd.DataFrame()
    epoch_path = run_dir / "reports" / "metrics_by_epoch.csv"
    seed_path = run_dir / "reports" / "metrics_by_seed.csv"
    epochs = pd.read_csv(epoch_path) if epoch_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    check("validation_predictions_present", not predictions.empty, len(predictions))
    check("epoch_metrics_present", not epochs.empty and len(epochs) >= 2, len(epochs))
    check("seed_metrics_present", not metrics.empty, len(metrics))
    check("test_predictions_absent", not list((run_dir / "predictions").glob("test_predictions_seed_*.csv")), "validation-only smoke")
    check("no_forbidden_metric_in_c17_outputs", not any("AUPRC" in path.read_text(encoding="utf-8", errors="ignore") for path in list((run_dir / "reports").glob("*.csv")) + list((run_dir / "predictions").glob("*.csv"))), "C17 output scan")

    required = ["label", "base_logit", "base_prob", "delta_logit", "final_logit", "final_prob"]
    check("residual_prediction_columns_present", all(column in predictions.columns for column in required), required)
    if not predictions.empty and all(column in predictions.columns for column in required):
        check("residual_predictions_finite", finite(predictions, required), required)
        delta = pd.to_numeric(predictions["delta_logit"], errors="coerce")
        check("residual_bound_holds", bool((delta.abs() <= 0.500001).all()), float(delta.abs().max()))
        check("predictions_not_constant", float(pd.to_numeric(predictions["final_prob"], errors="coerce").std(ddof=1)) > 1e-6, "final probability std")
        check("residual_is_nonzero", float(delta.std(ddof=1)) > 1e-8, float(delta.std(ddof=1)))
        labels = pd.to_numeric(predictions["label"], errors="coerce")
        positive_delta = delta[labels == 1]
        check("positive_residual_not_globally_negative", float(positive_delta.mean()) >= -0.20 if not positive_delta.empty else False, float(positive_delta.mean()) if not positive_delta.empty else "missing")
    else:
        for name in ("residual_predictions_finite", "residual_bound_holds", "predictions_not_constant", "residual_is_nonzero", "positive_residual_not_globally_negative"):
            check(name, False, "prediction columns unavailable")

    if not epochs.empty:
        numeric_columns = [column for column in epochs.columns if column not in {"split", "selected_by_val_auc"}]
        check("epoch_metrics_finite", bool(np.isfinite(epochs[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all()), numeric_columns)
        selected = pd.to_numeric(epochs.get("selected_by_val_auc", pd.Series(dtype=float)), errors="coerce").fillna(0)
        check("one_selected_epoch", int(selected.sum()) == 1, int(selected.sum()))
    else:
        check("epoch_metrics_finite", False, "missing metrics_by_epoch.csv")
        check("one_selected_epoch", False, "missing metrics_by_epoch.csv")

    summary = {
        "route": args.route,
        "run_dir": str(run_dir),
        "selection_scope": "validation_only",
        "test_used_for_selection": False,
        "status": "PASS" if checks and all(item["pass"] for item in checks) else "FAIL",
        "passed": sum(int(item["pass"]) for item in checks),
        "total": len(checks),
        "checks": checks,
    }
    (output_dir / "c17_run_health.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(output_dir / "c17_run_health.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_pass and summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
