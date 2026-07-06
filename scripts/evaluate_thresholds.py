from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.metrics import compute_binary_metrics


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def metrics_at_threshold(labels: Iterable[int], probs: Iterable[float], threshold: float, split: str, name: str) -> Dict[str, Any]:
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    return {
        "split": split,
        "threshold_name": name,
        "threshold": float(threshold),
        **metrics,
    }


def scan_thresholds(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    candidates = sorted(set(float(x) for x in np.linspace(0.01, 0.99, 99)) | set(float(x) for x in probs))
    rows = [metrics_at_threshold(labels, probs, threshold, "val", "scan") for threshold in candidates]
    best_f1 = max(rows, key=lambda row: (row["F1"], row["Balanced_ACC"], row["threshold"]))
    best_youden = max(rows, key=lambda row: (row["Sensitivity"] + row["Specificity"] - 1.0, row["Balanced_ACC"], row["threshold"]))
    feasible = [row for row in rows if row["Sensitivity"] >= 0.75]
    if feasible:
        target_sens = max(feasible, key=lambda row: (row["Specificity"], row["F1"], -abs(row["threshold"] - 0.5)))
    else:
        target_sens = max(rows, key=lambda row: (row["Sensitivity"], row["Specificity"], row["F1"]))
    return {
        "threshold_0.5": 0.5,
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_youden_threshold": float(best_youden["threshold"]),
        "target_sensitivity_threshold_0.75": float(target_sens["threshold"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-derived thresholds on validation and test predictions.")
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_df = pd.read_csv(args.val_predictions)
    test_df = pd.read_csv(args.test_predictions)
    val_prob_col = prob_column(val_df)
    test_prob_col = prob_column(test_df)
    val_labels = val_df["label"].astype(int).to_numpy()
    val_probs = val_df[val_prob_col].astype(float).to_numpy()
    test_labels = test_df["label"].astype(int).to_numpy()
    test_probs = test_df[test_prob_col].astype(float).to_numpy()

    thresholds = scan_thresholds(val_labels, val_probs)
    rows: List[Dict[str, Any]] = []
    for name, threshold in thresholds.items():
        rows.append(metrics_at_threshold(val_labels, val_probs, threshold, "val", name))
        rows.append(metrics_at_threshold(test_labels, test_probs, threshold, "test", name))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "threshold_report.csv", index=False)
    (out_dir / "threshold_report.json").write_text(
        json.dumps({"thresholds": thresholds, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
