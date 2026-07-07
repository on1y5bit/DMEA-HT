from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


GOOD_SEEDS = {0, 2, 4, 3407}
BAD_SEEDS = {1, 3, 42}


def seed_group(seed: int) -> str:
    if seed in GOOD_SEEDS:
        return "good"
    if seed in BAD_SEEDS:
        return "bad"
    return "other"


def value(row: pd.Series, col: str) -> Any:
    return row[col] if col in row and not pd.isna(row[col]) else pd.NA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze available C1 extended-seed loss/checkpoint dynamics.")
    parser.add_argument("--c1-extended-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.c1_extended_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "reports" / "metrics_by_seed.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics_by_seed.csv: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    rows: List[Dict[str, Any]] = []
    missing_note = (
        "Per-epoch train/validation losses and loss components are unavailable in current train.py outputs; "
        "only best_epoch and final best-state metrics_by_seed diagnostics are available."
    )
    for seed in sorted(metrics["seed"].astype(int).unique()):
        val_rows = metrics[(metrics["seed"].astype(int) == seed) & (metrics["split"].astype(str).str.lower() == "val")]
        if val_rows.empty:
            continue
        row = val_rows.iloc[0]
        rows.append(
            {
                "seed": seed,
                "seed_group": seed_group(seed),
                "best_epoch": value(row, "best_epoch"),
                "last_epoch": pd.NA,
                "best_val_auc": value(row, "AUC"),
                "last_val_auc": pd.NA,
                "best_minus_last_val_auc": pd.NA,
                "min_train_loss": pd.NA,
                "min_val_loss": value(row, "loss"),
                "final_train_loss": pd.NA,
                "final_val_loss": value(row, "loss"),
                "mean_cls_loss": pd.NA,
                "mean_text_morphology_loss": value(row, "text_morphology_loss"),
                "evidence_loss_to_total_loss_ratio": pd.NA,
                "notes": missing_note,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "c1_loss_dynamics_by_seed.csv", index=False)

    lines = [
        "# Phase C5 C1 Loss Dynamics",
        "",
        missing_note,
        "",
        "Available diagnostics therefore answer checkpoint questions only partially.",
        "",
        "## Available Columns",
        "",
        frame.to_markdown(index=False),
        "",
        "Interpretation:",
        "",
        "- Bad seed overfitting cannot be verified without epoch curves.",
        "- Unstable checkpoint selection can only be approximated from best_epoch spread.",
        "- Evidence loss dominance cannot be computed because total epoch loss components were not logged.",
        "- This favors an optimization/logging stabilization follow-up before stronger claims about evidence-loss strength.",
    ]
    (out_dir / "c1_loss_dynamics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase C5 loss dynamics diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
