#!/usr/bin/env python3
"""Summarize C17 validation-only epoch dynamics for the two fixed routes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DYNAMICS_COLUMNS = [
    "route",
    "seed",
    "epoch",
    "train_total_loss",
    "train_classification_loss",
    "train_residual_regularization_loss",
    "train_positive_preservation_loss",
    "val_auc",
    "val_sensitivity",
    "val_specificity",
    "val_balanced_accuracy",
    "val_positive_probability_mean",
    "val_negative_probability_mean",
    "val_positive_negative_gap",
    "mean_delta_logit",
    "mean_positive_delta_logit",
    "mean_negative_delta_logit",
    "std_delta_logit",
    "fraction_delta_at_lower_bound",
    "fraction_delta_at_upper_bound",
    "fraction_positive_delta_below_minus_0_10",
    "support_strength",
    "opposition_strength",
    "uncertainty_strength",
    "conflict_score",
    "selected_by_val_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-run-dir", required=True)
    parser.add_argument("--rank-run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c17_dema")
    return parser.parse_args()


def stage(epoch: int) -> str:
    if epoch <= 3:
        return "epochs_1_3"
    if epoch <= 8:
        return "epochs_4_8"
    return "post_ramp"


def read_route(run_dir: Path, route: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    if not path.exists():
        return pd.DataFrame(columns=DYNAMICS_COLUMNS + ["stage"])
    frame = pd.read_csv(path)
    frame["route"] = route
    frame["stage"] = [stage(int(epoch)) for epoch in pd.to_numeric(frame["epoch"], errors="coerce").fillna(0)]
    for column in DYNAMICS_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[DYNAMICS_COLUMNS + ["stage"]]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = read_route(Path(args.core_run_dir), "DEMA-R")
    rank = read_route(Path(args.rank_run_dir), "DEMA-RP")
    combined = pd.concat([core, rank], ignore_index=True)

    core.to_csv(output_dir / "c17_epoch_dynamics_core.csv", index=False)
    rank.to_csv(output_dir / "c17_epoch_dynamics_rank.csv", index=False)
    suppression_columns = [
        "route",
        "seed",
        "epoch",
        "stage",
        "val_auc",
        "val_sensitivity",
        "val_specificity",
        "val_balanced_accuracy",
        "val_positive_probability_mean",
        "val_negative_probability_mean",
        "val_positive_negative_gap",
        "mean_delta_logit",
        "mean_positive_delta_logit",
        "mean_negative_delta_logit",
        "fraction_positive_delta_below_minus_0_10",
    ]
    combined[suppression_columns].to_csv(output_dir / "c17_positive_suppression_by_epoch.csv", index=False)
    mechanism_columns = [
        "route",
        "seed",
        "epoch",
        "stage",
        "support_strength",
        "opposition_strength",
        "uncertainty_strength",
        "conflict_score",
        "mean_delta_logit",
        "std_delta_logit",
        "fraction_delta_at_lower_bound",
        "fraction_delta_at_upper_bound",
    ]
    combined[mechanism_columns].to_csv(output_dir / "c17_mechanism_diagnostics_by_epoch.csv", index=False)

    loss_columns = [
        "train_total_loss",
        "train_classification_loss",
        "train_residual_regularization_loss",
        "train_positive_preservation_loss",
    ]
    if combined.empty:
        loss_effects = pd.DataFrame(columns=["route", "stage", "n_epochs", *loss_columns])
    else:
        loss_effects = (
            combined.groupby(["route", "stage"], dropna=False)[loss_columns]
            .agg(["mean", "std"])
            .reset_index()
        )
        loss_effects.columns = [
            "_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column)
            for column in loss_effects.columns
        ]
        loss_effects = loss_effects.rename(columns={"route_": "route", "stage_": "stage"})
        counts = combined.groupby(["route", "stage"], dropna=False).size().reset_index(name="n_epochs")
        loss_effects = counts.merge(loss_effects, on=["route", "stage"], how="left")
    loss_effects.to_csv(output_dir / "c17_loss_activation_effects.csv", index=False)
    print(
        {
            "status": "PASS" if not combined.empty else "INCOMPLETE",
            "core_epochs": int(len(core)),
            "rank_epochs": int(len(rank)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
