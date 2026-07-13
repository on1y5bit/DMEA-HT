#!/usr/bin/env python3
"""Describe validation-only C17 inversion transitions for C18 targeting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c18_dema")
    return parser.parse_args()


def read_predictions(prediction_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(prediction_dir.glob("val_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            digits = "".join(character for character in path.stem if character.isdigit())
            frame["seed"] = int(digits) if digits else 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pairwise_transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    for seed, seed_frame in frame.groupby("seed", sort=True):
        positives = seed_frame[seed_frame["label"] == 1]
        negatives = seed_frame[seed_frame["label"] == 0]
        for _, positive in positives.iterrows():
            for _, negative in negatives.iterrows():
                base_margin = float(positive["base_logit"] - negative["base_logit"])
                final_margin = float(positive["final_logit"] - negative["final_logit"])
                base_inversion = int(base_margin <= 0.0)
                final_inversion = int(final_margin <= 0.0)
                if base_inversion and not final_inversion:
                    transition = "repaired"
                elif not base_inversion and final_inversion:
                    transition = "introduced"
                elif base_inversion:
                    transition = "remained_inverted"
                else:
                    transition = "remained_correct"
                pair_rows.append(
                    {
                        "seed": int(seed),
                        "positive_patient_id": positive["patient_id"],
                        "negative_patient_id": negative["patient_id"],
                        "base_margin": base_margin,
                        "final_margin": final_margin,
                        "margin_delta": final_margin - base_margin,
                        "base_inversion": base_inversion,
                        "final_inversion": final_inversion,
                        "transition": transition,
                        "positive_support_strength": float(positive.get("patient_support_strength", np.nan)),
                        "positive_opposition_strength": float(positive.get("patient_opposition_strength", np.nan)),
                        "positive_uncertainty": float(positive.get("patient_uncertainty_strength", np.nan)),
                        "positive_conflict_score": float(positive.get("patient_conflict_score", np.nan)),
                        "negative_support_strength": float(negative.get("patient_support_strength", np.nan)),
                        "negative_opposition_strength": float(negative.get("patient_opposition_strength", np.nan)),
                        "negative_uncertainty": float(negative.get("patient_uncertainty_strength", np.nan)),
                        "negative_conflict_score": float(negative.get("patient_conflict_score", np.nan)),
                    }
                )
        seed_pairs = pd.DataFrame([row for row in pair_rows if row["seed"] == int(seed)])
        for transition, group in seed_pairs.groupby("transition", sort=True):
            summary_rows.append(
                {
                    "seed": int(seed),
                    "transition": transition,
                    "n_pairs": int(len(group)),
                    "mean_base_margin": float(group["base_margin"].mean()),
                    "mean_final_margin": float(group["final_margin"].mean()),
                    "mean_margin_delta": float(group["margin_delta"].mean()),
                }
            )
        summary_rows.append(
            {
                "seed": int(seed),
                "transition": "__total__",
                "n_pairs": int(len(seed_pairs)),
                "base_inversions": int(seed_pairs["base_inversion"].sum()),
                "final_inversions": int(seed_pairs["final_inversion"].sum()),
                "repaired_inversions": int(((seed_pairs["base_inversion"] == 1) & (seed_pairs["final_inversion"] == 0)).sum()),
                "introduced_inversions": int(((seed_pairs["base_inversion"] == 0) & (seed_pairs["final_inversion"] == 1)).sum()),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(prediction_dir)
    pairs, summary = pairwise_transitions(predictions)
    pairs.to_csv(output_dir / "c18_c17_inversion_transition_pairs.csv", index=False)
    summary.to_csv(output_dir / "c18_c17_inversion_transition_summary.csv", index=False)
    observed = sorted(int(value) for value in predictions.get("seed", pd.Series(dtype=int)).unique())
    result = {
        "status": "PASS" if observed == list(EXPECTED_SEEDS) and not pairs.empty else "FAIL",
        "seeds": observed,
        "pair_count": int(len(pairs)),
        "output_dir": str(output_dir),
    }
    (output_dir / "c18_c17_inversion_transition_metadata.json").write_text(
        __import__("json").dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
