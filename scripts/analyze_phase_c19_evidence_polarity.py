#!/usr/bin/env python3
"""Audit C17 evidence polarity and C18 residual direction using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (0, 42, 3407)
ROUTES = {
    "C18-D": "runs/dema_ht_c18_directional_multiseed/predictions",
    "C18-DH": "runs/dema_ht_c18_directional_hardrank_multiseed/predictions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--c18-directional-prediction-dir", default=ROUTES["C18-D"])
    parser.add_argument("--c18-hardrank-prediction-dir", default=ROUTES["C18-DH"])
    parser.add_argument("--output-dir", default="analysis_reports/phase_c19_dema")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def read_validation(path: Path, seed: int) -> pd.DataFrame:
    prediction_path = path / f"val_predictions_seed_{seed}.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"missing validation prediction: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    if "patient_id" not in frame or "label" not in frame:
        raise ValueError(f"prediction is missing patient_id/label: {prediction_path}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype(int)
    return frame


def numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(pair["left"].rank().corr(pair["right"].rank()))


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return frame.to_string(index=False)


def c17_polarity_summary(frame: pd.DataFrame, seed: int) -> Dict[str, object]:
    support = numeric(frame, "patient_support_strength")
    opposition = numeric(frame, "patient_opposition_strength")
    gap = support - opposition
    positive = frame["label"] == 1
    negative = frame["label"] == 0
    return {
        "route": "C17",
        "seed": seed,
        "n_patients": int(len(frame)),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "positive_support_dominant_rate": mean_or_nan((support[positive] > opposition[positive]).astype(float)),
        "negative_opposition_dominant_rate": mean_or_nan((opposition[negative] > support[negative]).astype(float)),
        "support_opposition_gap_std": float(gap.std(ddof=1)) if len(gap) > 1 else 0.0,
        "support_opposition_gap_min": float(gap.min()),
        "support_opposition_gap_max": float(gap.max()),
        "support_opposition_gap_nonconstant": bool(gap.nunique(dropna=True) > 1),
    }


def pair_table(c17: pd.DataFrame, c18: pd.DataFrame, route: str, seed: int) -> pd.DataFrame:
    left = c17.set_index("patient_id")
    right = c18.set_index("patient_id")
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValueError(f"no common validation patients for {route} seed {seed}")
    left = left.loc[common]
    right = right.loc[common]
    if not np.array_equal(left["label"].to_numpy(), right["label"].to_numpy()):
        raise ValueError(f"C17/C18 validation labels differ for {route} seed {seed}")

    c17_logit = numeric(left, "final_logit", numeric(left, "logit", 0.0))
    c18_logit = numeric(right, "final_logit", numeric(right, "logit", 0.0))
    c17_support = numeric(left, "patient_support_strength")
    c17_opposition = numeric(left, "patient_opposition_strength")
    c18_delta = numeric(right, "directional_delta", numeric(right, "delta_logit", 0.0))
    c18_support_delta = numeric(right, "effective_support_delta", numeric(right, "support_delta", 0.0))
    c18_opposition_delta = numeric(right, "effective_opposition_delta", numeric(right, "opposition_delta", 0.0))
    c18_conflict = numeric(left, "patient_conflict_score")
    c18_uncertainty = numeric(left, "patient_uncertainty_strength")

    positives = np.flatnonzero(left["label"].to_numpy() == 1)
    negatives = np.flatnonzero(left["label"].to_numpy() == 0)
    rows: List[Dict[str, object]] = []
    for positive_index in positives:
        for negative_index in negatives:
            c17_margin = float(c17_logit.iloc[positive_index] - c17_logit.iloc[negative_index])
            c18_margin = float(c18_logit.iloc[positive_index] - c18_logit.iloc[negative_index])
            base_inversion = c17_margin <= 0.0
            final_inversion = c18_margin <= 0.0
            rows.append(
                {
                    "route": route,
                    "seed": seed,
                    "positive_patient_id": str(left.index[positive_index]),
                    "negative_patient_id": str(left.index[negative_index]),
                    "c17_margin": c17_margin,
                    "c18_margin": c18_margin,
                    "base_inversion": int(base_inversion),
                    "final_inversion": int(final_inversion),
                    "repaired": int(base_inversion and not final_inversion),
                    "introduced": int((not base_inversion) and final_inversion),
                    "positive_support_strength": float(c17_support.iloc[positive_index]),
                    "positive_opposition_strength": float(c17_opposition.iloc[positive_index]),
                    "negative_support_strength": float(c17_support.iloc[negative_index]),
                    "negative_opposition_strength": float(c17_opposition.iloc[negative_index]),
                    "positive_c18_delta": float(c18_delta.iloc[positive_index]),
                    "negative_c18_delta": float(c18_delta.iloc[negative_index]),
                    "positive_effective_support_delta": float(c18_support_delta.iloc[positive_index]),
                    "positive_effective_opposition_delta": float(c18_opposition_delta.iloc[positive_index]),
                    "negative_effective_support_delta": float(c18_support_delta.iloc[negative_index]),
                    "negative_effective_opposition_delta": float(c18_opposition_delta.iloc[negative_index]),
                    "positive_conflict": float(c18_conflict.iloc[positive_index]),
                    "negative_conflict": float(c18_conflict.iloc[negative_index]),
                    "positive_uncertainty": float(c18_uncertainty.iloc[positive_index]),
                    "negative_uncertainty": float(c18_uncertainty.iloc[negative_index]),
                }
            )
    return pd.DataFrame(rows)


def patient_polarity_table(c17: pd.DataFrame, c18: pd.DataFrame, route: str, seed: int) -> pd.DataFrame:
    c17_keep = [
        "patient_id",
        "label",
        "patient_support_strength",
        "patient_opposition_strength",
        "patient_uncertainty_strength",
        "patient_conflict_score",
        "text_temporal_conflict_score",
        "morphology_alignment_cosine",
        "final_logit",
        "final_prob",
    ]
    c18_keep = [
        "patient_id",
        "directional_delta",
        "effective_support_delta",
        "effective_opposition_delta",
        "support_delta",
        "opposition_delta",
        "support_gate",
        "opposition_gate",
        "conflict_suppression",
        "final_logit",
        "final_prob",
    ]
    left = c17[[column for column in c17_keep if column in c17.columns]].copy()
    right = c18[[column for column in c18_keep if column in c18.columns]].copy()
    merged = left.merge(right, on="patient_id", how="inner", suffixes=("_c17", "_c18"))
    merged["route"] = route
    merged["seed"] = seed
    merged["c17_evidence_gap"] = numeric(merged, "patient_support_strength") - numeric(merged, "patient_opposition_strength")
    merged["c18_delta_sign_matches_c17_gap"] = (
        np.sign(numeric(merged, "directional_delta")) == np.sign(merged["c17_evidence_gap"])
    ).astype(int)
    merged["c18_both_branches_nonzero"] = (
        (numeric(merged, "effective_support_delta") > 1e-8)
        & (numeric(merged, "effective_opposition_delta") > 1e-8)
    ).astype(int)
    merged["support_dominant"] = (numeric(merged, "patient_support_strength") > numeric(merged, "patient_opposition_strength")).astype(int)
    merged["opposition_dominant"] = (numeric(merged, "patient_opposition_strength") > numeric(merged, "patient_support_strength")).astype(int)
    merged["abs_c18_delta"] = numeric(merged, "directional_delta").abs()
    return merged


def branch_summary(patient_table: pd.DataFrame, pairs: pd.DataFrame, route: str, seed: int) -> Dict[str, object]:
    labels = numeric(patient_table, "label").astype(int)
    delta = numeric(patient_table, "directional_delta")
    conflict = numeric(patient_table, "patient_conflict_score")
    high_conflict = conflict >= 0.35
    low_conflict = conflict < 0.35
    positive = labels == 1
    negative = labels == 0
    support_effective = numeric(patient_table, "effective_support_delta")
    opposition_effective = numeric(patient_table, "effective_opposition_delta")
    return {
        "route": route,
        "seed": seed,
        "positive_support_dominant_rate": mean_or_nan(patient_table.loc[positive, "support_dominant"]),
        "negative_opposition_dominant_rate": mean_or_nan(patient_table.loc[negative, "opposition_dominant"]),
        "polarity_sign_match_rate": mean_or_nan(patient_table["c18_delta_sign_matches_c17_gap"]),
        "both_branches_nonzero_rate": mean_or_nan(patient_table["c18_both_branches_nonzero"]),
        "support_delta_spearman": spearman(numeric(patient_table, "patient_support_strength"), support_effective),
        "opposition_delta_spearman": spearman(numeric(patient_table, "patient_opposition_strength"), opposition_effective),
        "positive_mean_delta": mean_or_nan(delta[positive]),
        "negative_mean_delta": mean_or_nan(delta[negative]),
        "positive_delta_below_minus_0_05_rate": mean_or_nan((delta[positive] < -0.05).astype(float)),
        "negative_delta_above_plus_0_05_rate": mean_or_nan((delta[negative] > 0.05).astype(float)),
        "high_conflict_abs_delta": mean_or_nan(patient_table.loc[high_conflict, "abs_c18_delta"]),
        "low_conflict_abs_delta": mean_or_nan(patient_table.loc[low_conflict, "abs_c18_delta"]),
        "high_conflict_delta_smaller": bool(
            np.isfinite(mean_or_nan(patient_table.loc[high_conflict, "abs_c18_delta"]))
            and np.isfinite(mean_or_nan(patient_table.loc[low_conflict, "abs_c18_delta"]))
            and mean_or_nan(patient_table.loc[high_conflict, "abs_c18_delta"])
            < mean_or_nan(patient_table.loc[low_conflict, "abs_c18_delta"])
        ),
        "repaired_inversions": int(pairs["repaired"].sum()),
        "introduced_inversions": int(pairs["introduced"].sum()),
        "base_inversions": int(pairs["base_inversion"].sum()),
        "final_inversions": int(pairs["final_inversion"].sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c17_dir = Path(args.c17_prediction_dir)
    route_dirs = {
        "C18-D": Path(args.c18_directional_prediction_dir),
        "C18-DH": Path(args.c18_hardrank_prediction_dir),
    }

    c17_rows = [c17_polarity_summary(read_validation(c17_dir, seed), seed) for seed in EXPECTED_SEEDS]
    c17_summary = pd.DataFrame(c17_rows)
    patient_tables: List[pd.DataFrame] = []
    pair_tables: List[pd.DataFrame] = []
    branch_rows: List[Dict[str, object]] = []
    for route, route_dir in route_dirs.items():
        for seed in EXPECTED_SEEDS:
            c17 = read_validation(c17_dir, seed)
            c18 = read_validation(route_dir, seed)
            patients = patient_polarity_table(c17, c18, route, seed)
            pairs = pair_table(c17, c18, route, seed)
            patient_tables.append(patients)
            pair_tables.append(pairs)
            branch_rows.append(branch_summary(patients, pairs, route, seed))

    patients_all = pd.concat(patient_tables, ignore_index=True)
    pairs_all = pd.concat(pair_tables, ignore_index=True)
    branch_all = pd.DataFrame(branch_rows)
    patients_all.to_csv(output_dir / "c19_c18_polarity_by_patient.csv", index=False)
    c17_summary.to_csv(output_dir / "c19_c18_polarity_by_seed.csv", index=False)
    pairs_all[pairs_all["repaired"] == 1].to_csv(output_dir / "c19_c18_repaired_pair_polarity.csv", index=False)
    pairs_all[pairs_all["introduced"] == 1].to_csv(output_dir / "c19_c18_introduced_pair_polarity.csv", index=False)
    branch_all.to_csv(output_dir / "c19_c18_branch_compensation.csv", index=False)

    c17_ok = bool(
        (c17_summary["positive_support_dominant_rate"] >= 0.60).all()
        and (c17_summary["negative_opposition_dominant_rate"] >= 0.60).all()
        and c17_summary["support_opposition_gap_nonconstant"].all()
        and (c17_summary["support_opposition_gap_std"] > 1e-8).all()
    )
    decision = "C17_EVIDENCE_POLARITY_USABLE_WITH_CONSTRAINTS" if c17_ok else "C19_POLARITY_BASE_INVALID"
    report = {
        "decision": decision,
        "validation_only": True,
        "expected_seeds": list(EXPECTED_SEEDS),
        "c17_polarity": c17_rows,
        "c18_branch_compensation": branch_rows,
        "answers": {
            "support_strength_effective_support_monotonic": bool(
                (branch_all["support_delta_spearman"].dropna() > 0).all()
            ),
            "opposition_strength_effective_opposition_monotonic": bool(
                (branch_all["opposition_delta_spearman"].dropna() > 0).all()
            ),
            "both_branches_simultaneously_active": bool((branch_all["both_branches_nonzero_rate"] > 0.95).any()),
            "seed_42_polarity_worse": bool(
                branch_all.loc[branch_all["seed"] == 42, "polarity_sign_match_rate"].mean()
                < branch_all.loc[branch_all["seed"] != 42, "polarity_sign_match_rate"].mean()
            ),
            "high_conflict_overcorrected": bool((~branch_all["high_conflict_delta_smaller"]).any()),
            "c17_evidence_usable": c17_ok,
        },
    }
    (output_dir / "c19_polarity_audit.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        "# C19-A Evidence Polarity Audit",
        "",
        f"- Decision: `{decision}`.",
        "- Scope: validation predictions only; no test prediction was read.",
        "- Seeds: `[0, 42, 3407]`.",
        "",
        "## C17 Polarity By Seed",
        "",
        markdown_table(c17_summary),
        "",
        "## C18 Branch Compensation",
        "",
        markdown_table(branch_all),
        "",
        "## Interpretation",
        "",
        f"- Support and opposition dominance satisfy the minimum 0.60/0.60 requirement: `{c17_ok}`.",
        f"- C17 evidence is admissible for a constrained polarity-locked residual: `{c17_ok}`.",
        "- Repaired and introduced pair polarity is stored in the two pair-level CSV reports.",
        "- High-conflict and low-conflict absolute residual comparisons are stored in the branch compensation report.",
    ]
    (output_dir / "c19_pretraining_polarity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    if args.require_pass and not c17_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
