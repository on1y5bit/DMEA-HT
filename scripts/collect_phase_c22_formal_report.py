#!/usr/bin/env python3
"""Consolidate the C22 formal run and apply the validation-AUC decision gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
)
SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c22-run-dir", default="runs/dema_ht_c22_stable_evidence_pooling_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c22_dema")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def probability_column(frame: pd.DataFrame) -> str:
    for column in ("prob", "final_prob", "pred_prob", "prediction", "y_prob"):
        if column in frame.columns:
            return column
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = int(frame["seed"].iloc[0]) if "seed" in frame.columns and len(frame) else seed_from_name(path)
        frame["split"] = split
        frame["route_prob"] = pd.to_numeric(frame[probability_column(frame)], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def auc(labels: Iterable[int], probs: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probs), dtype=float)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, p))


def safe_std(values: Iterable[float]) -> float:
    values_np = np.asarray(list(values), dtype=float)
    return float(values_np.std(ddof=1)) if values_np.size > 1 else 0.0


def pairwise_table(c22_seed: pd.DataFrame, c17_seed: pd.DataFrame) -> pd.DataFrame:
    c17 = c17_seed[["patient_id", "route_prob"]].rename(columns={"route_prob": "c17_prob"})
    merged = c22_seed.merge(c17, on="patient_id", how="inner", validate="one_to_one")
    if len(merged) != len(c22_seed):
        raise RuntimeError("C17/C22 validation patient alignment failed")
    positives = merged[merged["label"].astype(int) == 1].sort_values("patient_id")
    negatives = merged[merged["label"].astype(int) == 0].sort_values("patient_id")
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c13_inversion = float(positive["base_prob"]) < float(negative["base_prob"])
            c17_inversion = float(positive["c17_prob"]) < float(negative["c17_prob"])
            c22_inversion = float(positive["route_prob"]) < float(negative["route_prob"])
            rows.append(
                {
                    "seed": int(c22_seed["seed"].iloc[0]),
                    "positive_patient_id": str(positive["patient_id"]),
                    "negative_patient_id": str(negative["patient_id"]),
                    "c13_positive_prob": float(positive["base_prob"]),
                    "c13_negative_prob": float(negative["base_prob"]),
                    "c17_positive_prob": float(positive["c17_prob"]),
                    "c17_negative_prob": float(negative["c17_prob"]),
                    "c22_positive_prob": float(positive["route_prob"]),
                    "c22_negative_prob": float(negative["route_prob"]),
                    "c13_inversion": int(c13_inversion),
                    "c17_inversion": int(c17_inversion),
                    "c22_inversion": int(c22_inversion),
                    "c22_repaired_vs_c17": int(c17_inversion and not c22_inversion),
                    "c22_introduced_vs_c17": int((not c17_inversion) and c22_inversion),
                }
            )
    return pd.DataFrame(rows)


def shortcut_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    present = [field for field in fields if field in frame.columns]
    if not present or frame["label"].nunique() < 2:
        return None
    matrix = pd.DataFrame(index=frame.index)
    for field in present:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if not values.dropna().empty else 0.0).astype(float)
    if matrix.shape[1] == 0:
        return None
    folds = min(5, int(frame["label"].value_counts().min()))
    if folds < 2:
        return None
    try:
        probabilities = cross_val_predict(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            matrix.to_numpy(),
            frame["label"].astype(int).to_numpy(),
            cv=StratifiedKFold(folds, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        return auc(frame["label"], probabilities)
    except Exception:
        return None


def write_shortcut_audit(c22: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed_value, seed_frame in list(c22.groupby("seed")) + [("pooled", c22)]:
        row: Dict[str, Any] = {"seed": seed_value, "n_rows": int(len(seed_frame))}
        row["selected_structure_shortcut_auc"] = shortcut_auc(seed_frame, SHORTCUT_FIELDS)
        for field in SHORTCUT_FIELDS:
            if field in seed_frame.columns:
                values = pd.to_numeric(seed_frame[field], errors="coerce")
                pair = pd.DataFrame({"delta": seed_frame["delta_c22"], "field": values}).dropna()
                row[f"delta_spearman_{field}"] = (
                    float(pair["delta"].corr(pair["field"], method="spearman")) if len(pair) >= 3 else None
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    args = parse_args()
    c22_dir = resolve_path(args.c22_run_dir)
    c17_dir = resolve_path(args.c17_run_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c22 = read_predictions(c22_dir, "val")
    c17 = read_predictions(c17_dir, "val")
    if c22.empty or c17.empty:
        raise RuntimeError("C22 and C17 validation predictions are required")
    if set(c22["seed"].astype(int)) != set(SEEDS) or set(c17["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Formal validation predictions do not contain exactly seeds 0, 42, 3407")

    c22["delta_c22"] = pd.to_numeric(c22["delta_c22"], errors="coerce")
    c22["base_prob"] = pd.to_numeric(c22["base_prob"], errors="coerce")
    c22["route_prob"] = pd.to_numeric(c22["route_prob"], errors="coerce")
    c17["route_prob"] = pd.to_numeric(c17["route_prob"], errors="coerce")
    diagnostics = c22.merge(
        c17[["patient_id", "seed", "route_prob"]].rename(columns={"route_prob": "c17_prob"}),
        on=["patient_id", "seed"],
        how="left",
        validate="one_to_one",
    )
    diagnostics.to_csv(output_dir / "c22_patient_diagnostics_val.csv", index=False)

    positive_rows: List[Dict[str, Any]] = []
    pairwise_frames: List[pd.DataFrame] = []
    comparison_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        c22_seed = c22[c22["seed"].astype(int) == seed].copy()
        c17_seed = c17[c17["seed"].astype(int) == seed].copy()
        if c22_seed.empty or c17_seed.empty:
            raise RuntimeError(f"missing validation seed {seed}")
        labels = c22_seed["label"].astype(int)
        c13_prob = c22_seed["base_prob"].astype(float)
        c17_prob = c17_seed.set_index("patient_id").loc[c22_seed["patient_id"], "route_prob"].astype(float).to_numpy()
        c22_prob = c22_seed["route_prob"].astype(float)
        comparison_rows.append(
            {
                "seed": seed,
                "c13_auc": auc(labels, c13_prob),
                "c17_auc": auc(labels, c17_prob),
                "c22_auc": auc(labels, c22_prob),
                "c22_minus_c13_auc": auc(labels, c22_prob) - auc(labels, c13_prob),
                "c22_minus_c17_auc": auc(labels, c22_prob) - auc(labels, c17_prob),
            }
        )
        positive_mask = labels.to_numpy() == 1
        base_pred = c13_prob.to_numpy() >= 0.5
        c17_pred = c17_prob >= 0.5
        c22_pred = c22_prob.to_numpy() >= 0.5
        positive_rows.append(
            {
                "seed": seed,
                "positive_count": int(positive_mask.sum()),
                "mean_positive_delta_c22": float(c22_seed.loc[positive_mask, "delta_c22"].mean()),
                "fraction_positive_delta_below_minus_0_10": float(
                    (c22_seed.loc[positive_mask, "delta_c22"] < -0.10).mean()
                ),
                "c13_tp_to_fn": int((positive_mask & base_pred & ~c22_pred).sum()),
                "c13_fn_to_tp": int((positive_mask & ~base_pred & c22_pred).sum()),
                "c17_tp_to_fn": int((positive_mask & c17_pred & ~c22_pred).sum()),
                "c17_fn_to_tp": int((positive_mask & ~c17_pred & c22_pred).sum()),
                "mean_positive_prob_change_vs_c13": float(
                    (c22_prob.to_numpy()[positive_mask] - c13_prob.to_numpy()[positive_mask]).mean()
                ),
            }
        )
        pairwise = pairwise_table(c22_seed, c17_seed)
        pairwise_frames.append(pairwise)
        for route, column in (("C13", "c13_inversion"), ("C17", "c17_inversion"), ("C22", "c22_inversion")):
            inversion_rows.append(
                {
                    "seed": seed,
                    "route": route,
                    "pairwise_inversions": int(pairwise[column].sum()),
                    "pairwise_rows": int(len(pairwise)),
                    "repaired_vs_c17": int(pairwise["c22_repaired_vs_c17"].sum()) if route == "C22" else 0,
                    "introduced_vs_c17": int(pairwise["c22_introduced_vs_c17"].sum()) if route == "C22" else 0,
                }
            )

    pd.DataFrame(positive_rows).to_csv(output_dir / "c22_positive_preservation_audit.csv", index=False)
    pairwise_frame = pd.concat(pairwise_frames, ignore_index=True)
    pairwise_frame.to_csv(output_dir / "c22_pairwise_ranking_val.csv", index=False)
    inversion_frame = pd.DataFrame(inversion_rows)
    inversion_frame.to_csv(output_dir / "c22_pairwise_inversion_summary.csv", index=False)
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(output_dir / "c22_c13_c17_comparison.csv", index=False)

    positive_frame = pd.DataFrame(positive_rows)
    health_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = c22[c22["seed"].astype(int) == seed]
        delta = seed_frame["delta_c22"].astype(float).to_numpy()
        health_rows.append(
            {
                "seed": seed,
                "mean_delta_c22": float(delta.mean()),
                "std_delta_c22": float(delta.std(ddof=1)),
                "min_delta_c22": float(delta.min()),
                "max_delta_c22": float(delta.max()),
                "fraction_at_lower_bound": float((delta <= -0.50 + 1e-5).mean()),
                "fraction_at_upper_bound": float((delta >= 0.50 - 1e-5).mean()),
                "nonzero_variance": bool(delta.std(ddof=1) > 1e-6),
            }
        )
    health_frame = pd.DataFrame(health_rows)
    health_frame.to_csv(output_dir / "c22_residual_health_audit.csv", index=False)
    shortcut_frame = write_shortcut_audit(c22, output_dir / "c22_shortcut_residual_audit.csv")

    c17_auc = comparison_frame["c17_auc"].to_numpy(dtype=float)
    c22_auc = comparison_frame["c22_auc"].to_numpy(dtype=float)
    c13_auc = comparison_frame["c13_auc"].to_numpy(dtype=float)
    auc_diff = c22_auc - c17_auc
    c17_mean = float(c17_auc.mean())
    c22_mean = float(c22_auc.mean())
    c17_std = safe_std(c17_auc)
    c22_std = safe_std(c22_auc)
    positive_safe = bool(
        (positive_frame["mean_positive_delta_c22"] >= -0.02).all()
        and (positive_frame["fraction_positive_delta_below_minus_0_10"] <= 0.25).all()
        and (positive_frame["c13_tp_to_fn"] == 0).all()
    )
    inversion_seed = inversion_frame[inversion_frame["route"] == "C22"].set_index("seed")
    c17_inversions = inversion_frame[inversion_frame["route"] == "C17"].set_index("seed")["pairwise_inversions"]
    inversion_safe = bool(all(int(inversion_seed.loc[seed, "pairwise_inversions"]) <= int(c17_inversions.loc[seed]) for seed in SEEDS))
    residual_healthy = bool(health_frame["nonzero_variance"].all() and (health_frame["fraction_at_lower_bound"] < 0.25).all() and (health_frame["fraction_at_upper_bound"] < 0.25).all())
    shortcut_values = pd.to_numeric(shortcut_frame["selected_structure_shortcut_auc"], errors="coerce").dropna()
    shortcut_max = float(shortcut_values.max()) if not shortcut_values.empty else float("nan")
    shortcut_safe = bool(np.isfinite(shortcut_max) and shortcut_max <= 0.55)
    training_valid = bool(np.isfinite(c22_auc).all() and len(c22) > 0 and len(pairwise_frame) > 0)
    two_of_three_above = int((auc_diff > 0).sum()) >= 2
    no_seed_drop = bool((auc_diff >= -0.005).all())
    auc_stability = bool(c22_std <= 0.02)

    if not training_valid:
        decision = "DEMA_C22_TRAINING_INVALID"
    elif not positive_safe:
        decision = "DEMA_C22_POSITIVE_SUPPRESSION"
    elif not inversion_safe:
        decision = "DEMA_C22_INVERSION_WORSENING"
    elif not residual_healthy:
        decision = "DEMA_C22_RESIDUAL_COLLAPSE"
    elif not shortcut_safe or not no_seed_drop or not auc_stability:
        decision = "DEMA_C22_FORMAL_FAIL_KEEP_C17"
    elif c22_mean > c17_mean and two_of_three_above:
        decision = "PROMOTE_DEMA_C22_STABLE_EVIDENCE_POOLING"
    elif c22_mean <= c17_mean and abs(c22_mean - c17_mean) < 0.002 and c22_std < c17_std:
        decision = "DEMA_C22_SIMPLIFICATION_COMPETITIVE_KEEP_C17"
    elif c22_mean < c17_mean:
        decision = "DEMA_C22_PROPAGATION_CONTAINS_NECESSARY_SIGNAL"
    else:
        decision = "DEMA_C22_FORMAL_FAIL_KEEP_C17"

    stability_lines = [
        "# Phase C22 Stable Evidence Pooling Seed Stability",
        "",
        "- official route: DEMA-HT C22 stable evidence pooling",
        "- formal seeds: `0, 42, 3407`",
        "- checkpoint selection: validation AUC only",
        "- test role: reporting-only after validation selection",
        "- shortcut fields: audit-only and excluded from the model",
        "",
        "## Validation AUC",
        "",
        f"- C13 mean/std: `{c13_auc.mean():.10f} +/- {safe_std(c13_auc):.10f}`",
        f"- C17 mean/std: `{c17_mean:.10f} +/- {c17_std:.10f}`",
        f"- C22 mean/std: `{c22_mean:.10f} +/- {c22_std:.10f}`",
        f"- C22 minus C17 mean: `{c22_mean - c17_mean:+.10f}`",
        f"- seeds above C17: `{int((auc_diff > 0).sum())}/3`",
        f"- largest seed drop versus C17: `{float(auc_diff.min()):+.10f}`",
        "",
        "## Safety Gates",
        "",
        f"- training validity: `{training_valid}`",
        f"- positive preservation: `{positive_safe}`",
        f"- inversion non-worsening: `{inversion_safe}`",
        f"- residual health: `{residual_healthy}`",
        f"- selected-structure shortcut-only AUC: `{shortcut_max:.10f}`; pass=`{shortcut_safe}`",
        f"- AUC stability: `{auc_stability}`",
        "",
        f"## Decision: `{decision}`",
        "",
        "The C22 head reads only the valid-mask mean of real pre-propagation image, text, and bio evidence projector nodes. No clinical mechanism node is claimed to be individually identified by this experiment.",
    ]
    (output_dir / "c22_seed_stability_report.md").write_text("\n".join(stability_lines) + "\n", encoding="utf-8")

    necessity_lines = [
        "# Phase C22 Mechanism Propagation Necessity Report",
        "",
        "C22 freezes the C13 predictor and C17 evidence projector parameters, bypasses mechanism propagation and downstream aggregation, and trains only a bounded residual over the stable pre-propagation pool.",
        "",
        f"- C17 validation AUC mean/std: `{c17_mean:.10f} +/- {c17_std:.10f}`",
        f"- C22 validation AUC mean/std: `{c22_mean:.10f} +/- {c22_std:.10f}`",
        f"- decision: `{decision}`",
        "",
        "Interpretation is route-level only: a competitive C22 result supports simplification of the unstable propagation stage, whereas a lower C22 result supports retaining the propagated C17 route. It does not prove that a particular graph node or clinical mechanism is necessary.",
    ]
    (output_dir / "c22_mechanism_propagation_necessity_report.md").write_text("\n".join(necessity_lines) + "\n", encoding="utf-8")

    final_lines = [
        "# Phase C22 DEMA-HT Final Report",
        "",
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`",
        "- runtime contract: `/home/linruixin/chen/conda/envs/ma`",
        "- data root: `/data/csb/DMEA-HT/HT_2025.12_25`",
        "- formal seeds: `0, 42, 3407`",
        "- validation AUC is the sole selection and decision metric",
        "- secondary ranking metrics are intentionally omitted from the formal route tables",
        "- test is reporting-only and was not used for selection or promotion",
        "- no branch or worktree was created",
        "",
        "## Design",
        "",
        "C22 uses the frozen C13 base logit, frozen C17 evidence projectors, a valid-mask mean over the 14 real pre-propagation projector nodes, and one zero-initialized bounded residual head. Mechanism propagation, downstream role scoring, conflict aggregation, and the final mechanism head are not called by the C22 forward path.",
        "",
        "## Results",
        "",
        f"- C13 validation AUC mean/std: `{c13_auc.mean():.10f} +/- {safe_std(c13_auc):.10f}`",
        f"- C17 validation AUC mean/std: `{c17_mean:.10f} +/- {c17_std:.10f}`",
        f"- C22 validation AUC mean/std: `{c22_mean:.10f} +/- {c22_std:.10f}`",
        f"- positive preservation gate: `{positive_safe}`",
        f"- inversion non-worsening gate: `{inversion_safe}`",
        f"- residual health gate: `{residual_healthy}`",
        f"- selected-structure shortcut audit gate: `{shortcut_safe}`",
        "",
        f"## Final Decision: `{decision}`",
        "",
        "The current strict-best route remains C17 unless the decision label explicitly promotes C22. All artifacts in this directory are reproducibility and audit outputs; shortcut variables are never classifier inputs.",
    ]
    (output_dir / "phase_c22_dema_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    decision_payload = {
        "phase": "C22",
        "decision": decision,
        "c13_mean_auc": float(c13_auc.mean()),
        "c17_mean_auc": c17_mean,
        "c17_std_auc": c17_std,
        "c22_mean_auc": c22_mean,
        "c22_std_auc": c22_std,
        "positive_preservation_pass": positive_safe,
        "inversion_non_worsening_pass": inversion_safe,
        "residual_health_pass": residual_healthy,
        "shortcut_pass": shortcut_safe,
        "test_used_for_decision": False,
        "artifacts": [path.name for path in sorted(output_dir.iterdir()) if path.is_file()],
    }
    (output_dir / "c22_final_decision.json").write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision_payload, ensure_ascii=True))
    if args.require_pass and decision not in {
        "PROMOTE_DEMA_C22_STABLE_EVIDENCE_POOLING",
        "DEMA_C22_SIMPLIFICATION_COMPETITIVE_KEEP_C17",
        "DEMA_C22_PROPAGATION_CONTAINS_NECESSARY_SIGNAL",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
