#!/usr/bin/env python3
"""Consolidate C23 formal outputs and apply its validation-AUC decision gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SEEDS = (0, 42, 3407)
SHORTCUT_FIELDS = (
    "selected_n_visits", "used_images", "image_padding_count", "has_bio",
    "bio_missing_count", "report_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c23-run-dir", default="runs/dema_ht_c23_confidence_gated_residual_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c23_dema")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else seed_from_name(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def probability_column(frame: pd.DataFrame) -> str:
    for column in ("prob", "final_prob", "pred_prob", "prediction", "y_prob"):
        if column in frame:
            return column
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def auc(labels: Iterable[int], probs: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(list(labels), dtype=int), np.asarray(list(probs), dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def shortcut_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    present = [field for field in fields if field in frame]
    if not present or frame["label"].nunique() < 2:
        return None
    matrix = pd.DataFrame(index=frame.index)
    for field in present:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if not values.dropna().empty else 0.0)
    folds = min(5, int(frame["label"].value_counts().min()))
    if folds < 2:
        return None
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"), matrix.to_numpy(),
        frame["label"].astype(int).to_numpy(),
        cv=StratifiedKFold(folds, shuffle=True, random_state=42), method="predict_proba",
    )[:, 1]
    return auc(frame["label"], probabilities)


def pairwise_table(seed_frame: pd.DataFrame) -> pd.DataFrame:
    positives = seed_frame[seed_frame["label"].astype(int) == 1].sort_values("patient_id")
    negatives = seed_frame[seed_frame["label"].astype(int) == 0].sort_values("patient_id")
    rows: List[Dict[str, Any]] = []
    seed = int(seed_frame["seed"].iloc[0])
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c17_inversion = float(positive["frozen_c17_prob"]) < float(negative["frozen_c17_prob"])
            c23_inversion = float(positive["prob"]) < float(negative["prob"])
            rows.append({
                "seed": seed,
                "positive_patient_id": positive["patient_id"],
                "negative_patient_id": negative["patient_id"],
                "c17_positive_prob": positive["frozen_c17_prob"],
                "c17_negative_prob": negative["frozen_c17_prob"],
                "c23_positive_prob": positive["prob"],
                "c23_negative_prob": negative["prob"],
                "c17_inversion": int(c17_inversion),
                "c23_inversion": int(c23_inversion),
                "c23_repaired_vs_c17": int(c17_inversion and not c23_inversion),
                "c23_introduced_vs_c17": int((not c17_inversion) and c23_inversion),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    c23_run, c17_run, output = map(resolve_path, (args.c23_run_dir, args.c17_run_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    c23, c17 = read_predictions(c23_run, "val"), read_predictions(c17_run, "val")
    if c23.empty or c17.empty:
        raise RuntimeError("C23 and C17 validation predictions are required")
    if set(c23["seed"].astype(int)) != set(SEEDS) or set(c17["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Formal predictions must contain exactly seeds 0, 42, 3407")
    c17 = c17.rename(columns={probability_column(c17): "c17_reference_prob"})
    diagnostics = c23.merge(
        c17[["patient_id", "seed", "c17_reference_prob"]], on=["patient_id", "seed"],
        how="left", validate="one_to_one",
    )
    if diagnostics["c17_reference_prob"].isna().any():
        raise RuntimeError("C17/C23 patient alignment failed")
    reproduction_error = float((diagnostics["frozen_c17_prob"] - diagnostics["c17_reference_prob"]).abs().max())
    diagnostics.to_csv(output / "c23_patient_diagnostics_val.csv", index=False)

    comparison_rows, preservation_rows, confidence_rows, high_rows, transition_frames = [], [], [], [], []
    pairwise_frames, inversion_rows, health_rows, shortcut_rows = [], [], [], []
    for seed in SEEDS:
        frame = diagnostics[diagnostics["seed"].astype(int) == seed].copy()
        labels = frame["label"].astype(int).to_numpy()
        c17_prob = frame["frozen_c17_prob"].astype(float).to_numpy()
        c23_prob = frame["prob"].astype(float).to_numpy()
        delta = frame["delta_c23"].astype(float).to_numpy()
        c17_pred, c23_pred = c17_prob >= 0.5, c23_prob >= 0.5
        positive, negative = labels == 1, labels == 0
        comparison_rows.append({
            "seed": seed, "c17_auc": auc(labels, c17_prob), "c23_auc": auc(labels, c23_prob),
            "c23_minus_c17_auc": auc(labels, c23_prob) - auc(labels, c17_prob),
        })
        transition = frame[["patient_id", "seed", "label", "frozen_c17_prob", "prob", "delta_c23", "confidence_group"]].copy()
        transition["c17_prediction"] = c17_pred.astype(int)
        transition["c23_prediction"] = c23_pred.astype(int)
        transition["transition"] = [f"{a}->{b}" for a, b in zip(c17_pred.astype(int), c23_pred.astype(int))]
        transition_frames.append(transition)
        preservation_rows.append({
            "seed": seed,
            "mean_positive_delta_c23": float(delta[positive].mean()),
            "fraction_positive_delta_below_minus_0_05": float((delta[positive] < -0.05).mean()),
            "c17_tp_to_fn": int((positive & c17_pred & ~c23_pred).sum()),
            "c17_fn_to_tp": int((positive & ~c17_pred & c23_pred).sum()),
            "mean_negative_delta_c23": float(delta[negative].mean()),
            "fraction_negative_delta_above_plus_0_05": float((delta[negative] > 0.05).mean()),
            "c17_tn_to_fp": int((negative & ~c17_pred & c23_pred).sum()),
            "c17_fp_to_tn": int((negative & c17_pred & ~c23_pred).sum()),
        })
        for group in ("low", "medium", "high"):
            group_frame = frame[frame["confidence_group"] == group]
            confidence_rows.append({
                "seed": seed, "confidence_group": group, "n": len(group_frame),
                "mean_abs_delta": float(group_frame["delta_c23"].abs().mean()) if len(group_frame) else 0.0,
                "mean_delta": float(group_frame["delta_c23"].mean()) if len(group_frame) else 0.0,
                "mean_gate": float(group_frame["confidence_gate"].mean()) if len(group_frame) else 0.0,
            })
        high = frame[frame["confidence_group"] == "high"]
        high_label = high["label"].astype(int).to_numpy()
        high_c17 = high["frozen_c17_prob"].to_numpy() >= 0.5
        high_c23 = high["prob"].to_numpy() >= 0.5
        high_rows.append({
            "seed": seed, "n": len(high),
            "mean_abs_delta": float(high["delta_c23"].abs().mean()) if len(high) else 0.0,
            "max_abs_delta": float(high["delta_c23"].abs().max()) if len(high) else 0.0,
            "tp_to_fn": int(((high_label == 1) & high_c17 & ~high_c23).sum()),
            "tn_to_fp": int(((high_label == 0) & ~high_c17 & high_c23).sum()),
        })
        pairwise = pairwise_table(frame)
        pairwise_frames.append(pairwise)
        inversion_rows.append({
            "seed": seed,
            "c17_inversions": int(pairwise["c17_inversion"].sum()),
            "c23_inversions": int(pairwise["c23_inversion"].sum()),
            "change": int(pairwise["c23_inversion"].sum() - pairwise["c17_inversion"].sum()),
            "repaired": int(pairwise["c23_repaired_vs_c17"].sum()),
            "introduced": int(pairwise["c23_introduced_vs_c17"].sum()),
        })
        health_rows.append({
            "seed": seed, "mean_delta": float(delta.mean()), "std_delta": float(delta.std(ddof=1)),
            "min_delta": float(delta.min()), "max_delta": float(delta.max()),
            "fraction_near_negative_bound": float((delta <= -0.15 + 1e-4).mean()),
            "fraction_near_positive_bound": float((delta >= 0.15 - 1e-4).mean()),
            "nonzero_variance": bool(delta.std(ddof=1) > 1e-6),
        })
        shortcut_row: Dict[str, Any] = {"seed": seed, "selected_structure_shortcut_auc": shortcut_auc(frame, SHORTCUT_FIELDS)}
        for field in SHORTCUT_FIELDS:
            if field in frame:
                shortcut_row[f"delta_spearman_{field}"] = frame["delta_c23"].corr(pd.to_numeric(frame[field], errors="coerce"), method="spearman")
        shortcut_rows.append(shortcut_row)

    comparison = pd.DataFrame(comparison_rows)
    preservation = pd.DataFrame(preservation_rows)
    confidence = pd.DataFrame(confidence_rows)
    high = pd.DataFrame(high_rows)
    transitions = pd.concat(transition_frames, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    inversions = pd.DataFrame(inversion_rows)
    health = pd.DataFrame(health_rows)
    shortcuts = pd.DataFrame(shortcut_rows)
    transitions.to_csv(output / "c23_c17_transition_audit.csv", index=False)
    preservation.to_csv(output / "c23_positive_negative_preservation_audit.csv", index=False)
    confidence.to_csv(output / "c23_confidence_group_audit.csv", index=False)
    high.to_csv(output / "c23_high_confidence_damage_audit.csv", index=False)
    pairwise.to_csv(output / "c23_pairwise_ranking_val.csv", index=False)
    inversions.to_csv(output / "c23_pairwise_inversion_summary.csv", index=False)
    health.to_csv(output / "c23_residual_health_audit.csv", index=False)
    shortcuts.to_csv(output / "c23_shortcut_residual_audit.csv", index=False)
    comparison.to_csv(output / "c23_c17_comparison.csv", index=False)

    for source, target in (
        ("metrics_by_epoch.csv", "c23_metrics_by_epoch.csv"),
        ("metrics_by_seed.csv", "c23_metrics_by_seed.csv"),
        ("metrics_summary.csv", "c23_metrics_summary.csv"),
    ):
        pd.read_csv(c23_run / "reports" / source).to_csv(output / target, index=False)

    c17_auc = comparison["c17_auc"].to_numpy(dtype=float)
    c23_auc = comparison["c23_auc"].to_numpy(dtype=float)
    difference = c23_auc - c17_auc
    positive_safe = bool(
        (preservation["mean_positive_delta_c23"] >= -0.005).all()
        and (preservation["fraction_positive_delta_below_minus_0_05"] <= 0.10).all()
        and int(preservation["c17_tp_to_fn"].sum()) == 0
    )
    negative_safe = bool(
        (preservation["mean_negative_delta_c23"] <= 0.005).all()
        and (preservation["fraction_negative_delta_above_plus_0_05"] <= 0.10).all()
        and int(preservation["c17_tn_to_fp"].sum()) <= int(preservation["c17_fp_to_tn"].sum())
    )
    high_safe = bool(
        (high["mean_abs_delta"] <= 0.02).all() and (high["max_abs_delta"] <= 0.05).all()
        and int(high["tp_to_fn"].sum()) == 0 and int(high["tn_to_fp"].sum()) == 0
    )
    inversion_safe = bool(
        int((inversions["change"] < 0).sum()) >= 2
        and int(inversions["repaired"].sum()) > int(inversions["introduced"].sum())
        and (inversions["change"] <= 3).all()
    )
    residual_healthy = bool(
        health["nonzero_variance"].all()
        and (health["fraction_near_negative_bound"] < 0.25).all()
        and (health["fraction_near_positive_bound"] < 0.25).all()
    )
    shortcut_max = float(pd.to_numeric(shortcuts["selected_structure_shortcut_auc"], errors="coerce").max())
    shortcut_safe = bool(np.isfinite(shortcut_max) and shortcut_max <= 0.55)
    training_valid = bool(np.isfinite(c23_auc).all() and reproduction_error <= 1e-5 and len(pairwise) > 0)
    auc_gate = bool(c23_auc.mean() > c17_auc.mean() and int((difference > 0).sum()) >= 2 and (difference >= -0.005).all() and safe_std(c23_auc) <= 0.02)

    if not training_valid:
        decision = "DEMA_C23_TRAINING_INVALID"
    elif not positive_safe:
        decision = "DEMA_C23_POSITIVE_SUPPRESSION"
    elif not negative_safe:
        decision = "DEMA_C23_NEGATIVE_INFLATION"
    elif not high_safe:
        decision = "DEMA_C23_HIGH_CONFIDENCE_DAMAGE"
    elif not inversion_safe:
        decision = "DEMA_C23_INVERSION_WORSENING"
    elif not residual_healthy:
        decision = "DEMA_C23_RESIDUAL_COLLAPSE"
    elif not shortcut_safe or (difference < -0.005).any() or safe_std(c23_auc) > 0.02:
        decision = "DEMA_C23_FORMAL_FAIL_KEEP_C17"
    elif auc_gate:
        decision = "PROMOTE_DEMA_C23_CONFIDENCE_GATED_LOCAL_RESIDUAL"
    else:
        decision = "DEMA_C23_NO_AUC_GAIN_KEEP_C17"

    common = [
        "- formal seeds: `0, 42, 3407`",
        "- checkpoint selection: validation AUC only",
        "- test role: reporting-only after validation selection",
        f"- C17 validation AUC mean/std: `{c17_auc.mean():.10f} +/- {safe_std(c17_auc):.10f}`",
        f"- C23 validation AUC mean/std: `{c23_auc.mean():.10f} +/- {safe_std(c23_auc):.10f}`",
        f"- C23 minus C17 mean: `{c23_auc.mean() - c17_auc.mean():+.10f}`",
        f"- positive preservation: `{positive_safe}`",
        f"- negative preservation: `{negative_safe}`",
        f"- high-confidence protection: `{high_safe}`",
        f"- inversion gate: `{inversion_safe}`",
        f"- residual health: `{residual_healthy}`",
        f"- selected-structure shortcut-only AUC: `{shortcut_max:.10f}`; pass=`{shortcut_safe}`",
        f"- decision: `{decision}`",
    ]
    (output / "c23_seed_stability_report.md").write_text("# C23 Seed Stability\n\n" + "\n".join(common) + "\n", encoding="utf-8")
    (output / "c23_confidence_gated_local_residual_report.md").write_text(
        "# C23 Confidence-Gated Local Residual\n\n"
        "C23 freezes the complete C17 predictor and reads only its frozen latent pathological-mechanism interaction representation. "
        "The deterministic gate is largest near the C17 decision boundary and cannot use labels or shortcut fields.\n\n"
        + "\n".join(common) + "\n", encoding="utf-8",
    )
    (output / "phase_c23_dema_final_report.md").write_text(
        "# Phase C23 DEMA-HT Final Report\n\n"
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`\n"
        "- runtime: `/home/linruixin/chen/conda/envs/ma`\n"
        "- data: `/data/csb/DMEA-HT/HT_2025.12_25`\n"
        "- current strict best remains C17 unless the decision explicitly promotes C23\n"
        + "\n".join(common) + "\n", encoding="utf-8",
    )
    payload = {
        "phase": "C23", "decision": decision,
        "c17_mean_auc": float(c17_auc.mean()), "c17_std_auc": safe_std(c17_auc),
        "c23_mean_auc": float(c23_auc.mean()), "c23_std_auc": safe_std(c23_auc),
        "positive_preservation_pass": positive_safe, "negative_preservation_pass": negative_safe,
        "high_confidence_pass": high_safe, "inversion_pass": inversion_safe,
        "residual_health_pass": residual_healthy, "shortcut_pass": shortcut_safe,
        "test_used_for_decision": False,
    }
    (output / "c23_final_decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if args.require_pass and decision != "PROMOTE_DEMA_C23_CONFIDENCE_GATED_LOCAL_RESIDUAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
