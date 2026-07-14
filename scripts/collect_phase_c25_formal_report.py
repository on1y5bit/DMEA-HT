#!/usr/bin/env python3
"""Consolidate C25 outputs and apply the final residual promotion gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SEEDS = (0, 42, 3407)
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits", "used_images", "image_padding_count", "has_bio",
    "bio_missing_count", "report_length",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits", "raw_n_images")
ALL_SHORTCUT_FIELDS = SELECTED_SHORTCUT_FIELDS + RAW_SHORTCUT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c25-run-dir", default="runs/dema_ht_c25_pairwise_ranking_residual_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--prior-comparison", default="analysis_reports/phase_c24_dema/c24_c17_c23_comparison.csv")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c25_dema")
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

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probs), dtype=float)
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


def pairwise_table(frame: pd.DataFrame) -> pd.DataFrame:
    positives = frame[frame["label"].astype(int) == 1].sort_values("patient_id")
    negatives = frame[frame["label"].astype(int) == 0].sort_values("patient_id")
    seed = int(frame["seed"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c17_inversion = float(positive["frozen_c17_prob"]) < float(negative["frozen_c17_prob"])
            c25_inversion = float(positive["prob"]) < float(negative["prob"])
            rows.append({
                "seed": seed,
                "positive_patient_id": positive["patient_id"],
                "negative_patient_id": negative["patient_id"],
                "c17_positive_prob": positive["frozen_c17_prob"],
                "c17_negative_prob": negative["frozen_c17_prob"],
                "c25_positive_prob": positive["prob"],
                "c25_negative_prob": negative["prob"],
                "c17_margin": float(positive["frozen_c17_logit"]) - float(negative["frozen_c17_logit"]),
                "c25_margin": float(positive["logit"]) - float(negative["logit"]),
                "c17_inversion": int(c17_inversion),
                "c25_inversion": int(c25_inversion),
                "c25_repaired_vs_c17": int(c17_inversion and not c25_inversion),
                "c25_introduced_vs_c17": int((not c17_inversion) and c25_inversion),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    c25_run = resolve_path(args.c25_run_dir)
    c17_run = resolve_path(args.c17_run_dir)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    c25 = read_predictions(c25_run, "val")
    c17 = read_predictions(c17_run, "val")
    if c25.empty or c17.empty:
        raise RuntimeError("C25 and C17 validation predictions are required")
    if set(c25["seed"].astype(int)) != set(SEEDS) or set(c17["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Formal predictions must contain exactly seeds 0, 42, 3407")
    required = {
        "patient_id", "seed", "label", "frozen_c17_logit", "frozen_c17_prob",
        "frozen_c17_correct_audit_only", "confidence_gate", "confidence_group",
        "delta_c25", "logit", "prob", "final_predicted_class",
    }
    missing = sorted(required - set(c25.columns))
    if missing:
        raise RuntimeError(f"C25 prediction export missing columns: {missing}")

    c17 = c17.rename(columns={probability_column(c17): "c17_reference_prob"})
    diagnostics = c25.merge(
        c17[["patient_id", "seed", "label", "c17_reference_prob"]],
        on=["patient_id", "seed"], how="left", validate="one_to_one", suffixes=("", "_reference"),
    )
    if diagnostics["c17_reference_prob"].isna().any() or not diagnostics["label"].astype(int).eq(diagnostics["label_reference"].astype(int)).all():
        raise RuntimeError("C17/C25 patient or label alignment failed")
    reproduction_error = float((diagnostics["frozen_c17_prob"] - diagnostics["c17_reference_prob"]).abs().max())
    diagnostics.to_csv(output / "c25_patient_diagnostics_val.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    transition_frames: List[pd.DataFrame] = []
    correct_rows: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []
    confidence_rows: List[Dict[str, Any]] = []
    pairwise_frames: List[pd.DataFrame] = []
    inversion_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    constant_prediction = False
    for seed in SEEDS:
        frame = diagnostics[diagnostics["seed"].astype(int) == seed].copy()
        labels = frame["label"].astype(int).to_numpy()
        c17_prob = frame["frozen_c17_prob"].astype(float).to_numpy()
        c25_prob = frame["prob"].astype(float).to_numpy()
        delta = frame["delta_c25"].astype(float).to_numpy()
        frozen_correct = frame["frozen_c17_correct_audit_only"].astype(bool).to_numpy()
        c17_pred, c25_pred = c17_prob >= 0.5, c25_prob >= 0.5
        positive, negative = labels == 1, labels == 0
        correct_positive, correct_negative = frozen_correct & positive, frozen_correct & negative
        constant_prediction = constant_prediction or np.unique(c25_prob).size <= 1
        c17_auc, c25_auc = auc(labels, c17_prob), auc(labels, c25_prob)
        comparison_rows.append({"seed": seed, "c17_auc": c17_auc, "c25_auc": c25_auc, "c25_minus_c17_auc": c25_auc - c17_auc})

        transition = frame[["patient_id", "seed", "label", "frozen_c17_prob", "prob", "delta_c25", "frozen_c17_correct_audit_only"]].copy()
        transition["c17_prediction"] = c17_pred.astype(int)
        transition["c25_prediction"] = c25_pred.astype(int)
        transition["transition"] = [f"{a}->{b}" for a, b in zip(c17_pred.astype(int), c25_pred.astype(int))]
        transition_frames.append(transition)
        correct_rows.append({
            "seed": seed,
            "correct_positive_n": int(correct_positive.sum()),
            "correct_positive_mean_delta": float(delta[correct_positive].mean()) if correct_positive.any() else 0.0,
            "correct_positive_wrong_direction_fraction": float((delta[correct_positive] < 0).mean()) if correct_positive.any() else 0.0,
            "correct_negative_n": int(correct_negative.sum()),
            "correct_negative_mean_delta": float(delta[correct_negative].mean()) if correct_negative.any() else 0.0,
            "correct_negative_wrong_direction_fraction": float((delta[correct_negative] > 0).mean()) if correct_negative.any() else 0.0,
            "tp_to_fn": int((positive & c17_pred & ~c25_pred).sum()),
            "fn_to_tp": int((positive & ~c17_pred & c25_pred).sum()),
            "tn_to_fp": int((negative & ~c17_pred & c25_pred).sum()),
            "fp_to_tn": int((negative & c17_pred & ~c25_pred).sum()),
        })
        positive_fraction = float((delta > 0).mean())
        negative_fraction = float((delta < 0).mean())
        positive_mean = float(delta[positive].mean())
        negative_mean = float(delta[negative].mean())
        global_rows.append({
            "seed": seed, "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "positive_delta_fraction": positive_fraction, "negative_delta_fraction": negative_fraction,
            "positive_class_mean_delta": positive_mean, "negative_class_mean_delta": negative_mean,
            "same_sign_fraction_ge_0_90": bool(max(positive_fraction, negative_fraction) >= 0.90),
            "both_class_means_same_sign_and_large": bool(np.sign(positive_mean) == np.sign(negative_mean) and abs(positive_mean) > 0.01 and abs(negative_mean) > 0.01),
        })
        for group in ("low", "medium", "high"):
            subset = frame[frame["confidence_group"].eq(group)]
            confidence_rows.append({
                "seed": seed, "confidence_group": group, "n": len(subset),
                "mean_abs_delta": float(subset["delta_c25"].abs().mean()) if len(subset) else 0.0,
                "mean_delta": float(subset["delta_c25"].mean()) if len(subset) else 0.0,
                "mean_gate": float(subset["confidence_gate"].mean()) if len(subset) else 0.0,
            })
        pairs = pairwise_table(frame)
        pairwise_frames.append(pairs)
        c17_inversions = int(pairs["c17_inversion"].sum())
        c25_inversions = int(pairs["c25_inversion"].sum())
        inversion_rows.append({
            "seed": seed, "c17_inversions": c17_inversions, "c25_inversions": c25_inversions,
            "change": c25_inversions - c17_inversions,
            "repaired": int(pairs["c25_repaired_vs_c17"].sum()),
            "introduced": int(pairs["c25_introduced_vs_c17"].sum()),
        })
        bound = 0.15 * frame["confidence_gate"].astype(float).to_numpy()
        low = frame["confidence_group"].eq("low").to_numpy()
        high = frame["confidence_group"].eq("high").to_numpy()
        health_rows.append({
            "seed": seed, "mean_delta": float(delta.mean()), "std_delta": float(delta.std(ddof=1)),
            "min_delta": float(delta.min()), "max_delta": float(delta.max()),
            "fraction_near_negative_bound": float((delta <= -0.99 * bound).mean()),
            "fraction_near_positive_bound": float((delta >= 0.99 * bound).mean()),
            "mean_abs_delta_low": float(np.abs(delta[low]).mean()) if low.any() else 0.0,
            "mean_abs_delta_high": float(np.abs(delta[high]).mean()) if high.any() else 0.0,
            "nonzero_variance": bool(delta.std(ddof=1) > 1e-6),
        })
        shortcut_row: Dict[str, Any] = {"seed": seed, "selected_structure_shortcut_auc": shortcut_auc(frame, SELECTED_SHORTCUT_FIELDS)}
        for field in RAW_SHORTCUT_FIELDS:
            if field in frame:
                raw = pd.DataFrame({"label": frame["label"], "value": pd.to_numeric(frame[field], errors="coerce")}).dropna()
                raw_auc = auc(raw["label"], raw["value"]) if raw["label"].nunique() > 1 else 0.5
                shortcut_row[f"{field}_orientation_invariant_label_auc_warning"] = max(raw_auc, 1.0 - raw_auc)
        for field in ALL_SHORTCUT_FIELDS:
            if field in frame:
                shortcut_row[f"delta_spearman_{field}"] = frame["delta_c25"].corr(pd.to_numeric(frame[field], errors="coerce"), method="spearman")
        shortcut_rows.append(shortcut_row)

    comparison = pd.DataFrame(comparison_rows)
    transitions = pd.concat(transition_frames, ignore_index=True)
    correct = pd.DataFrame(correct_rows)
    global_shift = pd.DataFrame(global_rows)
    confidence = pd.DataFrame(confidence_rows)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    inversions = pd.DataFrame(inversion_rows)
    health = pd.DataFrame(health_rows)
    shortcuts = pd.DataFrame(shortcut_rows)
    transitions.to_csv(output / "c25_c17_transition_audit.csv", index=False)
    correct.to_csv(output / "c25_correct_case_direction_audit.csv", index=False)
    global_shift.to_csv(output / "c25_global_shift_audit.csv", index=False)
    confidence.to_csv(output / "c25_confidence_group_audit.csv", index=False)
    pairwise.to_csv(output / "c25_pairwise_ranking_val.csv", index=False)
    inversions.to_csv(output / "c25_pairwise_inversion_summary.csv", index=False)
    health.to_csv(output / "c25_residual_health_audit.csv", index=False)
    shortcuts.to_csv(output / "c25_shortcut_residual_audit.csv", index=False)

    epoch_metrics = pd.read_csv(c25_run / "reports" / "metrics_by_epoch.csv")
    coverage_rows = []
    for seed in SEEDS:
        rows = epoch_metrics[epoch_metrics["seed"].astype(int) == seed]
        mixed = int(pd.to_numeric(rows["train_mixed_class_batch_count"], errors="coerce").fillna(0).sum())
        single = int(pd.to_numeric(rows["train_single_class_batch_count"], errors="coerce").fillna(0).sum())
        pairs = int(pd.to_numeric(rows["train_pair_count"], errors="coerce").fillna(0).sum())
        coverage_rows.append({"seed": seed, "mixed_class_batches": mixed, "single_class_batches": single, "mixed_class_batch_fraction": mixed / max(mixed + single, 1), "pair_count": pairs})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "c25_rank_batch_coverage_audit.csv", index=False)
    for source, target in (("metrics_by_epoch.csv", "c25_metrics_by_epoch.csv"), ("metrics_by_seed.csv", "c25_metrics_by_seed.csv"), ("metrics_summary.csv", "c25_metrics_summary.csv")):
        pd.read_csv(c25_run / "reports" / source).to_csv(output / target, index=False)

    prior = pd.read_csv(resolve_path(args.prior_comparison))
    prior_columns = [column for column in ("seed", "c17_auc", "c23_auc", "c24_auc") if column in prior]
    four_way = comparison.merge(prior[prior_columns], on="seed", how="left", suffixes=("", "_prior"), validate="one_to_one")
    if "c17_auc_prior" in four_way:
        four_way = four_way.drop(columns="c17_auc_prior")
    if "c23_auc" not in four_way or "c24_auc" not in four_way:
        raise RuntimeError("C17/C23/C24 prior comparison is incomplete")
    four_way["c25_minus_c23_auc"] = four_way["c25_auc"] - four_way["c23_auc"]
    four_way["c25_minus_c24_auc"] = four_way["c25_auc"] - four_way["c24_auc"]
    four_way.to_csv(output / "c25_c17_c23_c24_comparison.csv", index=False)

    c17_auc = comparison["c17_auc"].to_numpy(dtype=float)
    c25_auc = comparison["c25_auc"].to_numpy(dtype=float)
    difference = c25_auc - c17_auc
    coverage_safe = bool((coverage["mixed_class_batch_fraction"] >= 0.90).all() and (coverage["pair_count"] > 0).all())
    positive_safe = bool(int(correct["tp_to_fn"].sum()) == 0 and (correct["correct_positive_mean_delta"] >= -0.005).all() and (correct["correct_positive_wrong_direction_fraction"] <= 0.10).all())
    negative_safe = bool(int(correct["tn_to_fp"].sum()) == 0 and (correct["correct_negative_mean_delta"] <= 0.005).all() and (correct["correct_negative_wrong_direction_fraction"] <= 0.10).all())
    global_safe = bool((global_shift["mean_delta"].abs() <= 0.01).all() and not global_shift["same_sign_fraction_ge_0_90"].any() and not global_shift["both_class_means_same_sign_and_large"].any())
    inversion_safe = bool(int((inversions["change"] < 0).sum()) >= 2 and int(inversions["repaired"].sum()) > int(inversions["introduced"].sum()) and (inversions["change"] <= 3).all())
    residual_healthy = bool(health["nonzero_variance"].all() and (health["fraction_near_negative_bound"] < 0.25).all() and (health["fraction_near_positive_bound"] < 0.25).all() and (health["mean_abs_delta_low"] > health["mean_abs_delta_high"]).all())
    shortcut_max = float(pd.to_numeric(shortcuts["selected_structure_shortcut_auc"], errors="coerce").max())
    shortcut_safe = bool(np.isfinite(shortcut_max) and shortcut_max <= 0.55)
    auc_safety = bool((difference >= -0.003).all() and safe_std(c25_auc) <= 0.02)
    auc_gate = bool(c25_auc.mean() > c17_auc.mean() and int((difference > 0).sum()) >= 2 and auc_safety)
    training_valid = bool(np.isfinite(c25_auc).all() and reproduction_error <= 1e-5 and len(pairwise) > 0 and not constant_prediction)

    if not training_valid:
        decision = "DEMA_C25_TRAINING_INVALID"
    elif not coverage_safe:
        decision = "DEMA_C25_RANK_BATCH_COVERAGE_INVALID"
    elif not positive_safe:
        decision = "DEMA_C25_POSITIVE_SUPPRESSION"
    elif not negative_safe:
        decision = "DEMA_C25_NEGATIVE_INFLATION"
    elif not global_safe:
        decision = "DEMA_C25_GLOBAL_SHIFT"
    elif not inversion_safe:
        decision = "DEMA_C25_INVERSION_WORSENING"
    elif not residual_healthy:
        decision = "DEMA_C25_RESIDUAL_COLLAPSE"
    elif not auc_safety or not shortcut_safe:
        decision = "DEMA_C25_FORMAL_FAIL_KEEP_C17"
    elif auc_gate:
        decision = "PROMOTE_DEMA_C25_PAIRWISE_RANKING_RESIDUAL"
    else:
        decision = "DEMA_C25_NO_AUC_GAIN_KEEP_C17"

    promoted = decision == "PROMOTE_DEMA_C25_PAIRWISE_RANKING_RESIDUAL"
    common = [
        "- formal seeds: `0, 42, 3407`",
        "- checkpoint selection: validation AUC only",
        "- test role: reporting-only after validation selection",
        f"- C17 validation AUC mean/std: `{c17_auc.mean():.10f} +/- {safe_std(c17_auc):.10f}`",
        f"- C25 validation AUC mean/std: `{c25_auc.mean():.10f} +/- {safe_std(c25_auc):.10f}`",
        f"- C25 minus C17 mean: `{c25_auc.mean() - c17_auc.mean():+.10f}`",
        f"- seeds above C17: `{int((difference > 0).sum())}/3`; worst seed delta: `{difference.min():+.10f}`",
        f"- mixed-class batch coverage gate: `{coverage_safe}`; minimum=`{coverage['mixed_class_batch_fraction'].min():.10f}`",
        f"- correct-positive preservation: `{positive_safe}`; aggregate TP->FN=`{int(correct['tp_to_fn'].sum())}`",
        f"- correct-negative preservation: `{negative_safe}`; aggregate TN->FP=`{int(correct['tn_to_fp'].sum())}`",
        f"- global-shift gate: `{global_safe}`",
        f"- inversion gate: `{inversion_safe}`; repaired=`{int(inversions['repaired'].sum())}`; introduced=`{int(inversions['introduced'].sum())}`",
        f"- residual health: `{residual_healthy}`",
        f"- selected-structure shortcut-only AUC: `{shortcut_max:.10f}`; pass=`{shortcut_safe}`",
        f"- decision: `{decision}`",
    ]
    (output / "c25_seed_stability_report.md").write_text("# C25 Seed Stability\n\n" + "\n".join(common) + "\n", encoding="utf-8")
    (output / "c25_pairwise_ranking_residual_report.md").write_text(
        "# C25 Pairwise-Ranking Residual\n\n"
        "C25 freezes the complete C17 predictor and trains only the unchanged C23 local residual head from detached `mea_mechanism_state`. "
        "Its primary objective ranks every positive-negative pair in each mixed training batch; no pointwise classification loss is used.\n\n"
        + "\n".join(common) + "\n", encoding="utf-8",
    )
    route_lines = ["# C25 Route Stop Decision", "", f"- decision: `{decision}`"]
    if not promoted:
        route_lines.extend(["- `STOP_FURTHER_RESIDUAL_REFINEMENT`", "- `KEEP_DEMA_C17_STRICT_BEST`", "- no C26 residual route is authorized"])
    else:
        route_lines.append("- C25 is promoted by the complete prespecified gate")
    (output / "c25_route_stop_decision.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    (output / "phase_c25_dema_final_report.md").write_text(
        "# Phase C25 DEMA-HT Final Report\n\n"
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`\n"
        "- runtime: `/home/linruixin/chen/conda/envs/ma`\n"
        "- data: `/data/csb/DMEA-HT/HT_2025.12_25`\n"
        "- current strict best remains C17 unless the decision explicitly promotes C25\n"
        + "\n".join(common) + "\n" + ("\nSTOP_FURTHER_RESIDUAL_REFINEMENT\nKEEP_DEMA_C17_STRICT_BEST\n" if not promoted else ""),
        encoding="utf-8",
    )
    payload = {
        "phase": "C25", "decision": decision,
        "c17_mean_auc": float(c17_auc.mean()), "c17_std_auc": safe_std(c17_auc),
        "c25_mean_auc": float(c25_auc.mean()), "c25_std_auc": safe_std(c25_auc),
        "coverage_pass": coverage_safe, "positive_preservation_pass": positive_safe,
        "negative_preservation_pass": negative_safe, "global_shift_pass": global_safe,
        "inversion_pass": inversion_safe, "residual_health_pass": residual_healthy,
        "shortcut_pass": shortcut_safe, "test_used_for_decision": False,
        "stop_further_residual_refinement": not promoted,
        "keep_dema_c17_strict_best": not promoted,
    }
    (output / "c25_final_decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if args.require_pass and not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
