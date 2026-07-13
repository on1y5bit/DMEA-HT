#!/usr/bin/env python3
"""Collect validation-gated C16-MEA pilot or formal reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_prediction_shortcut_residual import (  # noqa: E402
    DEFAULT_SHORTCUT_FIELDS,
    audit_frame,
    merge_shortcuts,
    read_manifest_frame,
    read_split_predictions,
)


C13_MEAN_AUC = 0.8664554097
C13_SEED0_AUC = 0.8655500226
FORMAL_SEEDS = [0, 42, 3407]
VALID_LABELS = {
    "PROMOTE_C16_MEA_CORE",
    "PROMOTE_C16_MEA_RANK",
    "C16_MEA_PARTIAL_IMPROVEMENT_NOT_STABLE",
    "C16_MEA_EVIDENCE_ROLE_COLLAPSE",
    "C16_MEA_MECHANISM_ALIGNMENT_FAIL",
    "C16_MEA_PILOT_FAIL_KEEP_C13",
    "C16_MEA_FORMAL_FAIL_KEEP_C13",
    "C16_MEA_TRAINING_INVALID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seed0", "formal"), required=True)
    parser.add_argument("--core-run")
    parser.add_argument("--rank-run")
    parser.add_argument("--formal-run")
    parser.add_argument("--selected-route", choices=("core", "rank"))
    parser.add_argument("--c13-run", default="runs/dmea_ht_v2_c13_temporal_focus_stress_seeds")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c16_mea")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_epochs(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        if "seed" not in frame.columns:
            frame["seed"] = seed_from_path(path)
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    view = frame[list(columns)].copy() if columns is not None and not frame.empty else frame.copy()
    if view.empty:
        return "_No rows available._"
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in view.iterrows():
        values = []
        for column in view.columns:
            value = row[column]
            if pd.isna(value):
                text = "NA"
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.6f}"
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pairwise_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for seed, group in predictions.groupby("seed"):
        positives = group[group["label"].astype(int) == 1]
        negatives = group[group["label"].astype(int) == 0]
        for _, positive in positives.iterrows():
            for _, negative in negatives.iterrows():
                positive_logit = float(positive["logit"])
                negative_logit = float(negative["logit"])
                margin = positive_logit - negative_logit
                rows.append(
                    {
                        "seed": int(seed),
                        "positive_patient_id": str(positive["patient_id"]),
                        "negative_patient_id": str(negative["patient_id"]),
                        "positive_logit": positive_logit,
                        "negative_logit": negative_logit,
                        "pair_margin": margin,
                        "is_inversion": int(margin <= 0.0),
                    }
                )
    return pd.DataFrame(rows)


def inversion_summary(pairs: pd.DataFrame, model_id: str) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby("seed", as_index=False)
        .agg(pair_count=("is_inversion", "size"), inversion_count=("is_inversion", "sum"), mean_pair_margin=("pair_margin", "mean"))
        .assign(model_id=model_id)
    )


def shortcut_audit(run_dir: Path, manifest_path: Path, model_id: str) -> pd.DataFrame:
    fields = list(DEFAULT_SHORTCUT_FIELDS)
    manifest = read_manifest_frame(manifest_path, fields)
    predictions = read_split_predictions(run_dir, "val")
    if predictions.empty:
        return pd.DataFrame()
    merged = merge_shortcuts(predictions, manifest, fields)
    rows = [audit_frame(model_id, "val", str(seed), group, fields) for seed, group in merged.groupby("seed")]
    rows.append(audit_frame(model_id, "val", "pooled", merged, fields))
    return pd.DataFrame(rows)


def health_status(run_dir: Path) -> str:
    path = run_dir / "reports" / "c16_mea_alignment_health.json"
    if not path.exists():
        return "MISSING"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status", "MISSING"))


def metric_row(run_dir: Path, seed: int = 0) -> pd.Series | None:
    metrics = read_metrics(run_dir)
    if metrics.empty:
        return None
    rows = metrics[(metrics["split"].astype(str) == "val") & (pd.to_numeric(metrics["seed"], errors="coerce") == seed)]
    return rows.iloc[0] if len(rows) == 1 else None


def seed0_stage(args: argparse.Namespace, output_dir: Path) -> str:
    if not args.core_run or not args.rank_run:
        raise ValueError("--core-run and --rank-run are required for the seed0 stage")
    c13_run = Path(args.c13_run)
    c13_row = metric_row(c13_run, 0)
    c13_predictions = read_predictions(c13_run, "val")
    c13_pairs = inversion_summary(pairwise_rows(c13_predictions[c13_predictions["seed"] == 0]), "C13")
    c13_inversions = int(c13_pairs.iloc[0]["inversion_count"]) if not c13_pairs.empty else None
    routes: List[Dict[str, Any]] = []
    for route, path_text in (("core", args.core_run), ("rank", args.rank_run)):
        run_dir = Path(path_text)
        row = metric_row(run_dir, 0)
        predictions = read_predictions(run_dir, "val")
        pair_summary = inversion_summary(pairwise_rows(predictions), route)
        inversions = int(pair_summary.iloc[0]["inversion_count"]) if not pair_summary.empty else None
        shortcut = shortcut_audit(run_dir, Path(args.manifest), route)
        pooled_shortcut = shortcut[shortcut["seed"].astype(str) == "pooled"] if not shortcut.empty else pd.DataFrame()
        if row is None or c13_row is None:
            passed = False
            reasons = ["missing validation metrics"]
            values: Dict[str, Any] = {}
        else:
            values = {key: float(row[key]) for key in ("AUC", "AUPRC", "Sensitivity", "Specificity", "pos_neg_gap", "positive_prob_mean")}
            reasons = []
            if values["AUC"] < C13_SEED0_AUC:
                reasons.append("AUC below C13 seed-0")
            if values["AUPRC"] < float(c13_row["AUPRC"]) - 0.005:
                reasons.append("AUPRC decrease exceeds 0.005")
            if values["Sensitivity"] < 0.55:
                reasons.append("sensitivity below 0.55")
            if values["Specificity"] < 0.75:
                reasons.append("specificity below 0.75")
            if values["pos_neg_gap"] < float(c13_row["pos_neg_gap"]) - 0.02:
                reasons.append("positive-negative gap materially decreased")
            if values["positive_prob_mean"] < float(c13_row["positive_prob_mean"]) - 0.05:
                reasons.append("positive probabilities globally suppressed")
            if c13_inversions is not None and inversions is not None and inversions > c13_inversions:
                reasons.append("pairwise inversions increased")
            if health_status(run_dir) != "PASS":
                reasons.append("alignment health gate failed")
            if not pooled_shortcut.empty:
                if float(pooled_shortcut.iloc[0]["max_abs_spearman"]) >= 0.20 or float(pooled_shortcut.iloc[0]["linear_r2_prob_from_shortcuts"]) >= 0.10:
                    reasons.append("new shortcut residual concern")
            else:
                reasons.append("shortcut residual audit unavailable")
            passed = not reasons
        routes.append(
            {
                "route": route,
                **values,
                "auc_delta_vs_c13_seed0": values.get("AUC", np.nan) - C13_SEED0_AUC,
                "inversion_count": inversions,
                "c13_inversion_count": c13_inversions,
                "health_status": health_status(run_dir),
                "pilot_gate_pass": passed,
                "reasons": "; ".join(reasons) if reasons else "all seed-0 pilot checks passed",
            }
        )
    route_frame = pd.DataFrame(routes)
    passing = route_frame[route_frame["pilot_gate_pass"] == True].copy()  # noqa: E712
    if passing.empty:
        selected_route = None
        decision = "C16_MEA_PILOT_FAIL_KEEP_C13"
    elif len(passing) == 1:
        selected_route = str(passing.iloc[0]["route"])
        decision = f"SELECT_{selected_route.upper()}_FOR_FORMAL"
    else:
        core_auc = float(passing[passing["route"] == "core"].iloc[0]["AUC"])
        rank_auc = float(passing[passing["route"] == "rank"].iloc[0]["AUC"])
        selected_route = "core" if abs(core_auc - rank_auc) < 0.003 or core_auc > rank_auc else "rank"
        decision = f"SELECT_{selected_route.upper()}_FOR_FORMAL"
    payload = {
        "stage": "seed0_validation_only",
        "test_read_for_selection": False,
        "selected_route": selected_route,
        "decision": decision,
        "routes": routes,
    }
    (output_dir / "c16_mea_seed0_route_decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    route_frame.to_csv(output_dir / "c16_mea_seed0_route_comparison.csv", index=False)
    (output_dir / "c16_mea_seed0_route_decision.md").write_text(
        "# C16-MEA Seed-0 Route Decision\n\n"
        f"- Decision: `{decision}`\n- Selected route: `{selected_route or 'none'}`\n"
        "- Selection used validation only; pilot test predictions were not generated or read.\n\n"
        + markdown_table(route_frame)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return decision


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    numeric_columns = ["AUC", "AUPRC", "Sensitivity", "Specificity", "Balanced_ACC", "pos_neg_gap", "positive_prob_mean", "negative_prob_mean"]
    for split, group in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split, "n_seeds": int(group["seed"].nunique())}
        for column in numeric_columns:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{column}_min"] = float(values.min())
                row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def add_confusion(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    pred = (pd.to_numeric(out["pred_prob"], errors="coerce") >= 0.5).astype(int)
    label = pd.to_numeric(out["label"], errors="coerce").astype(int)
    out["correct"] = (pred == label).astype(int)
    out["confusion_type"] = np.select(
        [(label == 1) & (pred == 1), (label == 1) & (pred == 0), (label == 0) & (pred == 1)],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return out


def grouped_means(frame: pd.DataFrame, groups: List[str], columns: List[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.groupby(groups, dropna=False)[available].mean(numeric_only=True).reset_index() if available else pd.DataFrame()


def formal_stage(args: argparse.Namespace, output_dir: Path) -> str:
    if not args.formal_run or not args.selected_route:
        raise ValueError("--formal-run and --selected-route are required for the formal stage")
    run_dir = Path(args.formal_run)
    c13_run = Path(args.c13_run)
    metrics = read_metrics(run_dir)
    epochs = read_epochs(run_dir)
    val = add_confusion(read_predictions(run_dir, "val"))
    test = read_predictions(run_dir, "test")
    c13_metrics = read_metrics(c13_run)
    c13_val = add_confusion(read_predictions(c13_run, "val"))
    complete = not metrics.empty and not epochs.empty and not val.empty and not test.empty
    formal_val = metrics[metrics["split"].astype(str) == "val"].copy() if not metrics.empty else pd.DataFrame()
    complete = complete and sorted(pd.to_numeric(formal_val["seed"], errors="coerce").astype(int).tolist()) == FORMAL_SEEDS

    metrics.to_csv(output_dir / "c16_mea_metrics_by_seed.csv", index=False)
    metric_summary(metrics).to_csv(output_dir / "c16_mea_metrics_summary.csv", index=False)
    epochs.to_csv(output_dir / "c16_mea_metrics_by_epoch.csv", index=False)
    val.to_csv(output_dir / "c16_mea_patient_diagnostics_val.csv", index=False)
    pairs = pairwise_rows(val)
    pairs.to_csv(output_dir / "c16_mea_pairwise_ranking_val.csv", index=False)
    pair_summary = inversion_summary(pairs, f"C16_MEA_{args.selected_route.upper()}")
    c13_pair_summary = inversion_summary(pairwise_rows(c13_val), "C13")
    inversion = pair_summary.merge(c13_pair_summary, on="seed", how="outer", suffixes=("_c16", "_c13"))
    if not inversion.empty:
        inversion["inversion_delta_c16_minus_c13"] = inversion["inversion_count_c16"] - inversion["inversion_count_c13"]
    inversion.to_csv(output_dir / "c16_mea_pairwise_inversion_summary.csv", index=False)

    role_columns = [
        "patient_support_strength", "patient_opposition_strength", "patient_uncertainty_strength", "patient_conflict_score",
        "image_support_score", "image_opposition_score", "image_uncertainty_score", "text_support_score", "text_opposition_score",
        "text_uncertainty_score", "bio_support_score", "bio_opposition_score", "bio_uncertainty_score", "evidence_role_entropy",
    ]
    role_health = grouped_means(val, ["seed", "label", "correct", "confusion_type"], role_columns)
    role_health.to_csv(output_dir / "c16_mea_evidence_role_health.csv", index=False)
    (output_dir / "c16_mea_evidence_role_health_report.md").write_text(
        "# C16-MEA Evidence Role Health\n\n" + markdown_table(role_health) + "\n", encoding="utf-8"
    )

    if not val.empty:
        val["conflict_group"] = val.groupby("seed")["patient_conflict_score"].transform(lambda series: np.where(series >= series.median(), "high", "low"))
    mechanism_columns = ["morphology_alignment_cosine", "morphology_alignment_available", "mechanism_state_norm", "mechanism_attention_max", "patient_conflict_score"]
    mechanism_health = grouped_means(val, ["seed", "label", "correct", "conflict_group"], mechanism_columns)
    mechanism_health.to_csv(output_dir / "c16_mea_mechanism_alignment_health.csv", index=False)
    (output_dir / "c16_mea_mechanism_alignment_report.md").write_text(
        "# C16-MEA Mechanism Alignment Health\n\n"
        "Only image-text morphology alignment is enabled; no unverified bio-text alignment is reported.\n\n"
        + markdown_table(mechanism_health)
        + "\n",
        encoding="utf-8",
    )

    temporal_columns = [
        "text_latest_support_score", "text_latest_opposition_score", "text_latest_available", "text_history_support_score",
        "text_history_opposition_score", "text_history_available", "text_temporal_conflict_score", "text_temporal_available",
    ]
    temporal = grouped_means(val, ["seed", "confusion_type"], temporal_columns)
    temporal.to_csv(output_dir / "c16_mea_temporal_evidence_audit.csv", index=False)

    compare = val.merge(
        c13_val[["patient_id", "seed", "label", "pred_prob", "confusion_type"]],
        on=["patient_id", "seed", "label"],
        how="inner",
        suffixes=("_c16", "_c13"),
    )
    if not compare.empty:
        compare["probability_delta_c16_minus_c13"] = compare["pred_prob_c16"] - compare["pred_prob_c13"]
        compare["absolute_error_delta_c16_minus_c13"] = (
            (compare["pred_prob_c16"] - compare["label"]).abs() - (compare["pred_prob_c13"] - compare["label"]).abs()
        )
        compare["transition"] = compare["confusion_type_c13"] + "->" + compare["confusion_type_c16"]
    compare.to_csv(output_dir / "c16_mea_positive_preservation_audit.csv", index=False)

    shortcut = shortcut_audit(run_dir, Path(args.manifest), f"C16_MEA_{args.selected_route.upper()}")
    shortcut.to_csv(output_dir / "c16_mea_shortcut_residual_audit.csv", index=False)

    summary = metric_summary(metrics)
    val_summary = summary[summary["split"].astype(str) == "val"].iloc[0] if not summary.empty and (summary["split"].astype(str) == "val").any() else None
    mean_auc = float(val_summary["AUC_mean"]) if val_summary is not None else float("nan")
    auc_std = float(val_summary["AUC_std"]) if val_summary is not None else float("nan")
    mean_auprc = float(val_summary["AUPRC_mean"]) if val_summary is not None else float("nan")
    c13_val_metrics = c13_metrics[c13_metrics["split"].astype(str) == "val"] if not c13_metrics.empty else pd.DataFrame()
    c13_mean_auprc = float(pd.to_numeric(c13_val_metrics["AUPRC"], errors="coerce").mean()) if not c13_val_metrics.empty else float("nan")
    improved_inversions = int((inversion.get("inversion_delta_c16_minus_c13", pd.Series(dtype=float)) < 0).sum())
    role_order = val.assign(order=val["patient_support_strength"] - val["patient_opposition_strength"]).groupby("label")["order"].mean() if not val.empty else pd.Series(dtype=float)
    role_order_valid = 1 in role_order.index and 0 in role_order.index and float(role_order.loc[1]) > 0.0 and float(role_order.loc[0]) < 0.0
    role_probability_columns = [
        "image_support_score", "text_support_score", "bio_support_score", "image_opposition_score", "text_opposition_score",
        "bio_opposition_score", "image_uncertainty_score", "text_uncertainty_score", "bio_uncertainty_score",
    ]
    role_means = val[role_probability_columns].mean() if not val.empty else pd.Series(dtype=float)
    support_mean = float(role_means[[name for name in role_means.index if "support" in name]].mean()) if not role_means.empty else 0.0
    opposition_mean = float(role_means[[name for name in role_means.index if "opposition" in name]].mean()) if not role_means.empty else 0.0
    uncertainty_mean = float(role_means[[name for name in role_means.index if "uncertainty" in name]].mean()) if not role_means.empty else 0.0
    role_fraction = max(support_mean, opposition_mean, uncertainty_mean) / max(support_mean + opposition_mean + uncertainty_mean, 1e-8)
    role_health_pass = role_fraction < 0.95 and role_order_valid
    modality_means = val[["image_evidence_weight", "text_evidence_weight", "bio_evidence_weight"]].mean() if not val.empty else pd.Series(dtype=float)
    mechanism_health_pass = not val.empty and float(val["mechanism_state_norm"].max()) < 100.0 and float(modality_means.max()) < 0.95
    pooled_shortcut = shortcut[shortcut["seed"].astype(str) == "pooled"] if not shortcut.empty else pd.DataFrame()
    shortcut_pass = not pooled_shortcut.empty and float(pooled_shortcut.iloc[0]["max_abs_spearman"]) < 0.20 and float(pooled_shortcut.iloc[0]["linear_r2_prob_from_shortcuts"]) < 0.10
    performance_pass = (
        complete
        and mean_auc > C13_MEAN_AUC
        and mean_auc - C13_MEAN_AUC >= 0.01
        and float(formal_val["AUC"].min()) >= 0.85
        and auc_std <= 0.02
        and mean_auprc >= c13_mean_auprc
        and improved_inversions >= 2
        and float(formal_val["Sensitivity"].mean()) >= 0.55
        and float(formal_val["Specificity"].mean()) >= 0.75
    )
    promotion_pass = performance_pass and role_health_pass and mechanism_health_pass and shortcut_pass
    if not complete:
        decision = "C16_MEA_TRAINING_INVALID"
    elif not role_health_pass:
        decision = "C16_MEA_EVIDENCE_ROLE_COLLAPSE"
    elif not mechanism_health_pass:
        decision = "C16_MEA_MECHANISM_ALIGNMENT_FAIL"
    elif promotion_pass:
        decision = "PROMOTE_C16_MEA_CORE" if args.selected_route == "core" else "PROMOTE_C16_MEA_RANK"
    elif mean_auc > C13_MEAN_AUC:
        decision = "C16_MEA_PARTIAL_IMPROVEMENT_NOT_STABLE"
    else:
        decision = "C16_MEA_FORMAL_FAIL_KEEP_C13"
    if decision not in VALID_LABELS:
        raise RuntimeError(f"Invalid decision label: {decision}")

    comparison = formal_val.merge(c13_val_metrics, on="seed", how="outer", suffixes=("_c16", "_c13"))
    for metric in ("AUC", "AUPRC", "Sensitivity", "Specificity", "Balanced_ACC", "pos_neg_gap"):
        if f"{metric}_c16" in comparison.columns and f"{metric}_c13" in comparison.columns:
            comparison[f"{metric}_delta"] = comparison[f"{metric}_c16"] - comparison[f"{metric}_c13"]
    (output_dir / "c16_mea_model_comparison_report.md").write_text(
        "# C16-MEA Model Comparison\n\nValidation-only comparison used for promotion. Test is reporting-only.\n\n"
        + markdown_table(comparison)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "c16_mea_seed_stability_report.md").write_text(
        "# C16-MEA Seed Stability\n\n"
        f"- Seeds: `{FORMAL_SEEDS}`\n- Validation AUC: `{mean_auc:.6f} +/- {auc_std:.6f}`\n"
        f"- Validation AUPRC: `{mean_auprc:.6f}`\n- Minimum/maximum validation AUC: `{formal_val['AUC'].min():.6f}` / `{formal_val['AUC'].max():.6f}`\n\n"
        + markdown_table(formal_val)
        + "\n",
        encoding="utf-8",
    )
    final_payload = {
        "decision": decision,
        "selected_route": args.selected_route,
        "formal_complete": bool(complete),
        "test_selection_role": "reporting_only",
        "mean_validation_auc": mean_auc,
        "validation_auc_std": auc_std,
        "auc_delta_vs_c13": mean_auc - C13_MEAN_AUC,
        "target_auc_0_90_reached": bool(mean_auc >= 0.90),
        "performance_gate_pass": bool(performance_pass),
        "evidence_role_gate_pass": bool(role_health_pass),
        "mechanism_health_gate_pass": bool(mechanism_health_pass),
        "shortcut_gate_pass": bool(shortcut_pass),
        "current_strict_best": f"C16_MEA_{args.selected_route.upper()}" if decision.startswith("PROMOTE") else "C13_TEMPORAL_FOCUS_DMEA_HT",
    }
    (output_dir / "phase_c16_mea_final_report.json").write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "phase_c16_mea_final_report.md").write_text(
        "# Phase C16-MEA Final Report\n\n"
        "- The previous shared-specific C16 plan was invalid, was reverted, and is not part of this project.\n"
        "- C16-MEA is the corrected disease-mechanism and evidence-aware phase.\n"
        f"- Selected route: `{args.selected_route}` by seed-0 validation only.\n"
        f"- Formal validation AUC: `{mean_auc:.6f} +/- {auc_std:.6f}`.\n"
        f"- Delta versus C13: `{mean_auc - C13_MEAN_AUC:.6f}`.\n"
        f"- Validation AUC 0.90 reached: `{mean_auc >= 0.90}`.\n"
        f"- Test role: reporting-only after route selection.\n"
        f"- Final decision: `{decision}`.\n"
        f"- Current strict best: `{final_payload['current_strict_best']}`.\n\n"
        "## Gate Summary\n\n"
        + markdown_table(pd.DataFrame([final_payload]))
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_payload, indent=2))
    return decision


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = seed0_stage(args, output_dir) if args.stage == "seed0" else formal_stage(args, output_dir)
    passed = decision.startswith("SELECT_") if args.stage == "seed0" else decision.startswith("PROMOTE_")
    if args.require_pass and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
