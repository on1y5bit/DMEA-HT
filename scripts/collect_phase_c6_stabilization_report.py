from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


BAD_SEEDS = {1, 3, 42}
SHORTCUT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]
ORIGINAL_BADSEED_AUC = 0.7430
ORIGINAL_BADSEED_AUPRC = 0.7347
ORIGINAL_BADSEED_GAP = 0.1430
STRICT_MVP_VAL_AUC = 0.7581


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            item = row[col]
            if isinstance(item, float):
                values.append(fmt(item))
            elif pd.isna(item):
                values.append("NA")
            else:
                values.append(str(item).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    keep = ["patient_id", "split", "label"] + [field for field in SHORTCUT_FIELDS if field in frame.columns]
    return frame[keep].drop_duplicates("patient_id")


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics_by_seed.csv: {path}")
    return pd.read_csv(path)


def read_epoch_metrics(run_dir: Path, candidate: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    if not path.exists():
        return pd.DataFrame([{"candidate_name": candidate, "notes": f"Missing {path}"}])
    frame = pd.read_csv(path)
    frame.insert(0, "candidate_name", candidate)
    return frame


def read_predictions(run_dir: Path, split: str = "val") -> pd.DataFrame:
    paths = sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing {split} prediction files under {run_dir / 'predictions'}")
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        if "seed" not in frame.columns:
            match = re.search(r"seed_(\d+)", path.name)
            frame["seed"] = int(match.group(1)) if match else -1
        frame["pred_prob"] = frame[prob_column(frame)].astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mvp_reference(mvp_run: Path) -> pd.DataFrame:
    preds = read_predictions(mvp_run, "val")
    return preds.groupby("patient_id", as_index=False).agg(mvp_pred_prob=("pred_prob", "mean"))


def shortcut_residual(candidate: str, preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    rows: List[Dict[str, Any]] = []
    for seed, group in merged.groupby("seed"):
        for field in SHORTCUT_FIELDS:
            pair = pd.DataFrame(
                {
                    "prob": pd.to_numeric(group["pred_prob"], errors="coerce"),
                    "field": pd.to_numeric(group[field], errors="coerce") if field in group.columns else pd.NA,
                }
            ).dropna()
            n_unique = int(pair["field"].nunique()) if not pair.empty else 0
            corr = pd.NA
            if len(pair) >= 3 and n_unique >= 2:
                corr = pair["prob"].corr(pair["field"], method="spearman")
            rows.append(
                {
                    "candidate_name": candidate,
                    "seed": int(seed),
                    "field": field,
                    "spearman": corr,
                    "abs_spearman": abs(float(corr)) if not pd.isna(corr) else pd.NA,
                    "n_unique_values": n_unique,
                    "warning_if_constant": n_unique < 2,
                }
            )
    return pd.DataFrame(rows)


def positive_preservation(candidate: str, preds: pd.DataFrame, manifest: pd.DataFrame, mvp_ref: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(manifest[["patient_id", "label"]], on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns:
        merged["label"] = merged["label"].fillna(merged["label_manifest"])
    merged["label"] = merged["label"].astype(int)
    merged = merged.merge(mvp_ref, on="patient_id", how="left")
    merged["c1_abs_error"] = (merged["pred_prob"] - merged["label"]).abs()
    merged["mvp_abs_error"] = (merged["mvp_pred_prob"] - merged["label"]).abs()
    merged["abs_error_delta_vs_mvp"] = merged["c1_abs_error"] - merged["mvp_abs_error"]
    rows: List[Dict[str, Any]] = []
    for (seed, label), group in merged.groupby(["seed", "label"]):
        rows.append(
            {
                "candidate_name": candidate,
                "seed": int(seed),
                "label": int(label),
                "n": len(group),
                "mean_pred_prob": group["pred_prob"].mean(),
                "mean_mvp_pred_prob": group["mvp_pred_prob"].mean(),
                "mean_abs_error_delta_vs_mvp": group["abs_error_delta_vs_mvp"].mean(),
                "fraction_improved_vs_mvp": float((group["abs_error_delta_vs_mvp"] < 0).mean()),
                "fraction_harmed_vs_mvp": float((group["abs_error_delta_vs_mvp"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def candidate_summary(candidate: str, run_dir: Path, manifest: pd.DataFrame, mvp_ref: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics = read_metrics(run_dir)
    val = metrics[(metrics["split"].astype(str).str.lower() == "val") & (metrics["seed"].astype(int).isin(BAD_SEEDS))].copy()
    preds = read_predictions(run_dir, "val")
    preds = preds[preds["seed"].astype(int).isin(BAD_SEEDS)].copy()
    residual = shortcut_residual(candidate, preds, manifest)
    pos = positive_preservation(candidate, preds, manifest, mvp_ref)
    max_residual = residual["abs_spearman"].dropna().max() if not residual.empty else pd.NA
    pos_label = pos[pos["label"] == 1]
    failure: List[str] = []
    val_auc_mean = float(val["AUC"].mean())
    val_auprc_mean = float(val["AUPRC"].mean())
    gap_mean = float(val["pos_neg_gap"].mean()) if "pos_neg_gap" in val.columns else 0.0
    sens_mean = float(val["Sensitivity"].mean())
    if val_auc_mean <= ORIGINAL_BADSEED_AUC:
        failure.append("mean validation AUC does not beat original bad-seed C1")
    if val_auc_mean < STRICT_MVP_VAL_AUC - 0.003:
        failure.append("mean validation AUC is not close to strict MVP reference")
    if val_auprc_mean < ORIGINAL_BADSEED_AUPRC - 0.02:
        failure.append("validation AUPRC decreases materially")
    if gap_mean <= ORIGINAL_BADSEED_GAP:
        failure.append("positive-negative prediction gap does not improve")
    if sens_mean < 0.45:
        failure.append("sensitivity is too low")
    if not pos_label.empty and float(pos_label["mean_abs_error_delta_vs_mvp"].mean()) > 0.20:
        failure.append("positive-case absolute error remains strongly worse than MVP")
    if pd.notna(max_residual) and float(max_residual) > 0.35:
        failure.append("shortcut residual association is elevated")
    if not failure:
        decision = "STABILIZATION_PASS_RECOMMEND_FORMAL"
        reason = "passes bad-seed validation, positive preservation, and shortcut residual gates"
    elif val_auc_mean > ORIGINAL_BADSEED_AUC and gap_mean > ORIGINAL_BADSEED_GAP:
        decision = "STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS"
        reason = "; ".join(failure)
    else:
        decision = "STABILIZATION_FAIL"
        reason = "; ".join(failure)
    summary = {
        "candidate_name": candidate,
        "run_dir": str(run_dir),
        "seeds": ",".join(str(seed) for seed in sorted(val["seed"].astype(int).unique())),
        "val_auc_mean": val_auc_mean,
        "val_auc_std": float(val["AUC"].std(ddof=1)) if len(val) > 1 else 0.0,
        "val_auprc_mean": val_auprc_mean,
        "val_auprc_std": float(val["AUPRC"].std(ddof=1)) if len(val) > 1 else 0.0,
        "val_sensitivity_mean": sens_mean,
        "val_specificity_mean": float(val["Specificity"].mean()),
        "val_balanced_accuracy_mean": float(val["Balanced_ACC"].mean()),
        "val_positive_prob_mean": float(val["positive_prob_mean"].mean()),
        "val_negative_prob_mean": float(val["negative_prob_mean"].mean()),
        "val_pos_neg_gap_mean": gap_mean,
        "max_abs_prediction_shortcut_residual_spearman": max_residual,
        "selected_epoch_mean": float(val["best_epoch"].mean()),
        "selected_epoch_min": int(val["best_epoch"].min()),
        "selected_epoch_max": int(val["best_epoch"].max()),
        "stabilization_decision": decision,
        "failure_reason": reason,
    }
    return summary, residual, pos


def write_report(path: Path, title: str, body: str, frame: pd.DataFrame | None = None) -> None:
    lines = [f"# {title}", "", body.strip(), ""]
    if frame is not None:
        lines.extend([frame_to_markdown(frame), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C6 bad-seed stabilization pilot reports.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mvp-run-dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="candidate_name=run_dir entries.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest_frame(Path(args.manifest))
    manifest_val = manifest[manifest["split"].astype(str).str.lower() == "val"].copy()
    mvp_ref = mvp_reference(Path(args.mvp_run_dir))

    summaries: List[Dict[str, Any]] = []
    residual_frames: List[pd.DataFrame] = []
    positive_frames: List[pd.DataFrame] = []
    epoch_frames: List[pd.DataFrame] = []
    for name, run_dir in [parse_run(value) for value in args.runs]:
        summary, residual, positive = candidate_summary(name, run_dir, manifest_val, mvp_ref)
        summaries.append(summary)
        residual_frames.append(residual)
        positive_frames.append(positive)
        epoch_frames.append(read_epoch_metrics(run_dir, name))

    summary_df = pd.DataFrame(summaries).sort_values(["stabilization_decision", "val_auc_mean"], ascending=[True, False])
    residual_df = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    positive_df = pd.concat(positive_frames, ignore_index=True) if positive_frames else pd.DataFrame()
    epoch_df = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()

    summary_df.to_csv(out_dir / "c6_badseed_pilot_summary.csv", index=False)
    epoch_df.to_csv(out_dir / "c6_epoch_dynamics.csv", index=False)
    positive_df.to_csv(out_dir / "c6_positive_preservation.csv", index=False)
    residual_df.to_csv(out_dir / "c6_shortcut_residual_audit.csv", index=False)
    summary_df[["candidate_name", "stabilization_decision", "failure_reason"]].to_csv(
        out_dir / "c6_decision_gate_summary.csv", index=False
    )

    write_report(
        out_dir / "c6_badseed_pilot_report.md",
        "Phase C6 Bad-Seed Pilot Summary",
        "Validation-only comparison against original C1 bad-seed mean AUC 0.7430, gap 0.1430, and strict MVP AUC 0.7581.",
        summary_df,
    )
    write_report(
        out_dir / "c6_epoch_dynamics_report.md",
        "Phase C6 Epoch Dynamics",
        "Per-epoch rows are produced by the C6 train.py logging change. selected_by_val_auc marks the checkpoint epoch.",
        epoch_df,
    )
    write_report(
        out_dir / "c6_positive_preservation_report.md",
        "Phase C6 Positive Preservation",
        "Positive label rows show whether a candidate still harms HT-positive patients relative to MVP.",
        positive_df,
    )
    write_report(
        out_dir / "c6_shortcut_residual_audit_report.md",
        "Phase C6 Shortcut Residual Audit",
        "Audit only. Shortcut fields are never model inputs or labels.",
        residual_df,
    )
    best = summary_df.iloc[0] if not summary_df.empty else None
    final_lines = [
        "# Phase C6 Final Report",
        "",
        "Phase C6 is a bad-seed stabilization pilot and does not promote a new main model.",
        "",
    ]
    if best is not None:
        final_lines.extend(
            [
                f"Top candidate by sorted decision/AUC: `{best['candidate_name']}`.",
                f"Decision: `{best['stabilization_decision']}`.",
                f"Validation AUC mean: {fmt(best['val_auc_mean'])}.",
                f"Validation AUPRC mean: {fmt(best['val_auprc_mean'])}.",
                f"Positive-negative gap mean: {fmt(best['val_pos_neg_gap_mean'])}.",
                f"Failure reason: {best['failure_reason']}.",
                "",
            ]
        )
    final_lines.extend(
        [
            "A passing C6 candidate is only eligible for later formal evaluation; it is not the new main model.",
            "",
            "## Decision Table",
            "",
            frame_to_markdown(summary_df),
        ]
    )
    (out_dir / "c6_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase C6 reports to {out_dir}")


if __name__ == "__main__":
    main()
