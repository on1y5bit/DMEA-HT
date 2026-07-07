from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


GOOD_SEEDS = {0, 2, 4, 3407}
BAD_SEEDS = {1, 3, 42}
SHORTCUT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]
MANIFEST_FIELDS = [
    "patient_id",
    "split",
    "label",
    "txt_morphology_label",
    "txt_morphology_confidence",
    "matched_morphology_terms",
    *SHORTCUT_FIELDS,
]


def seed_group(seed: int) -> str:
    if seed in GOOD_SEEDS:
        return "good"
    if seed in BAD_SEEDS:
        return "bad"
    return "other"


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return -1


def prediction_files(run_dir: Path, split: str) -> List[Path]:
    searched = [
        run_dir / "predictions" / f"{split}_predictions_seed_*.csv",
        run_dir / "**" / f"{split}_predictions_seed_*.csv",
        run_dir / "**" / f"*{split}*prediction*.csv",
        run_dir / "**" / f"*prediction*{split}*.csv",
    ]
    files: List[Path] = []
    for pattern in searched:
        files.extend(sorted(pattern.parent.glob(pattern.name)) if "**" not in str(pattern) else sorted(run_dir.glob(str(pattern.relative_to(run_dir)))))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    if not unique:
        patterns = "\n".join(str(pattern) for pattern in searched)
        raise FileNotFoundError(f"No {split} prediction CSVs found under {run_dir}. Searched:\n{patterns}")
    return [Path(path) for path in unique]


def read_predictions(run_dir: Path, split: str = "val") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in prediction_files(run_dir, split):
        frame = pd.read_csv(path)
        if "patient_id" not in frame.columns:
            continue
        if "seed" not in frame.columns:
            frame["seed"] = seed_from_path(path)
        if "split" not in frame.columns:
            frame["split"] = split
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["pred_prob"] = frame[prob_column(frame)].astype(float)
        frames.append(frame)
    if not frames:
        raise ValueError(f"Prediction files were found under {run_dir}, but none contained patient_id.")
    return pd.concat(frames, ignore_index=True)


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return pd.read_csv(path)


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    for field in MANIFEST_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame[MANIFEST_FIELDS].drop_duplicates("patient_id")


def numeric_series(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce") if field in frame.columns else pd.Series(dtype=float)
    return values


def max_abs_shortcut_spearman(frame: pd.DataFrame) -> float | None:
    values: List[float] = []
    for field in SHORTCUT_FIELDS:
        if field not in frame.columns:
            continue
        pair = pd.DataFrame({"prob": frame["pred_prob"], "field": numeric_series(frame, field)}).dropna()
        if len(pair) < 3 or pair["field"].nunique() < 2:
            continue
        corr = pair["prob"].corr(pair["field"], method="spearman")
        if not pd.isna(corr):
            values.append(abs(float(corr)))
    return max(values) if values else None


def metric_value(metrics: pd.DataFrame, seed: int, split: str, col: str) -> Any:
    rows = metrics[(metrics["seed"].astype(int) == int(seed)) & (metrics["split"].astype(str).str.lower() == split)]
    if rows.empty or col not in rows.columns:
        return pd.NA
    return rows.iloc[0][col]


def build_seed_summary(c1_preds: pd.DataFrame, metrics: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    merged = c1_preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns:
        merged["label"] = merged["label"].fillna(merged["label_manifest"]) if "label" in merged.columns else merged["label_manifest"]
    merged["label"] = merged["label"].astype(int)
    for seed, group in merged.groupby("seed"):
        seed_int = int(seed)
        pos = group[group["label"] == 1]["pred_prob"].astype(float)
        neg = group[group["label"] == 0]["pred_prob"].astype(float)
        val_auc = metric_value(metrics, seed_int, "val", "AUC")
        row = {
            "seed": seed_int,
            "seed_group": seed_group(seed_int),
            "val_auc": val_auc,
            "val_auprc": metric_value(metrics, seed_int, "val", "AUPRC"),
            "val_f1": metric_value(metrics, seed_int, "val", "F1"),
            "val_sensitivity": metric_value(metrics, seed_int, "val", "Sensitivity"),
            "val_specificity": metric_value(metrics, seed_int, "val", "Specificity"),
            "val_balanced_accuracy": metric_value(metrics, seed_int, "val", "Balanced_ACC"),
            "val_precision": metric_value(metrics, seed_int, "val", "Precision"),
            "val_recall": metric_value(metrics, seed_int, "val", "Recall"),
            "val_threshold_0p5_tp": metric_value(metrics, seed_int, "val", "TP"),
            "val_threshold_0p5_fp": metric_value(metrics, seed_int, "val", "FP"),
            "val_threshold_0p5_tn": metric_value(metrics, seed_int, "val", "TN"),
            "val_threshold_0p5_fn": metric_value(metrics, seed_int, "val", "FN"),
            "positive_pred_mean": pos.mean(),
            "negative_pred_mean": neg.mean(),
            "positive_pred_std": pos.std(ddof=1),
            "negative_pred_std": neg.std(ddof=1),
            "pos_neg_pred_gap": pos.mean() - neg.mean(),
            "max_abs_prediction_shortcut_spearman": max_abs_shortcut_spearman(group),
            "selected_checkpoint_epoch": metric_value(metrics, seed_int, "val", "best_epoch"),
            "last_epoch_val_auc": pd.NA,
            "best_minus_last_val_auc": pd.NA,
            "notes": "Per-epoch validation history is unavailable; best_minus_last_val_auc cannot be computed.",
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def mvp_patient_reference(mvp_preds: pd.DataFrame) -> pd.DataFrame:
    ref = (
        mvp_preds.groupby("patient_id", as_index=False)
        .agg(mvp_pred_prob=("pred_prob", "mean"), mvp_pred_prob_std=("pred_prob", "std"), mvp_reference_n_seeds=("seed", "nunique"))
    )
    return ref


def build_patient_delta(c1_preds: pd.DataFrame, mvp_preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    c1 = c1_preds[["patient_id", "seed", "label", "pred_prob"]].copy()
    c1["patient_id"] = c1["patient_id"].astype(str)
    c1 = c1.rename(columns={"pred_prob": "c1_pred_prob"})
    c1["seed"] = c1["seed"].astype(int)
    ref = mvp_patient_reference(mvp_preds)
    merged = c1.merge(ref, on="patient_id", how="left").merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns:
        merged["label"] = merged["label"].fillna(merged["label_manifest"])
    merged["label"] = merged["label"].astype(int)
    merged["seed_group"] = merged["seed"].map(seed_group)
    merged["prob_delta"] = merged["c1_pred_prob"] - merged["mvp_pred_prob"]
    merged["mvp_abs_error"] = (merged["mvp_pred_prob"] - merged["label"]).abs()
    merged["c1_abs_error"] = (merged["c1_pred_prob"] - merged["label"]).abs()
    merged["abs_error_delta"] = merged["c1_abs_error"] - merged["mvp_abs_error"]
    output_cols = [
        "patient_id",
        "seed",
        "seed_group",
        "label",
        "mvp_pred_prob",
        "c1_pred_prob",
        "prob_delta",
        "mvp_abs_error",
        "c1_abs_error",
        "abs_error_delta",
        "txt_morphology_label",
        "txt_morphology_confidence",
        "matched_morphology_terms",
        *SHORTCUT_FIELDS,
    ]
    return merged[[col for col in output_cols if col in merged.columns]].sort_values(["seed_group", "seed", "patient_id"])


def confidence_bin(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "unknown_or_missing"
    if pd.isna(x):
        return "unknown_or_missing"
    if x < 0.4:
        return "low"
    if x < 0.7:
        return "medium"
    return "high"


def quantile_bin(series: pd.Series, value: Any, prefix: str) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return f"{prefix}_missing"
    if pd.isna(x) or numeric.dropna().nunique() < 2:
        return f"{prefix}_missing"
    qs = numeric.quantile([0.33, 0.67]).tolist()
    if x <= qs[0]:
        return f"{prefix}_low"
    if x <= qs[1]:
        return f"{prefix}_mid"
    return f"{prefix}_high"


def terms_present(value: Any) -> str:
    if value is None or pd.isna(value):
        return "absent"
    text = str(value).strip()
    return "present" if text and text not in {"[]", "nan", "None"} else "absent"


def build_stratified_delta(delta: pd.DataFrame) -> pd.DataFrame:
    frame = delta.copy()
    frame["txt_morphology_confidence_bin"] = frame.get("txt_morphology_confidence", pd.Series(pd.NA, index=frame.index)).map(confidence_bin)
    frame["matched_morphology_terms_present"] = frame.get("matched_morphology_terms", pd.Series(pd.NA, index=frame.index)).map(terms_present)
    for field in ("report_length", "selected_n_visits", "used_images"):
        if field in frame.columns:
            frame[f"{field}_bin"] = [quantile_bin(frame[field], value, field) for value in frame[field]]
        else:
            frame[f"{field}_bin"] = f"{field}_missing"
    group_cols = [
        "label",
        "seed_group",
        "txt_morphology_label",
        "txt_morphology_confidence_bin",
        "matched_morphology_terms_present",
        "report_length_bin",
        "selected_n_visits_bin",
        "used_images_bin",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        item.update(
            {
                "n": len(group),
                "mean_mvp_pred_prob": group["mvp_pred_prob"].mean(),
                "mean_c1_pred_prob": group["c1_pred_prob"].mean(),
                "mean_prob_delta": group["prob_delta"].mean(),
                "mean_mvp_abs_error": group["mvp_abs_error"].mean(),
                "mean_c1_abs_error": group["c1_abs_error"].mean(),
                "mean_abs_error_delta": group["abs_error_delta"].mean(),
                "fraction_improved": float((group["abs_error_delta"] < 0).mean()),
                "fraction_harmed": float((group["abs_error_delta"] > 0).mean()),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["seed_group", "label", "n"], ascending=[True, True, False])


def prediction_distribution(c1_preds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (seed, label), group in c1_preds.groupby(["seed", "label"]):
        probs = group["pred_prob"].astype(float)
        rows.append(
            {
                "seed": int(seed),
                "seed_group": seed_group(int(seed)),
                "label": int(label),
                "n": len(probs),
                "mean": probs.mean(),
                "std": probs.std(ddof=1),
                "min": probs.min(),
                "p10": probs.quantile(0.10),
                "p25": probs.quantile(0.25),
                "median": probs.median(),
                "p75": probs.quantile(0.75),
                "p90": probs.quantile(0.90),
                "max": probs.max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "label"])


def shortcut_residual_rows(c1_preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    merged = c1_preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    rows: List[Dict[str, Any]] = []
    for seed, group in merged.groupby("seed"):
        for field in SHORTCUT_FIELDS:
            if field not in group.columns:
                rows.append(
                    {
                        "seed": int(seed),
                        "seed_group": seed_group(int(seed)),
                        "field": field,
                        "spearman": pd.NA,
                        "abs_spearman": pd.NA,
                        "n_unique_values": 0,
                        "warning_if_constant": True,
                    }
                )
                continue
            pair = pd.DataFrame({"prob": group["pred_prob"], "field": numeric_series(group, field)}).dropna()
            n_unique = int(pair["field"].nunique()) if not pair.empty else 0
            constant = n_unique < 2
            corr = pd.NA
            if len(pair) >= 3 and not constant:
                corr = pair["prob"].corr(pair["field"], method="spearman")
            rows.append(
                {
                    "seed": int(seed),
                    "seed_group": seed_group(int(seed)),
                    "field": field,
                    "spearman": corr,
                    "abs_spearman": abs(float(corr)) if not pd.isna(corr) else pd.NA,
                    "n_unique_values": n_unique,
                    "warning_if_constant": bool(constant),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "field"])


def write_seed_report(summary: pd.DataFrame, out_dir: Path) -> None:
    good = summary[summary["seed_group"] == "good"]
    bad = summary[summary["seed_group"] == "bad"]
    lines = [
        "# Phase C5 Seed Failure Summary",
        "",
        "Validation split only. C4 already marked C1 extended-seed stability as failed.",
        "",
        f"Good seed mean AUC: {fmt(good['val_auc'].mean())}; bad seed mean AUC: {fmt(bad['val_auc'].mean())}.",
        f"Good seed mean pos-neg prediction gap: {fmt(good['pos_neg_pred_gap'].mean())}; bad seed mean gap: {fmt(bad['pos_neg_pred_gap'].mean())}.",
        f"Good seed mean sensitivity/specificity: {fmt(good['val_sensitivity'].mean())} / {fmt(good['val_specificity'].mean())}.",
        f"Bad seed mean sensitivity/specificity: {fmt(bad['val_sensitivity'].mean())} / {fmt(bad['val_specificity'].mean())}.",
        "",
        "Interpretation:",
        "",
        "- Bad seeds are primarily ranking failures when validation AUC and positive-negative prediction gaps both drop.",
        "- Threshold behavior is also relevant when sensitivity or specificity collapses at 0.5.",
        "- Per-epoch curves are unavailable in this training code, so checkpoint instability can only be approximated from selected best_epoch and final best-state metrics.",
    ]
    (out_dir / "c1_seed_failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_patient_delta_report(delta: pd.DataFrame, out_dir: Path) -> None:
    grouped = delta.groupby(["seed_group", "label"], dropna=False).agg(
        n=("patient_id", "count"),
        mean_prob_delta=("prob_delta", "mean"),
        mean_abs_error_delta=("abs_error_delta", "mean"),
        fraction_improved=("abs_error_delta", lambda x: float((x < 0).mean())),
        fraction_harmed=("abs_error_delta", lambda x: float((x > 0).mean())),
    )
    improved = delta.nsmallest(10, "abs_error_delta")[["patient_id", "seed", "seed_group", "label", "abs_error_delta", "prob_delta"]]
    harmed = delta.nlargest(10, "abs_error_delta")[["patient_id", "seed", "seed_group", "label", "abs_error_delta", "prob_delta"]]
    lines = [
        "# Phase C5 C1 vs MVP Patient Delta",
        "",
        "MVP prediction is the per-patient mean strict-MVP validation prediction across available MVP seeds.",
        "",
        "## Mean Effects",
        "",
        grouped.to_markdown(),
        "",
        "## Most Improved",
        "",
        improved.to_markdown(index=False),
        "",
        "## Most Harmed",
        "",
        harmed.to_markdown(index=False),
    ]
    (out_dir / "c1_vs_mvp_patient_delta_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stratified_report(strata: pd.DataFrame, out_dir: Path) -> None:
    focus = strata.sort_values("mean_abs_error_delta").head(20)
    harmed = strata.sort_values("mean_abs_error_delta", ascending=False).head(20)
    lines = [
        "# Phase C5 Stratified Evidence Effect",
        "",
        "Negative mean_abs_error_delta means C1 improves over MVP in that stratum.",
        "",
        "## Most Improved Strata",
        "",
        focus.to_markdown(index=False),
        "",
        "## Most Harmed Strata",
        "",
        harmed.to_markdown(index=False),
    ]
    (out_dir / "c1_vs_mvp_stratified_delta_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution_report(dist: pd.DataFrame, seed_summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Phase C5 Prediction Distribution Diagnostics",
        "",
        "Failed seeds are expected to show weaker positive/negative separation when pos_neg_pred_gap and AUC both drop.",
        "",
        "## Seed-Level Separation",
        "",
        seed_summary[["seed", "seed_group", "val_auc", "positive_pred_mean", "negative_pred_mean", "pos_neg_pred_gap"]].to_markdown(index=False),
        "",
        "## Distribution By Seed And Label",
        "",
        dist.to_markdown(index=False),
    ]
    (out_dir / "c1_prediction_distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shortcut_report(residual: pd.DataFrame, out_dir: Path) -> None:
    max_rows = residual.dropna(subset=["abs_spearman"]).groupby(["seed", "seed_group"], as_index=False)["abs_spearman"].max()
    lines = [
        "# Phase C5 Shortcut Residual Audit",
        "",
        "Audit only. Shortcut fields are not model inputs and are not used for labels.",
        "",
        "## Per-Seed Max Absolute Spearman",
        "",
        max_rows.to_markdown(index=False),
        "",
        "## Field-Level Residuals",
        "",
        residual.to_markdown(index=False),
    ]
    (out_dir / "c1_seed_shortcut_residual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze C1 seed-level failure modes against strict MVP.")
    parser.add_argument("--mvp-run-dir", required=True)
    parser.add_argument("--c1-original-run-dir", required=True)
    parser.add_argument("--c1-extended-run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest_frame(Path(args.manifest))
    manifest_val = manifest[manifest["split"].astype(str).str.lower() == "val"].copy()

    c1_preds = read_predictions(Path(args.c1_extended_run_dir), "val")
    mvp_preds = read_predictions(Path(args.mvp_run_dir), "val")
    metrics = read_metrics(Path(args.c1_extended_run_dir))

    seed_summary = build_seed_summary(c1_preds, metrics, manifest_val)
    seed_summary.to_csv(out_dir / "c1_seed_failure_summary.csv", index=False)
    write_seed_report(seed_summary, out_dir)

    delta = build_patient_delta(c1_preds, mvp_preds, manifest_val)
    delta.to_csv(out_dir / "c1_vs_mvp_patient_delta_val.csv", index=False)
    write_patient_delta_report(delta, out_dir)

    strata = build_stratified_delta(delta)
    strata.to_csv(out_dir / "c1_vs_mvp_stratified_delta.csv", index=False)
    write_stratified_report(strata, out_dir)

    dist = prediction_distribution(c1_preds)
    dist.to_csv(out_dir / "c1_prediction_distribution_by_seed.csv", index=False)
    write_distribution_report(dist, seed_summary, out_dir)

    residual = shortcut_residual_rows(c1_preds, manifest_val)
    residual.to_csv(out_dir / "c1_seed_shortcut_residual.csv", index=False)
    write_shortcut_report(residual, out_dir)

    print(f"Wrote Phase C5 seed failure diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
