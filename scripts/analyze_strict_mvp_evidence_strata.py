from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dmea_ht.metrics import compute_binary_metrics
from analyze_strict_mvp_error_taxonomy import (
    SHORTCUT_FIELDS,
    confidence_group,
    confusion_type,
    frame_to_markdown,
    merge_manifest,
    read_manifest_frame,
    read_predictions,
)


STRATA_FIELDS = [
    "txt_morphology_label",
    "txt_morphology_confidence_bin",
    "matched_morphology_terms_present",
    "txt_negative_label",
    "txt_negative_confidence_bin",
    "report_length_bin",
    "selected_n_visits_bin",
    "used_images_bin",
    "has_bio",
    "bio_missing_count_bin",
]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def confidence_bin(value: Any) -> str:
    try:
        if pd.isna(value):
            return "missing"
        conf = float(value)
    except (TypeError, ValueError):
        return "missing"
    if conf >= 0.7:
        return "high"
    if conf >= 0.4:
        return "medium"
    return "low"


def terms_present(value: Any) -> str:
    if value is None or pd.isna(value):
        return "absent"
    text = str(value).strip()
    if not text or text in {"[]", "NA", "nan", "None"}:
        return "absent"
    return "present"


def quantile_bin(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce") if field in frame.columns else pd.Series(pd.NA, index=frame.index)
    out = pd.Series("missing", index=frame.index, dtype=object)
    valid = values.dropna()
    if valid.empty:
        return out
    if valid.nunique() <= 2:
        return values.astype(object).where(~values.isna(), "missing").astype(str)
    try:
        labels = ["q1_low", "q2_midlow", "q3_midhigh", "q4_high"]
        binned = pd.qcut(values, q=4, labels=labels, duplicates="drop")
        return binned.astype(object).where(~binned.isna(), "missing")
    except ValueError:
        median = valid.median()
        out = pd.Series(["high" if value >= median else "low" for value in values.fillna(median)], index=frame.index)
        return out.where(~values.isna(), "missing")


def bio_missing_bin(value: Any) -> str:
    try:
        if pd.isna(value):
            return "missing"
        count = float(value)
    except (TypeError, ValueError):
        return "missing"
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    return "2plus"


def prepare_split(run_dir: Path, manifest: pd.DataFrame, split: str, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    preds = read_predictions(run_dir, split, input_rows)
    merged = merge_manifest(preds, manifest, split)
    if merged.empty:
        return merged
    merged = merged.dropna(subset=["label", "pred_prob"]).copy()
    merged["label"] = merged["label"].astype(int)
    merged["pred_prob"] = pd.to_numeric(merged["pred_prob"], errors="coerce")
    merged = merged.dropna(subset=["pred_prob"])
    merged["pred_label"] = (merged["pred_prob"] >= 0.5).astype(int)
    merged["confusion_type"] = [confusion_type(int(y), int(p)) for y, p in zip(merged["label"], merged["pred_label"])]
    merged["is_error"] = merged["confusion_type"].isin(["FP", "FN"])
    merged["confidence_group"] = merged["pred_prob"].map(confidence_group)
    merged["txt_morphology_confidence_bin"] = merged["txt_morphology_confidence"].map(confidence_bin) if "txt_morphology_confidence" in merged.columns else "missing"
    merged["txt_negative_confidence_bin"] = merged["txt_negative_confidence"].map(confidence_bin) if "txt_negative_confidence" in merged.columns else "missing"
    merged["matched_morphology_terms_present"] = merged["matched_morphology_terms"].map(terms_present) if "matched_morphology_terms" in merged.columns else "absent"
    merged["report_length_bin"] = quantile_bin(merged, "report_length")
    merged["selected_n_visits_bin"] = quantile_bin(merged, "selected_n_visits")
    merged["used_images_bin"] = quantile_bin(merged, "used_images")
    merged["bio_missing_count_bin"] = merged["bio_missing_count"].map(bio_missing_bin) if "bio_missing_count" in merged.columns else "missing"
    if "has_bio" in merged.columns:
        merged["has_bio"] = merged["has_bio"].astype(object).where(~merged["has_bio"].isna(), "missing").astype(str)
    else:
        merged["has_bio"] = "missing"
    for field in STRATA_FIELDS + SHORTCUT_FIELDS:
        if field not in merged.columns:
            merged[field] = pd.NA
    return merged


def safe_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    y = frame["label"].astype(int)
    p = frame["pred_prob"].astype(float)
    out: Dict[str, Any] = {}
    if y.nunique() >= 2:
        metrics = compute_binary_metrics(y, p)
        out["auc_if_defined"] = metrics["AUC"]
        out["auprc_if_defined"] = metrics["AUPRC"]
    else:
        out["auc_if_defined"] = pd.NA
        out["auprc_if_defined"] = pd.NA
    preds = (p >= 0.5).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    out.update(
        {
            "accuracy_at_0p5": (tp + tn) / max(len(frame), 1),
            "sensitivity_at_0p5": sensitivity,
            "specificity_at_0p5": specificity,
            "f1_at_0p5": f1,
            "false_negative_count": fn,
            "false_positive_count": fp,
            "false_negative_rate": fn / max(int((y == 1).sum()), 1),
            "false_positive_rate": fp / max(int((y == 0).sum()), 1),
        }
    )
    return out


def build_strata(split: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for stratum in STRATA_FIELDS:
        for value, group in frame.groupby(stratum, dropna=False):
            pos = group[group["label"] == 1]["pred_prob"]
            neg = group[group["label"] == 0]["pred_prob"]
            row: Dict[str, Any] = {
                "split": split,
                "stratum_name": stratum,
                "stratum_value": "NA" if pd.isna(value) else str(value),
                "n": len(group),
                "n_positive": int((group["label"] == 1).sum()),
                "n_negative": int((group["label"] == 0).sum()),
                "mean_pred_prob": group["pred_prob"].mean(),
                "mean_pred_prob_positive": pos.mean() if not pos.empty else pd.NA,
                "mean_pred_prob_negative": neg.mean() if not neg.empty else pd.NA,
                "positive_negative_gap": (pos.mean() - neg.mean()) if (not pos.empty and not neg.empty) else pd.NA,
            }
            row.update(safe_metrics(group))
            rows.append(row)
    return pd.DataFrame(rows)


def overall_metrics(split: str, frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "split": split,
            "n": 0,
            "warning": "missing_predictions",
        }
    pos = frame[frame["label"] == 1]["pred_prob"]
    neg = frame[frame["label"] == 0]["pred_prob"]
    metrics = safe_metrics(frame)
    return {
        "split": split,
        "n": len(frame),
        "n_positive": int((frame["label"] == 1).sum()),
        "n_negative": int((frame["label"] == 0).sum()),
        "auc_if_defined": metrics["auc_if_defined"],
        "auprc_if_defined": metrics["auprc_if_defined"],
        "accuracy_at_0p5": metrics["accuracy_at_0p5"],
        "sensitivity_at_0p5": metrics["sensitivity_at_0p5"],
        "specificity_at_0p5": metrics["specificity_at_0p5"],
        "f1_at_0p5": metrics["f1_at_0p5"],
        "mean_pred_prob": frame["pred_prob"].mean(),
        "mean_pred_prob_positive": pos.mean() if not pos.empty else pd.NA,
        "mean_pred_prob_negative": neg.mean() if not neg.empty else pd.NA,
        "positive_negative_gap": (pos.mean() - neg.mean()) if (not pos.empty and not neg.empty) else pd.NA,
        "false_negative_count": metrics["false_negative_count"],
        "false_positive_count": metrics["false_positive_count"],
        "warning": "",
    }


def shortcut_strata(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    for field in SHORTCUT_FIELDS:
        work[f"{field}_audit_bin"] = quantile_bin(work, field)
        for value, group in work.groupby(f"{field}_audit_bin", dropna=False):
            y = group["label"].astype(int)
            p = group["pred_prob"].astype(float)
            auc = compute_binary_metrics(y, p)["AUC"] if y.nunique() >= 2 else pd.NA
            rows.append(
                {
                    "field": field,
                    "bin": "NA" if pd.isna(value) else str(value),
                    "n": len(group),
                    "error_rate": float(group["is_error"].mean()),
                    "fn_rate": float((group["confusion_type"] == "FN").sum() / max((y == 1).sum(), 1)),
                    "fp_rate": float((group["confusion_type"] == "FP").sum() / max((y == 0).sum(), 1)),
                    "mean_pred_prob": p.mean(),
                    "mean_label": y.mean(),
                    "auc_if_defined": auc,
                }
            )
    return pd.DataFrame(rows)


def write_report(val_strata: pd.DataFrame, test_strata: pd.DataFrame, shortcut: pd.DataFrame, out_dir: Path) -> None:
    warnings = val_strata[val_strata["auc_if_defined"].isna()][["stratum_name", "stratum_value", "n"]] if not val_strata.empty else pd.DataFrame()
    top_fn = val_strata.sort_values(["false_negative_rate", "n"], ascending=[False, False]).head(10) if not val_strata.empty else pd.DataFrame()
    top_fp = val_strata.sort_values(["false_positive_rate", "n"], ascending=[False, False]).head(10) if not val_strata.empty else pd.DataFrame()
    shortcut_top = shortcut.sort_values(["error_rate", "n"], ascending=[False, False]).head(12) if not shortcut.empty else pd.DataFrame()
    lines = [
        "# Strict MVP Evidence Diagnostics Report",
        "",
        "Target model: strict structural matched DMEA-MVP. Evidence fields and shortcut fields are analysis-only.",
        "",
        "## Undefined Validation AUC/AUPRC Strata",
        "",
        frame_to_markdown(warnings.head(30)),
        "",
        "## Validation Strata With Highest False-Negative Rates",
        "",
        frame_to_markdown(top_fn[["stratum_name", "stratum_value", "n", "false_negative_rate", "mean_pred_prob_positive"]].head(10) if not top_fn.empty else top_fn),
        "",
        "## Validation Strata With Highest False-Positive Rates",
        "",
        frame_to_markdown(top_fp[["stratum_name", "stratum_value", "n", "false_positive_rate", "mean_pred_prob_negative"]].head(10) if not top_fp.empty else top_fp),
        "",
        "## Shortcut Audit Strata",
        "",
        "These bins are audit-only and are not causal evidence.",
        "",
        frame_to_markdown(shortcut_top),
        "",
        "## Test Reporting-Only",
        "",
        f"Test reporting-only strata rows: {len(test_strata)}.",
    ]
    (out_dir / "strict_mvp_evidence_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze strict MVP evidence strata.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, str]] = []
    manifest = read_manifest_frame(Path(args.manifest), input_rows)
    val = prepare_split(Path(args.run_dir), manifest, "val", input_rows)
    test = prepare_split(Path(args.run_dir), manifest, "test", input_rows)
    val_strata = build_strata("val", val)
    test_strata = build_strata("test_reporting_only", test)
    shortcut = shortcut_strata(val)
    overall = pd.DataFrame([overall_metrics("val", val), overall_metrics("test_reporting_only", test)])
    val_strata.to_csv(out_dir / "strict_mvp_evidence_strata_val.csv", index=False)
    test_strata.to_csv(out_dir / "strict_mvp_evidence_strata_test_reporting_only.csv", index=False)
    shortcut.to_csv(out_dir / "strict_mvp_shortcut_strata_val.csv", index=False)
    overall.to_csv(out_dir / "strict_mvp_overall_metrics.csv", index=False)
    existing_inputs = out_dir / "inputs_used_and_missing.csv"
    if existing_inputs.exists():
        combined = pd.concat([pd.read_csv(existing_inputs), pd.DataFrame(input_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(input_rows)
    combined.drop_duplicates().to_csv(existing_inputs, index=False)
    write_report(val_strata, test_strata, shortcut, out_dir)
    print(f"Wrote strict MVP evidence strata outputs to {out_dir}")


if __name__ == "__main__":
    main()
