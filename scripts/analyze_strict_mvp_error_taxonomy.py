from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


EVIDENCE_FIELDS = [
    "txt_morphology_label",
    "txt_morphology_confidence",
    "matched_morphology_terms",
    "txt_negative_label",
    "txt_negative_confidence",
    "matched_negative_terms",
]
SHORTCUT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]
OUTPUT_COLUMNS = [
    "patient_id",
    "split",
    "seed",
    "label",
    "pred_prob",
    "pred_label",
    "is_error",
    "confusion_type",
    "confidence_group",
    "abs_error",
    "error_margin",
    "error_type",
    *EVIDENCE_FIELDS,
    *SHORTCUT_FIELDS,
    "report_length_bin",
]


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
        values: List[str] = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value))
            elif pd.isna(value):
                values.append("NA")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def prob_column(frame: pd.DataFrame) -> str:
    for column in ("pred_prob", "prob", "prediction_prob", "score"):
        if column in frame.columns:
            return column
    raise ValueError("Prediction CSV must contain pred_prob, prob, prediction_prob, or score.")


def precomputed_label_column(frame: pd.DataFrame) -> str | None:
    for column in ("pred_label", "prediction", "pred", "y_pred"):
        if column in frame.columns:
            return column
    return None


def prediction_files(run_dir: Path, split: str) -> List[Path]:
    patterns = [
        run_dir / "predictions" / f"{split}_predictions_seed_*.csv",
        run_dir / "**" / f"{split}_predictions_seed_*.csv",
        run_dir / "**" / f"*{split}*prediction*.csv",
        run_dir / "**" / f"*prediction*{split}*.csv",
    ]
    files: List[Path] = []
    for pattern in patterns:
        if "**" in str(pattern):
            files.extend(sorted(run_dir.glob(str(pattern.relative_to(run_dir)))))
        else:
            files.extend(sorted(pattern.parent.glob(pattern.name)))
    return sorted({path.resolve() for path in files if path.is_file()})


def read_predictions(run_dir: Path, split: str, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    files = prediction_files(run_dir, split)
    if not files:
        input_rows.append({"path": str(run_dir), "status": f"missing_{split}_predictions", "notes": "No prediction CSVs discovered."})
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
            if "patient_id" not in frame.columns:
                input_rows.append({"path": str(path), "status": "skipped", "notes": "Missing patient_id column."})
                continue
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["seed"] = frame["seed"] if "seed" in frame.columns else seed_from_path(path)
            frame["split"] = frame["split"] if "split" in frame.columns else split
            frame["pred_prob"] = pd.to_numeric(frame[prob_column(frame)], errors="coerce")
            label_col = precomputed_label_column(frame)
            if label_col:
                frame["pred_label"] = pd.to_numeric(frame[label_col], errors="coerce")
            frames.append(frame)
            input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows"})
        except Exception as exc:
            input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_manifest_frame(path: Path, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    try:
        frame = pd.DataFrame(read_manifest(path))
        input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} manifest rows"})
    except Exception as exc:
        input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
        return pd.DataFrame()
    frame["patient_id"] = frame["patient_id"].astype(str)
    for field in ["patient_id", "split", "label", *EVIDENCE_FIELDS, *SHORTCUT_FIELDS]:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame[["patient_id", "split", "label", *EVIDENCE_FIELDS, *SHORTCUT_FIELDS]].drop_duplicates("patient_id")


def merge_manifest(preds: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    if preds.empty:
        return preds
    merged = preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    for field in ["split", "label", *EVIDENCE_FIELDS, *SHORTCUT_FIELDS]:
        manifest_col = f"{field}_manifest"
        if field not in merged.columns and manifest_col in merged.columns:
            merged[field] = merged[manifest_col]
        elif field in merged.columns and manifest_col in merged.columns:
            merged[field] = merged[field].where(~merged[field].isna(), merged[manifest_col])
    merged["split"] = merged["split"].fillna(split)
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce").astype("Int64")
    merged["pred_label"] = pd.to_numeric(merged.get("pred_label", pd.NA), errors="coerce")
    merged["pred_label"] = merged["pred_label"].where(~merged["pred_label"].isna(), (merged["pred_prob"] >= 0.5).astype(int)).astype(int)
    return merged


def bin_numeric_split_local(frame: pd.DataFrame, field: str, out_field: str) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce") if field in frame.columns else pd.Series(pd.NA, index=frame.index)
    out = pd.Series("missing", index=frame.index, dtype=object)
    valid = values.dropna()
    if valid.empty:
        return out
    try:
        labels = ["q1_low", "q2_midlow", "q3_midhigh", "q4_high"]
        binned = pd.qcut(values, q=4, labels=labels, duplicates="drop")
        out = binned.astype(object).where(~binned.isna(), "missing")
    except ValueError:
        median = valid.median()
        out = pd.Series(["high" if value >= median else "low" for value in values.fillna(median)], index=frame.index)
        out = out.where(~values.isna(), "missing")
    return out.rename(out_field)


def confidence_group(prob: float) -> str:
    if prob >= 0.8:
        return "high_confidence_positive"
    if prob >= 0.6:
        return "medium_confidence_positive"
    if prob >= 0.4:
        return "borderline"
    if prob >= 0.2:
        return "medium_confidence_negative"
    return "high_confidence_negative"


def confusion_type(label: int, pred_label: int) -> str:
    if label == 1 and pred_label == 1:
        return "TP"
    if label == 0 and pred_label == 0:
        return "TN"
    if label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def assign_taxonomy(row: pd.Series, report_q3: float | None, visits_q3: float | None) -> str:
    if not bool(row["is_error"]):
        return "correct"
    morph_label = as_float(row.get("txt_morphology_label"))
    morph_conf = as_float(row.get("txt_morphology_confidence"))
    pred_prob = float(row["pred_prob"])
    report_length = as_float(row.get("report_length"))
    visits = as_float(row.get("selected_n_visits"))
    has_bio = as_float(row.get("has_bio"))
    bio_missing = as_float(row.get("bio_missing_count"))
    if row["confusion_type"] == "FN" and morph_label == 1 and (morph_conf is None or morph_conf >= 0.6):
        return "morphology_positive_false_negative"
    if row["confusion_type"] == "FP" and (morph_label != 1 or (morph_conf is not None and morph_conf < 0.6)):
        return "morphology_low_confidence_false_positive"
    if row["confusion_type"] == "FN" and pred_prob < 0.2:
        return "high_confidence_false_negative"
    if row["confusion_type"] == "FP" and pred_prob >= 0.8:
        return "high_confidence_false_positive"
    if (report_q3 is not None and report_length is not None and report_length >= report_q3) or (
        visits_q3 is not None and visits is not None and visits >= visits_q3
    ):
        return "long_report_or_multivisit_uncertainty"
    if (has_bio == 0 or (bio_missing is not None and bio_missing > 0)) and (pred_prob >= 0.8 or pred_prob < 0.2):
        return "bio_missing_overconfident_error"
    if 0.4 <= pred_prob < 0.6:
        return "borderline_error"
    return "other_error"


def enrich_cases(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame = frame.dropna(subset=["label", "pred_prob"])
    frame["label"] = frame["label"].astype(int)
    frame["confidence_group"] = frame["pred_prob"].astype(float).map(confidence_group)
    frame["confusion_type"] = [confusion_type(int(y), int(p)) for y, p in zip(frame["label"], frame["pred_label"])]
    frame["is_error"] = frame["confusion_type"].isin(["FP", "FN"])
    frame["abs_error"] = (frame["label"] - frame["pred_prob"]).abs()
    frame["error_margin"] = frame.apply(lambda row: 1.0 - row["pred_prob"] if int(row["label"]) == 1 else row["pred_prob"], axis=1)
    frame["report_length_bin"] = bin_numeric_split_local(frame, "report_length", "report_length_bin")
    report_q3 = pd.to_numeric(frame.get("report_length", pd.Series(dtype=float)), errors="coerce").quantile(0.75)
    visits_q3 = pd.to_numeric(frame.get("selected_n_visits", pd.Series(dtype=float)), errors="coerce").quantile(0.75)
    report_q3 = None if pd.isna(report_q3) else float(report_q3)
    visits_q3 = None if pd.isna(visits_q3) else float(visits_q3)
    frame["error_type"] = frame.apply(lambda row: assign_taxonomy(row, report_q3, visits_q3), axis=1)
    for col in OUTPUT_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    errors = frame[frame["is_error"]].copy()
    errors = errors.sort_values(["abs_error", "seed", "patient_id"], ascending=[False, True, True])
    return errors[OUTPUT_COLUMNS]


def taxonomy_summary(val_errors: pd.DataFrame, test_errors: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, frame in [("val", val_errors), ("test_reporting_only", test_errors)]:
        total = len(frame)
        if frame.empty:
            continue
        for error_type, group in frame.groupby("error_type"):
            rows.append(
                {
                    "split": split,
                    "error_type": error_type,
                    "n_errors": len(group),
                    "proportion_of_errors": len(group) / max(total, 1),
                    "false_negative_count": int((group["confusion_type"] == "FN").sum()),
                    "false_positive_count": int((group["confusion_type"] == "FP").sum()),
                    "mean_pred_prob": group["pred_prob"].mean(),
                    "mean_abs_error": group["abs_error"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "n_errors", "error_type"], ascending=[True, False, True]) if rows else pd.DataFrame()


def write_report(summary: pd.DataFrame, val_errors: pd.DataFrame, test_errors: pd.DataFrame, out_dir: Path) -> None:
    val_table = summary[summary["split"] == "val"].copy() if not summary.empty else pd.DataFrame()
    fn = val_errors[val_errors["confusion_type"] == "FN"]["error_type"].value_counts().reset_index()
    fp = val_errors[val_errors["confusion_type"] == "FP"]["error_type"].value_counts().reset_index()
    if not fn.empty:
        fn.columns = ["error_type", "n"]
    if not fp.empty:
        fp.columns = ["error_type", "n"]
    lines = [
        "# Strict MVP Error Taxonomy Report",
        "",
        "Target model: strict structural matched DMEA-MVP. This is analysis-only; no training was performed.",
        "",
        "## Validation Error Taxonomy",
        "",
        frame_to_markdown(val_table),
        "",
        "## Top False-Negative Categories",
        "",
        frame_to_markdown(fn.head(8)),
        "",
        "## Top False-Positive Categories",
        "",
        frame_to_markdown(fp.head(8)),
        "",
        "## Test Reporting-Only Note",
        "",
        f"Test reporting-only error rows: {len(test_errors)}. These rows are for transparency and manual review only.",
    ]
    (out_dir / "strict_mvp_error_taxonomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze strict MVP patient-level error taxonomy.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, str]] = []
    manifest = read_manifest_frame(Path(args.manifest), input_rows)
    outputs: Dict[str, pd.DataFrame] = {}
    for split, suffix in [("val", "val"), ("test", "test_reporting_only")]:
        preds = read_predictions(run_dir, split, input_rows)
        merged = merge_manifest(preds, manifest, split)
        outputs[suffix] = enrich_cases(merged, split)
    outputs["val"].to_csv(out_dir / "strict_mvp_error_cases_val.csv", index=False)
    outputs["test_reporting_only"].to_csv(out_dir / "strict_mvp_error_cases_test_reporting_only.csv", index=False)
    high_val = outputs["val"][
        ((outputs["val"]["confusion_type"] == "FN") & (outputs["val"]["pred_prob"] < 0.2))
        | ((outputs["val"]["confusion_type"] == "FP") & (outputs["val"]["pred_prob"] >= 0.8))
    ].sort_values("abs_error", ascending=False)
    high_test = outputs["test_reporting_only"][
        ((outputs["test_reporting_only"]["confusion_type"] == "FN") & (outputs["test_reporting_only"]["pred_prob"] < 0.2))
        | ((outputs["test_reporting_only"]["confusion_type"] == "FP") & (outputs["test_reporting_only"]["pred_prob"] >= 0.8))
    ].sort_values("abs_error", ascending=False)
    high_val.to_csv(out_dir / "strict_mvp_high_confidence_errors_val.csv", index=False)
    high_test.to_csv(out_dir / "strict_mvp_high_confidence_errors_test_reporting_only.csv", index=False)
    summary = taxonomy_summary(outputs["val"], outputs["test_reporting_only"])
    summary.to_csv(out_dir / "strict_mvp_error_taxonomy_summary.csv", index=False)
    pd.DataFrame(input_rows).to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    write_report(summary, outputs["val"], outputs["test_reporting_only"], out_dir)
    print(f"Wrote strict MVP error taxonomy outputs to {out_dir}")


if __name__ == "__main__":
    main()
