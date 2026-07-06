from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


EVIDENCE_COLUMNS = [
    "txt_morphology_label",
    "txt_morphology_confidence",
    "matched_morphology_terms",
    "image_morphology_weak_label",
    "image_morphology_weak_confidence",
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = seed_from_path(path)
        if "split" not in frame.columns:
            frame["split"] = split
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["prob_for_analysis"] = frame[prob_column(frame)].astype(float)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {split} prediction CSV files found under {run_dir / 'predictions'}")
    return pd.concat(frames, ignore_index=True)


def manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    keep = ["patient_id", "split", "label", "report_text"] + [col for col in EVIDENCE_COLUMNS if col in frame.columns]
    return frame[[col for col in keep if col in frame.columns]].drop_duplicates("patient_id")


def case_group(label: int, mvp_prob: float, c1_prob: float) -> str:
    mvp_correct = int(mvp_prob >= 0.5) == int(label)
    c1_correct = int(c1_prob >= 0.5) == int(label)
    if mvp_correct and c1_correct:
        return "both_correct"
    if (not mvp_correct) and c1_correct:
        return "mvp_wrong_c1_correct"
    if mvp_correct and (not c1_correct):
        return "mvp_correct_c1_wrong"
    return "both_wrong"


def truncate_text(value: Any, limit: int = 1000) -> str:
    text = "" if pd.isna(value) else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_effects(mvp: pd.DataFrame, c1: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    merged = mvp.merge(
        c1,
        on=["patient_id", "seed", "split"],
        suffixes=("_mvp", "_c1"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping patients/seeds for split {split}.")
    label_col = "label_mvp" if "label_mvp" in merged.columns else "label"
    merged["label"] = merged[label_col].astype(int)
    merged["mvp_prob"] = merged["prob_for_analysis_mvp"].astype(float)
    merged["c1_prob"] = merged["prob_for_analysis_c1"].astype(float)
    merged["c1_minus_mvp_prob"] = merged["c1_prob"] - merged["mvp_prob"]
    merged["mvp_abs_error"] = (merged["mvp_prob"] - merged["label"]).abs()
    merged["c1_abs_error"] = (merged["c1_prob"] - merged["label"]).abs()
    merged["c1_minus_mvp_abs_error"] = merged["c1_abs_error"] - merged["mvp_abs_error"]
    merged["mvp_brier"] = (merged["mvp_prob"] - merged["label"]) ** 2
    merged["c1_brier"] = (merged["c1_prob"] - merged["label"]) ** 2
    merged["c1_minus_mvp_brier"] = merged["c1_brier"] - merged["mvp_brier"]
    merged["case_group"] = [
        case_group(label, mvp_prob, c1_prob)
        for label, mvp_prob, c1_prob in zip(merged["label"], merged["mvp_prob"], merged["c1_prob"])
    ]
    merged = merged.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "report_text" in merged.columns:
        merged["report_text_if_available_or_omit_if_not_safe"] = merged["report_text"].map(truncate_text)
    output_cols = [
        "patient_id",
        "split",
        "seed",
        "label",
        "mvp_prob",
        "c1_prob",
        "c1_minus_mvp_prob",
        "mvp_abs_error",
        "c1_abs_error",
        "c1_minus_mvp_abs_error",
        "mvp_brier",
        "c1_brier",
        "c1_minus_mvp_brier",
        "case_group",
    ]
    output_cols.extend(col for col in EVIDENCE_COLUMNS if col in merged.columns and col not in output_cols)
    if "report_text_if_available_or_omit_if_not_safe" in merged.columns:
        output_cols.append("report_text_if_available_or_omit_if_not_safe")
    return merged[[col for col in output_cols if col in merged.columns]].sort_values(["seed", "case_group", "patient_id"])


def summarize_effects(frame: pd.DataFrame, split: str) -> List[str]:
    lines = [
        f"## {split}",
        "",
        f"Rows: {len(frame)} patient-seed predictions.",
        f"Mean C1-MVP probability delta: {frame['c1_minus_mvp_prob'].mean():.4f}.",
        f"Mean C1-MVP absolute-error delta: {frame['c1_minus_mvp_abs_error'].mean():.4f} (negative is better).",
        "",
        "| case_group | n | mean prob delta | mean abs-error delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    grouped = frame.groupby("case_group", dropna=False)
    for name, group in grouped:
        lines.append(
            f"| {name} | {len(group)} | {group['c1_minus_mvp_prob'].mean():.4f} | "
            f"{group['c1_minus_mvp_abs_error'].mean():.4f} |"
        )
    if "txt_morphology_label" in frame.columns:
        lines.extend(["", "| txt_morphology_label | n | mean abs-error delta |", "| --- | ---: | ---: |"])
        for name, group in frame.groupby("txt_morphology_label", dropna=False):
            lines.append(f"| {name} | {len(group)} | {group['c1_minus_mvp_abs_error'].mean():.4f} |")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction-level effects from strict MVP to C1 text evidence.")
    parser.add_argument("--mvp-run", required=True)
    parser.add_argument("--c1-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_frame(Path(args.manifest))
    report_lines = [
        "# Phase C3 C1 Evidence Effects",
        "",
        "This analysis compares strict MVP predictions against C1 text morphology predictions.",
        "Validation split is decision-relevant; test split is reporting-only.",
        "",
    ]
    for split, filename in (
        ("val", "c1_evidence_effects_val.csv"),
        ("test", "c1_evidence_effects_test_reporting_only.csv"),
    ):
        mvp = read_predictions(Path(args.mvp_run), split)
        c1 = read_predictions(Path(args.c1_run), split)
        effects = build_effects(mvp, c1, manifest, split)
        effects.to_csv(out_dir / filename, index=False)
        report_lines.extend(summarize_effects(effects, split))
    (out_dir / "c1_evidence_effects_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote C1 evidence effects to {out_dir}")


if __name__ == "__main__":
    main()
