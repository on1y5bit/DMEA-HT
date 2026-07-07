from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


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


def read_csv(path: Path, required: bool, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    if not path.exists():
        input_rows.append({"path": str(path), "status": "missing_required" if required else "missing_optional", "notes": "Input not found."})
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
        return pd.DataFrame()
    input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows"})
    return frame


def read_text(path: Path, required: bool, input_rows: List[Dict[str, str]]) -> str:
    if not path.exists():
        input_rows.append({"path": str(path), "status": "missing_required" if required else "missing_optional", "notes": "Input not found."})
        return ""
    text = path.read_text(encoding="utf-8")
    input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(text)} chars"})
    return text


def get_overall(overall: pd.DataFrame, split: str) -> pd.Series:
    if overall.empty or "split" not in overall.columns:
        return pd.Series(dtype=object)
    rows = overall[overall["split"].astype(str) == split]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def top_error_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    val = summary[summary["split"].astype(str) == "val"].copy()
    top = val.sort_values("n_errors", ascending=False).head(10)
    fn = val[val["false_negative_count"] > 0].sort_values("false_negative_count", ascending=False).head(8)
    fp = val[val["false_positive_count"] > 0].sort_values("false_positive_count", ascending=False).head(8)
    return top, fn, fp


def high_conf_examples(high_conf: pd.DataFrame) -> pd.DataFrame:
    keep = ["patient_id", "label", "pred_prob", "error_type", "matched_morphology_terms", "report_length", "selected_n_visits"]
    if high_conf.empty:
        return pd.DataFrame(columns=keep)
    frame = high_conf.copy()
    for col in keep:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame.sort_values("abs_error", ascending=False)[keep].head(12)


def strata_focus(strata: pd.DataFrame, name: str) -> pd.DataFrame:
    if strata.empty:
        return pd.DataFrame()
    frame = strata[strata["stratum_name"].astype(str) == name].copy()
    if frame.empty:
        return frame
    cols = [
        "stratum_name",
        "stratum_value",
        "n",
        "auc_if_defined",
        "auprc_if_defined",
        "sensitivity_at_0p5",
        "specificity_at_0p5",
        "positive_negative_gap",
        "false_negative_rate",
        "false_positive_rate",
    ]
    return frame[[col for col in cols if col in frame.columns]].sort_values("n", ascending=False)


def shortcut_interpretation(shortcut: pd.DataFrame, overall_error_rate: float) -> tuple[str, pd.DataFrame]:
    if shortcut.empty:
        return "No shortcut strata rows were available.", shortcut
    frame = shortcut.copy()
    frame["error_rate_delta_vs_overall"] = pd.to_numeric(frame["error_rate"], errors="coerce") - overall_error_rate
    top = frame.sort_values(["error_rate_delta_vs_overall", "n"], ascending=[False, False]).head(12)
    elevated = top[(top["n"] >= 10) & (top["error_rate_delta_vs_overall"] >= 0.20)]
    if elevated.empty:
        text = "Validation errors do not show a large audit-bin concentration by the configured threshold. Shortcut fields remain audit-only."
    else:
        text = "Some validation errors are concentrated in selected structural audit bins. This is a diagnostic caution signal, not proof of shortcut or a training feature."
    return text, top


def recommend(summary: pd.DataFrame, high_conf: pd.DataFrame, shortcut_top: pd.DataFrame) -> str:
    if summary.empty:
        return "RETURN_TO_DATA_AUDIT"
    val = summary[summary["split"].astype(str) == "val"].copy()
    total_errors = int(val["n_errors"].sum()) if "n_errors" in val.columns else 0
    if total_errors <= 0:
        return "NO_TRAIN_CONTINUE_ANALYSIS"
    morph_fn = val[val["error_type"].astype(str) == "morphology_positive_false_negative"]["n_errors"].sum()
    borderline = val[val["error_type"].astype(str) == "borderline_error"]["n_errors"].sum()
    high_conf_n = len(high_conf)
    elevated_shortcut = (
        not shortcut_top.empty
        and "error_rate_delta_vs_overall" in shortcut_top.columns
        and bool(((shortcut_top["n"] >= 10) & (shortcut_top["error_rate_delta_vs_overall"] >= 0.20)).any())
    )
    if elevated_shortcut:
        return "RETURN_TO_DATA_AUDIT"
    if morph_fn >= 3 and morph_fn / total_errors >= 0.20:
        return "ALLOW_SMALL_PILOT_POSITIVE_PRESERVATION"
    if borderline / total_errors >= 0.35:
        return "ALLOW_SMALL_PILOT_CALIBRATION_ONLY"
    if high_conf_n >= 5:
        return "RETURN_TO_DATA_AUDIT"
    return "NO_TRAIN_CONTINUE_ANALYSIS"


def write_final_report(
    out_dir: Path,
    phase_c7_text: str,
    overall: pd.DataFrame,
    taxonomy: pd.DataFrame,
    strata: pd.DataFrame,
    high_conf: pd.DataFrame,
    shortcut: pd.DataFrame,
    inputs: pd.DataFrame,
) -> str:
    val_overall = get_overall(overall, "val")
    top_errors, top_fn, top_fp = top_error_tables(taxonomy)
    overall_error_rate = 0.0
    if not val_overall.empty and "false_negative_count" in val_overall and "false_positive_count" in val_overall and "n" in val_overall:
        overall_error_rate = (float(val_overall["false_negative_count"]) + float(val_overall["false_positive_count"])) / max(float(val_overall["n"]), 1.0)
    shortcut_text, shortcut_top = shortcut_interpretation(shortcut, overall_error_rate)
    recommendation = recommend(taxonomy, high_conf, shortcut_top)
    morphology = strata_focus(strata, "txt_morphology_label")
    morph_conf = strata_focus(strata, "txt_morphology_confidence_bin")
    negative = strata_focus(strata, "txt_negative_label")
    negative_conf = strata_focus(strata, "txt_negative_confidence_bin")
    structure = pd.concat(
        [
            strata_focus(strata, "report_length_bin").head(8),
            strata_focus(strata, "selected_n_visits_bin").head(8),
            strata_focus(strata, "used_images_bin").head(8),
            strata_focus(strata, "has_bio").head(8),
            strata_focus(strata, "bio_missing_count_bin").head(8),
        ],
        ignore_index=True,
    )
    examples = high_conf_examples(high_conf)
    c7_note = "Phase C7 report was loaded." if phase_c7_text else "Phase C7 report was not available to this collector."
    lines = [
        "# Phase C8 Final Report",
        "",
        "## Route Status",
        "",
        "- Current main path: strict structural matched DMEA-MVP.",
        "- C1/C2/C6 remain ablation-only and are not revived in Phase C8.",
        "- No training was performed in C8.",
        f"- {c7_note}",
        "",
        "## Validation-Only Strict MVP Performance Recap",
        "",
        f"- Validation AUC: {fmt(val_overall.get('auc_if_defined'))}.",
        f"- Validation AUPRC: {fmt(val_overall.get('auprc_if_defined'))}.",
        f"- Sensitivity/specificity at threshold 0.5: {fmt(val_overall.get('sensitivity_at_0p5'))} / {fmt(val_overall.get('specificity_at_0p5'))}.",
        f"- Positive-negative prediction gap: {fmt(val_overall.get('positive_negative_gap'))}.",
        f"- Validation false negatives / false positives: {fmt(val_overall.get('false_negative_count'), 0)} / {fmt(val_overall.get('false_positive_count'), 0)}.",
        "",
        "## Error Taxonomy Summary",
        "",
        frame_to_markdown(top_errors),
        "",
        "### Top False-Negative Categories",
        "",
        frame_to_markdown(top_fn),
        "",
        "### Top False-Positive Categories",
        "",
        frame_to_markdown(top_fp),
        "",
        "## Evidence Strata Findings",
        "",
        "### Morphology Evidence Label",
        "",
        frame_to_markdown(morphology),
        "",
        "### Morphology Evidence Confidence",
        "",
        frame_to_markdown(morph_conf),
        "",
        "### Negative Evidence Label",
        "",
        frame_to_markdown(negative),
        "",
        "### Negative Evidence Confidence",
        "",
        frame_to_markdown(negative_conf),
        "",
        "### Report, Visit/Image, And Bio Availability Strata",
        "",
        frame_to_markdown(structure),
        "",
        "## High-Confidence Error Examples",
        "",
        frame_to_markdown(examples),
        "",
        "## Shortcut Audit Interpretation",
        "",
        shortcut_text,
        "",
        frame_to_markdown(shortcut_top),
        "",
        "## Next-Phase Recommendation",
        "",
        f"`{recommendation}`.",
        "",
        "This recommendation is validation-based. Test outputs are reporting-only and did not drive the recommendation.",
        "",
        "## Suggested Future Gate",
        "",
        "- A future pilot may be considered only if the validation-set error pattern is concrete and reproducible.",
        "- No test tuning.",
        "- No shortcut variables as classifier inputs.",
        "- Bad-seed or stress-seed pilot before formal training.",
        "- Positive-preservation check before formal training.",
        "- Formal seeds remain 0, 42, and 3407 only after the pilot gate passes.",
        "",
        "## Inputs Used",
        "",
        frame_to_markdown(inputs),
    ]
    (out_dir / "phase_c8_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C8 strict MVP evidence diagnostics.")
    parser.add_argument("--phase-c8-dir", required=True)
    parser.add_argument("--phase-c7-dir", required=False, default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_c8 = Path(args.phase_c8_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, str]] = []
    taxonomy = read_csv(phase_c8 / "strict_mvp_error_taxonomy_summary.csv", True, input_rows)
    overall = read_csv(phase_c8 / "strict_mvp_overall_metrics.csv", False, input_rows)
    strata = read_csv(phase_c8 / "strict_mvp_evidence_strata_val.csv", True, input_rows)
    high_conf = read_csv(phase_c8 / "strict_mvp_high_confidence_errors_val.csv", True, input_rows)
    shortcut = read_csv(phase_c8 / "strict_mvp_shortcut_strata_val.csv", True, input_rows)
    if args.phase_c7_dir:
        phase_c7_text = read_text(Path(args.phase_c7_dir) / "phase_c7_final_report.md", False, input_rows)
    else:
        phase_c7_text = ""
    discovered = read_csv(phase_c8 / "inputs_used_and_missing.csv", False, input_rows)
    inputs = pd.concat([pd.DataFrame(input_rows), discovered], ignore_index=True) if not discovered.empty else pd.DataFrame(input_rows)
    inputs.drop_duplicates().to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    recommendation = write_final_report(out_dir, phase_c7_text, overall, taxonomy, strata, high_conf, shortcut, inputs)
    print(f"Wrote Phase C8 final report to {out_dir / 'phase_c8_final_report.md'}")
    print(f"Next-phase recommendation: {recommendation}")


if __name__ == "__main__":
    main()
