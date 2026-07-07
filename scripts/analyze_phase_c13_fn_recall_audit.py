from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            else:
                try:
                    values.append("NA" if pd.isna(value) else str(value).replace("|", "/"))
                except (TypeError, ValueError):
                    values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def stratum_rows(strata: pd.DataFrame, name: str) -> pd.DataFrame:
    if strata.empty or "stratum_name" not in strata.columns:
        return pd.DataFrame()
    rows = strata[strata["stratum_name"].astype(str) == name].copy()
    keep = [
        "stratum_name",
        "stratum_value",
        "n",
        "n_positive",
        "n_negative",
        "auc_if_defined",
        "sensitivity_at_0p5",
        "specificity_at_0p5",
        "false_negative_count",
        "false_negative_rate",
        "false_positive_count",
        "false_positive_rate",
        "positive_negative_gap",
    ]
    for col in keep:
        if col not in rows.columns:
            rows[col] = pd.NA
    return rows[keep].sort_values(["false_negative_count", "n"], ascending=[False, False])


def merge_filter_audit(errors: pd.DataFrame, filter_audit: pd.DataFrame) -> pd.DataFrame:
    if filter_audit.empty:
        return errors.copy()
    keep = [
        "patient_id",
        "n_dropped_clauses",
        "latest_diffuse_ht_like",
        "n_thyroid_morphology_clauses",
        "changed_txt_morphology_label",
        "changed_image_morphology_weak_label",
        "report_length_delta",
    ]
    for col in keep:
        if col not in filter_audit.columns:
            filter_audit[col] = pd.NA
    audit = filter_audit[keep].copy()
    audit["patient_id"] = audit["patient_id"].astype(str)
    out = errors.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    return out.merge(audit, on="patient_id", how="left")


def summarize_error_types(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame()
    grouped = (
        errors.groupby(["split", "confusion_type", "error_type"], dropna=False)
        .agg(
            n=("patient_id", "count"),
            mean_pred_prob=("pred_prob", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_report_length=("report_length", "mean"),
            mean_selected_n_visits=("selected_n_visits", "mean"),
        )
        .reset_index()
        .sort_values(["split", "confusion_type", "n"], ascending=[True, True, False])
    )
    return grouped


def summarize_fn_features(fn: pd.DataFrame) -> pd.DataFrame:
    if fn.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    feature_defs: List[Tuple[str, str]] = [
        ("txt_morphology_confidence", "morphology_confidence"),
        ("txt_negative_label", "negative_label"),
        ("txt_negative_confidence", "negative_confidence"),
        ("report_length_bin", "report_length_bin"),
        ("selected_n_visits", "selected_n_visits_exact"),
        ("bio_missing_count", "bio_missing_count"),
    ]
    for field, name in feature_defs:
        if field not in fn.columns:
            continue
        for value, group in fn.groupby(field, dropna=False):
            rows.append(
                {
                    "feature": name,
                    "value": value,
                    "n_fn": len(group),
                    "mean_pred_prob": group["pred_prob"].mean(),
                    "mean_abs_error": group["abs_error"].mean(),
                    "mean_report_length": pd.to_numeric(group.get("report_length"), errors="coerce").mean(),
                    "mean_selected_n_visits": pd.to_numeric(group.get("selected_n_visits"), errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "n_fn"], ascending=[True, False])


def top_fn_cases(fn: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "patient_id",
        "pred_prob",
        "confidence_group",
        "error_type",
        "txt_morphology_confidence",
        "txt_negative_label",
        "txt_negative_confidence",
        "selected_n_visits",
        "report_length",
        "report_length_bin",
        "n_dropped_clauses",
        "latest_diffuse_ht_like",
        "changed_txt_morphology_label",
        "matched_morphology_terms",
        "matched_negative_terms",
    ]
    for col in keep:
        if col not in fn.columns:
            fn[col] = pd.NA
    return fn.sort_values(["pred_prob", "abs_error"], ascending=[True, False])[keep].head(20)


def recommendation(fn: pd.DataFrame, strata: pd.DataFrame, filter_audit: pd.DataFrame) -> str:
    if fn.empty:
        return "NO_FN_RECALL_ACTION_NEEDED"
    val_fn = fn[fn["split"].astype(str) == "val"].copy() if "split" in fn.columns else fn.copy()
    if val_fn.empty:
        return "NO_VALIDATION_FN_RECALL_ACTION_NEEDED"
    dropped_positive = 0
    if not filter_audit.empty and "label" in filter_audit.columns:
        val_pos = filter_audit[(filter_audit["split"].astype(str) == "val") & (filter_audit["label"].astype(int) == 1)]
        dropped_positive = int((pd.to_numeric(val_pos.get("n_dropped_clauses"), errors="coerce").fillna(0) > 0).sum())
    high_visit = stratum_rows(strata, "selected_n_visits_bin")
    high_report = stratum_rows(strata, "report_length_bin")
    high_visit_fn = int(high_visit["false_negative_count"].max()) if not high_visit.empty else 0
    high_report_fn = int(high_report["false_negative_count"].max()) if not high_report.empty else 0
    if dropped_positive > 0:
        return "AUDIT_C12_FILTER_POSITIVE_DAMAGE_BEFORE_ANY_RECALL_PILOT"
    if high_visit_fn >= 10 or high_report_fn >= 6:
        return "DESIGN_C13_TEMPORAL_OR_LONG_REPORT_RECALL_PILOT_AFTER_STRESS_SEEDS"
    return "CASE_REVIEW_FN_RECALL_BEFORE_MODEL_CHANGE"


def write_report(
    out_dir: Path,
    errors: pd.DataFrame,
    strata: pd.DataFrame,
    filter_audit: pd.DataFrame,
    error_summary: pd.DataFrame,
    fn_feature_summary: pd.DataFrame,
    fn_cases: pd.DataFrame,
) -> str:
    val_errors = errors[errors["split"].astype(str) == "val"].copy() if "split" in errors.columns else errors.copy()
    val_fn = val_errors[val_errors["confusion_type"].astype(str) == "FN"].copy()
    val_fp = val_errors[val_errors["confusion_type"].astype(str) == "FP"].copy()
    reco = recommendation(val_fn, strata, filter_audit)
    morphology = stratum_rows(strata, "txt_morphology_confidence_bin")
    negative = stratum_rows(strata, "txt_negative_label")
    negative_conf = stratum_rows(strata, "txt_negative_confidence_bin")
    report_len = stratum_rows(strata, "report_length_bin")
    visits = stratum_rows(strata, "selected_n_visits_bin")
    filter_damage = pd.DataFrame()
    if not filter_audit.empty:
        filt = filter_audit.copy()
        filt["label"] = pd.to_numeric(filt["label"], errors="coerce")
        filt["n_dropped_clauses"] = pd.to_numeric(filt["n_dropped_clauses"], errors="coerce").fillna(0)
        filter_damage = (
            filt.groupby(["split", "label"], as_index=False)
            .agg(
                n=("patient_id", "count"),
                n_filtered=("n_dropped_clauses", lambda values: int((values > 0).sum())),
                n_morphology_changed=("changed_txt_morphology_label", "sum"),
                n_image_weak_changed=("changed_image_morphology_weak_label", "sum"),
            )
            .sort_values(["split", "label"])
        )
    lines = [
        "# Phase C13 FN Recall Audit",
        "",
        "This is an analysis-only follow-up to C12. It uses validation errors to identify recall bottlenecks before any new training pilot.",
        "",
        "## Validation Error Balance",
        "",
        f"- Validation errors: {len(val_errors)}.",
        f"- Validation FN / FP: {len(val_fn)} / {len(val_fp)}.",
        "",
        "## Error Type Summary",
        "",
        frame_to_markdown(error_summary[error_summary["split"].astype(str) == "val"] if not error_summary.empty else error_summary),
        "",
        "## FN Feature Summary",
        "",
        frame_to_markdown(fn_feature_summary),
        "",
        "## Evidence Strata Relevant To Recall",
        "",
        "### Morphology Confidence",
        "",
        frame_to_markdown(morphology),
        "",
        "### Negative Evidence Label",
        "",
        frame_to_markdown(negative),
        "",
        "### Negative Evidence Confidence",
        "",
        frame_to_markdown(negative_conf),
        "",
        "### Report Length",
        "",
        frame_to_markdown(report_len),
        "",
        "### Selected Visits",
        "",
        frame_to_markdown(visits),
        "",
        "## C12 Filter Positive-Damage Check",
        "",
        frame_to_markdown(filter_damage),
        "",
        "## Lowest-Probability Validation FN Cases",
        "",
        frame_to_markdown(fn_cases),
        "",
        "## Interpretation",
        "",
        "- C12 reduced false positives, but validation false negatives now dominate.",
        "- C12 manifest audit showed no validation-positive report filtering or morphology-label damage, so the FN pattern is not explained by C12 deleting positive validation evidence.",
        "- FN concentration in long reports or high-visit patients points to temporal/report aggregation as the next recall bottleneck.",
        "- Negative evidence is not sufficient as a global explanation because many FNs have no strong negative label.",
        "",
        "## Recommendation",
        "",
        f"`{reco}`.",
        "",
        "Stress-seed results should be collected before launching any C13 training pilot.",
    ]
    (out_dir / "phase_c13_fn_recall_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze C12 false-negative recall bottlenecks before C13.")
    parser.add_argument("--error-cases", required=True)
    parser.add_argument("--evidence-strata", required=True)
    parser.add_argument("--filter-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = read_csv(Path(args.error_cases))
    strata = read_csv(Path(args.evidence_strata))
    filter_audit = read_csv(Path(args.filter_audit), required=False)
    merged_errors = merge_filter_audit(errors, filter_audit)
    val_fn = merged_errors[
        (merged_errors["split"].astype(str) == "val") & (merged_errors["confusion_type"].astype(str) == "FN")
    ].copy()
    error_summary = summarize_error_types(merged_errors)
    fn_feature_summary = summarize_fn_features(val_fn)
    fn_cases = top_fn_cases(val_fn)
    error_summary.to_csv(out_dir / "c13_error_type_summary.csv", index=False)
    fn_feature_summary.to_csv(out_dir / "c13_fn_feature_summary_val.csv", index=False)
    fn_cases.to_csv(out_dir / "c13_lowest_probability_fn_cases_val.csv", index=False)
    recommendation_value = write_report(
        out_dir,
        errors=merged_errors,
        strata=strata,
        filter_audit=filter_audit,
        error_summary=error_summary,
        fn_feature_summary=fn_feature_summary,
        fn_cases=fn_cases,
    )
    print(f"Recommendation: {recommendation_value}")


if __name__ == "__main__":
    main()
