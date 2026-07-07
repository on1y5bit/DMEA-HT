from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


AUDIT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]
EVIDENCE_FIELDS = [
    "txt_morphology_label",
    "txt_morphology_confidence",
    "matched_morphology_terms",
    "txt_negative_label",
    "txt_negative_confidence",
    "matched_negative_terms",
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
            elif isinstance(value, (list, tuple, set)):
                values.append(str(list(value)).replace("|", "/"))
            elif isinstance(value, dict):
                values.append(json.dumps(value, ensure_ascii=False).replace("|", "/"))
            else:
                try:
                    if pd.isna(value):
                        values.append("NA")
                    else:
                        values.append(str(value).replace("|", "/"))
                except (TypeError, ValueError):
                    values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    parsed = parse_maybe_list(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None", "NA"}:
        return []
    return [text]


def to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def text_value(row: pd.Series) -> str:
    for field in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        if field in row and not pd.isna(row[field]):
            return str(row[field])
    for field in ("visits", "records", "followups", "reports"):
        if field in row and not pd.isna(row[field]):
            return json.dumps(row[field], ensure_ascii=False)[:6000]
    return ""


def split_report_segments(text: str) -> List[str]:
    if not text:
        return []
    pieces = re.split(r"(?:\n+|。|；|;|\|\|\||\t+)", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def count_term_segments(segments: List[str], terms: Iterable[str]) -> int:
    terms = [term for term in terms if term]
    if not segments or not terms:
        return 0
    return sum(1 for segment in segments if any(term in segment for term in terms))


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    for field in ["patient_id", "split", "label", *EVIDENCE_FIELDS, *AUDIT_FIELDS, "report_text"]:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame.drop_duplicates("patient_id")


def read_error_cases(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    if "confusion_type" not in frame.columns:
        raise ValueError(f"Missing confusion_type in {path}")
    return frame


def quartile_threshold(frame: pd.DataFrame, field: str) -> float | None:
    if field not in frame.columns:
        return None
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.quantile(0.75))


def bool_int(value: bool) -> int:
    return 1 if value else 0


def build_patient_audit(errors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    fp = errors[errors["confusion_type"].astype(str) == "FP"].copy()
    if fp.empty:
        return pd.DataFrame()
    merged = fp.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    for field in [*EVIDENCE_FIELDS, *AUDIT_FIELDS, "label", "split"]:
        manifest_col = f"{field}_manifest"
        if manifest_col in merged.columns:
            if field not in merged.columns:
                merged[field] = merged[manifest_col]
            else:
                merged[field] = merged[field].where(~merged[field].isna(), merged[manifest_col])
    report_q3 = quartile_threshold(merged, "report_length")
    visits_q3 = quartile_threshold(merged, "selected_n_visits")
    rows: List[Dict[str, Any]] = []
    for patient_id, group in merged.groupby("patient_id"):
        first = group.iloc[0]
        morphology_terms = as_list(first.get("matched_morphology_terms"))
        negative_terms = as_list(first.get("matched_negative_terms"))
        report_text = text_value(first)
        segments = split_report_segments(report_text)
        report_length = to_float(first.get("report_length"))
        n_visits = to_float(first.get("selected_n_visits"))
        high_conf_count = int((pd.to_numeric(group["pred_prob"], errors="coerce") >= 0.8).sum())
        seed_count = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        morph_label = to_float(first.get("txt_morphology_label"))
        neg_label = to_float(first.get("txt_negative_label"))
        neg_conf = to_float(first.get("txt_negative_confidence"))
        morph_conf = to_float(first.get("txt_morphology_confidence"))
        long_report = report_q3 is not None and report_length is not None and report_length >= report_q3
        multi_visit = visits_q3 is not None and n_visits is not None and n_visits >= visits_q3
        evidence_overlap = bool(morphology_terms and negative_terms)
        negative_evidence_positive = neg_label == 1
        aggregation_suspected = evidence_overlap and (long_report or multi_visit or seed_count >= 2)
        weak_negative_conflict = negative_evidence_positive and neg_conf is not None and neg_conf >= 0.7 and morph_label == 1
        morphology_only_fp = bool(morphology_terms) and not negative_terms
        rows.append(
            {
                "patient_id": patient_id,
                "label": int(first.get("label", 0)),
                "n_fp_seed_rows": len(group),
                "n_unique_fp_seeds": seed_count,
                "mean_pred_prob": pd.to_numeric(group["pred_prob"], errors="coerce").mean(),
                "max_pred_prob": pd.to_numeric(group["pred_prob"], errors="coerce").max(),
                "high_confidence_fp_seed_count": high_conf_count,
                "persistent_fp_all_formal_seeds": bool_int(seed_count >= 3),
                "txt_morphology_label": morph_label,
                "txt_morphology_confidence": morph_conf,
                "matched_morphology_terms": morphology_terms,
                "txt_negative_label": neg_label,
                "txt_negative_confidence": neg_conf,
                "matched_negative_terms": negative_terms,
                "morphology_negative_evidence_overlap": bool_int(evidence_overlap),
                "negative_evidence_positive": bool_int(negative_evidence_positive),
                "weak_negative_conflict": bool_int(weak_negative_conflict),
                "morphology_only_fp": bool_int(morphology_only_fp),
                "selected_n_visits": n_visits,
                "used_images": to_float(first.get("used_images")),
                "image_padding_count": to_float(first.get("image_padding_count")),
                "has_bio": to_float(first.get("has_bio")),
                "bio_missing_count": to_float(first.get("bio_missing_count")),
                "report_length": report_length,
                "long_report_q4": bool_int(long_report),
                "multi_visit_q4": bool_int(multi_visit),
                "report_segment_count_estimate": len(segments),
                "segments_with_morphology_terms": count_term_segments(segments, morphology_terms),
                "segments_with_negative_terms": count_term_segments(segments, negative_terms),
                "aggregation_artifact_suspected": bool_int(aggregation_suspected),
                "report_text_preview": report_text[:800].replace("\n", " "),
                "audit_priority": audit_priority(high_conf_count, aggregation_suspected, weak_negative_conflict, seed_count),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["audit_priority", "max_pred_prob", "n_unique_fp_seeds"], ascending=[False, False, False])


def audit_priority(high_conf_count: int, aggregation: bool, weak_negative_conflict: bool, seed_count: int) -> int:
    score = 0
    score += 3 if high_conf_count > 0 else 0
    score += 2 if aggregation else 0
    score += 2 if weak_negative_conflict else 0
    score += 1 if seed_count >= 2 else 0
    return score


def summarize_flags(patient_audit: pd.DataFrame) -> pd.DataFrame:
    if patient_audit.empty:
        return pd.DataFrame()
    flags = [
        "persistent_fp_all_formal_seeds",
        "high_confidence_fp_seed_count",
        "morphology_negative_evidence_overlap",
        "negative_evidence_positive",
        "weak_negative_conflict",
        "morphology_only_fp",
        "long_report_q4",
        "multi_visit_q4",
        "aggregation_artifact_suspected",
    ]
    rows: List[Dict[str, Any]] = []
    n = len(patient_audit)
    for flag in flags:
        if flag not in patient_audit.columns:
            continue
        if flag == "high_confidence_fp_seed_count":
            count = int((patient_audit[flag] > 0).sum())
        else:
            count = int((patient_audit[flag].astype(float) > 0).sum())
        rows.append({"flag": flag, "n_patients": count, "fraction_of_fp_patients": count / max(n, 1)})
    return pd.DataFrame(rows)


def write_report(patient_audit: pd.DataFrame, flag_summary: pd.DataFrame, out_dir: Path) -> str:
    if patient_audit.empty:
        recommendation = "RETURN_TO_DATA_AUDIT"
        lines = ["# Phase C9 False-Positive Data Audit", "", "No validation false positives were available for audit."]
        (out_dir / "phase_c9_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return recommendation
    n_patients = len(patient_audit)
    high_conf = int((patient_audit["high_confidence_fp_seed_count"] > 0).sum())
    overlap = int((patient_audit["morphology_negative_evidence_overlap"] > 0).sum())
    agg = int((patient_audit["aggregation_artifact_suspected"] > 0).sum())
    neg_pos = int((patient_audit["negative_evidence_positive"] > 0).sum())
    persistent = int((patient_audit["persistent_fp_all_formal_seeds"] > 0).sum())
    if agg / max(n_patients, 1) >= 0.30 or overlap / max(n_patients, 1) >= 0.50:
        recommendation = "DATA_CONSTRUCTION_AUDIT_BEFORE_MODEL_CHANGE"
    elif high_conf >= 5 and neg_pos / max(n_patients, 1) >= 0.50:
        recommendation = "EVIDENCE_LABEL_AUDIT_BEFORE_MODEL_CHANGE"
    else:
        recommendation = "NO_MODEL_CHANGE_CONTINUE_CASE_REVIEW"
    top_cols = [
        "patient_id",
        "n_unique_fp_seeds",
        "mean_pred_prob",
        "max_pred_prob",
        "high_confidence_fp_seed_count",
        "matched_morphology_terms",
        "matched_negative_terms",
        "report_length",
        "selected_n_visits",
        "aggregation_artifact_suspected",
        "audit_priority",
    ]
    top_cases = patient_audit[[col for col in top_cols if col in patient_audit.columns]].head(15)
    lines = [
        "# Phase C9 False-Positive Data Audit",
        "",
        "Phase C9 is analysis-only. No model, data loader, label, split, manifest, or training code was changed.",
        "",
        "## Validation False-Positive Patient Summary",
        "",
        f"- Unique validation FP patients: {n_patients}.",
        f"- FP patients present across all three formal seeds: {persistent}.",
        f"- FP patients with at least one high-confidence FP seed: {high_conf}.",
        f"- FP patients with morphology/negative-evidence overlap: {overlap}.",
        f"- FP patients with txt_negative_label=1: {neg_pos}.",
        f"- FP patients with aggregation-artifact suspicion: {agg}.",
        "",
        "## Flag Summary",
        "",
        frame_to_markdown(flag_summary),
        "",
        "## Highest-Priority Patient Cases",
        "",
        frame_to_markdown(top_cases),
        "",
        "## Interpretation",
        "",
        "- This audit does not prove shortcut causality and does not use audit-only fields as classifier inputs.",
        "- Repeated high-confidence false positives across seeds are stronger data-audit targets than single-seed false positives.",
        "- Morphology and negative-evidence overlap is a likely source of misleading patient-level report aggregation.",
        "- Long-report or multi-visit concentration should be treated as a report-construction and case-review signal before changing the model.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
        "",
        "Before any model or data-construction change, manually review the high-priority patient cases and verify whether positive morphology terms are historical, negated, contradicted, or mixed with later negative evidence.",
    ]
    (out_dir / "phase_c9_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict MVP validation false positives for Phase C9.")
    parser.add_argument("--c8-error-cases-val", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_cases = read_error_cases(Path(args.c8_error_cases_val))
    manifest = read_manifest_frame(Path(args.manifest))
    patient_audit = build_patient_audit(error_cases, manifest)
    flag_summary = summarize_flags(patient_audit)
    patient_audit.to_csv(out_dir / "c9_fp_patient_audit_val.csv", index=False)
    flag_summary.to_csv(out_dir / "c9_fp_flag_summary_val.csv", index=False)
    if not patient_audit.empty:
        patient_audit.head(30).to_csv(out_dir / "c9_fp_high_priority_cases_val.csv", index=False)
    recommendation = write_report(patient_audit, flag_summary, out_dir)
    print(f"Wrote Phase C9 false-positive audit to {out_dir}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
