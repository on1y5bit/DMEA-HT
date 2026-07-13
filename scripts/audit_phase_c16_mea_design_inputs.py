#!/usr/bin/env python3
"""Audit real C13 inputs and prior diagnostics before any C16-MEA model work."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found as a literal assignment in {path}")


SHORTCUT_FIELDS = literal_assignment(DEFAULT_REPO_ROOT / "dmea_ht/data.py", "SHORTCUT_FIELDS")
BIO_COLUMNS = literal_assignment(DEFAULT_REPO_ROOT / "scripts/build_distmatch_manifest.py", "BIO_COLUMNS")
MORPHOLOGY_HIGH_CONF = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "MORPHOLOGY_HIGH_CONF"
)
MORPHOLOGY_MED_CONF = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "MORPHOLOGY_MED_CONF"
)
NEGATIVE_STRONG_TERMS = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "NEGATIVE_STRONG_TERMS"
)
NEGATIVE_WEAK_TERMS = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "NEGATIVE_WEAK_TERMS"
)
UNCERTAIN_TERMS = literal_assignment(DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "UNCERTAIN_TERMS")
DIAG_HINT_TERMS = literal_assignment(DEFAULT_REPO_ROOT / "scripts/build_evidence_weak_labels.py", "DIAG_HINT_TERMS")
MORPHOLOGY_CUES = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/analyze_phase_c11_report_filter_hypotheses.py", "MORPHOLOGY_CUES"
)
DIFFUSE_HT_CUES = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/analyze_phase_c11_report_filter_hypotheses.py", "DIFFUSE_HT_CUES"
)
BENIGN_NODULE_CUES = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/analyze_phase_c11_report_filter_hypotheses.py", "BENIGN_NODULE_CUES"
)
NEGATIVE_THYROID_CUES = literal_assignment(
    DEFAULT_REPO_ROOT / "scripts/analyze_phase_c11_report_filter_hypotheses.py", "NEGATIVE_THYROID_CUES"
)


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() in {".jsonl", ".ndjson"}:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("samples", "records", "patients"):
                if key in data:
                    return list(data[key])
        return list(data)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_maybe_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[:1] in "[{":
            try:
                parsed = ast.literal_eval(stripped)
                return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                pass
        for separator in ("|", ";"):
            if separator in stripped:
                return [item for item in stripped.split(separator) if item]
        return [stripped]
    return [value]


REQUIRED_MODEL_FIELDS = {
    "patient_id",
    "label",
    "split",
    "image_paths",
    "report_text",
    "bio_values",
    "bio_missing_mask",
    "bio_abnormal_flags",
}

WEAK_LABEL_PREFIXES = (
    "txt_",
    "image_morphology_weak_",
    "bio_immune_abnormal_",
    "bio_function_abnormal_",
    "bio_missing_label",
    "discordance_state_",
    "matched_",
    "evidence_label_source",
)

C13_TEMPORAL_FIELDS = [
    "phase_c13_temporal_focus",
    "phase_c13_focus_prefix_chars",
    "phase_c13_focus_latest_date",
    "phase_c13_n_latest_focus_clauses",
    "phase_c13_n_history_focus_clauses",
    "phase_c13_n_visits_parsed",
    "phase_c13_focus_source",
    "selected_visit_dates",
]

EXTRA_SHORTCUT_FIELDS = [
    "selected_visit_dates",
    "phase_c13_focus_prefix_chars",
    "phase_c13_n_latest_focus_clauses",
    "phase_c13_n_history_focus_clauses",
    "phase_c13_n_visits_parsed",
]

C14_ARTIFACTS = [
    {
        "phase": "C14-A",
        "path": "analysis_reports/phase_c14a/c14a_positive_patient_token_exposure_val.csv",
        "purpose": "text evidence exposure and latest/full/first-window term counts",
        "reusable": "first-window exposure; morphology/diffuse/negative term counts; cross-seed FN/TP groups",
        "restriction": "audit stratification only; never a model target or selector",
    },
    {
        "phase": "C14-B",
        "path": "analysis_reports/phase_c14b/c14b_representation_diagnostics_val.csv",
        "purpose": "C13 representation, contribution, evidence-score, and discordance diagnostics",
        "reusable": "encoder/global norms; classifier contributions; evidence scores; pairwise cosines",
        "restriction": "validation diagnostics only; do not feed saved diagnostics into C16-MEA",
    },
    {
        "phase": "C14-B",
        "path": "analysis_reports/phase_c14b/c14b_modality_masking_val.csv",
        "purpose": "diagnostic modality masking and single-modality-like probabilities",
        "reusable": "delta_mask_text/image/bio and contribution direction as audit comparators",
        "restriction": "distribution-shift diagnostic, not a training ablation target",
    },
    {
        "phase": "C14-B",
        "path": "analysis_reports/phase_c14b/c14b_text_occlusion_val.csv",
        "purpose": "diffuse, negative, and temporal-prefix text occlusion diagnostics",
        "reusable": "delta_remove_diffuse/negative and prefix effects",
        "restriction": "validation diagnostic only; no patient labels derived from occlusion",
    },
    {
        "phase": "C14-C",
        "path": "analysis_reports/phase_c14c/c14c_pairwise_inversions_by_seed.csv",
        "purpose": "validation positive-negative inversion inventory",
        "reusable": "pair IDs, margins, and inversion counts for later comparison",
        "restriction": "reporting/audit only; no validation pairs in training",
    },
    {
        "phase": "C14-D",
        "path": "analysis_reports/phase_c14d/c14d_hard_patient_profiles.csv",
        "purpose": "hard-patient multimodal and evidence profile",
        "reusable": "hard-patient cohorts and diagnostic fields for post-training audit",
        "restriction": "must not become sample weights, model inputs, or route labels",
    },
    {
        "phase": "C14-E",
        "path": "analysis_reports/phase_c14e/c14e_candidate_mechanism_coverage.csv",
        "purpose": "candidate mechanism coverage and matched-control generalizability",
        "reusable": "coverage limitations and proposed mechanism names",
        "restriction": "failed generalizability gate; cannot supervise or justify a broad fix",
    },
    {
        "phase": "C14-E",
        "path": "analysis_reports/phase_c14e/c14e_route_gate_summary.csv",
        "purpose": "final C14 evidence gate",
        "reusable": "DATA_LIMIT_NO_GENERAL_MODEL_FIX limitation",
        "restriction": "C16-MEA is a separately authorized hypothesis and must retain this limitation",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-table", help="Optional all_patients.xlsx used only for schema/field verification.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--output-dir", default="analysis_reports/phase_c16_mea_design")
    parser.add_argument("--text-max-length", type=int, default=256)
    return parser.parse_args()


def is_empty(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def compact_example(field: str, value: Any) -> str:
    if is_empty(value):
        return ""
    if field in {"report_text", "text", "report"}:
        return f"<text_chars={len(str(value))}>"
    if field in {"image_paths", "images", "selected_visit_dates"}:
        return f"<list_len={len(parse_maybe_list(value))}>"
    if field.startswith("bio_"):
        return f"<list_len={len(parse_maybe_list(value))}>"
    if isinstance(value, dict):
        return "<dict_keys=" + ",".join(sorted(str(key) for key in value)[:12]) + ">"
    text = str(value).replace("\n", " ")
    return text[:80]


def field_policy(field: str) -> tuple[str, int, str]:
    if field == "patient_id":
        return "identifier", 0, "grouping/export only"
    if field == "label":
        return "training_target", 1, "patient-level binary target only"
    if field == "split":
        return "partition", 0, "loader partition only"
    if field in {"image_paths", "report_text", "bio_values"}:
        return "model_input", 1, "unchanged C13 modality input"
    if field == "bio_missing_mask":
        return "validity_mask", 1, "mask invalid bio computations; never reduce to a predictive count"
    if field == "bio_abnormal_flags":
        return "untrusted_legacy_tensor", 0, "zero-filled placeholder unless an explicit trusted source exists"
    if field in set(SHORTCUT_FIELDS) | set(EXTRA_SHORTCUT_FIELDS):
        return "shortcut_audit_only", 0, "audit/export only; prohibited from model, gates, losses, or reliability scalars"
    if field.startswith(WEAK_LABEL_PREFIXES):
        return "legacy_weak_label_audit_only", 0, "dictionary audit metadata only; no BCE or target use"
    if field.startswith("phase_c13_"):
        return "temporal_audit_metadata", 0, "offline verification only; derive model masks from report markers"
    if field == "sample_weight":
        return "legacy_training_control", 0, "preserve C13 behavior; not an evidence representation"
    return "manifest_metadata_review", 0, "not authorized as a C16-MEA input without a separate audit"


def audit_input_fields(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    fields = sorted({str(key) for row in rows for key in row.keys()})
    records: List[Dict[str, Any]] = []
    for field in fields:
        values = [row.get(field) for row in rows if field in row]
        nonempty = [value for value in values if not is_empty(value)]
        category, allowed, rule = field_policy(field)
        records.append(
            {
                "field": field,
                "rows_present": len(values),
                "rows_nonempty": len(nonempty),
                "nonempty_fraction": len(nonempty) / len(rows) if rows else 0.0,
                "observed_types": "|".join(sorted({type(value).__name__ for value in nonempty})),
                "example_shape_or_value": compact_example(field, nonempty[0]) if nonempty else "",
                "policy_category": category,
                "allowed_as_c16_mea_input": allowed,
                "policy_rule": rule,
            }
        )
    for field in sorted(REQUIRED_MODEL_FIELDS - set(fields)):
        category, allowed, rule = field_policy(field)
        records.append(
            {
                "field": field,
                "rows_present": 0,
                "rows_nonempty": 0,
                "nonempty_fraction": 0.0,
                "observed_types": "",
                "example_shape_or_value": "",
                "policy_category": category,
                "allowed_as_c16_mea_input": allowed,
                "policy_rule": rule,
            }
        )
    return pd.DataFrame(records).sort_values(["policy_category", "field"]).reset_index(drop=True)


def numeric_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def missing_flag(value: Any) -> int:
    numeric = numeric_or_nan(value)
    return int(numeric == 1) if np.isfinite(numeric) else 1


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def semantic_group(field: str) -> str:
    key = normalized_name(field)
    if key in {"tgab", "tpoab"}:
        return "immune_observed"
    if key in {"ft3", "ft4", "tsh"}:
        return "thyroid_function_observed"
    return "other_observed"


def reference_columns(source_columns: Sequence[str], field: str) -> List[str]:
    key = normalized_name(field)
    markers = ("low", "lower", "high", "upper", "ref", "range", "normal", "min", "max")
    out = []
    for column in source_columns:
        normalized = normalized_name(column)
        if key and key in normalized and normalized != key and any(marker in normalized for marker in markers):
            out.append(str(column))
    return out


def trusted_abnormal_row(row: Mapping[str, Any]) -> bool:
    for key in ("bio_abnormal_flags_trusted", "bio_abnormal_source", "bio_reference_range_source"):
        value = row.get(key)
        if value not in (None, "", 0, "0", False, "false", "False"):
            return True
    return False


def audit_bio_fields(
    rows: Sequence[Mapping[str, Any]], source_table: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    source_columns = list(source_table.columns.astype(str)) if source_table is not None else []
    source_by_normalized = {normalized_name(column): column for column in source_columns}
    trusted_rows = sum(1 for row in rows if trusted_abnormal_row(row))
    vector_lengths = Counter(len(parse_maybe_list(row.get("bio_values"))) for row in rows)
    mask_lengths = Counter(len(parse_maybe_list(row.get("bio_missing_mask"))) for row in rows)
    flag_lengths = Counter(len(parse_maybe_list(row.get("bio_abnormal_flags"))) for row in rows)
    records: List[Dict[str, Any]] = []
    for index, field in enumerate(BIO_COLUMNS):
        observed_values: List[float] = []
        abnormal_ones = 0
        flag_values = 0
        subgroup_total: Counter[str] = Counter()
        subgroup_observed: Counter[str] = Counter()
        for row in rows:
            values = parse_maybe_list(row.get("bio_values"))
            missing = parse_maybe_list(row.get("bio_missing_mask"))
            flags = parse_maybe_list(row.get("bio_abnormal_flags"))
            subgroup = f"{str(row.get('split', '')).lower()}_label{int(float(row.get('label', 0)))}"
            subgroup_total[subgroup] += 1
            is_missing_value = index >= len(missing) or missing_flag(missing[index]) == 1
            if index < len(values) and not is_missing_value:
                numeric = numeric_or_nan(values[index])
                if np.isfinite(numeric):
                    observed_values.append(numeric)
                    subgroup_observed[subgroup] += 1
            if index < len(flags):
                flag = numeric_or_nan(flags[index])
                if np.isfinite(flag):
                    flag_values += 1
                    abnormal_ones += int(flag == 1)
        source_column = source_by_normalized.get(normalized_name(field), "")
        ref_columns = reference_columns(source_columns, field)
        source_observed = 0
        source_dtype = ""
        if source_table is not None and source_column:
            source_series = source_table[source_column]
            source_observed = int(source_series.notna().sum())
            source_dtype = str(source_series.dtype)
        group = semantic_group(field)
        record = {
                "bio_index": index,
                "field_name": field,
                "semantic_group": group,
                "manifest_rows": len(rows),
                "manifest_observed_count": len(observed_values),
                "manifest_observed_fraction": len(observed_values) / len(rows) if rows else 0.0,
                "manifest_min": min(observed_values) if observed_values else np.nan,
                "manifest_median": float(np.median(observed_values)) if observed_values else np.nan,
                "manifest_max": max(observed_values) if observed_values else np.nan,
                "source_column_present": int(bool(source_column)),
                "source_column": source_column,
                "source_dtype": source_dtype,
                "source_observed_count": source_observed,
                "reference_range_columns": "|".join(ref_columns),
                "reference_range_available": int(bool(ref_columns)),
                "trusted_abnormal_metadata_rows": trusted_rows,
                "abnormal_flag_value_count": flag_values,
                "abnormal_flag_one_count": abnormal_ones,
                "allowed_c16_mea_semantics": f"{group}_continuous_values_with_validity_mask",
                "allowed_rule_direction": "latent_only; no rule-based abnormal/support direction",
                "prohibited_use": "untrusted abnormal flags, invented reference ranges, or missingness count as evidence",
        }
        for subgroup in ("train_label0", "train_label1", "val_label0", "val_label1", "test_label0", "test_label1"):
            record[f"{subgroup}_observed_fraction"] = (
                subgroup_observed[subgroup] / subgroup_total[subgroup] if subgroup_total[subgroup] else np.nan
            )
        records.append(record)
    fields = pd.DataFrame(records)
    grouping_rows: List[Dict[str, Any]] = []
    for group, frame in fields.groupby("semantic_group", sort=False):
        grouping_rows.append(
            {
                "semantic_group": group,
                "field_names": "|".join(frame["field_name"].astype(str)),
                "bio_indices": "|".join(frame["bio_index"].astype(str)),
                "all_source_fields_present": int((frame["source_column_present"] == 1).all()),
                "mean_manifest_observed_fraction": frame["manifest_observed_fraction"].mean(),
                "any_reference_range_available": int((frame["reference_range_available"] == 1).any()),
                "trusted_abnormal_metadata_rows": trusted_rows,
                "implementation_path": (
                    "group observed continuous values; learn evidence roles from patient supervision"
                    if (frame["source_column_present"] == 1).all()
                    else "fallback to neutral bio_observed evidence until field semantics are repaired"
                ),
                "blocked_claim": "no abnormal/normal/support/opposition claim without verified ranges",
            }
        )
    summary = {
        "bio_columns": BIO_COLUMNS,
        "bio_vector_length_counts": {str(key): int(value) for key, value in sorted(vector_lengths.items())},
        "bio_mask_length_counts": {str(key): int(value) for key, value in sorted(mask_lengths.items())},
        "bio_flag_length_counts": {str(key): int(value) for key, value in sorted(flag_lengths.items())},
        "source_table_available": source_table is not None,
        "source_columns": source_columns,
        "all_bio_source_fields_present": bool((fields["source_column_present"] == 1).all()),
        "any_reference_range_available": bool((fields["reference_range_available"] == 1).any()),
        "trusted_abnormal_metadata_rows": trusted_rows,
        "all_abnormal_flags_zero": bool(fields["abnormal_flag_one_count"].sum() == 0),
    }
    return fields, pd.DataFrame(grouping_rows), summary


def unique_terms(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        term = str(value).strip()
        if term and term not in seen:
            out.append(term)
            seen.add(term)
    return out


def audit_text_terms(rows: Sequence[Mapping[str, Any]], visible_chars: int) -> pd.DataFrame:
    dictionaries = [
        ("morphology_high", "support_or_morphology", MORPHOLOGY_HIGH_CONF, "build_evidence_weak_labels.py"),
        ("morphology_medium", "support_or_morphology", MORPHOLOGY_MED_CONF, "build_evidence_weak_labels.py"),
        ("diffuse_ht_like", "support_or_morphology", DIFFUSE_HT_CUES, "analyze_phase_c11_report_filter_hypotheses.py"),
        ("opposition_strong", "opposition_or_normal", NEGATIVE_STRONG_TERMS, "build_evidence_weak_labels.py"),
        ("opposition_weak", "nonspecific_negative", NEGATIVE_WEAK_TERMS, "build_evidence_weak_labels.py"),
        ("uncertainty", "uncertainty", UNCERTAIN_TERMS, "build_evidence_weak_labels.py"),
        ("diagnostic_hint", "diagnostic_support_hint", DIAG_HINT_TERMS, "build_evidence_weak_labels.py"),
        ("benign_nodular", "nonspecific_or_benign_morphology", BENIGN_NODULE_CUES, "analyze_phase_c11_report_filter_hypotheses.py"),
        ("thyroid_morphology_general", "nonspecific_morphology", MORPHOLOGY_CUES, "analyze_phase_c11_report_filter_hypotheses.py"),
        ("opposition_thyroid_general", "opposition_or_normal", NEGATIVE_THYROID_CUES, "analyze_phase_c11_report_filter_hypotheses.py"),
    ]
    texts = [str(row.get("report_text") or row.get("text") or "") for row in rows]
    records: List[Dict[str, Any]] = []
    for dictionary_name, role, terms, source in dictionaries:
        for term in unique_terms(terms):
            full_count = sum(term.upper() in text.upper() if term.upper() == "HT" else term in text for text in texts)
            first_count = sum(
                term.upper() in text[:visible_chars].upper() if term.upper() == "HT" else term in text[:visible_chars]
                for text in texts
            )
            records.append(
                {
                    "dictionary": dictionary_name,
                    "term": term,
                    "semantic_role": role,
                    "source_file": source,
                    "patients_with_term_full_report": full_count,
                    "patients_with_term_model_window": first_count,
                    "requires_negation_or_context_handling": int(role in {"support_or_morphology", "diagnostic_support_hint"}),
                    "allowed_c16_mea_use": "character-position mask for guided pooling plus learned empty-mask fallback",
                    "allowed_as_patient_target": 0,
                    "prohibited_use": "weak-label BCE, image pseudo-label, or guaranteed clinical slot claim",
                }
            )
    return pd.DataFrame(records).sort_values(["dictionary", "term"]).reset_index(drop=True)


def report_text(row: Mapping[str, Any]) -> str:
    return str(row.get("report_text") or row.get("text") or row.get("report") or "")


def audit_temporal_fields(rows: Sequence[Mapping[str, Any]], visible_chars: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    texts = [report_text(row) for row in rows]
    marker_specs = [
        ("latest_focus_marker", "[C13_LATEST_THYROID", "latest thyroid-focused section"),
        ("history_focus_marker", "[C13_HISTORY_THYROID]", "historical thyroid-focused section"),
        ("full_report_marker", "[C13_FULL_REPORT]", "boundary before the unchanged full report"),
    ]
    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"manifest_rows": len(rows), "model_visible_chars": visible_chars}
    for name, marker, semantics in marker_specs:
        present = sum(marker in text for text in texts)
        visible = sum(0 <= text.find(marker) < visible_chars for text in texts)
        summary[f"{name}_count"] = present
        summary[f"{name}_fraction"] = present / len(rows) if rows else 0.0
        summary[f"{name}_visible_count"] = visible
        records.append(
            {
                "item_type": "report_marker",
                "field_or_construct": marker,
                "rows_present_or_derivable": present,
                "fraction_present_or_derivable": present / len(rows) if rows else 0.0,
                "rows_visible_in_model_window": visible,
                "semantics": semantics,
                "allowed_c16_mea_use": "derive character-position masks from the unchanged C13 report text",
                "restriction": "do not fabricate dates or use marker counts as predictive scalars",
            }
        )
    for field in C13_TEMPORAL_FIELDS:
        present = sum(field in row and not is_empty(row.get(field)) for row in rows)
        records.append(
            {
                "item_type": "manifest_metadata",
                "field_or_construct": field,
                "rows_present_or_derivable": present,
                "fraction_present_or_derivable": present / len(rows) if rows else 0.0,
                "rows_visible_in_model_window": 0,
                "semantics": "C13 construction audit metadata",
                "allowed_c16_mea_use": "offline audit and reconstruction verification only",
                "restriction": "not a classifier, gate, loss, or reliability input",
            }
        )
    latest_count = summary.get("latest_focus_marker_count", 0)
    history_count = summary.get("history_focus_marker_count", 0)
    full_count = summary.get("full_report_marker_count", 0)
    derived = [
        ("latest_support_or_opposition", latest_count, "dictionary masks inside the explicit latest-focus section"),
        ("historical_support_or_opposition", history_count, "dictionary masks inside the explicit history-focus section"),
        ("latest_history_conflict", min(latest_count, history_count), "opposing dictionary evidence across explicit latest/history sections"),
        ("full_report_fallback", full_count, "learned pooling over the unchanged full-report section"),
    ]
    for construct, count, semantics in derived:
        records.append(
            {
                "item_type": "derived_temporal_state",
                "field_or_construct": construct,
                "rows_present_or_derivable": count,
                "fraction_present_or_derivable": count / len(rows) if rows else 0.0,
                "rows_visible_in_model_window": np.nan,
                "semantics": semantics,
                "allowed_c16_mea_use": "masked evidence pooling with learned fallback when the section is absent",
                "restriction": "absence is unavailable evidence, not negative disease evidence",
            }
        )
    summary["latest_history_joint_count"] = sum(
        "[C13_LATEST_THYROID" in text and "[C13_HISTORY_THYROID]" in text for text in texts
    )
    return pd.DataFrame(records), summary


def audit_c14_mapping(repo_root: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    available_count = 0
    for spec in C14_ARTIFACTS:
        path = repo_root / spec["path"]
        exists = path.is_file()
        rows = 0
        columns = ""
        route_or_status = ""
        if exists:
            available_count += 1
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                rows = len(frame)
                columns = "|".join(str(column) for column in frame.columns)
                for candidate in ("route", "allowed_next_step", "final_status"):
                    if candidate in frame.columns and not frame.empty:
                        route_or_status += f"{candidate}={frame.iloc[0][candidate]};"
        records.append(
            {
                **spec,
                "available": int(exists),
                "row_count": rows,
                "columns": columns,
                "route_or_status": route_or_status,
            }
        )
    return pd.DataFrame(records), {"available_count": available_count, "required_count": len(C14_ARTIFACTS)}


def audit_shortcuts(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    fields = list(dict.fromkeys(list(SHORTCUT_FIELDS) + EXTRA_SHORTCUT_FIELDS))
    records: List[Dict[str, Any]] = []
    for field in fields:
        present = sum(field in row for row in rows)
        records.append(
            {
                "field": field,
                "item_type": "shortcut_or_structural_metadata",
                "manifest_rows_present": present,
                "allowed_as_model_input": 0,
                "allowed_as_loss_or_gate_input": 0,
                "allowed_as_audit_field": 1,
                "implementation_rule": "retain only in batch shortcuts/export tables; never tensorize for C16-MEA",
                "verification": "absence from mechanism_evidence_alignment forward signature and classifier tensors",
            }
        )
    for field, rule in [
        ("image_mask", "mask padded image tokens before evidence pooling"),
        ("report_attention_mask", "mask padded text tokens and empty dictionary masks"),
        ("bio_missing_mask", "mask unavailable bio values per field; do not sum or encode the count"),
    ]:
        records.append(
            {
                "field": field,
                "item_type": "validity_mask",
                "manifest_rows_present": len(rows) if field == "bio_missing_mask" else np.nan,
                "allowed_as_model_input": 1,
                "allowed_as_loss_or_gate_input": 1,
                "allowed_as_audit_field": 1,
                "implementation_rule": rule,
                "verification": "mask effects only; no learned scalar derived from valid/missing counts",
            }
        )
    return pd.DataFrame(records)


def code_path_summary(repo_root: Path) -> Dict[str, Any]:
    models_text = (repo_root / "dmea_ht/models.py").read_text(encoding="utf-8")
    data_text = (repo_root / "dmea_ht/data.py").read_text(encoding="utf-8")
    forbidden_terms = ["DSSAAlignment", "use_dssa", "shared_specific", "shared-specific"]
    return {
        "image_tokens_available": "return tokens, global_token" in models_text,
        "text_tokens_available": "return tokens, global_token" in models_text,
        "bio_tokens_available": "bio_tokens = self.token_proj" in models_text,
        "patient_anchor_available": "class PatientAnchorFusion" in models_text,
        "classifier_contributions_available": all(term in models_text for term in ("e_img", "e_text", "e_bio", "e_synergy")),
        "current_generic_evidence_roles": ["morphology", "immune", "function", "negative", "uncertain", "temporal"],
        "current_role_alignment_loss_is_zero": "role_loss = evidence_tokens.new_tensor(0.0)" in models_text,
        "text_tokenization_is_character_position_preserving": "tokens = list(text.strip())" in data_text,
        "forbidden_dssa_symbols_present": [term for term in forbidden_terms if term in models_text or term in data_text],
        "paths": [
            "image files -> ImageEncoder -> per-image tokens + image global",
            "C13 report text -> stable character IDs -> text tokens + text global",
            "seven bio values -> BioEncoder -> per-field tokens + bio global",
            "all tokens -> existing generic EvidenceRoleAlignment + PatientAnchorFusion",
            "image/text/bio globals + patient anchor + negative token -> EvidenceConservationClassifier",
        ],
        "c16_mea_boundary": "operate on evidence tokens between unchanged encoders and final patient classification; do not align raw modality globals",
    }


def split_label_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter[int]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        counts.setdefault(split, Counter())[int(float(row.get("label", 0)))] += 1
    return {split: {str(label): int(value) for label, value in sorted(counter.items())} for split, counter in counts.items()}


def write_outputs(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("C13 manifest is empty")
    source_path = Path(args.source_table) if args.source_table else None
    if source_path is None:
        source_table = None
    elif source_path.suffix.lower() == ".csv":
        source_table = pd.read_csv(source_path)
    else:
        source_table = pd.read_excel(source_path)
    visible_chars = max(int(args.text_max_length) - 2, 1)

    input_fields = audit_input_fields(rows)
    bio_fields, bio_grouping, bio_summary = audit_bio_fields(rows, source_table)
    text_terms = audit_text_terms(rows, visible_chars)
    temporal_fields, temporal_summary = audit_temporal_fields(rows, visible_chars)
    c14_mapping, c14_summary = audit_c14_mapping(repo_root)
    shortcut_map = audit_shortcuts(rows)
    code_summary = code_path_summary(repo_root)

    input_fields.to_csv(output_dir / "c16_mea_available_input_fields.csv", index=False)
    bio_fields.to_csv(output_dir / "c16_mea_available_bio_fields.csv", index=False)
    bio_grouping.to_csv(output_dir / "c16_mea_bio_semantic_grouping.csv", index=False)
    text_terms.to_csv(output_dir / "c16_mea_existing_text_evidence_terms.csv", index=False)
    temporal_fields.to_csv(output_dir / "c16_mea_existing_temporal_fields.csv", index=False)
    c14_mapping.to_csv(output_dir / "c16_mea_c14_diagnostic_mapping.csv", index=False)
    shortcut_map.to_csv(output_dir / "c16_mea_shortcut_exclusion_map.csv", index=False)

    required_present = {
        field: any(field in row and not is_empty(row.get(field)) for row in rows) for field in REQUIRED_MODEL_FIELDS
    }
    checks = [
        {"check": "required_manifest_fields", "pass": all(required_present.values()), "detail": required_present},
        {
            "check": "bio_vector_order_and_length",
            "pass": bio_summary["bio_vector_length_counts"] == {str(len(BIO_COLUMNS)): len(rows)},
            "detail": bio_summary["bio_vector_length_counts"],
        },
        {
            "check": "real_bio_source_fields_present",
            "pass": bio_summary["all_bio_source_fields_present"],
            "detail": BIO_COLUMNS,
        },
        {
            "check": "temporal_markers_available",
            "pass": temporal_summary.get("latest_focus_marker_count", 0) > 0
            and temporal_summary.get("full_report_marker_count", 0) > 0,
            "detail": temporal_summary,
        },
        {"check": "text_dictionaries_available", "pass": not text_terms.empty, "detail": len(text_terms)},
        {
            "check": "c14_diagnostics_available",
            "pass": c14_summary["available_count"] == c14_summary["required_count"],
            "detail": c14_summary,
        },
        {
            "check": "shortcut_map_complete",
            "pass": set(SHORTCUT_FIELDS).issubset(set(shortcut_map["field"].astype(str))),
            "detail": sorted(SHORTCUT_FIELDS),
        },
        {
            "check": "no_mistaken_dssa_symbols",
            "pass": not code_summary["forbidden_dssa_symbols_present"],
            "detail": code_summary["forbidden_dssa_symbols_present"],
        },
    ]
    summary = {
        "manifest": str(Path(args.manifest)),
        "source_table": str(Path(args.source_table)) if args.source_table else "",
        "manifest_rows": len(rows),
        "unique_patients": len({str(row.get("patient_id")) for row in rows}),
        "split_label_counts": split_label_counts(rows),
        "text_max_length": int(args.text_max_length),
        "model_visible_report_chars": visible_chars,
        "required_manifest_fields": required_present,
        "bio": bio_summary,
        "temporal": temporal_summary,
        "c14": c14_summary,
        "code_path": code_summary,
        "checks": checks,
        "audit_pass": all(bool(check["pass"]) for check in checks),
    }
    (output_dir / "c16_mea_design_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = write_outputs(args)
    print(
        json.dumps(
            {
                "audit_pass": summary["audit_pass"],
                "manifest_rows": summary["manifest_rows"],
                "output_dir": str(Path(args.output_dir)),
                "checks": {check["check"]: check["pass"] for check in summary["checks"]},
            },
            ensure_ascii=False,
        )
    )
    if not summary["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
