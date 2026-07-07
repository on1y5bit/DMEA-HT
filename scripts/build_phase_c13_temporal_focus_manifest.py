from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest
from scripts.analyze_phase_c11_report_filter_hypotheses import (
    BENIGN_NODULE_CUES,
    DIFFUSE_HT_CUES,
    MORPHOLOGY_CUES,
    NEGATIVE_THYROID_CUES,
    THYROID_CUES,
    any_term,
    parse_visits,
    split_clauses,
)
from scripts.build_evidence_weak_labels import DEFAULT_BIO_COLUMNS, add_evidence_labels


TEXT_LABEL_FIELDS = [
    "txt_morphology_label",
    "txt_negative_label",
    "txt_uncertain_label",
    "txt_diag_hint_label",
    "image_morphology_weak_label",
    "discordance_state_label",
]


def report_text(row: Dict[str, Any]) -> str:
    for key in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


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


def unique_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def is_thyroid_clause(clause: str) -> bool:
    return any_term(clause, THYROID_CUES)


def clause_priority(clause: str) -> int:
    if any_term(clause, DIFFUSE_HT_CUES):
        return 0
    if any_term(clause, MORPHOLOGY_CUES):
        return 1
    if any_term(clause, NEGATIVE_THYROID_CUES):
        return 2
    if any_term(clause, BENIGN_NODULE_CUES):
        return 3
    return 4


def thyroid_focus_clauses(block: str) -> List[str]:
    clauses = [clause for clause in split_clauses(block) if is_thyroid_clause(clause)]
    clauses = unique_in_order(clauses)
    return sorted(clauses, key=clause_priority)


def truncate_join(parts: Sequence[str], max_chars: int) -> str:
    out: List[str] = []
    used = 0
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        sep = "\n" if out else ""
        remaining = max_chars - used - len(sep)
        if remaining <= 0:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        out.append(piece)
        used += len(sep) + len(piece)
        if used >= max_chars:
            break
    return "\n".join(out)


def build_focus_prefix(text: str, max_prefix_chars: int) -> Dict[str, Any]:
    visits = parse_visits(text)
    if not visits and text.strip():
        visits = [("unknown", text)]
    if not visits:
        return {
            "prefix": "",
            "n_visits_parsed": 0,
            "n_latest_focus_clauses": 0,
            "n_history_focus_clauses": 0,
            "latest_date": "",
        }
    latest_date, latest_block = visits[-1]
    latest_clauses = thyroid_focus_clauses(latest_block)
    history_clauses: List[str] = []
    for date, block in visits[:-1]:
        for clause in thyroid_focus_clauses(block):
            history_clauses.append(f"{date} {clause}" if date != "unknown" else clause)
    history_clauses = unique_in_order(history_clauses)
    parts: List[str] = []
    if latest_clauses:
        parts.append(f"[C13_LATEST_THYROID {latest_date}] " + " ".join(latest_clauses))
    if history_clauses:
        parts.append("[C13_HISTORY_THYROID] " + " ".join(history_clauses))
    prefix = truncate_join(parts, max_prefix_chars)
    return {
        "prefix": prefix,
        "n_visits_parsed": len(visits),
        "n_latest_focus_clauses": len(latest_clauses),
        "n_history_focus_clauses": len(history_clauses),
        "latest_date": latest_date,
    }


def term_counts(text: str) -> Dict[str, int]:
    return {
        "morphology": sum(1 for term in MORPHOLOGY_CUES if term and term in text),
        "diffuse": sum(1 for term in DIFFUSE_HT_CUES if term and term in text),
        "negative": sum(1 for term in NEGATIVE_THYROID_CUES if term and term in text),
        "benign": sum(1 for term in BENIGN_NODULE_CUES if term and term in text),
    }


def label_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("split", ""))][int(float(row.get("label", 0)))] += 1
    return {split: {str(label): int(count) for label, count in by_label.items()} for split, by_label in counts.items()}


def validate_invariance(input_rows: Sequence[Dict[str, Any]], output_rows: Sequence[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    if len(input_rows) != len(output_rows):
        issues.append(f"row_count_changed:{len(input_rows)}->{len(output_rows)}")
        return issues
    for idx, (before, after) in enumerate(zip(input_rows, output_rows)):
        for field in ("patient_id", "label", "split"):
            if str(before.get(field)) != str(after.get(field)):
                issues.append(f"{field}_changed_at_row_{idx}:{before.get(field)}->{after.get(field)}")
        for field in ("image_paths", "images", "image_path", "bio_values", "bio_missing_mask"):
            if field in before and json.dumps(before.get(field), ensure_ascii=False, sort_keys=True, default=str) != json.dumps(
                after.get(field), ensure_ascii=False, sort_keys=True, default=str
            ):
                issues.append(f"{field}_changed_at_row_{idx}:{before.get('patient_id')}")
    return issues


def build_rows(
    rows: Sequence[Dict[str, Any]],
    max_prefix_chars: int,
    bio_columns: Sequence[str],
    trust_abnormal_flags: bool,
    negation_window: int,
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    out_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for row in rows:
        original = report_text(row)
        focus = build_focus_prefix(original, max_prefix_chars=max_prefix_chars)
        prefix = focus["prefix"]
        focused_text = f"{prefix}\n[C13_FULL_REPORT]\n{original}" if prefix else original
        before_labels = {field: row.get(field) for field in TEXT_LABEL_FIELDS}
        before_first = term_counts(original[:256])
        after_first = term_counts(focused_text[:256])
        out = dict(row)
        out["report_text"] = focused_text
        out["report_length"] = len(focused_text)
        out["phase_c13_temporal_focus"] = 1
        out["phase_c13_focus_prefix_chars"] = len(prefix)
        out["phase_c13_focus_latest_date"] = focus["latest_date"]
        out["phase_c13_n_latest_focus_clauses"] = focus["n_latest_focus_clauses"]
        out["phase_c13_n_history_focus_clauses"] = focus["n_history_focus_clauses"]
        out["phase_c13_n_visits_parsed"] = focus["n_visits_parsed"]
        out["phase_c13_focus_source"] = {
            "phase": "C13",
            "uses_labels": False,
            "uses_predictions": False,
            "uses_test_selection": False,
            "max_prefix_chars": max_prefix_chars,
        }
        out = add_evidence_labels(
            out,
            bio_columns=bio_columns,
            trust_abnormal_flags=trust_abnormal_flags,
            negation_window=negation_window,
        )
        out_rows.append(out)
        audit_row: Dict[str, Any] = {
            "patient_id": str(row.get("patient_id")),
            "split": str(row.get("split", "")),
            "label": int(float(row.get("label", 0))),
            "original_report_length": len(original),
            "focused_report_length": len(focused_text),
            "focus_prefix_chars": len(prefix),
            "n_visits_parsed": focus["n_visits_parsed"],
            "n_latest_focus_clauses": focus["n_latest_focus_clauses"],
            "n_history_focus_clauses": focus["n_history_focus_clauses"],
            "first256_morphology_before": before_first["morphology"],
            "first256_morphology_after": after_first["morphology"],
            "first256_diffuse_before": before_first["diffuse"],
            "first256_diffuse_after": after_first["diffuse"],
            "first256_negative_before": before_first["negative"],
            "first256_negative_after": after_first["negative"],
            "first256_benign_before": before_first["benign"],
            "first256_benign_after": after_first["benign"],
        }
        for field in TEXT_LABEL_FIELDS:
            audit_row[f"before_{field}"] = before_labels.get(field)
            audit_row[f"after_{field}"] = out.get(field)
            audit_row[f"changed_{field}"] = int(str(before_labels.get(field)) != str(out.get(field)))
        audit_rows.append(audit_row)
    return out_rows, pd.DataFrame(audit_rows)


def summarize_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_split_label = (
        audit.groupby(["split", "label"], as_index=False)
        .agg(
            n=("patient_id", "count"),
            mean_prefix_chars=("focus_prefix_chars", "mean"),
            n_with_prefix=("focus_prefix_chars", lambda values: int((values > 0).sum())),
            mean_first256_morphology_before=("first256_morphology_before", "mean"),
            mean_first256_morphology_after=("first256_morphology_after", "mean"),
            mean_first256_diffuse_before=("first256_diffuse_before", "mean"),
            mean_first256_diffuse_after=("first256_diffuse_after", "mean"),
            txt_morphology_changed_rate=("changed_txt_morphology_label", "mean"),
            image_weak_changed_rate=("changed_image_morphology_weak_label", "mean"),
        )
        .sort_values(["split", "label"])
    )
    val_positive = audit[(audit["split"] == "val") & (audit["label"] == 1)].copy()
    positive_focus = pd.DataFrame(
        [
            {
                "split": "val",
                "label": 1,
                "n_positive": len(val_positive),
                "n_with_prefix": int((val_positive["focus_prefix_chars"] > 0).sum()) if len(val_positive) else 0,
                "mean_first256_morphology_before": val_positive["first256_morphology_before"].mean()
                if len(val_positive)
                else 0.0,
                "mean_first256_morphology_after": val_positive["first256_morphology_after"].mean()
                if len(val_positive)
                else 0.0,
                "mean_first256_diffuse_before": val_positive["first256_diffuse_before"].mean()
                if len(val_positive)
                else 0.0,
                "mean_first256_diffuse_after": val_positive["first256_diffuse_after"].mean()
                if len(val_positive)
                else 0.0,
                "n_txt_morphology_changed": int(val_positive["changed_txt_morphology_label"].sum())
                if len(val_positive)
                else 0,
                "n_image_weak_changed": int(val_positive["changed_image_morphology_weak_label"].sum())
                if len(val_positive)
                else 0,
            }
        ]
    )
    return by_split_label, positive_focus


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    out_dir: Path,
    input_rows: Sequence[Dict[str, Any]],
    output_rows: Sequence[Dict[str, Any]],
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    positive_focus: pd.DataFrame,
    invariance_issues: Sequence[str],
    max_prefix_chars: int,
) -> str:
    if invariance_issues:
        recommendation = "DO_NOT_TRAIN_INVARIANCE_FAILED"
    elif float(positive_focus.iloc[0].get("mean_first256_morphology_after", 0.0)) > float(
        positive_focus.iloc[0].get("mean_first256_morphology_before", 0.0)
    ):
        recommendation = "ALLOW_C13_SINGLE_SEED_TEMPORAL_FOCUS_PILOT"
    else:
        recommendation = "AUDIT_MORE_BEFORE_C13_TRAINING"
    top_changed = audit.sort_values(["focus_prefix_chars", "first256_morphology_after"], ascending=[False, False]).head(15)
    lines = [
        "# Phase C13 Temporal-Focus Manifest Audit",
        "",
        "C13 is a data-construction pilot that moves thyroid-relevant latest and historical clauses before the full report text.",
        "",
        "## Input And Output",
        "",
        f"- Input rows: {len(input_rows)}.",
        f"- Output rows: {len(output_rows)}.",
        f"- Max focus prefix chars: {max_prefix_chars}.",
        f"- Invariance issues: {len(invariance_issues)}.",
        "",
        "## Split/Label Counts",
        "",
        f"- Input: `{json.dumps(label_counts(input_rows), ensure_ascii=False, sort_keys=True)}`.",
        f"- Output: `{json.dumps(label_counts(output_rows), ensure_ascii=False, sort_keys=True)}`.",
        "",
        "## First-256 Evidence Exposure Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Validation Positive Focus Check",
        "",
        frame_to_markdown(positive_focus),
        "",
        "## Highest Prefix Patients",
        "",
        frame_to_markdown(
            top_changed[
                [
                    "patient_id",
                    "split",
                    "label",
                    "focus_prefix_chars",
                    "n_latest_focus_clauses",
                    "n_history_focus_clauses",
                    "first256_morphology_before",
                    "first256_morphology_after",
                    "first256_diffuse_before",
                    "first256_diffuse_after",
                    "changed_txt_morphology_label",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- The C13 pilot uses report text only, not labels, predictions, or test-selected information.",
        "- Patient IDs, labels, splits, image paths, and bio values must remain invariant.",
        "- The intended mechanism is to reduce truncation loss from long reports under `text_max_length=256`.",
        "- Shortcut and audit fields remain outside the classifier.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
    ]
    if invariance_issues:
        lines.extend(["", "## Invariance Issues", "", "\n".join(f"- {issue}" for issue in invariance_issues[:50])])
    (out_dir / "phase_c13_temporal_focus_manifest_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase C13 temporal-focus report manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--max-prefix-chars", type=int, default=220)
    parser.add_argument("--negation-window", type=int, default=10)
    parser.add_argument("--bio-columns", default=",".join(DEFAULT_BIO_COLUMNS))
    parser.add_argument("--trust-bio-abnormal-flags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    out_dir = Path(args.audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bio_columns = [part.strip() for part in str(args.bio_columns).split(",") if part.strip()]
    rows = read_manifest(input_path)
    output_rows, audit = build_rows(
        rows,
        max_prefix_chars=int(args.max_prefix_chars),
        bio_columns=bio_columns,
        trust_abnormal_flags=bool(args.trust_bio_abnormal_flags),
        negation_window=int(args.negation_window),
    )
    invariance_issues = validate_invariance(rows, output_rows)
    summary, positive_focus = summarize_audit(audit)
    write_jsonl(output_rows, output_path)
    audit.to_csv(out_dir / "c13_temporal_focus_patient_audit.csv", index=False)
    summary.to_csv(out_dir / "c13_temporal_focus_split_label_summary.csv", index=False)
    positive_focus.to_csv(out_dir / "c13_temporal_focus_positive_focus_val.csv", index=False)
    recommendation = write_report(
        out_dir,
        input_rows=rows,
        output_rows=output_rows,
        audit=audit,
        summary=summary,
        positive_focus=positive_focus,
        invariance_issues=invariance_issues,
        max_prefix_chars=int(args.max_prefix_chars),
    )
    print(json.dumps({"output": str(output_path), "rows": len(output_rows), "recommendation": recommendation}))


if __name__ == "__main__":
    main()
