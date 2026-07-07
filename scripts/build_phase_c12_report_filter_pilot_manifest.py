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
    thyroid_text,
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
    lines = [
        "| " + " | ".join(str(col) for col in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
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


def has_latest_diffuse(visits: Sequence[Tuple[str, str]]) -> bool:
    if not visits:
        return False
    latest_thyroid = thyroid_text(visits[-1][1])
    return any_term(latest_thyroid, DIFFUSE_HT_CUES)


def is_thyroid_clause(clause: str) -> bool:
    return any_term(clause, THYROID_CUES)


def should_drop_clause(clause: str, latest_diffuse: bool, mode: str) -> tuple[bool, str]:
    thyroid = is_thyroid_clause(clause)
    if not thyroid:
        return False, ""
    morph = any_term(clause, MORPHOLOGY_CUES)
    diffuse = any_term(clause, DIFFUSE_HT_CUES)
    benign = any_term(clause, BENIGN_NODULE_CUES)
    negative = any_term(clause, NEGATIVE_THYROID_CUES)
    if not morph or diffuse or negative or latest_diffuse:
        return False, ""
    if mode == "benign_only" and benign:
        return True, "benign_nodule_without_latest_diffuse"
    if mode == "latest_diffuse_gate":
        return True, "morphology_without_latest_diffuse"
    if mode == "combined_low_risk" and (benign or morph):
        return True, "combined_low_risk_without_latest_diffuse"
    return False, ""


def filter_visit_block(block: str, latest_diffuse: bool, mode: str) -> tuple[str, Counter[str], int, int]:
    kept: List[str] = []
    reasons: Counter[str] = Counter()
    n_thyroid_morphology = 0
    n_dropped = 0
    for clause in split_clauses(block):
        if is_thyroid_clause(clause) and any_term(clause, MORPHOLOGY_CUES):
            n_thyroid_morphology += 1
        drop, reason = should_drop_clause(clause, latest_diffuse=latest_diffuse, mode=mode)
        if drop:
            reasons[reason] += 1
            n_dropped += 1
            continue
        kept.append(clause)
    return " ".join(kept), reasons, n_thyroid_morphology, n_dropped


def filter_report(text: str, mode: str) -> Dict[str, Any]:
    visits = parse_visits(text)
    if not visits and text.strip():
        visits = [("unknown", text)]
    latest_diffuse = has_latest_diffuse(visits)
    parts: List[str] = []
    reason_counts: Counter[str] = Counter()
    n_morphology_clauses = 0
    n_dropped_clauses = 0
    for date, block in visits:
        filtered_block, reasons, n_morph, n_dropped = filter_visit_block(block, latest_diffuse, mode)
        reason_counts.update(reasons)
        n_morphology_clauses += n_morph
        n_dropped_clauses += n_dropped
        if filtered_block.strip():
            if date == "unknown":
                parts.append(filtered_block.strip())
            else:
                parts.append(f"[{date}] {filtered_block.strip()}")
    filtered_text = "\n".join(parts)
    return {
        "filtered_text": filtered_text,
        "latest_diffuse_ht_like": int(latest_diffuse),
        "n_visits_parsed_for_filter": len(visits),
        "n_thyroid_morphology_clauses": n_morphology_clauses,
        "n_dropped_clauses": n_dropped_clauses,
        "drop_reasons": dict(reason_counts),
    }


def label_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("split", ""))][int(float(row.get("label", 0)))] += 1
    return {split: {str(label): int(count) for label, count in by_label.items()} for split, by_label in counts.items()}


def build_rows(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    bio_columns: Sequence[str],
    trust_abnormal_flags: bool,
    negation_window: int,
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    out_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for row in rows:
        original = report_text(row)
        filtered = filter_report(original, mode)
        pilot_row = dict(row)
        before_labels = {field: pilot_row.get(field) for field in TEXT_LABEL_FIELDS}
        pilot_row["report_text"] = filtered["filtered_text"]
        pilot_row["report_length"] = len(filtered["filtered_text"])
        pilot_row["phase_c12_report_filter"] = 1
        pilot_row["phase_c12_report_filter_mode"] = mode
        pilot_row["phase_c12_original_report_length"] = len(original)
        pilot_row["phase_c12_filtered_report_length"] = len(filtered["filtered_text"])
        pilot_row["phase_c12_report_length_delta"] = len(filtered["filtered_text"]) - len(original)
        pilot_row["phase_c12_latest_diffuse_ht_like"] = filtered["latest_diffuse_ht_like"]
        pilot_row["phase_c12_n_dropped_clauses"] = filtered["n_dropped_clauses"]
        pilot_row["phase_c12_drop_reasons"] = filtered["drop_reasons"]
        pilot_row["phase_c12_n_thyroid_morphology_clauses"] = filtered["n_thyroid_morphology_clauses"]
        pilot_row["phase_c12_filter_source"] = {
            "phase": "C12",
            "mode": mode,
            "uses_labels": False,
            "uses_predictions": False,
            "uses_test_selection": False,
        }
        pilot_row = add_evidence_labels(
            pilot_row,
            bio_columns=bio_columns,
            trust_abnormal_flags=trust_abnormal_flags,
            negation_window=negation_window,
        )
        audit_row: Dict[str, Any] = {
            "patient_id": str(row.get("patient_id")),
            "split": str(row.get("split", "")),
            "label": int(float(row.get("label", 0))),
            "original_report_length": len(original),
            "filtered_report_length": len(filtered["filtered_text"]),
            "report_length_delta": len(filtered["filtered_text"]) - len(original),
            "n_dropped_clauses": filtered["n_dropped_clauses"],
            "latest_diffuse_ht_like": filtered["latest_diffuse_ht_like"],
            "n_thyroid_morphology_clauses": filtered["n_thyroid_morphology_clauses"],
            "drop_reasons": json.dumps(filtered["drop_reasons"], ensure_ascii=False, sort_keys=True),
        }
        for field in TEXT_LABEL_FIELDS:
            audit_row[f"before_{field}"] = before_labels.get(field)
            audit_row[f"after_{field}"] = pilot_row.get(field)
            audit_row[f"changed_{field}"] = int(str(before_labels.get(field)) != str(pilot_row.get(field)))
        out_rows.append(pilot_row)
        audit_rows.append(audit_row)

    return out_rows, pd.DataFrame(audit_rows)


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


def summarize_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_split_label = (
        audit.groupby(["split", "label"], as_index=False)
        .agg(
            n=("patient_id", "count"),
            n_filtered=("n_dropped_clauses", lambda values: int((values > 0).sum())),
            filtered_rate=("n_dropped_clauses", lambda values: float((values > 0).mean())),
            mean_original_report_length=("original_report_length", "mean"),
            mean_filtered_report_length=("filtered_report_length", "mean"),
            mean_report_length_delta=("report_length_delta", "mean"),
            median_report_length_delta=("report_length_delta", "median"),
            morphology_label_changed_rate=("changed_txt_morphology_label", "mean"),
            negative_label_changed_rate=("changed_txt_negative_label", "mean"),
            image_weak_label_changed_rate=("changed_image_morphology_weak_label", "mean"),
        )
        .sort_values(["split", "label"])
    )
    label_change_rows: List[Dict[str, Any]] = []
    for field in TEXT_LABEL_FIELDS:
        for split, split_frame in audit.groupby("split"):
            for label, label_frame in split_frame.groupby("label"):
                label_change_rows.append(
                    {
                        "field": field,
                        "split": split,
                        "label": int(label),
                        "n_changed": int(label_frame[f"changed_{field}"].sum()),
                        "changed_rate": float(label_frame[f"changed_{field}"].mean()),
                    }
                )
    label_changes = pd.DataFrame(label_change_rows)
    val_positive = audit[(audit["split"] == "val") & (audit["label"] == 1)].copy()
    positive_risk = pd.DataFrame(
        [
            {
                "split": "val",
                "label": 1,
                "n_positive": len(val_positive),
                "n_filtered": int((val_positive["n_dropped_clauses"] > 0).sum()),
                "filtered_positive_rate": float((val_positive["n_dropped_clauses"] > 0).mean())
                if len(val_positive)
                else 0.0,
                "n_txt_morphology_changed": int(val_positive["changed_txt_morphology_label"].sum())
                if len(val_positive)
                else 0,
                "txt_morphology_changed_rate": float(val_positive["changed_txt_morphology_label"].mean())
                if len(val_positive)
                else 0.0,
                "n_image_weak_changed": int(val_positive["changed_image_morphology_weak_label"].sum())
                if len(val_positive)
                else 0,
                "image_weak_changed_rate": float(val_positive["changed_image_morphology_weak_label"].mean())
                if len(val_positive)
                else 0.0,
            }
        ]
    )
    return by_split_label, label_changes, positive_risk


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
    by_split_label: pd.DataFrame,
    label_changes: pd.DataFrame,
    positive_risk: pd.DataFrame,
    invariance_issues: Sequence[str],
    mode: str,
) -> str:
    val_pos_risk = positive_risk.iloc[0].to_dict() if not positive_risk.empty else {}
    if invariance_issues:
        recommendation = "DO_NOT_TRAIN_INVARIANCE_FAILED"
    elif float(val_pos_risk.get("txt_morphology_changed_rate", 1.0)) <= 0.15 and float(
        val_pos_risk.get("filtered_positive_rate", 1.0)
    ) <= 0.20:
        recommendation = "ALLOW_C12_SINGLE_SEED_TRAINING_PILOT"
    else:
        recommendation = "AUDIT_MORE_BEFORE_TRAINING"

    top_changed = audit.sort_values(["n_dropped_clauses", "report_length_delta"], ascending=[False, True]).head(15)
    lines = [
        "# Phase C12 Report-Construction Pilot Manifest Audit",
        "",
        "C12 builds a deterministic report-filter pilot manifest before any model or architecture change.",
        "",
        "## Input And Output",
        "",
        f"- Input rows: {len(input_rows)}.",
        f"- Output rows: {len(output_rows)}.",
        f"- Filter mode: `{mode}`.",
        f"- Invariance issues: {len(invariance_issues)}.",
        "",
        "## Split/Label Counts",
        "",
        f"- Input: `{json.dumps(label_counts(input_rows), ensure_ascii=False, sort_keys=True)}`.",
        f"- Output: `{json.dumps(label_counts(output_rows), ensure_ascii=False, sort_keys=True)}`.",
        "",
        "## Report Length And Label Change Summary",
        "",
        frame_to_markdown(by_split_label),
        "",
        "## Text Evidence Label Changes",
        "",
        frame_to_markdown(label_changes),
        "",
        "## Validation Positive Preservation Risk",
        "",
        frame_to_markdown(positive_risk),
        "",
        "## Most Changed Patients",
        "",
        frame_to_markdown(
            top_changed[
                [
                    "patient_id",
                    "split",
                    "label",
                    "original_report_length",
                    "filtered_report_length",
                    "report_length_delta",
                    "n_dropped_clauses",
                    "latest_diffuse_ht_like",
                    "changed_txt_morphology_label",
                    "changed_image_morphology_weak_label",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- Patient IDs, labels, splits, image paths, and bio values must remain invariant.",
        "- The report filter uses report text only, not labels, predictions, or test-selected information.",
        "- Shortcut and audit fields remain outside the classifier.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
    ]
    if invariance_issues:
        lines.extend(["", "## Invariance Issues", "", "\n".join(f"- {issue}" for issue in invariance_issues[:50])])
    (out_dir / "phase_c12_manifest_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase C12 report-filter pilot manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument(
        "--mode",
        default="combined_low_risk",
        choices=["benign_only", "latest_diffuse_gate", "combined_low_risk"],
    )
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
        mode=args.mode,
        bio_columns=bio_columns,
        trust_abnormal_flags=bool(args.trust_bio_abnormal_flags),
        negation_window=int(args.negation_window),
    )
    invariance_issues = validate_invariance(rows, output_rows)
    by_split_label, label_changes, positive_risk = summarize_audit(audit)
    write_jsonl(output_rows, output_path)
    audit.to_csv(out_dir / "c12_report_filter_patient_audit.csv", index=False)
    by_split_label.to_csv(out_dir / "c12_report_filter_split_label_summary.csv", index=False)
    label_changes.to_csv(out_dir / "c12_report_filter_label_change_summary.csv", index=False)
    positive_risk.to_csv(out_dir / "c12_report_filter_positive_preservation_val.csv", index=False)
    recommendation = write_report(
        out_dir,
        input_rows=rows,
        output_rows=output_rows,
        audit=audit,
        by_split_label=by_split_label,
        label_changes=label_changes,
        positive_risk=positive_risk,
        invariance_issues=invariance_issues,
        mode=args.mode,
    )
    print(json.dumps({"output": str(output_path), "rows": len(output_rows), "recommendation": recommendation}))


if __name__ == "__main__":
    main()
