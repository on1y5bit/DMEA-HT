from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


AUDIT_FIELDS = [
    "patient_id",
    "split",
    "label",
    "report_text",
    "report_text_truncated",
    "txt_morphology_label",
    "txt_negative_label",
    "txt_uncertain_label",
    "txt_diag_hint_label",
    "image_morphology_weak_label",
    "bio_immune_abnormal_label",
    "bio_function_abnormal_label",
    "bio_missing_label",
    "discordance_state_label",
    "matched_morphology_terms",
    "matched_negative_terms",
    "matched_uncertain_terms",
    "matched_diag_hint_terms",
    "txt_morphology_confidence",
    "txt_negative_confidence",
    "txt_uncertain_confidence",
    "txt_diag_hint_confidence",
    "image_morphology_weak_confidence",
    "audit_group",
]
GROUP_ORDER = ["morph1_neg0", "morph0_neg1", "morph1_neg1", "morph0_neg0"]


def int_label(row: Dict[str, Any], field: str) -> int:
    try:
        return int(float(row.get(field, 0)))
    except (TypeError, ValueError):
        return 0


def audit_group(row: Dict[str, Any]) -> str:
    return f"morph{int_label(row, 'txt_morphology_label')}_neg{int_label(row, 'txt_negative_label')}"


def term_string(value: Any) -> str:
    terms = [str(term) for term in parse_maybe_list(value)]
    return "|".join(terms)


def report_text(row: Dict[str, Any]) -> str:
    return str(row.get("report_text") or row.get("text") or row.get("report") or "")


def export_row(row: Dict[str, Any], max_report_chars: int, include_text: bool) -> Dict[str, Any]:
    text = report_text(row)
    out: Dict[str, Any] = {}
    for field in AUDIT_FIELDS:
        if field == "report_text":
            out[field] = text if include_text else ""
        elif field == "report_text_truncated":
            out[field] = text[:max_report_chars]
        elif field == "audit_group":
            out[field] = audit_group(row)
        elif field.startswith("matched_"):
            out[field] = term_string(row.get(field, []))
        else:
            out[field] = row.get(field, "")
    return out


def sample_rows(rows: Sequence[Dict[str, Any]], samples_per_group: int, seed: int, per_split: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prefix = str(row.get("split", "missing")).lower() if per_split else "all"
        buckets[f"{prefix}:{audit_group(row)}"].append(row)

    sampled: List[Dict[str, Any]] = []
    prefixes = sorted({key.split(":", 1)[0] for key in buckets})
    for prefix in prefixes:
        for group in GROUP_ORDER:
            candidates = list(buckets.get(f"{prefix}:{group}", []))
            rng.shuffle(candidates)
            sampled.extend(candidates[:samples_per_group])
    return sampled


def print_summary(rows: Sequence[Dict[str, Any]], exported: Sequence[Dict[str, Any]], output: str) -> None:
    print(f"Loaded rows: {len(rows)}")
    print("Group counts:")
    counts = Counter(audit_group(row) for row in rows)
    for group in GROUP_ORDER:
        print(f"  {group}: {counts[group]}")
    print("Split-wise group counts:")
    split_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        split_counts[str(row.get("split", "missing")).lower()][audit_group(row)] += 1
    for split in sorted(split_counts):
        pieces = [f"{group}: {split_counts[split][group]}" for group in GROUP_ORDER]
        print(f"  {split}: " + "; ".join(pieces))
    print(f"Rows exported: {len(exported)}")
    print(f"Output CSV: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export samples for manual evidence weak-label quality audit.")
    parser.add_argument("--manifest", required=True, help="Evidence manifest JSON/JSONL/CSV path.")
    parser.add_argument("--output", required=True, help="Audit CSV output path.")
    parser.add_argument("--samples-per-group", type=int, default=30, help="Maximum samples per audit group.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--per-split", action="store_true", help="Sample each audit group separately inside each split.")
    parser.add_argument("--include-text", dest="include_text", action="store_true", default=True, help="Include full report_text column.")
    parser.add_argument("--no-include-text", dest="include_text", action="store_false", help="Do not include full report_text column.")
    parser.add_argument("--max-report-chars", type=int, default=1000, help="Characters kept in report_text_truncated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    sampled = sample_rows(rows, int(args.samples_per_group), int(args.seed), bool(args.per_split))
    exported = [export_row(row, int(args.max_report_chars), bool(args.include_text)) for row in sampled]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exported)
    print_summary(rows, exported, str(output))


if __name__ == "__main__":
    main()
