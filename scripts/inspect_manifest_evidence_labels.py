from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


EVIDENCE_LABEL_FIELDS = [
    "txt_morphology_label",
    "txt_negative_label",
    "txt_uncertain_label",
    "txt_diag_hint_label",
    "bio_immune_abnormal_label",
    "bio_function_abnormal_label",
    "bio_missing_label",
    "image_morphology_weak_label",
    "discordance_state_label",
]
CONFIDENCE_FIELDS = [
    "txt_morphology_confidence",
    "txt_negative_confidence",
    "txt_uncertain_confidence",
    "txt_diag_hint_confidence",
    "image_morphology_weak_confidence",
]
TERM_FIELDS = [
    "matched_morphology_terms",
    "matched_negative_terms",
    "matched_uncertain_terms",
    "matched_diag_hint_terms",
]
DISCORDANCE_NAMES = {
    0: "consistent_negative",
    1: "consistent_positive",
    2: "morphology_positive_bio_negative",
    3: "bio_positive_morphology_negative",
    4: "morphology_positive_bio_missing",
    5: "uncertain_or_insufficient",
}


def normalize_value(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def value_name(field: str, value: Any) -> str:
    normalized = normalize_value(value)
    if field == "discordance_state_label":
        return DISCORDANCE_NAMES.get(normalized, str(normalized))
    return str(normalized)


def count_values(rows: Iterable[Dict[str, Any]], field: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[normalize_value(row.get(field, "missing"))] += 1
    return counts


def format_counts(counts: Counter, total: int, field: str) -> str:
    parts: List[str] = []
    for key in sorted(counts, key=lambda item: str(item)):
        pct = counts[key] / total * 100.0 if total else 0.0
        parts.append(f"{value_name(field, key)}: {counts[key]} ({pct:.1f}%)")
    return "; ".join(parts)


def split_groups(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("split", "missing")).lower()].append(row)
    return [(key, groups[key]) for key in ("train", "val", "test") if key in groups]


def label_groups(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("label", "missing"))].append(row)
    return [(key, groups[key]) for key in sorted(groups)]


def joint_key(row: Dict[str, Any]) -> str:
    morph = int(float(row.get("txt_morphology_label", 0)))
    neg = int(float(row.get("txt_negative_label", 0)))
    return f"morph{morph}_neg{neg}"


def print_distribution_section(title: str, groups: Sequence[Tuple[str, Sequence[Dict[str, Any]]]], fields: Sequence[str]) -> None:
    print(f"\n## {title}")
    for group_name, group in groups:
        print(f"\n[{group_name}] n={len(group)}")
        for field in fields:
            counts = count_values(group, field)
            print(f"{field}: {format_counts(counts, len(group), field)}")


def print_joint_counts(rows: Sequence[Dict[str, Any]]) -> None:
    print("\n## Morphology/negative joint table")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[joint_key(row)].append(row)
    for key in ("morph1_neg0", "morph0_neg1", "morph1_neg1", "morph0_neg0"):
        print(f"{key}: {len(groups[key])}")
    print("\nSplit-wise joint counts")
    for split, split_rows in split_groups(rows):
        counts = Counter(joint_key(row) for row in split_rows)
        print(f"{split}: " + "; ".join(f"{key}: {counts[key]}" for key in ("morph1_neg0", "morph0_neg1", "morph1_neg1", "morph0_neg0")))
    print("\nLabel-wise joint counts")
    for label, label_rows in label_groups(rows):
        counts = Counter(joint_key(row) for row in label_rows)
        print(f"label={label}: " + "; ".join(f"{key}: {counts[key]}" for key in ("morph1_neg0", "morph0_neg1", "morph1_neg1", "morph0_neg0")))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_confidence_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("\n## Confidence summary")
    groups: List[Tuple[str, Sequence[Dict[str, Any]]]] = [("all", rows)]
    groups.extend((f"split={name}", group) for name, group in split_groups(rows))
    groups.extend((f"label={name}", group) for name, group in label_groups(rows))
    for name, group in groups:
        parts = []
        for field in CONFIDENCE_FIELDS:
            values = [safe_float(row.get(field, 0.0)) for row in group]
            parts.append(f"{field}: {mean(values):.4f}" if values else f"{field}: 0.0000")
        print(f"{name}: " + "; ".join(parts))


def term_counter(rows: Sequence[Dict[str, Any]], field: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for term in parse_maybe_list(row.get(field)):
            counter[str(term)] += 1
    return counter


def print_top_terms(rows: Sequence[Dict[str, Any]], field: str, title: str, top_k: int) -> None:
    print(f"\n## {title}")
    counter = term_counter(rows, field)
    if not counter:
        print("none")
        return
    for term, count in counter.most_common(top_k):
        print(f"{term}: {count}")


def print_unknown_summary(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    print("\n## Unknown-label summary")
    for field in fields:
        count = sum(1 for row in rows if normalize_value(row.get(field)) == -1)
        pct = count / len(rows) * 100.0 if rows else 0.0
        print(f"{field}: {count} ({pct:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DMEA-HT v2 evidence weak-label distributions.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON/JSONL/CSV path.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top matched terms to print.")
    parser.add_argument(
        "--fields",
        default=",".join(EVIDENCE_LABEL_FIELDS),
        help="Comma-separated label fields to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    fields = [part.strip() for part in str(args.fields).split(",") if part.strip()]
    print(f"manifest: {args.manifest}")
    print(f"rows: {len(rows)}")
    missing_fields = [field for field in fields + CONFIDENCE_FIELDS + TERM_FIELDS if any(field not in row for row in rows)]
    if missing_fields:
        print(f"missing_fields: {sorted(set(missing_fields))}")

    print_distribution_section("Overall label distribution", [("all", rows)], fields)
    print_distribution_section("Split-wise label distribution", [(f"split={name}", group) for name, group in split_groups(rows)], fields)
    print_distribution_section("Label-by-class distribution", [(f"label={name}", group) for name, group in label_groups(rows)], fields)
    print_joint_counts(rows)
    print_confidence_summary(rows)
    print_top_terms(rows, "matched_morphology_terms", "Top matched morphology terms", int(args.top_k))
    print_top_terms(rows, "matched_negative_terms", "Top matched negative terms", int(args.top_k))
    print_unknown_summary(rows, fields)


if __name__ == "__main__":
    main()
