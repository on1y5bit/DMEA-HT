from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


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


def count_values(rows: Iterable[Dict[str, Any]], field: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[normalize_value(row.get(field, "missing"))] += 1
    return counts


def format_counts(counts: Counter, total: int, field: str) -> str:
    parts: List[str] = []
    for key in sorted(counts, key=lambda item: str(item)):
        label = DISCORDANCE_NAMES.get(key, str(key)) if field == "discordance_state_label" else str(key)
        pct = counts[key] / total * 100.0 if total else 0.0
        parts.append(f"{label}: {counts[key]} ({pct:.1f}%)")
    return "; ".join(parts)


def group_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    groups["all"] = list(rows)
    for row in rows:
        split = str(row.get("split", "missing")).lower()
        label = str(row.get("label", "missing"))
        groups[f"split={split}"].append(row)
        groups[f"label={label}"].append(row)
        groups[f"split={split},label={label}"].append(row)
    ordered = ["all"]
    for split in ("train", "val", "test"):
        ordered.append(f"split={split}")
    for label in ("0", "1"):
        ordered.append(f"label={label}")
    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            ordered.append(f"split={split},label={label}")
    return [(key, groups[key]) for key in ordered if key in groups]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DMEA-HT v2 evidence weak-label distributions.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON/JSONL/CSV path.")
    parser.add_argument(
        "--fields",
        default=",".join(EVIDENCE_LABEL_FIELDS),
        help="Comma-separated fields to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    fields = [part.strip() for part in str(args.fields).split(",") if part.strip()]
    print(f"manifest: {args.manifest}")
    print(f"rows: {len(rows)}")
    missing_fields = [field for field in fields if any(field not in row for row in rows)]
    if missing_fields:
        print(f"missing_fields: {missing_fields}")
    for group_name, group in group_rows(rows):
        print(f"\n[{group_name}] n={len(group)}")
        for field in fields:
            counts = count_values(group, field)
            print(f"{field}: {format_counts(counts, len(group), field)}")


if __name__ == "__main__":
    main()
