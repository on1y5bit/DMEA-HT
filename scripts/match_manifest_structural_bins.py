from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def add_report_length_bins(rows: List[Dict[str, Any]], n_bins: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    frame = pd.DataFrame(rows)
    if frame.empty or "report_length" not in frame:
        return [dict(row, report_length_bin="missing") for row in rows]

    for split, split_frame in frame.groupby("split", dropna=False):
        lengths = pd.to_numeric(split_frame["report_length"], errors="coerce").fillna(0)
        try:
            bins = pd.qcut(lengths.rank(method="first"), q=min(n_bins, len(lengths)), labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series([0] * len(split_frame), index=split_frame.index)
        for idx, bin_value in zip(split_frame.index, bins):
            row = dict(rows[int(idx)])
            row["report_length_bin"] = f"q{int(bin_value)}"
            out.append(row)
    return out


def structural_key(row: Dict[str, Any], include_report_length: bool, include_bio_missing: bool) -> Tuple[Any, ...]:
    key: List[Any] = [
        str(row.get("split", "")),
        to_int(row.get("selected_n_visits", row.get("n_visits", 0))),
        to_int(row.get("used_images", row.get("n_images", 0))),
        to_int(row.get("image_padding_count", row.get("padding_count", 0))),
    ]
    if include_bio_missing:
        key.extend(
            [
                to_int(row.get("has_bio", 0)),
                to_int(row.get("bio_missing_count", 0)),
            ]
        )
    if include_report_length:
        key.append(str(row.get("report_length_bin", "")))
    return tuple(key)


def match_rows(
    rows: List[Dict[str, Any]],
    include_report_length: bool,
    include_bio_missing: bool,
    seed: int,
    min_per_label: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[Any, ...], Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[structural_key(row, include_report_length, include_bio_missing)][to_int(row["label"])].append(row)

    matched: List[Dict[str, Any]] = []
    summary = {
        "seed": seed,
        "include_report_length": include_report_length,
        "include_bio_missing": include_bio_missing,
        "input_rows": len(rows),
        "kept_groups": 0,
        "dropped_groups": 0,
        "matched_by_split_label": defaultdict(Counter),
    }
    for key, by_label in grouped.items():
        pos = by_label.get(1, [])
        neg = by_label.get(0, [])
        target = min(len(pos), len(neg))
        if target < min_per_label:
            summary["dropped_groups"] += 1
            continue
        summary["kept_groups"] += 1
        selected_pos = rng.sample(pos, target)
        selected_neg = rng.sample(neg, target)
        matched.extend(selected_pos)
        matched.extend(selected_neg)
        split = str(key[0])
        summary["matched_by_split_label"][split][1] += target
        summary["matched_by_split_label"][split][0] += target

    summary["output_rows"] = len(matched)
    summary["matched_by_split_label"] = {
        split: {str(label): int(count) for label, count in counts.items()}
        for split, counts in summary["matched_by_split_label"].items()
    }
    return matched, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a manifest by selected structure, bio missingness, and report-length bins.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--report-length-bins", type=int, default=10)
    parser.add_argument("--no-report-length", action="store_true")
    parser.add_argument("--no-bio-missing", action="store_true")
    parser.add_argument("--min-per-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    rows = add_report_length_bins(rows, args.report_length_bins)
    matched, summary = match_rows(
        rows,
        include_report_length=not args.no_report_length,
        include_bio_missing=not args.no_bio_missing,
        seed=args.seed,
        min_per_label=args.min_per_label,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in matched:
            row = dict(row)
            row["structural_match"] = 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(matched)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

