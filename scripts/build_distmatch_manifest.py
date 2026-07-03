from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import patient_split, read_manifest


BIO_COLUMNS = ["sex", "age", "TgAb", "FT3", "FT4", "TPOAb", "TSH"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VisitRecord:
    date: str
    image_paths: List[str]
    report_text: str = ""


def is_missing(value: Any) -> bool:
    return pd.isna(value) or value == ""


def normalize_date(value: Any) -> str:
    if is_missing(value):
        return ""
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def label_patient_ids(data_root: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for label in (0, 1):
        label_root = data_root / str(label)
        if not label_root.exists():
            continue
        for patient_dir in label_root.iterdir():
            if patient_dir.is_dir():
                labels[patient_dir.name] = label
    return labels


def table_by_patient(data_root: Path) -> Dict[str, pd.DataFrame]:
    table = pd.read_excel(data_root / "all_patients.xlsx")
    table = table.rename(columns={"patient_Id": "patient_id"})
    table["patient_id"] = table["patient_id"].astype(str)
    table["_date"] = table["time"].map(normalize_date)
    return {patient_id: group.copy() for patient_id, group in table.groupby("patient_id")}


def latest_bio(group: pd.DataFrame | None) -> tuple[List[float], List[int]]:
    if group is None or group.empty:
        return [0.0] * len(BIO_COLUMNS), [1] * len(BIO_COLUMNS)
    ordered = group.sort_values("_date")
    latest = ordered.iloc[-1]
    values: List[float] = []
    missing: List[int] = []
    for column in BIO_COLUMNS:
        value = latest.get(column)
        if is_missing(value):
            values.append(0.0)
            missing.append(1)
        else:
            values.append(float(value))
            missing.append(0)
    return values, missing


def reports_by_date(group: pd.DataFrame | None) -> Dict[str, str]:
    if group is None or group.empty:
        return {}
    reports: Dict[str, List[str]] = defaultdict(list)
    for _, row in group.sort_values("_date").iterrows():
        report = row.get("report")
        date = row.get("_date")
        if date and not is_missing(report):
            reports[str(date)].append(str(report).strip())
    return {date: "\n".join(parts) for date, parts in reports.items()}


def visit_records(data_root: Path, label: int, patient_id: str, reports: Dict[str, str]) -> List[VisitRecord]:
    patient_dir = data_root / str(label) / patient_id
    if not patient_dir.exists():
        return []
    visits: List[VisitRecord] = []
    for visit_dir in patient_dir.iterdir():
        if not visit_dir.is_dir():
            continue
        image_paths = sorted(
            path.relative_to(data_root).as_posix()
            for path in visit_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            continue
        date = visit_dir.name
        visits.append(VisitRecord(date=date, image_paths=image_paths, report_text=reports.get(date, "")))
    return sorted(visits, key=lambda item: item.date)


def historical_visits(visits: List[VisitRecord], exclude_latest_anchor: bool) -> List[VisitRecord]:
    if not exclude_latest_anchor or len(visits) <= 1:
        return visits
    history = visits[:-1]
    return history or visits


def split_map_from_base(base_manifest: Path | None, rows_for_split: List[Dict[str, Any]], seed: int) -> Dict[str, str]:
    if base_manifest and base_manifest.exists():
        mapping = {}
        for row in read_manifest(base_manifest):
            split = row.get("split")
            if split:
                mapping[str(row["patient_id"])] = str(split).lower()
        if mapping:
            return mapping
    splits = patient_split(rows_for_split, seed=seed)
    return {str(row["patient_id"]): split for row, split in zip(rows_for_split, splits)}


def sample_reference_count(reference_counts: List[int], max_count: int, rng: random.Random) -> int:
    feasible = [count for count in reference_counts if count <= max_count]
    if not feasible:
        return max(1, min(max_count, min(reference_counts) if reference_counts else max_count))
    return rng.choice(feasible)


def sample_negative_visits(visits: List[VisitRecord], target_count: int, rng: random.Random) -> List[VisitRecord]:
    if target_count >= len(visits):
        return visits
    selected = rng.sample(visits, target_count)
    return sorted(selected, key=lambda item: item.date)


def fixed_visit_images(paths: List[str], max_images_per_visit: int, rng: random.Random) -> tuple[List[str], int]:
    if not paths:
        return [], max_images_per_visit
    if len(paths) >= max_images_per_visit:
        return sorted(rng.sample(paths, max_images_per_visit)), 0
    padded = list(paths)
    while len(padded) < max_images_per_visit:
        padded.append(paths[(len(padded) - len(paths)) % len(paths)])
    return padded, max_images_per_visit - len(paths)


def build_row(
    patient_id: str,
    label: int,
    split: str,
    visits: List[VisitRecord],
    selected_visits: List[VisitRecord],
    bio_values: List[float],
    bio_missing_mask: List[int],
    max_images_per_visit: int,
    rng: random.Random,
) -> Dict[str, Any]:
    image_paths: List[str] = []
    padding_count = 0
    for visit in selected_visits:
        fixed_paths, added_padding = fixed_visit_images(visit.image_paths, max_images_per_visit, rng)
        image_paths.extend(fixed_paths)
        padding_count += added_padding

    report_parts = []
    for visit in selected_visits:
        if visit.report_text:
            report_parts.append(f"[{visit.date}] {visit.report_text}")
    report_text = "\n".join(report_parts)

    return {
        "patient_id": patient_id,
        "label": label,
        "split": split,
        "image_paths": image_paths,
        "report_text": report_text,
        "bio_values": bio_values,
        "bio_missing_mask": bio_missing_mask,
        "bio_abnormal_flags": [0] * len(BIO_COLUMNS),
        "n_images": len(image_paths),
        "n_visits": len(selected_visits),
        "selected_n_visits": len(selected_visits),
        "raw_n_visits": len(visits),
        "used_images": len(image_paths),
        "raw_n_images": sum(len(visit.image_paths) for visit in visits),
        "image_padding_count": padding_count,
        "padding_count": padding_count,
        "max_images_per_visit": max_images_per_visit,
        "selected_visit_dates": [visit.date for visit in selected_visits],
        "has_bio": int(any(mask == 0 for mask in bio_missing_mask)),
        "bio_missing_count": int(sum(bio_missing_mask)),
        "report_length": len(report_text),
        "source_folder": "",
        "distmatch": 1,
    }


def build_distmatch_manifest(
    data_root: Path,
    base_manifest: Path | None,
    max_images_per_visit: int,
    seed: int,
    exclude_latest_anchor: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    labels = label_patient_ids(data_root)
    table = table_by_patient(data_root)

    patient_info: Dict[str, Dict[str, Any]] = {}
    split_seed_rows: List[Dict[str, Any]] = []
    for patient_id, label in labels.items():
        group = table.get(patient_id)
        reports = reports_by_date(group)
        visits = historical_visits(visit_records(data_root, label, patient_id, reports), exclude_latest_anchor)
        if not visits:
            continue
        bio_values, bio_missing_mask = latest_bio(group)
        patient_info[patient_id] = {
            "patient_id": patient_id,
            "label": label,
            "visits": visits,
            "bio_values": bio_values,
            "bio_missing_mask": bio_missing_mask,
        }
        split_seed_rows.append({"patient_id": patient_id, "label": label})

    split_map = split_map_from_base(base_manifest, split_seed_rows, seed=seed)
    by_split_label: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for patient_id, info in patient_info.items():
        split = split_map.get(patient_id)
        if not split:
            continue
        by_split_label[split][int(info["label"])].append(patient_id)

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"seed": seed, "max_images_per_visit": max_images_per_visit, "splits": {}}
    for split, label_groups in sorted(by_split_label.items()):
        pos_ids = label_groups.get(1, [])
        neg_ids = label_groups.get(0, [])
        pos_counts = [len(patient_info[pid]["visits"]) for pid in pos_ids]
        split_summary = {
            "label1_reference_counts": dict(Counter(pos_counts)),
            "label0_available_counts": dict(Counter(len(patient_info[pid]["visits"]) for pid in neg_ids)),
            "n_label1": len(pos_ids),
            "n_label0": len(neg_ids),
        }

        for patient_id in pos_ids:
            info = patient_info[patient_id]
            rows.append(
                build_row(
                    patient_id=patient_id,
                    label=1,
                    split=split,
                    visits=info["visits"],
                    selected_visits=info["visits"],
                    bio_values=info["bio_values"],
                    bio_missing_mask=info["bio_missing_mask"],
                    max_images_per_visit=max_images_per_visit,
                    rng=rng,
                )
            )

        neg_selected_counts = []
        for patient_id in neg_ids:
            info = patient_info[patient_id]
            visits = info["visits"]
            target_count = sample_reference_count(pos_counts, max_count=len(visits), rng=rng)
            selected_visits = sample_negative_visits(visits, target_count, rng=rng)
            neg_selected_counts.append(len(selected_visits))
            rows.append(
                build_row(
                    patient_id=patient_id,
                    label=0,
                    split=split,
                    visits=visits,
                    selected_visits=selected_visits,
                    bio_values=info["bio_values"],
                    bio_missing_mask=info["bio_missing_mask"],
                    max_images_per_visit=max_images_per_visit,
                    rng=rng,
                )
            )

        split_summary["label0_selected_counts"] = dict(Counter(neg_selected_counts))
        summary["splits"][split] = split_summary

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visit-level distmatch manifest for DMEA-HT.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--max-images-per-visit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-latest-anchor", action="store_true")
    args = parser.parse_args()

    rows, summary = build_distmatch_manifest(
        data_root=Path(args.data_root),
        base_manifest=Path(args.base_manifest) if args.base_manifest else None,
        max_images_per_visit=args.max_images_per_visit,
        seed=args.seed,
        exclude_latest_anchor=not args.include_latest_anchor,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["out"] = str(out)
    summary["rows"] = len(rows)
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

