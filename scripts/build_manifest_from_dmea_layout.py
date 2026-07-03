from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import patient_split


BIO_COLUMNS = ["sex", "age", "TgAb", "FT3", "FT4", "TPOAb", "TSH"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def is_missing(value: Any) -> bool:
    return pd.isna(value) or value == ""


def latest_bio(group: pd.DataFrame) -> tuple[List[float], List[int]]:
    ordered = group.sort_values("time")
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


def joined_reports(group: pd.DataFrame) -> str:
    parts = []
    for _, row in group.sort_values("time").iterrows():
        report = row.get("report")
        if not is_missing(report):
            parts.append(f"[{row.get('time')}] {str(report).strip()}")
    return "\n".join(parts)


def image_paths_for_patient(data_root: Path, label: int, patient_id: str) -> List[str]:
    patient_dir = data_root / str(label) / patient_id
    if not patient_dir.exists():
        return []
    paths = []
    for path in patient_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path.relative_to(data_root).as_posix())
    return sorted(paths)


def visit_count_for_patient(data_root: Path, label: int, patient_id: str, table_visits: int) -> int:
    patient_dir = data_root / str(label) / patient_id
    folder_visits = 0
    if patient_dir.exists():
        folder_visits = sum(1 for item in patient_dir.iterdir() if item.is_dir())
    return max(folder_visits, table_visits)


def label_patient_ids(data_root: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for label in (0, 1):
        root = data_root / str(label)
        if not root.exists():
            continue
        for patient_dir in root.iterdir():
            if patient_dir.is_dir():
                labels[patient_dir.name] = label
    return labels


def maybe_limit(rows: List[Dict[str, Any]], limit_per_label: int | None, seed: int) -> List[Dict[str, Any]]:
    if not limit_per_label:
        return rows
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    limited: List[Dict[str, Any]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        limited.extend(label_rows[:limit_per_label])
    return limited


def build_manifest(data_root: Path, make_split: bool, limit_per_label: int | None, seed: int) -> List[Dict[str, Any]]:
    table_path = data_root / "all_patients.xlsx"
    table = pd.read_excel(table_path)
    table = table.rename(columns={"patient_Id": "patient_id"})
    table["patient_id"] = table["patient_id"].astype(str)
    labels = label_patient_ids(data_root)

    rows: List[Dict[str, Any]] = []
    for patient_id, group in table.groupby("patient_id"):
        if patient_id not in labels:
            continue
        label = labels[patient_id]
        images = image_paths_for_patient(data_root, label, patient_id)
        report_text = joined_reports(group)
        bio_values, bio_missing_mask = latest_bio(group)
        n_visits = visit_count_for_patient(data_root, label, patient_id, group["time"].nunique())
        rows.append(
            {
                "patient_id": patient_id,
                "label": label,
                "image_paths": images,
                "report_text": report_text,
                "bio_values": bio_values,
                "bio_missing_mask": bio_missing_mask,
                "bio_abnormal_flags": [0] * len(BIO_COLUMNS),
                "n_images": len(images),
                "n_visits": int(n_visits),
                "has_bio": int(any(mask == 0 for mask in bio_missing_mask)),
                "bio_missing_count": int(sum(bio_missing_mask)),
                "report_length": len(report_text),
                "source_folder": "",
            }
        )

    rows = maybe_limit(rows, limit_per_label=limit_per_label, seed=seed)
    if make_split:
        splits = patient_split(rows, seed=seed)
        for row, split in zip(rows, splits):
            row["split"] = split
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for /label/patient/date/image DMEA-HT layout.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--make-split", action="store_true")
    parser.add_argument("--limit-per-label", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_manifest(Path(args.data_root), args.make_split, args.limit_per_label, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

