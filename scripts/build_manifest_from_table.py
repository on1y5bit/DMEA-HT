from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def optional_value(row: pd.Series, column: str | None, default: Any = "") -> Any:
    if not column or column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patient-level JSONL manifest from a CSV/XLSX table.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patient-id-col", default="patient_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col")
    parser.add_argument("--image-paths-col")
    parser.add_argument("--report-text-col")
    parser.add_argument("--bio-values-col")
    parser.add_argument("--source-folder-col")
    args = parser.parse_args()

    table = Path(args.table)
    if table.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(table)
    else:
        frame = pd.read_csv(table)

    rows = []
    for _, row in frame.iterrows():
        item: Dict[str, Any] = {
            "patient_id": str(row[args.patient_id_col]),
            "label": int(row[args.label_col]),
            "image_paths": optional_value(row, args.image_paths_col, ""),
            "report_text": optional_value(row, args.report_text_col, ""),
            "bio_values": optional_value(row, args.bio_values_col, ""),
            "source_folder": optional_value(row, args.source_folder_col, ""),
        }
        split = optional_value(row, args.split_col, "")
        if split:
            item["split"] = str(split).lower()
        item["n_images"] = len([p for p in str(item["image_paths"]).replace(";", "|").split("|") if p])
        item["report_length"] = len(str(item["report_text"]))
        item["has_bio"] = int(bool(str(item["bio_values"]).strip()))
        rows.append(item)

    with open(args.out, "w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
