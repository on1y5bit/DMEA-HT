from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from .shortcut_bins import add_shortcut_bins


DEFAULT_FIELDS = [
    "n_images_bin",
    "n_visits_bin",
    "selected_n_visits",
    "raw_n_visits",
    "used_images",
    "raw_n_images",
    "has_bio_bin",
    "bio_missing_count_bin",
    "report_length_bin",
    "image_padding_count",
    "source_folder",
]


def distribution_report(rows: Iterable[Dict[str, Any]], fields: List[str] | None = None) -> pd.DataFrame:
    binned = [add_shortcut_bins(row) for row in rows]
    fields = fields or DEFAULT_FIELDS
    records = []
    for field in fields:
        frame = pd.DataFrame({"label": [int(r["label"]) for r in binned], field: [r.get(field, "") for r in binned]})
        counts = frame.groupby(["label", field]).size().reset_index(name="count")
        totals = frame.groupby("label").size().to_dict()
        for item in counts.to_dict("records"):
            records.append(
                {
                    "field": field,
                    "value": item[field],
                    "label": item["label"],
                    "count": item["count"],
                    "proportion": item["count"] / max(totals.get(item["label"], 0), 1),
                }
            )
    return pd.DataFrame(records)
