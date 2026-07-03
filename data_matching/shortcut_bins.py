from __future__ import annotations

from typing import Any, Dict, Tuple


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bin_n_images(value: Any) -> str:
    x = to_float(value)
    if x <= 1:
        return "1"
    if x <= 3:
        return "2-3"
    if x <= 6:
        return "4-6"
    return "7+"


def bin_n_visits(value: Any) -> str:
    x = to_float(value)
    if x <= 1:
        return "1"
    if x == 2:
        return "2"
    return "3+"


def bin_report_length(value: Any) -> str:
    x = to_float(value)
    if x <= 80:
        return "short"
    if x <= 200:
        return "medium"
    return "long"


def bin_bio_missing_count(value: Any) -> str:
    x = to_float(value)
    if x <= 0:
        return "0"
    if x <= 2:
        return "1-2"
    return "3+"


def add_shortcut_bins(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["n_images_bin"] = bin_n_images(row.get("n_images", 0))
    out["n_visits_bin"] = bin_n_visits(row.get("n_visits", 0))
    out["report_length_bin"] = bin_report_length(row.get("report_length", 0))
    out["bio_missing_count_bin"] = bin_bio_missing_count(row.get("bio_missing_count", 0))
    out["has_bio_bin"] = str(int(to_float(row.get("has_bio", 0))))
    return out


def matching_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    binned = add_shortcut_bins(row)
    return (
        binned["n_visits_bin"],
        binned["n_images_bin"],
        binned["has_bio_bin"],
        binned["bio_missing_count_bin"],
    )

