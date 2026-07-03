from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .shortcut_bins import add_shortcut_bins


def stratified_patient_split(rows: Iterable[Dict[str, Any]], seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        item = add_shortcut_bins(row)
        key = (int(item["label"]), item["n_visits_bin"], item["n_images_bin"], item["has_bio_bin"])
        groups[key].append(item)

    output: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        for i, row in enumerate(group_rows):
            row = dict(row)
            if i < n_test:
                row["split"] = "test"
            elif i < n_test + n_val:
                row["split"] = "val"
            else:
                row["split"] = "train"
            output.append(row)
    return output

