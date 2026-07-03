from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from .shortcut_bins import matching_key


def add_sample_weights(rows: Iterable[Dict[str, Any]], min_weight: float = 0.2, max_weight: float = 5.0) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    label_key_counts = defaultdict(Counter)
    target_counts = Counter()
    for row in rows:
        key = matching_key(row)
        label = int(row["label"])
        label_key_counts[label][key] += 1
        target_counts[key] += 1

    total = max(len(rows), 1)
    target_dist = {key: count / total for key, count in target_counts.items()}
    label_totals = {label: sum(counter.values()) for label, counter in label_key_counts.items()}

    for row in rows:
        label = int(row["label"])
        key = matching_key(row)
        observed = label_key_counts[label][key] / max(label_totals[label], 1)
        weight = target_dist[key] / max(observed, 1e-12)
        row["sample_weight"] = max(min(weight, max_weight), min_weight)
    return rows

