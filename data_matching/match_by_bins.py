from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .shortcut_bins import matching_key


def match_by_bins(rows: Iterable[Dict[str, Any]], seed: int = 42, preserve_positive: bool = True) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        grouped[matching_key(row)][int(row["label"])].append(dict(row))

    matched: List[Dict[str, Any]] = []
    for group in grouped.values():
        neg = group[0]
        pos = group[1]
        if not neg or not pos:
            continue
        target = min(len(neg), len(pos))
        if preserve_positive and len(pos) <= len(neg):
            selected_pos = pos
            selected_neg = rng.sample(neg, min(len(neg), max(target, len(pos))))
        else:
            selected_pos = rng.sample(pos, target)
            selected_neg = rng.sample(neg, target)
        matched.extend(selected_pos)
        matched.extend(selected_neg)
    return matched

