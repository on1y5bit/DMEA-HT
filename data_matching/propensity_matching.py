from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from .shortcut_bins import add_shortcut_bins


def shortcut_propensity_scores(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    binned = [add_shortcut_bins(row) for row in rows]
    frame = pd.DataFrame(binned)
    fields = ["n_images_bin", "n_visits_bin", "has_bio_bin", "bio_missing_count_bin", "report_length_bin"]
    x = pd.get_dummies(frame[fields].astype(str), dummy_na=True)
    y = frame["label"].astype(int)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x, y)
        frame["shortcut_propensity"] = model.predict_proba(x)[:, 1]
    except Exception:
        frame["shortcut_propensity"] = float(y.mean()) if len(y) else 0.0
    return frame[["patient_id", "label", "shortcut_propensity"]]

