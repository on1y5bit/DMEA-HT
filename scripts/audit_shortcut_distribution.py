from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_matching.distribution_report import distribution_report
from dmea_ht.data import read_manifest
from dmea_ht.metrics import compute_binary_metrics


DEFAULT_PROXY_FIELDS = [
    "n_images",
    "n_visits",
    "selected_n_visits",
    "raw_n_visits",
    "used_images",
    "raw_n_images",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "image_padding_count",
    "source_folder",
]


def shortcut_proxy_auc(rows: List[Dict[str, Any]], fields: List[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    y = frame["label"].astype(int)
    fields = fields or DEFAULT_PROXY_FIELDS
    existing = [field for field in fields if field in frame.columns]
    if not existing or y.nunique() < 2:
        return pd.DataFrame([{"shortcut_only_AUC": 0.0, "shortcut_only_AUPRC": 0.0}])
    x = pd.get_dummies(frame[existing].fillna("").astype(str), dummy_na=True)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        folds = min(5, int(y.value_counts().min()))
        if folds < 2:
            return pd.DataFrame([{"shortcut_only_AUC": 0.0, "shortcut_only_AUPRC": 0.0}])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        probs = cross_val_predict(model, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        metrics = compute_binary_metrics(y, probs)
        return pd.DataFrame([{"shortcut_only_AUC": metrics["AUC"], "shortcut_only_AUPRC": metrics["AUPRC"]}])
    except Exception as exc:
        return pd.DataFrame([{"shortcut_only_AUC": 0.0, "shortcut_only_AUPRC": 0.0, "error": str(exc)}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fields", help="Comma-separated shortcut fields for proxy AUC. Defaults to all known audit fields.")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)

    distribution_report(rows).to_csv(out_dir / "shortcut_distribution_before_matching.csv", index=False)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()] if args.fields else None
    proxy = shortcut_proxy_auc(rows, fields=fields)
    proxy.to_csv(out_dir / "shortcut_proxy_auc.csv", index=False)

    if args.predictions:
        preds = pd.read_csv(args.predictions)
        shortcut_cols = [col for col in ["n_images", "n_visits", "has_bio", "bio_missing_count", "report_length"] if col in preds.columns]
        corr_rows = []
        for col in shortcut_cols:
            corr = preds[["prob", col]].apply(pd.to_numeric, errors="coerce").corr(method="spearman").iloc[0, 1]
            corr_rows.append({"field": col, "spearman_prob": corr})
        pd.DataFrame(corr_rows).to_csv(out_dir / "prediction_shortcut_correlation.csv", index=False)

    summary = {
        "n_rows": len(rows),
        "shortcut_only_AUC": float(proxy.iloc[0].get("shortcut_only_AUC", 0.0)),
        "shortcut_only_AUPRC": float(proxy.iloc[0].get("shortcut_only_AUPRC", 0.0)),
    }
    (out_dir / "shortcut_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
