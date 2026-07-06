from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest
from dmea_ht.metrics import compute_binary_metrics


DEFAULT_SHORTCUT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def numeric_matrix(frame: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for field in fields:
        if field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce")
            if values.isna().all():
                values = pd.Series(np.zeros(len(frame)), index=frame.index)
            else:
                values = values.fillna(values.median())
            out[field] = values.astype(float)
    return out


def linear_r2(x: pd.DataFrame, y: pd.Series) -> float | None:
    if x.empty or len(x) < 3:
        return None
    try:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x.to_numpy(), y.to_numpy(dtype=float))
        return float(model.score(x.to_numpy(), y.to_numpy(dtype=float)))
    except Exception:
        return None


def shortcut_auc(x: pd.DataFrame, y: pd.Series) -> float | None:
    if x.empty or y.nunique() < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        folds = min(5, int(y.value_counts().min()))
        if folds < 2:
            return None
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        probs = cross_val_predict(
            model,
            x.to_numpy(),
            y.astype(int).to_numpy(),
            cv=StratifiedKFold(folds, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        return float(compute_binary_metrics(y.astype(int), probs)["AUC"])
    except Exception:
        return None


def read_manifest_frame(path: Path, fields: List[str]) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    keep = ["patient_id", "split", "label"] + [field for field in fields if field in frame.columns]
    return frame[keep].drop_duplicates("patient_id")


def read_split_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = frame["seed"] if "seed" in frame.columns else seed_from_path(path)
        frame["split"] = frame["split"] if "split" in frame.columns else split
        frame["pred_prob_for_audit"] = frame[prob_column(frame)].astype(float)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_shortcuts(preds: pd.DataFrame, manifest: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    merged = preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns and "label" not in merged.columns:
        merged["label"] = merged["label_manifest"]
    for field in fields:
        manifest_col = f"{field}_manifest"
        if field not in merged.columns and manifest_col in merged.columns:
            merged[field] = merged[manifest_col]
        elif field in merged.columns and manifest_col in merged.columns:
            merged[field] = merged[field].where(~merged[field].isna(), merged[manifest_col])
    return merged


def audit_frame(model_id: str, split: str, seed_label: str, frame: pd.DataFrame, fields: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model_id": model_id,
        "split": split,
        "seed": seed_label,
        "n_rows": int(len(frame)),
    }
    y_prob = pd.to_numeric(frame["pred_prob_for_audit"], errors="coerce")
    for field in fields:
        if field not in frame.columns:
            row[f"spearman_{field}"] = None
            continue
        pair = pd.DataFrame({"prob": y_prob, "field": pd.to_numeric(frame[field], errors="coerce")}).dropna()
        row[f"spearman_{field}"] = float(pair["prob"].corr(pair["field"], method="spearman")) if len(pair) >= 3 else None
    spearman_values = [abs(float(row[f"spearman_{field}"])) for field in fields if row.get(f"spearman_{field}") is not None]
    row["max_abs_spearman"] = max(spearman_values) if spearman_values else None
    x = numeric_matrix(frame, fields)
    row["linear_r2_prob_from_shortcuts"] = linear_r2(x, y_prob)
    if "label" in frame.columns:
        row["shortcut_only_label_auc_audit_only"] = shortcut_auc(x, frame["label"].astype(int))
    else:
        row["shortcut_only_label_auc_audit_only"] = None
    return row


def write_report(frame: pd.DataFrame, out_dir: Path, fields: List[str]) -> None:
    lines = [
        "# Phase C3 Prediction Shortcut Residual Audit",
        "",
        "This is an audit-only analysis. Shortcut fields are never fed into the classifier.",
        "",
        f"Fields: {', '.join(fields)}.",
        "",
    ]
    if frame.empty:
        lines.append("No audit rows were generated.")
    else:
        pooled = frame[frame["seed"].astype(str) == "pooled"].copy()
        lines.extend(
            [
                "| model_id | split | max abs Spearman | linear R2 | shortcut-only label AUC audit-only |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in pooled.sort_values(["model_id", "split"]).iterrows():
            lines.append(
                f"| {row['model_id']} | {row['split']} | {row.get('max_abs_spearman', float('nan')):.4f} | "
                f"{row.get('linear_r2_prob_from_shortcuts', float('nan')):.4f} | "
                f"{row.get('shortcut_only_label_auc_audit_only', float('nan')):.4f} |"
            )
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "- Chance-level shortcut-only label AUC supports that selected structural fields alone do not recover labels.",
                "- Prediction-shortcut Spearman and linear R2 measure residual association in model outputs.",
                "- High residual association should trigger a pilot, not immediate formal promotion.",
            ]
        )
    (out_dir / "shortcut_residual_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit residual association between predictions and selected shortcut fields.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=run_dir entries.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fields", default=",".join(DEFAULT_SHORTCUT_FIELDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest_frame(Path(args.manifest), fields)
    rows: List[Dict[str, Any]] = []
    for model_id, run_dir in [parse_named_path(value) for value in args.runs]:
        for split in ("val", "test"):
            preds = read_split_predictions(run_dir, split)
            if preds.empty:
                continue
            merged = merge_shortcuts(preds, manifest, fields)
            for seed, seed_frame in merged.groupby("seed"):
                rows.append(audit_frame(model_id, split, str(seed), seed_frame, fields))
            rows.append(audit_frame(model_id, split, "pooled", merged, fields))
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "shortcut_residual_audit.csv", index=False)
    write_report(frame, out_dir, fields)
    print(json.dumps({"rows": len(frame), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
