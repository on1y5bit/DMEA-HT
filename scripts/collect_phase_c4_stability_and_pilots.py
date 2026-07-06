from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest
from dmea_ht.metrics import compute_binary_metrics


SHORTCUT_FIELDS = [
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def parse_weight_run(value: str) -> Tuple[float, Path]:
    if "=" not in value:
        raise ValueError(f"Weight run must be weight=path, got: {value}")
    weight, path = value.split("=", 1)
    return float(weight), Path(path)


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def prob_column(frame: pd.DataFrame) -> str:
    if "pred_prob" in frame.columns:
        return "pred_prob"
    if "prob" in frame.columns:
        return "prob"
    raise ValueError("Prediction CSV must contain pred_prob or prob.")


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    keep = ["patient_id", "split", "label"] + [field for field in SHORTCUT_FIELDS if field in frame.columns]
    return frame[keep].drop_duplicates("patient_id")


def numeric_shortcuts(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for field in SHORTCUT_FIELDS:
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        if values.isna().all():
            values = pd.Series(np.zeros(len(frame)), index=frame.index)
        else:
            values = values.fillna(values.median())
        out[field] = values.astype(float)
    return out


def shortcut_proxy_auc(manifest: pd.DataFrame) -> float | None:
    x = numeric_shortcuts(manifest)
    if x.empty or "label" not in manifest.columns or manifest["label"].nunique() < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        y = manifest["label"].astype(int)
        folds = min(5, int(y.value_counts().min()))
        if folds < 2:
            return None
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        probs = cross_val_predict(
            model,
            x.to_numpy(),
            y.to_numpy(),
            cv=StratifiedKFold(folds, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        return float(compute_binary_metrics(y, probs)["AUC"])
    except Exception:
        return None


def read_metrics_by_seed(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics_by_seed.csv: {path}")
    return pd.read_csv(path)


def read_prediction(run_dir: Path, split: str, seed: int) -> pd.DataFrame:
    path = run_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["pred_prob_for_audit"] = frame[prob_column(frame)].astype(float)
    return frame


def max_abs_shortcut_spearman(run_dir: Path, seed: int, manifest: pd.DataFrame, split: str = "val") -> float | None:
    preds = read_prediction(run_dir, split, seed)
    merged = preds.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    values: List[float] = []
    for field in SHORTCUT_FIELDS:
        manifest_col = f"{field}_manifest"
        if field not in merged.columns and manifest_col in merged.columns:
            merged[field] = merged[manifest_col]
        if field not in merged.columns:
            continue
        pair = pd.DataFrame(
            {
                "prob": pd.to_numeric(merged["pred_prob_for_audit"], errors="coerce"),
                "field": pd.to_numeric(merged[field], errors="coerce"),
            }
        ).dropna()
        if len(pair) < 3 or pair["field"].nunique() < 2:
            continue
        corr = pair["prob"].corr(pair["field"], method="spearman")
        if not pd.isna(corr):
            values.append(abs(float(corr)))
    return max(values) if values else None


def metric_row(metrics: pd.DataFrame, seed: int, split: str) -> pd.Series:
    rows = metrics[(metrics["seed"].astype(int) == int(seed)) & (metrics["split"].astype(str).str.lower() == split)]
    if rows.empty:
        raise ValueError(f"Missing metrics for seed={seed}, split={split}")
    return rows.iloc[0]


def build_seed_summary(run_dir: Path, manifest: pd.DataFrame, selected_proxy_auc: float | None) -> pd.DataFrame:
    metrics = read_metrics_by_seed(run_dir)
    seeds = sorted(metrics["seed"].astype(int).unique().tolist())
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        val = metric_row(metrics, seed, "val")
        test = metric_row(metrics, seed, "test")
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(val.get("best_epoch", -1)),
                "val_auc": float(val["AUC"]),
                "val_auprc": float(val["AUPRC"]),
                "val_f1": float(val["F1"]),
                "val_sensitivity": float(val["Sensitivity"]),
                "val_specificity": float(val["Specificity"]),
                "val_balanced_accuracy": float(val["Balanced_ACC"]),
                "test_auc_reporting_only": float(test["AUC"]),
                "max_abs_prediction_shortcut_spearman_val": max_abs_shortcut_spearman(run_dir, seed, manifest, "val"),
                "selected_structure_shortcut_proxy_auc": selected_proxy_auc,
            }
        )
    return pd.DataFrame(rows)


def stability_status(summary: pd.DataFrame, strict_mvp_val_auc: float, residual_limit: float) -> Tuple[str, str]:
    mean_auc = float(summary["val_auc"].mean())
    min_auc = float(summary["val_auc"].min())
    mean_auprc = float(summary["val_auprc"].mean())
    max_residual = float(summary["max_abs_prediction_shortcut_spearman_val"].max())
    weak_seeds = int((summary["val_auc"] < strict_mvp_val_auc).sum())
    reasons: List[str] = []
    status = "STABILITY_PASS"
    if mean_auc <= strict_mvp_val_auc:
        status = "STABILITY_FAIL"
        reasons.append("mean validation AUC is not above strict MVP reference")
    if weak_seeds >= 2:
        status = "STABILITY_FAIL"
        reasons.append("multiple seeds fall below strict MVP reference")
    elif min_auc < strict_mvp_val_auc:
        status = "STABILITY_WARNING"
        reasons.append("one seed falls below strict MVP reference")
    if mean_auprc < 0.65:
        status = "STABILITY_WARNING" if status == "STABILITY_PASS" else status
        reasons.append("mean validation AUPRC is low")
    if max_residual > residual_limit:
        status = "STABILITY_WARNING" if status == "STABILITY_PASS" else status
        reasons.append("shortcut residual exceeds configured threshold")
    if not reasons:
        reasons.append("extended-seed validation metrics remain above strict MVP without new shortcut concern")
    return status, "; ".join(reasons)


def write_stability_report(summary: pd.DataFrame, out_dir: Path, status: str, reason: str) -> None:
    lines = [
        "# Phase C4 C1 Extended-Seed Stability",
        "",
        f"Status: `{status}`.",
        f"Reason: {reason}.",
        "",
        f"Mean validation AUC: {summary['val_auc'].mean():.4f} +/- {summary['val_auc'].std(ddof=1):.4f}.",
        f"Median validation AUC: {summary['val_auc'].median():.4f}.",
        f"Min/max validation AUC: {summary['val_auc'].min():.4f} / {summary['val_auc'].max():.4f}.",
        f"Mean validation AUPRC: {summary['val_auprc'].mean():.4f} +/- {summary['val_auprc'].std(ddof=1):.4f}.",
        f"Mean sensitivity/specificity: {summary['val_sensitivity'].mean():.4f} / {summary['val_specificity'].mean():.4f}.",
        f"Max validation prediction-shortcut residual Spearman: {summary['max_abs_prediction_shortcut_spearman_val'].max():.4f}.",
        "",
        "| seed | val AUC | val AUPRC | sensitivity | specificity | residual Spearman |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['val_auc']:.4f} | {row['val_auprc']:.4f} | "
            f"{row['val_sensitivity']:.4f} | {row['val_specificity']:.4f} | "
            f"{fmt(row['max_abs_prediction_shortcut_spearman_val'])} |"
        )
    (out_dir / "c1_extended_seed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pilot_status(row: pd.Series, baseline_seed0: pd.Series, residual_limit: float) -> Tuple[str, str]:
    reasons: List[str] = []
    status = "PILOT_PASS_RECOMMEND_FORMAL"
    if float(row["val_auc"]) + 1e-12 < float(baseline_seed0["val_auc"]):
        status = "PILOT_FAIL"
        reasons.append("validation AUC is below current C1 seed-0 baseline")
    if float(row["val_auprc"]) < float(baseline_seed0["val_auprc"]) - 0.02:
        status = "PILOT_FAIL"
        reasons.append("validation AUPRC drops by more than 0.02")
    residual = row.get("max_abs_prediction_shortcut_spearman_val")
    if residual is not None and not pd.isna(residual) and float(residual) > residual_limit:
        status = "PILOT_FAIL"
        reasons.append("shortcut residual exceeds configured threshold")
    if min(float(row["val_sensitivity"]), float(row["val_specificity"])) < 0.20:
        status = "PILOT_FAIL"
        reasons.append("sensitivity/specificity imbalance is severe")
    if not reasons:
        reasons.append("pilot matches or improves seed-0 validation baseline without shortcut concern")
    return status, "; ".join(reasons)


def build_pilot_summary(
    weight_runs: Iterable[Tuple[float, Path]],
    manifest: pd.DataFrame,
    selected_proxy_auc: float | None,
    baseline_seed0: pd.Series,
    residual_limit: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for weight, run_dir in weight_runs:
        run_summary = build_seed_summary(run_dir, manifest, selected_proxy_auc)
        if len(run_summary) != 1:
            raise ValueError(f"Pilot run should contain exactly one seed: {run_dir}")
        row = run_summary.iloc[0].to_dict()
        row["weight"] = weight
        status, reason = pilot_status(pd.Series(row), baseline_seed0, residual_limit)
        row["pilot_gate_status"] = status
        row["pilot_gate_reason"] = reason
        rows.append(row)
    frame = pd.DataFrame(rows)
    cols = [
        "weight",
        "seed",
        "best_epoch",
        "val_auc",
        "val_auprc",
        "val_f1",
        "val_sensitivity",
        "val_specificity",
        "val_balanced_accuracy",
        "test_auc_reporting_only",
        "max_abs_prediction_shortcut_spearman_val",
        "selected_structure_shortcut_proxy_auc",
        "pilot_gate_status",
        "pilot_gate_reason",
    ]
    return frame[cols].sort_values(["val_auc", "weight"], ascending=[False, True])


def write_pilot_report(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Phase C4 C1 Text Morphology Weight Pilot",
        "",
        "Pilot selection uses validation metrics only. Test AUC is reporting-only.",
        "",
        "| weight | seed | val AUC | val AUPRC | sensitivity | specificity | residual Spearman | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['weight']:.3f} | {int(row['seed'])} | {row['val_auc']:.4f} | {row['val_auprc']:.4f} | "
            f"{row['val_sensitivity']:.4f} | {row['val_specificity']:.4f} | "
            f"{fmt(row['max_abs_prediction_shortcut_spearman_val'])} | {row['pilot_gate_status']} |"
        )
    passing = summary[summary["pilot_gate_status"] == "PILOT_PASS_RECOMMEND_FORMAL"]
    lines.extend(["", "Recommendation:", ""])
    if passing.empty:
        lines.append("No pilot weight is recommended for future formal three-seed evaluation.")
    else:
        best = passing.iloc[0]
        lines.append(
            f"Recommend weight {best['weight']:.3f} for a later formal three-seed evaluation only; "
            "do not launch it automatically in Phase C4."
        )
    (out_dir / "c1_weight_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_reports(
    out_dir: Path,
    stability_status_value: str,
    stability_reason: str,
    pilot_summary: pd.DataFrame,
    c1_remains_main: bool,
) -> None:
    rows: List[Dict[str, Any]] = [
        {
            "item": "c1_text_morphology_only_extended_seeds",
            "status": stability_status_value,
            "reason": stability_reason,
        }
    ]
    for _, row in pilot_summary.iterrows():
        rows.append(
            {
                "item": f"text_morphology_weight_{row['weight']:.3f}",
                "status": row["pilot_gate_status"],
                "reason": row["pilot_gate_reason"],
            }
        )
    decision = pd.DataFrame(rows)
    decision.to_csv(out_dir / "decision_gate_phase_c4_summary.csv", index=False)

    lines = [
        "# Phase C4 Decision Gate",
        "",
        f"C1 remains current main candidate: `{str(c1_remains_main)}`.",
        "",
        "| item | status | reason |",
        "| --- | --- | --- |",
    ]
    for _, row in decision.iterrows():
        lines.append(f"| {row['item']} | {row['status']} | {row['reason']} |")
    (out_dir / "decision_gate_phase_c4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    out_dir: Path,
    stability_summary: pd.DataFrame,
    stability_status_value: str,
    stability_reason: str,
    pilot_summary: pd.DataFrame,
    c1_remains_main: bool,
) -> None:
    passing = pilot_summary[pilot_summary["pilot_gate_status"] == "PILOT_PASS_RECOMMEND_FORMAL"]
    shortcut_max = max(
        float(stability_summary["max_abs_prediction_shortcut_spearman_val"].max()),
        float(pilot_summary["max_abs_prediction_shortcut_spearman_val"].max()),
    )
    lines = [
        "# Phase C4 Final Report",
        "",
        "Phase C4 verifies C1 stability and runs a one-seed pilot-only text morphology weight sweep.",
        "",
        f"1. Does C1 remain the current main candidate? `{str(c1_remains_main)}`.",
        f"2. Is C1 stable under extended seeds? `{stability_status_value}`: {stability_reason}.",
    ]
    if passing.empty:
        lines.append("3. Weight recommendation: no pilot weight is recommended for formal evaluation.")
    else:
        best = passing.iloc[0]
        lines.append(f"3. Weight recommendation: {best['weight']:.3f} may deserve later formal three-seed evaluation.")
    lines.extend(
        [
            f"4. Shortcut concern: max residual Spearman across C4 summaries is {shortcut_max:.4f}.",
            "5. Next phase should run formal training only if the user explicitly approves a selected pilot recommendation.",
            "6. Still forbidden without new evidence: image morphology BCE, text negative loss, bio losses, discordance loss, counterfactual training, matched contrastive training, new architecture modules, new splits, and test-based selection.",
            "",
            "See `c1_extended_seed_report.md`, `c1_weight_pilot_report.md`, and `decision_gate_phase_c4_report.md` for details.",
        ]
    )
    (out_dir / "phase_c4_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DMEA-HT Phase C4 stability and pilot weight reports.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--extended-run", required=True)
    parser.add_argument("--weight-runs", nargs="+", required=True, help="Entries like 0.03=runs/path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strict-mvp-val-auc", type=float, default=0.7581107590161461)
    parser.add_argument("--shortcut-residual-limit", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest_frame(Path(args.manifest))
    selected_proxy_auc = shortcut_proxy_auc(manifest)

    stability_summary = build_seed_summary(Path(args.extended_run), manifest, selected_proxy_auc)
    stability_summary.to_csv(out_dir / "c1_extended_seed_summary.csv", index=False)
    stability_status_value, stability_reason = stability_status(
        stability_summary,
        strict_mvp_val_auc=args.strict_mvp_val_auc,
        residual_limit=args.shortcut_residual_limit,
    )
    write_stability_report(stability_summary, out_dir, stability_status_value, stability_reason)

    seed0_rows = stability_summary[stability_summary["seed"].astype(int) == 0]
    if seed0_rows.empty:
        raise ValueError("Extended run must contain seed 0 for pilot baseline comparison.")
    pilot_summary = build_pilot_summary(
        [parse_weight_run(value) for value in args.weight_runs],
        manifest,
        selected_proxy_auc,
        seed0_rows.iloc[0],
        residual_limit=args.shortcut_residual_limit,
    )
    pilot_summary.to_csv(out_dir / "c1_weight_pilot_summary.csv", index=False)
    write_pilot_report(pilot_summary, out_dir)

    c1_remains_main = stability_status_value in {"STABILITY_PASS", "STABILITY_WARNING"}
    write_decision_reports(out_dir, stability_status_value, stability_reason, pilot_summary, c1_remains_main)
    write_final_report(out_dir, stability_summary, stability_status_value, stability_reason, pilot_summary, c1_remains_main)
    print(f"Wrote Phase C4 reports to {out_dir}")


if __name__ == "__main__":
    main()
