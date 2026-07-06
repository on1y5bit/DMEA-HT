from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def infer_c2_name(path: Path) -> str:
    match = re.search(r"w(\d+)", path.name)
    if not match:
        return path.name
    digits = match.group(1)
    weight = int(digits) / 100.0
    return f"c2_text_anchor_w{weight:.2f}"


def metric_value(row: pd.Series, key: str) -> float | None:
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def read_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "reports" / "metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {summary_path}")
    frame = pd.read_csv(summary_path)
    out: Dict[str, Any] = {"run_dir": str(run_dir)}
    for split in ("val", "test"):
        split_rows = frame[frame["split"].astype(str).str.lower() == split]
        if split_rows.empty:
            continue
        row = split_rows.iloc[0]
        for metric in ("AUC", "AUPRC", "ACC", "F1", "Sensitivity", "Specificity", "Balanced_ACC"):
            out[f"{split}_{metric.lower()}_mean"] = metric_value(row, f"{metric}_mean")
            out[f"{split}_{metric.lower()}_std"] = metric_value(row, f"{metric}_std")
        for metric in ("text_morphology_auc", "image_morphology_auc"):
            out[f"{split}_{metric}_mean"] = metric_value(row, f"{metric}_mean")
            out[f"{split}_{metric}_std"] = metric_value(row, f"{metric}_std")
    return out


def read_seed_count(run_dir: Path) -> int:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if "seed" not in frame.columns:
        return 0
    return int(frame["seed"].nunique())


def maybe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_shortcut_summary(run_dir: Path) -> Dict[str, Any]:
    candidates = [
        run_dir / "shortcut_audit_selected" / "shortcut_audit_summary.json",
        run_dir / "shortcut_audit" / "shortcut_audit_summary.json",
    ]
    for candidate in candidates:
        data = maybe_read_json(candidate)
        if data:
            return {
                "selected_shortcut_only_auc": data.get("shortcut_only_AUC"),
                "selected_shortcut_only_auprc": data.get("shortcut_only_AUPRC"),
            }
    return {"selected_shortcut_only_auc": None, "selected_shortcut_only_auprc": None}


def collect_rows(named_runs: Iterable[Tuple[str, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, run_dir in named_runs:
        row = {"model_id": name}
        row.update(read_summary(run_dir))
        row.update(read_shortcut_summary(run_dir))
        row["n_seeds"] = read_seed_count(run_dir)
        row["test_usage"] = "reporting_only"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    best_val = frame["val_auc_mean"].astype(float).max()
    frame["val_auc_delta_vs_best"] = frame["val_auc_mean"].astype(float) - best_val
    frame["is_best_by_validation_auc"] = frame["val_auc_mean"].astype(float) == best_val
    frame["promotion_status"] = frame.apply(promotion_status, axis=1)
    return frame.sort_values(["val_auc_mean", "model_id"], ascending=[False, True])


def promotion_status(row: pd.Series) -> str:
    model_id = str(row["model_id"])
    if model_id == "strict_mvp":
        return "reference_baseline"
    if bool(row.get("is_best_by_validation_auc", False)):
        return "current_main_candidate"
    return "not_promoted"


def write_report(frame: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Phase C3 Model Comparison",
        "",
        "Selection rule: validation AUC only. Test metrics are reporting-only.",
        "",
    ]
    if frame.empty:
        lines.append("No runs were collected.")
    else:
        best = frame.iloc[0]
        lines.extend(
            [
                f"Current main candidate: `{best['model_id']}`.",
                f"Validation AUC: {best['val_auc_mean']:.4f} +/- {best.get('val_auc_std', 0.0):.4f}.",
                "",
                "| model_id | val AUC | val AUPRC | test AUC reporting-only | status |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in frame.iterrows():
            val_auc = f"{fmt(row.get('val_auc_mean'))} +/- {fmt(row.get('val_auc_std'))}"
            val_auprc = fmt(row.get("val_auprc_mean"))
            test_auc = f"{fmt(row.get('test_auc_mean'))} +/- {fmt(row.get('test_auc_std'))}"
            lines.append(f"| {row['model_id']} | {val_auc} | {val_auprc} | {test_auc} | {row['promotion_status']} |")
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "- C1/C2 variants are promoted only if they improve validation AUC.",
                "- Test AUC is listed for transparency and must not decide the winner.",
                "- Shortcut residual audits are reported separately in `shortcut_residual_audit_report.md`.",
            ]
        )
    (out_dir / "model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect completed DMEA-HT Phase C3 model comparison metrics.")
    parser.add_argument("--strict-mvp-run", required=True)
    parser.add_argument("--c1-text-run", required=True)
    parser.add_argument("--c1-text-image-run", required=True)
    parser.add_argument("--c2-runs", nargs="*", default=[])
    parser.add_argument("--extra-runs", nargs="*", default=[], help="Optional name=path entries.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_runs: List[Tuple[str, Path]] = [
        ("strict_mvp", Path(args.strict_mvp_run)),
        ("c1_text_morphology_only", Path(args.c1_text_run)),
        ("c1_text_image_evidence", Path(args.c1_text_image_run)),
    ]
    named_runs.extend((infer_c2_name(Path(path)), Path(path)) for path in args.c2_runs)
    named_runs.extend(parse_named_path(value) for value in args.extra_runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_rows(named_runs)
    frame.to_csv(out_dir / "model_comparison_table.csv", index=False)
    write_report(frame, out_dir)
    if not frame.empty:
        print(frame[["model_id", "val_auc_mean", "val_auc_std", "promotion_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
