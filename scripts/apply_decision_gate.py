from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data if isinstance(data, list) else [data])
    return pd.read_csv(path)


def maybe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_model_id(value: Any) -> str:
    text = str(value)
    aliases = {
        "c1_text": "c1_text_morphology_only",
        "c1_text_image": "c1_text_image_evidence",
        "c2_w001": "c2_text_anchor_w0.01",
        "c2_w003": "c2_text_anchor_w0.03",
        "c2_w005": "c2_text_anchor_w0.05",
        "c2_w010": "c2_text_anchor_w0.10",
    }
    return aliases.get(text, text)


def attach_shortcut_audit(frame: pd.DataFrame, audit_path: str | None) -> pd.DataFrame:
    if not audit_path:
        return frame
    path = Path(audit_path)
    if not path.exists():
        return frame
    audit = pd.read_csv(path)
    if audit.empty or "model_id" not in audit.columns:
        return frame
    audit["model_id"] = audit["model_id"].map(normalize_model_id)
    pooled = audit[(audit["split"].astype(str) == "val") & (audit["seed"].astype(str) == "pooled")]
    keep = pooled[["model_id", "max_abs_spearman", "linear_r2_prob_from_shortcuts", "shortcut_only_label_auc_audit_only"]]
    keep = keep.rename(
        columns={
            "max_abs_spearman": "max_abs_prediction_shortcut_spearman",
            "linear_r2_prob_from_shortcuts": "linear_r2_prob_from_shortcuts_val",
            "shortcut_only_label_auc_audit_only": "shortcut_only_label_auc_audit_only_val",
        }
    )
    frame = frame.copy()
    frame["model_id"] = frame["model_id"].map(normalize_model_id)
    return frame.merge(keep, on="model_id", how="left")


def gate_rows_from_comparison(
    frame: pd.DataFrame,
    min_val_auc_delta: float,
    max_residual_spearman: float,
    shortcut_audit: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame = attach_shortcut_audit(frame, shortcut_audit)
    best_val = float(frame["val_auc_mean"].max())
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        model_id = str(row["model_id"])
        val_auc = float(row["val_auc_mean"])
        delta = val_auc - best_val
        residual = maybe_float(row.get("max_abs_prediction_shortcut_spearman"))
        reasons: List[str] = []
        if model_id == "strict_mvp":
            status = "REFERENCE"
            reasons.append("reference baseline, not a promotion candidate")
        elif delta >= -abs(min_val_auc_delta):
            status = "PASS_CURRENT"
            reasons.append("best validation AUC under current completed results")
        else:
            status = "FAIL"
            reasons.append("does not beat current best validation AUC")
        if residual is not None and residual > max_residual_spearman:
            status = "FAIL"
            reasons.append("residual shortcut association exceeds configured threshold")
        rows.append(
            {
                "model_id": model_id,
                "gate_status": status,
                "val_auc_mean": val_auc,
                "val_auc_delta_vs_best": delta,
                "test_usage": "reporting_only",
                "max_abs_prediction_shortcut_spearman": residual,
                "decision_reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["gate_status", "val_auc_mean"], ascending=[True, False])


def gate_rows_from_proposal(path: Path, min_val_auc_delta: float, max_residual_spearman: float) -> pd.DataFrame:
    frame = read_table(path)
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        reasons: List[str] = []
        status = "PASS"
        model_id = str(row.get("model_id", row.get("name", "proposal")))
        if str(row.get("static_compile_passed", "")).lower() not in {"true", "1", "yes", "passed"}:
            status = "FAIL"
            reasons.append("static compile not marked passed")
        if str(row.get("formal_training_complete", "")).lower() not in {"true", "1", "yes", "complete", "completed"}:
            status = "FAIL"
            reasons.append("formal training not marked complete")
        val_delta = maybe_float(row.get("val_auc_delta_vs_current_main"))
        if val_delta is None or val_delta < min_val_auc_delta:
            status = "FAIL"
            reasons.append("validation AUC delta does not pass threshold")
        residual = maybe_float(row.get("max_abs_prediction_shortcut_spearman"))
        if residual is not None and residual > max_residual_spearman:
            status = "FAIL"
            reasons.append("residual shortcut association exceeds threshold")
        if not reasons:
            reasons.append("all configured gate checks passed")
        rows.append(
            {
                "model_id": model_id,
                "gate_status": status,
                "val_auc_delta_vs_current_main": val_delta,
                "test_usage": "reporting_only",
                "max_abs_prediction_shortcut_spearman": residual,
                "decision_reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def write_decision_gate_doc(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Phase C3 Decision Gate",
        "",
        "Promotion rule:",
        "",
        "1. Candidate must be analysis-clean: no label, split, data, or test-selection changes.",
        "2. Candidate must pass static compile and complete the planned run scope.",
        "3. Candidate must beat the current main candidate by validation AUC.",
        "4. Test metrics are reporting-only.",
        "5. Shortcut residual audit must not show a new structural shortcut concern.",
        "",
        "Decision summary:",
        "",
    ]
    if summary.empty:
        lines.append("No decision rows were generated.")
    else:
        lines.extend(["| model_id | status | validation delta | reasons |", "| --- | --- | ---: | --- |"])
        delta_col = "val_auc_delta_vs_best" if "val_auc_delta_vs_best" in summary.columns else "val_auc_delta_vs_current_main"
        for _, row in summary.iterrows():
            delta = row.get(delta_col)
            delta_text = "" if pd.isna(delta) else f"{float(delta):.4f}"
            lines.append(f"| {row['model_id']} | {row['gate_status']} | {delta_text} | {row['decision_reasons']} |")
    (out_dir / "decision_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_report_if_exists(path: Path) -> str:
    if not path.exists():
        return f"Missing report: {path.name}"
    text = path.read_text(encoding="utf-8").strip()
    return text[:3000]


def write_final_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Phase C3 Final Report",
        "",
        "Phase C3 consolidated completed results only. No new formal training is claimed here.",
        "",
    ]
    if not summary.empty:
        current = summary[summary["gate_status"].astype(str).isin(["PASS_CURRENT", "PASS"])]
        if not current.empty:
            best = current.iloc[0]
            lines.append(f"Current main candidate: `{best['model_id']}`.")
        else:
            lines.append("No new candidate passed the decision gate.")
        lines.append("")
    lines.extend(
        [
            "## Model Comparison",
            "",
            read_report_if_exists(out_dir / "model_comparison_report.md"),
            "",
            "## C1 Evidence Effects",
            "",
            read_report_if_exists(out_dir / "c1_evidence_effects_report.md"),
            "",
            "## Shortcut Residual Audit",
            "",
            read_report_if_exists(out_dir / "shortcut_residual_audit_report.md"),
            "",
            "## Decision",
            "",
            read_report_if_exists(out_dir / "decision_gate.md"),
            "",
            "Future training recommendation: only launch a new formal run after a candidate passes the documented pilot gate on validation AUC and shortcut residual checks.",
        ]
    )
    (out_dir / "phase_c3_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply DMEA-HT Phase C3 decision gate to completed results or a proposal.")
    parser.add_argument("--comparison-table")
    parser.add_argument("--proposal")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shortcut-audit", help="Optional shortcut_residual_audit.csv to merge into comparison decisions.")
    parser.add_argument("--min-val-auc-delta", type=float, default=0.0)
    parser.add_argument("--max-residual-spearman", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    if args.comparison_table:
        frames.append(
            gate_rows_from_comparison(
                read_table(Path(args.comparison_table)),
                args.min_val_auc_delta,
                args.max_residual_spearman,
                args.shortcut_audit,
            )
        )
    if args.proposal:
        frames.append(gate_rows_from_proposal(Path(args.proposal), args.min_val_auc_delta, args.max_residual_spearman))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary.to_csv(out_dir / "decision_gate_summary.csv", index=False)
    write_decision_gate_doc(out_dir, summary)
    write_final_report(out_dir, summary)
    print(summary.to_string(index=False) if not summary.empty else "No decision rows generated.")


if __name__ == "__main__":
    main()
