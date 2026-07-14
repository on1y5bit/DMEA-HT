#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def value_map(frame: pd.DataFrame) -> Dict[str, float]:
    return {str(row["metric"]): float(row["value"]) for _, row in frame.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the C27 visit reconstruction design decision.")
    parser.add_argument("--input-dir", default="analysis_reports/phase_c27_visit_design")
    args = parser.parse_args()
    root = Path(args.input_dir)
    coverage = pd.read_csv(root / "c27_visit_reconstruction_coverage.csv")
    invariance = json.loads((root / "c27_visit_manifest_invariance.json").read_text(encoding="utf-8"))
    values = value_map(coverage)
    visit_coverage = values["visit_report_reconstruction_coverage"]
    multi_val_coverage = values["multi_visit_validation_two_block_coverage"]
    coverage_pass = visit_coverage >= 0.80 and multi_val_coverage >= 0.70
    hard_pass = bool(invariance.get("hard_pass", False))
    if not coverage_pass:
        decision = "C27_VISIT_RECONSTRUCTION_INSUFFICIENT"
    elif not hard_pass:
        decision = "DEMA_C27_PATH_GATE_FAIL"
    else:
        decision = "C27_VISIT_RECONSTRUCTION_PASS"
    payload: Dict[str, Any] = {
        "phase": "C27-VTME",
        "decision": decision,
        "patients_reconstructed": int(values["patients"]),
        "selected_visits_reconstructed": int(values["total_selected_visits"]),
        "visit_report_coverage": visit_coverage,
        "multi_visit_validation_two_block_coverage": multi_val_coverage,
        "cross_patient_image_leakage_count": int(values["cross_patient_image_leakage_count"]),
        "split_label_invariance": bool(invariance["hard_checks"]["split_label_invariance"]),
        "manifest_sha256": invariance["visit_manifest_sha256"],
        "test_used_for_rule_design": False,
        "training_authorized": decision == "C27_VISIT_RECONSTRUCTION_PASS",
    }
    (root / "c27_visit_reconstruction_decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# C27 Visit Reconstruction Report",
        "",
        f"- decision: `{decision}`",
        f"- patients reconstructed: `{payload['patients_reconstructed']}`",
        f"- selected visits reconstructed: `{payload['selected_visits_reconstructed']}`",
        f"- visit-level report coverage: `{visit_coverage:.10f}` (required `>=0.80`)",
        f"- multi-visit validation two-block coverage: `{multi_val_coverage:.10f}` (required `>=0.70`)",
        f"- cross-patient image leakage count: `{payload['cross_patient_image_leakage_count']}`",
        f"- split/label invariance: `{payload['split_label_invariance']}`",
        f"- manifest SHA256: `{payload['manifest_sha256']}`",
        "- visit boundaries: C13 selected real visit dates only",
        "- image grouping: original C13 selected image paths grouped by source visit directory",
        "- report blocks: exact patient-date rows from `all_patients.xlsx`",
        "- dated bio: exact patient-date source row only",
        "- missing reports: empty with source reason; patient concatenated report is never copied",
        "- test role: invariance audit only; no reconstruction rule or threshold was selected on test",
        "",
        decision,
    ]
    (root / "c27_visit_reconstruction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if decision != "C27_VISIT_RECONSTRUCTION_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
