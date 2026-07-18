#!/usr/bin/env python3
"""Fail-closed authorization gate for C66-A's leakage-free nested protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as common  # noqa: E402


def record(checks: List[Dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def write_decision(path: Path, status: str, checks: List[Dict[str, Any]], audit: Dict[str, Any]) -> None:
    lines = [
        "# C66-A Leakage-Free Nested-CV Decision",
        "",
        f"Status: `{status}`",
        "",
        "## Protocol Result",
        "",
        "- C66-A reconstructed prior C61 Train/Validation exposure solely from C64's development-only inventory and frozen five-fold assignment artifacts.",
        "- The full manifest and locked Test records were not opened. Test was not loaded, parsed, or evaluated.",
        "- Outer-validation patients are excluded from fold-local source learning, inner validation, route selection, epoch selection, and outer-train refitting.",
        "",
        "## Gate Checks",
        "",
    ]
    for check in checks:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{marker}` {check['name']}: `{check['detail']}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- This decision authorizes the leakage-free nested-CV protocol only when it passes.",
            "- It does not certify a placeholder source initialization: the source-training implementation must record immutable public image/text source, version, and SHA256 before any C66 training starts.",
            "- No C13-C65 task checkpoint, historical prediction, or historical representation may enter C66 initialization or model input.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    try:
        config = common.load_c66_config(args.config)
        report_dir = common.report_dir(config)
        audit_path = report_dir / "c66a_prior_cv_leakage_audit.json"
        integrity_path = report_dir / "c66a_nested_split_integrity.json"
        overlap_path = report_dir / "c66a_prior_checkpoint_overlap_by_fold.csv"
        init_path = report_dir / "c66a_initialization_inventory.csv"
        required_paths = [audit_path, integrity_path, overlap_path, init_path]
        for path in required_paths:
            record(checks, f"artifact_present:{path.name}", path.exists(), str(path))

        if not all(path.exists() for path in required_paths):
            raise RuntimeError("C66-A required audit artifact is missing")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        overlap = pd.read_csv(overlap_path)
        initialization = pd.read_csv(init_path)

        record(checks, "prior_c61_exposure_reconstructed", bool(audit.get("prior_c61_exposure_reconstructed")), audit.get("prior_c61_exposure_totals"))
        record(checks, "nested_cv_isolation", bool(integrity.get("all_pass")), integrity.get("all_pass"))
        record(checks, "five_overlap_rows", len(overlap) == 5, len(overlap))
        record(checks, "prior_c61_train_total", int(overlap["prior_c61_train_patient_count"].sum()) == 602, int(overlap["prior_c61_train_patient_count"].sum()))
        record(checks, "prior_c61_validation_total", int(overlap["prior_c61_validation_patient_count"].sum()) == 94, int(overlap["prior_c61_validation_patient_count"].sum()))
        record(checks, "no_never_used_development_patients", int(overlap["never_used_by_c61_patient_count"].sum()) == 0, int(overlap["never_used_by_c61_patient_count"].sum()))
        record(checks, "test_not_loaded", audit.get("test_loaded") is False and int(audit.get("test_rows_read", -1)) == 0, {"test_loaded": audit.get("test_loaded"), "test_rows_read": audit.get("test_rows_read")})
        record(checks, "manifest_not_opened", audit.get("manifest_opened") is False, audit.get("manifest_opened"))
        record(checks, "no_task_checkpoint_loaded", audit.get("task_checkpoint_loaded") is False, audit.get("task_checkpoint_loaded"))
        record(checks, "no_historical_prediction_or_representation_input", audit.get("historical_prediction_or_representation_input") is False, audit.get("historical_prediction_or_representation_input"))
        record(checks, "all_initialization_rows_forbid_task_checkpoint", not initialization["task_checkpoint_used"].astype(bool).any(), int(initialization["task_checkpoint_used"].astype(bool).sum()))
        record(checks, "all_initialization_rows_forbid_historical_inputs", not initialization["historical_prediction_input"].astype(bool).any() and not initialization["historical_representation_input"].astype(bool).any(), {"prediction": int(initialization["historical_prediction_input"].astype(bool).sum()), "representation": int(initialization["historical_representation_input"].astype(bool).sum())})
        record(checks, "public_image_text_runtime_provenance_required", set(initialization.loc[initialization["module"].isin(["image_encoder", "text_encoder"]), "public_sha256"]) == {"runtime_preflight_required"}, initialization.loc[initialization["module"].isin(["image_encoder", "text_encoder"]), "public_sha256"].tolist())
    except Exception as exc:  # Fail closed and leave a machine-readable reason.
        audit = {"exception": str(exc)}
        record(checks, "gate_execution", False, str(exc))

    passed = all(check["passed"] for check in checks)
    status = "C66A_LEAKAGE_FREE_NESTED_CV_AUTHORIZED" if passed else "C66A_PROTOCOL_RECONSTRUCTION_FAIL"
    if "config" in locals():
        report_dir = common.report_dir(config)
    else:
        report_dir = REPO_ROOT / "analysis_reports" / "phase_c66a_dema"
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "phase": "C66-LFFC",
        "stage": "C66-A leakage-free nested-CV authorization",
        "status": status,
        "checks": checks,
        "test_loaded": False,
        "source_training_authorized_by_this_gate": False,
        "next_required_preflight": "record immutable public image/text source, version, and SHA256 before source training",
    }
    common.write_json(report_dir / "c66a_gate_decision.json", decision)
    write_decision(report_dir / "c66a_route_decision.md", status, checks, audit)
    print(json.dumps({"phase": "C66-LFFC", "status": status, "passed_checks": sum(check["passed"] for check in checks), "total_checks": len(checks), "test_loaded": False}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
