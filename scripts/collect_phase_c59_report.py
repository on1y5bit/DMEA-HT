#!/usr/bin/env python3
"""Freeze and report C59-PMESE Validation and reporting-only Test evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402
from scripts import collect_phase_c54_report as shared  # noqa: E402


REPORT_FILES = (
    "c54_metrics_by_seed.csv",
    "c54_metrics_summary.csv",
    "c54_metrics_by_epoch.csv",
    "c54_parameter_drift.csv",
    "c54_patient_diagnostics_val.csv",
    "c54_training_health.csv",
    "c54_positive_preservation.csv",
    "c54_pairwise_inversion_summary.csv",
    "c54_shortcut_audit.csv",
    "c54_validation_decision.json",
    "c54_route_decision.md",
    "phase_c54_dema_final_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c59_pmese_multiseed.yaml")
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    translated = dict(config)
    translated["c54"] = dict(config["c59"])
    return translated


def transform(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: transform(item) for key, item in value.items()}
    if isinstance(value, list):
        return [transform(item) for item in value]
    if isinstance(value, str):
        return value.replace("C54", "C59").replace("c54", "c59").replace("LRRA", "PMESE")
    return value


def materialize_c59(report_dir: Path) -> None:
    for source_name in REPORT_FILES:
        source = report_dir / source_name
        if not source.exists():
            continue
        target_name = source_name.replace("c54", "c59").replace("C54", "C59")
        target = report_dir / target_name
        if source.suffix == ".csv":
            frame = pd.read_csv(source)
            frame = frame.rename(columns={column: transform(column) for column in frame.columns})
            for column in frame.columns:
                if frame[column].dtype == object:
                    frame[column] = frame[column].map(transform)
            frame.to_csv(target, index=False)
        elif source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            target.write_text(json.dumps(transform(payload), indent=2) + "\n", encoding="utf-8")
        else:
            target.write_text(transform(source.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c59":
        raise RuntimeError("C59 report requires the formal C59 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    translated = runtime_config(config)
    if args.stage == "validation":
        decision = shared.freeze_validation_decision(translated, run_dir, report_dir)
        materialize_c59(report_dir)
        print(json.dumps({"status": "C59_VALIDATION_DECISION_FROZEN", "decision": transform(decision["decision_label"])}))
    else:
        decision = shared.write_final_report(translated, run_dir, report_dir)
        materialize_c59(report_dir)
        print(json.dumps({"status": "C59_FINAL_REPORT_COMPLETE", "decision": transform(decision["decision_label"])}))


if __name__ == "__main__":
    main()
