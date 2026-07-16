#!/usr/bin/env python3
"""Unattended C64 Stage-A -> CV -> final -> reporting driver."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-config", default="configs/dema_ht_c64_stage_a_head_only.yaml")
    parser.add_argument("--projector-config", default="configs/dema_ht_c64_stage_a_projector_cbpi.yaml")
    parser.add_argument("--full-config", default="configs/dema_ht_c64_stage_a_full_finetune.yaml")
    parser.add_argument("--cv-config", default="configs/dema_ht_c64_cv.yaml")
    parser.add_argument("--final-config", default="configs/dema_ht_c64_final.yaml")
    parser.add_argument("--skip-gate", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    head = common.load_c64_config(args.head_config)
    report_dir = common.resolve_path(head["project"]["report_dir"])
    status_path = report_dir / "c64_autorun_status.json"
    status: Dict[str, Any] = {
        "phase": "C64-STCV",
        "status": "RUNNING",
        "started_at": timestamp(),
        "completed_steps": [],
        "test_loaded": False,
    }

    def write_status() -> None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def run_step(name: str, command: Sequence[str]) -> None:
        status["current_step"] = name
        write_status()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(list(command), cwd=REPO_ROOT, env=environment, check=True)
        status["completed_steps"].append(name)
        status["current_step"] = None
        write_status()

    write_status()
    run_step(
        "build_folds",
        [sys.executable, str(REPO_ROOT / "scripts" / "build_phase_c64_folds.py"), "--config", str(common.resolve_path(args.cv_config))],
    )
    if not args.skip_gate:
        run_step(
            "gate",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "gate_phase_c64.py"),
                "--head-config",
                str(common.resolve_path(args.head_config)),
                "--projector-config",
                str(common.resolve_path(args.projector_config)),
                "--full-config",
                str(common.resolve_path(args.full_config)),
                "--cv-config",
                str(common.resolve_path(args.cv_config)),
                "--final-config",
                str(common.resolve_path(args.final_config)),
            ],
        )
    else:
        gate_path = report_dir / "c64_gate.json"
        if not gate_path.exists() or json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "C64_STAGED_TUNING_CV_AUTHORIZED":
            raise RuntimeError("--skip-gate requires an authorized C64 gate")
        status["completed_steps"].append("gate_prechecked")
        write_status()
    run_step(
        "stage_a",
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_phase_c64_stage_a.py"),
            "--stage",
            "direct-multiseed",
            "--config",
            str(common.resolve_path(args.head_config)),
            "--projector-config",
            str(common.resolve_path(args.projector_config)),
            "--full-config",
            str(common.resolve_path(args.full_config)),
        ],
    )
    run_step(
        "cv",
        [sys.executable, str(REPO_ROOT / "scripts" / "train_phase_c64_cv.py"), "--stage", "direct-multiseed", "--config", str(common.resolve_path(args.cv_config))],
    )
    run_step(
        "final_development_training",
        [sys.executable, str(REPO_ROOT / "scripts" / "train_phase_c64_final.py"), "--stage", "direct-multiseed", "--config", str(common.resolve_path(args.final_config))],
    )
    run_step(
        "reporting_test_once",
        [sys.executable, str(REPO_ROOT / "scripts" / "collect_phase_c64_final.py"), "--config", str(common.resolve_path(args.final_config))],
    )
    status.update({"status": "COMPLETE", "current_step": None, "finished_at": timestamp(), "test_loaded": True})
    write_status()
    print(json.dumps({"status": status["status"], "completed_steps": status["completed_steps"]}))


if __name__ == "__main__":
    main()
