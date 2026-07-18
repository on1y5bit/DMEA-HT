#!/usr/bin/env python3
"""Reconcile completed C66 inner-source health from immutable raw audit files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as protocol  # noqa: E402
from scripts import c66_training_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 42, 3407), choices=(0, 42, 3407))
    return parser.parse_args()


def reconcile_one(config: Dict[str, Any], fold: int, seed: int) -> Dict[str, Any]:
    run_dir = common.inner_source_dir(config, fold, seed)
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("phase") != "C66-LFFC" or status.get("stage") != "inner_source" or status.get("status") != "COMPLETE":
        raise RuntimeError(f"C66 source reconciliation requires a completed inner-source status: {status_path}")
    if status.get("test_loaded") is not False or int(status.get("test_rows_read", -1)) != 0:
        raise RuntimeError(f"C66 source reconciliation refuses a run with Test access: {status_path}")

    gradient = pd.read_csv(run_dir / "gradient_connectivity.csv")
    updates = pd.read_csv(run_dir / "parameter_update_audit.csv")
    details = common.training_health_details(
        gradient,
        updates,
        "source",
        int(status["best_epoch"]),
    )
    if not bool(details["training_health_pass"].astype(bool).all()):
        raise RuntimeError(f"C66 source raw-audit health remains failed: {run_dir}")

    previous_health = bool(status.get("training_health_pass"))
    status.update(
        {
            "training_health_pass": True,
            "training_health_rule": "selected_epoch_nonzero_group_gradients_and_finite_nonzero_group_updates",
            "health_reconciled_from_raw_audits": True,
        }
    )
    protocol.write_json(status_path, status)
    payload = {
        "phase": "C66-LFFC",
        "stage": "inner_source_health_reconciliation",
        "status": "C66_INNER_SOURCE_HEALTH_RECONCILED",
        "outer_fold": int(fold),
        "seed": int(seed),
        "prior_training_health_pass": previous_health,
        "training_health_pass": True,
        "test_loaded": False,
        "test_rows_read": 0,
        "details": details.to_dict(orient="records"),
    }
    protocol.write_json(run_dir / "health_reconciliation.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    results = [reconcile_one(config, args.fold, int(seed)) for seed in args.seeds]
    print(
        json.dumps(
            {
                "status": "C66_INNER_SOURCE_HEALTH_RECONCILIATION_COMPLETE",
                "fold": int(args.fold),
                "seeds": [int(seed) for seed in args.seeds],
                "test_loaded": False,
                "reconciled_runs": len(results),
            }
        )
    )


if __name__ == "__main__":
    main()
