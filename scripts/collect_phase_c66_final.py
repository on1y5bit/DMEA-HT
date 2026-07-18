#!/usr/bin/env python3
"""Irreversibly evaluate the frozen C66 final checkpoints on Test exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_reporting as reporting  # noqa: E402
from scripts import c66_common as protocol  # noqa: E402
from scripts import c66_training_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    return parser.parse_args()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def final_dir(config: Mapping[str, Any]) -> Path:
    return protocol.resolve_path(config["project"]["final_output_dir"])


def read_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = final_dir(config) / "final_training_contract.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C66_FINAL_TRAINING_AUTHORIZED" or payload.get("test_loaded") is not False:
        raise RuntimeError("C66 final Test requires the frozen OOF-authorized final contract")
    return payload


def verify_final_seed(config: Mapping[str, Any], seed: int, selected_route: str) -> Path:
    seed_dir = common.final_seed_dir(config, seed)
    status_path = seed_dir / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE_TRAINED_TEST_LOCKED" or str(status.get("selected_route")) != selected_route:
        raise RuntimeError(f"C66 final seed is not frozen for Test: {status_path}")
    if not bool_value(status.get("source_training_health_pass")) or not bool_value(status.get("route_training_health_pass")):
        raise RuntimeError(f"C66 final seed health failed: {status_path}")
    if status.get("test_loaded") is not False:
        raise RuntimeError(f"C66 final seed had unexpected Test access: {status_path}")
    checkpoint = seed_dir / "route" / "checkpoints" / "final_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    contract = read_contract(config)
    selected_route = str(contract["selected_route"])
    checkpoints = {seed: verify_final_seed(config, seed, selected_route) for seed in protocol.SEEDS}
    lock_path = final_dir(config) / "test_evaluation_lock.json"
    if lock_path.exists():
        raise RuntimeError(f"C66 Test was already opened or a prior Test collection interrupted: {lock_path}")
    # Fail closed: after this durable marker exists, a partial Test collection is never silently retried.
    protocol.write_json(
        lock_path,
        {
            "phase": "C66-LFFC",
            "status": "C66_TEST_EVALUATION_STARTED",
            "test_loaded": True,
            "selected_route": selected_route,
            "checkpoint_paths": {str(seed): str(path) for seed, path in checkpoints.items()},
        },
    )

    rows = common.test_rows(config)
    target_device = common.device()
    metrics: List[Dict[str, Any]] = []
    for seed, checkpoint in checkpoints.items():
        payload = common.torch_load(checkpoint)
        if payload.get("stage") != "final_route" or int(payload.get("seed", -1)) != seed or str(payload.get("route")) != selected_route:
            raise RuntimeError(f"C66 final checkpoint contract mismatch: {checkpoint}")
        model = common.route_model(config, seed, target_device, selected_route)
        model.load_state_dict(payload["model"], strict=True)
        loader = common.build_loaders(config, rows, seed + 80000, ("test",))["test"]
        result = common.run_epoch(model, loader, None, target_device, "route")
        frame = result["predictions"]
        common.save_predictions(frame, common.final_seed_dir(config, seed) / "route" / "predictions" / "test_predictions.csv")
        metrics.append(
            {
                "seed": seed,
                "Test_AUC": result["metrics"]["AUC"],
                "Sensitivity": result["metrics"]["Sensitivity"],
                "Specificity": result["metrics"]["Specificity"],
                "Balanced_ACC": result["metrics"]["Balanced_ACC"],
                "TP": result["metrics"]["TP"],
                "FN": result["metrics"]["FN"],
                "TN": result["metrics"]["TN"],
                "FP": result["metrics"]["FP"],
                "positive_sensitivity_damage": result["metrics"]["positive_sensitivity_damage"],
                "pairwise_inversion_count": result["metrics"]["pairwise_inversion_count"],
            }
        )
        del model

    frame = pd.DataFrame(metrics).sort_values("seed").reset_index(drop=True)
    gate = dict(config["final_test_gate"])
    mean_auc = float(frame["Test_AUC"].mean())
    std_auc = float(frame["Test_AUC"].std(ddof=1))
    success = bool(
        mean_auc >= float(gate["mean_auc_min"])
        and int((frame["Test_AUC"] >= float(gate["seed_auc_min"])).sum()) >= int(gate["seed_auc_count_min"])
        and std_auc <= float(gate["std_auc_max"])
    )
    status = "GOAL_REACHED_DEMA_HT_TEST_AUC_090_PLUS" if success else "DEMA_C66_TEST_AUC_TARGET_NOT_REACHED"
    out_dir = final_dir(config) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "c66_final_test_metrics_by_seed.csv", index=False)
    decision = {
        "phase": "C66-LFFC",
        "stage": "single_historical_internal_holdout_test",
        "status": status,
        "selected_route": selected_route,
        "mean_test_auc": mean_auc,
        "std_test_auc": std_auc,
        "seed_auc_count_at_least_0.9000": int((frame["Test_AUC"] >= float(gate["seed_auc_min"])).sum()),
        "test_loaded": True,
        "test_evaluated_once_per_final_checkpoint": True,
        "test_based_selection": False,
        "promotion": "PROMOTE_DEMA_C66_LEAKAGE_FREE_COADAPTATION" if success else "no_test_based_retraining",
    }
    protocol.write_json(out_dir / "c66_final_test_decision.json", decision)
    protocol.write_json(
        lock_path,
        {
            "phase": "C66-LFFC",
            "status": "C66_TEST_EVALUATION_COMPLETE",
            "final_status": status,
            "test_loaded": True,
            "selected_route": selected_route,
            "mean_test_auc": mean_auc,
        },
    )
    print(json.dumps({"status": status, "mean_test_auc": mean_auc, "std_test_auc": std_auc, "test_loaded": True}))


if __name__ == "__main__":
    main()
