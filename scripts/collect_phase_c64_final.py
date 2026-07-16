#!/usr/bin/env python3
"""Evaluate the frozen C64 final checkpoints on Test exactly once for reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402
from scripts import c64_reporting as reporting  # noqa: E402


TEST_AUC_TARGET = 0.90
TEST_SEED_COUNT = 2
TEST_STD_MAX = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c64_final.yaml")
    return parser.parse_args()


def contract(config: Dict[str, Any]) -> Dict[str, Any]:
    path = common.resolve_path(config["project"]["report_dir"]) / "c64_final_training_contract.json"
    if not path.exists():
        raise RuntimeError(f"C64 final contract missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C64_FINAL_TRAINING_CONTRACT_FROZEN" or payload.get("test_loaded", True):
        raise RuntimeError(f"C64 final contract is not frozen: {payload.get('status')}")
    return payload


def final_health(run_dir: Path, candidate: str, seed: int) -> Dict[str, Any]:
    inventory = pd.read_csv(run_dir / "trainable_parameter_inventory.csv")
    active = inventory[inventory["requires_grad"].astype(bool)]
    expected = set(common.expected_trainable_groups(candidate))
    actual = set(active["optimizer_group"].astype(str))
    updates = pd.read_csv(run_dir / "parameter_update_audit.csv")
    summaries = updates[updates["kind"].astype(str) == "module_summary"]
    update_pass = all(
        len(summaries[summaries["optimizer_group"].astype(str) == group]) > 0
        and summaries.loc[summaries["optimizer_group"].astype(str) == group, "updated"].map(reporting.bool_value).all()
        and summaries.loc[summaries["optimizer_group"].astype(str) == group, "finite"].map(reporting.bool_value).all()
        for group in expected
    )
    gradient = pd.read_csv(run_dir / "gradient_connectivity.csv")
    gradient_pass = True
    for group in set(active["module_group"].astype(str)):
        rows = gradient[gradient["module_group"].astype(str) == group]
        gradient_pass &= len(rows) > 0 and (pd.to_numeric(rows["gradient_norm"], errors="coerce") > 0.0).any()
    passed = actual == expected and update_pass and gradient_pass
    return {
        "candidate": candidate,
        "seed": seed,
        "expected_optimizer_groups": sorted(expected),
        "actual_optimizer_groups": sorted(actual),
        "parameter_update_pass": bool(update_pass),
        "gradient_connectivity_pass": bool(gradient_pass),
        "health_pass": bool(passed),
    }


def main() -> None:
    args = parse_args()
    config = common.load_c64_config(args.config)
    frozen = contract(config)
    candidate = str(frozen["selected_candidate"])
    rows = common.manifest_rows(config)
    output_root = common.resolve_path(config["project"]["final_output_dir"])
    report_dir = common.resolve_path(config["project"]["report_dir"])
    seed_dirs = {seed: output_root / "seed_runs" / f"seed_{seed}" for seed in common.SEEDS}
    for seed, run_dir in seed_dirs.items():
        status_path = run_dir / "run_status.json"
        if not status_path.exists():
            raise RuntimeError(f"C64 final run status missing: {run_dir}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE" or status.get("test_loaded", True):
            raise RuntimeError(f"C64 final development training is not complete or was already evaluated: {run_dir}")
        checkpoint = run_dir / f"{candidate}_seed_{seed}_final.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"C64 final checkpoint missing: {checkpoint}")

    # This is the only function in the C64 pipeline that constructs a Test loader.
    metric_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    for seed, run_dir in seed_dirs.items():
        checkpoint = run_dir / f"{candidate}_seed_{seed}_final.pt"
        metric = common.evaluate_test_seed(config, candidate, seed, rows, checkpoint, run_dir)
        metric_rows.append(metric)
        prediction = reporting.read_prediction(run_dir / "predictions" / f"test_predictions_seed_{seed}.csv")
        if len(prediction) != 84 or int(prediction["label"].sum()) != 42:
            raise RuntimeError(f"C64 Test cardinality failed for seed {seed}")
        shortcut_rows.append(reporting.shortcut_row(prediction, config, {"candidate": candidate, "seed": seed, "split": "test"}))
        health_rows.append(final_health(run_dir, candidate, seed))
        status_path = run_dir / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"test_loaded": True, "test_evaluation": "reporting_only_after_frozen_contract"})
        reporting.write_json(status_path, status)

    metrics = pd.DataFrame(metric_rows).sort_values("seed")
    shortcuts = pd.DataFrame(shortcut_rows).sort_values("seed")
    health = pd.DataFrame(health_rows).sort_values("seed")
    auc_values = metrics["AUC"].to_numpy(dtype=float)
    mean_auc = float(auc_values.mean())
    std_auc = float(auc_values.std(ddof=1))
    seed_count = int((auc_values >= TEST_AUC_TARGET).sum())
    health_pass = bool(health["health_pass"].astype(bool).all())
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(bool).all())
    target_pass = bool(mean_auc >= TEST_AUC_TARGET and seed_count >= TEST_SEED_COUNT and std_auc <= TEST_STD_MAX and health_pass and shortcut_pass)
    deployment_seed = 0
    decision = {
        "phase": "C64-STCV",
        "status": "C64_AUC_TARGET_REACHED" if target_pass else "C64_FINAL_AUC_TARGET_NOT_REACHED",
        "selected_candidate": candidate,
        "test_AUC_mean": mean_auc,
        "test_AUC_std": std_auc,
        "test_seed_count_at_least_0.9000": seed_count,
        "target_checks": {
            "mean_auc_pass": mean_auc >= TEST_AUC_TARGET,
            "seed_auc_pass": seed_count >= TEST_SEED_COUNT,
            "std_pass": std_auc <= TEST_STD_MAX,
            "shortcut_pass": shortcut_pass,
            "parameter_health_pass": health_pass,
        },
        "deployment_seed_predeclared": deployment_seed,
        "deployment_checkpoint": str(seed_dirs[deployment_seed] / f"{candidate}_seed_{deployment_seed}_final.pt") if target_pass else None,
        "test_loaded": True,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "threshold_tuned": False,
        "ensemble": False,
        "prediction_averaging": False,
        "one_checkpoint_one_model_one_forward": True,
        "final_contract_frozen_before_test": True,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "c64_test_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": candidate,
                "Test_AUC_mean": mean_auc,
                "Test_AUC_std": std_auc,
                "seed_count_at_least_0.9000": seed_count,
            }
        ]
    ).to_csv(report_dir / "c64_test_metrics_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c64_final_shortcut_audit.csv", index=False)
    health.to_csv(report_dir / "c64_final_training_health.csv", index=False)
    reporting.write_json(report_dir / "c64_final_route_decision.json", decision)
    reporting.write_markdown(
        report_dir / "phase_c64_dema_final_report.md",
        [
            "# DMEA-HT Phase C64-STCV Final Report",
            "",
            f"- Decision: `{decision['status']}`.",
            f"- Candidate: `{candidate}`; fixed development epochs: `{frozen['selected_epochs_by_seed']}`.",
            f"- Reporting-only Test AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`; seeds at or above 0.9000: `{seed_count}/3`.",
            f"- Predeclared deployment seed: `{deployment_seed}`; checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
            "- Test was loaded once per final checkpoint only after Stage-A route and CV epoch contract were frozen.",
            "- Test was not used for checkpoint, hyperparameter, epoch, threshold, or ensemble selection.",
            "- Deployment contract is one checkpoint, one model, one forward, with no ensemble or prediction averaging.",
        ],
    )
    print(json.dumps({"status": decision["status"], "test_AUC_mean": mean_auc, "test_AUC_std": std_auc}))


if __name__ == "__main__":
    main()
