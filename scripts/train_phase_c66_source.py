#!/usr/bin/env python3
"""Train or fixed-epoch refit C66's fold-local public source stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as protocol  # noqa: E402
from scripts import c66_training_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    parser.add_argument("--mode", choices=("inner", "refit"), required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 42, 3407))
    return parser.parse_args()


def completed(run_dir: Path) -> bool:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") == "COMPLETE":
        print(json.dumps({"status": "C66_SOURCE_ALREADY_COMPLETE", "run_dir": str(run_dir)}))
        return True
    raise RuntimeError(f"C66 source run directory contains a non-complete prior status: {status_path}")


def health(gradient: pd.DataFrame, updates: pd.DataFrame, stage: str, selected_epoch: int | None) -> bool:
    if selected_epoch is None:
        raise ValueError("C66 source health requires a selected epoch")
    return common.training_health_pass(gradient, updates, stage, int(selected_epoch))


def checkpoint_payload(
    config: Mapping[str, Any], model: Any, seed: int, fold: int, stage: str, epoch: int
) -> Dict[str, Any]:
    return {
        "phase": "C66-LFFC",
        "stage": stage,
        "seed": int(seed),
        "outer_fold": int(fold),
        "source_epoch": int(epoch),
        "sources": {name: value.detach().cpu() for name, value in model.sources.state_dict().items()},
        "source_evidence_stack": {
            name: value.detach().cpu() for name, value in model.source_evidence_stack.state_dict().items()
        },
        "generic_provenance": common.generic_provenance(config, verify_hashes=False),
        "task_checkpoint_loaded": False,
        "historical_prediction_or_representation_input": False,
        "test_loaded": False,
    }


def inner(config: Dict[str, Any], fold: int, seed: int) -> None:
    run_dir = common.inner_source_dir(config, fold, seed)
    if completed(run_dir):
        return
    rows = common.fold_inner_rows(config, common.development_rows(config), fold)
    target_device = common.device()
    model = common.source_model(config, seed, target_device)
    optimizer, inventory, optimizer_audit = common.optimizer_and_inventory(model, config, "source")
    before = common.initial_state(model)
    loaders = common.build_loaders(config, rows, seed + fold * 1000, ("train", "val"))
    section = dict(config["source_learning"])
    result = common.early_stop_train(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        target_device,
        "source",
        int(section["max_epochs"]),
        int(section["patience"]),
    )
    updates = common.parameter_update_audit(model, before, "source")
    run_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(run_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(run_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(model, "source").to_csv(run_dir / "initialization_inventory.csv", index=False)
    result["history"].to_csv(run_dir / "metrics_by_epoch.csv", index=False)
    result["gradient"].to_csv(run_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(run_dir / "parameter_update_audit.csv", index=False)
    metric = {"seed": seed, "outer_fold": fold, "best_epoch": result["best_epoch"], **result["val"]["metrics"]}
    pd.DataFrame([metric]).to_csv(run_dir / "metrics.csv", index=False)
    common.save_predictions(result["val"]["predictions"], run_dir / "predictions" / "inner_val_predictions.csv")
    common.torch_save(
        checkpoint_payload(config, model, seed, fold, "inner_source", result["best_epoch"]),
        run_dir / "checkpoints" / "source_best.pt",
    )
    healthy = health(result["gradient"], updates, "source", result["best_epoch"])
    common.save_run_status(
        run_dir / "run_status.json",
        {
            "phase": "C66-LFFC",
            "stage": "inner_source",
            "status": "COMPLETE",
            "seed": seed,
            "outer_fold": fold,
            "best_epoch": result["best_epoch"],
            "training_health_pass": healthy,
            "test_loaded": False,
            "test_rows_read": 0,
            "task_checkpoint_loaded": False,
        },
    )
    print(json.dumps({"status": "C66_INNER_SOURCE_COMPLETE", "fold": fold, "seed": seed, "best_epoch": result["best_epoch"]}))


def refit(config: Dict[str, Any], fold: int, seed: int) -> None:
    root = common.outer_refit_dir(config, fold, seed)
    run_dir = root / "source"
    if completed(run_dir):
        return
    decision_path = common.fold_decision_path(config, fold)
    if not decision_path.exists():
        raise RuntimeError(f"C66 source refit requires the frozen inner route decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    fixed_epochs = int(decision["source_epochs_by_seed"][str(seed)])
    rows = common.fold_outer_rows(config, common.development_rows(config), fold)
    target_device = common.device()
    model = common.source_model(config, seed, target_device)
    optimizer, inventory, optimizer_audit = common.optimizer_and_inventory(model, config, "source")
    before = common.initial_state(model)
    loaders = common.build_loaders(config, rows, seed + fold * 1000, ("train",))
    result = common.fixed_epoch_train(model, loaders["train"], optimizer, target_device, "source", fixed_epochs)
    updates = common.parameter_update_audit(model, before, "source")
    run_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(run_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(run_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(model, "source").to_csv(run_dir / "initialization_inventory.csv", index=False)
    result["history"].to_csv(run_dir / "metrics_by_epoch.csv", index=False)
    result["gradient"].to_csv(run_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(run_dir / "parameter_update_audit.csv", index=False)
    pd.DataFrame([{"seed": seed, "outer_fold": fold, "fixed_epoch": fixed_epochs}]).to_csv(run_dir / "metrics.csv", index=False)
    common.torch_save(
        checkpoint_payload(config, model, seed, fold, "outer_refit_source", fixed_epochs),
        run_dir / "checkpoints" / "source_refit.pt",
    )
    healthy = health(result["gradient"], updates, "source", fixed_epochs)
    common.save_run_status(
        run_dir / "run_status.json",
        {
            "phase": "C66-LFFC",
            "stage": "outer_refit_source",
            "status": "COMPLETE",
            "seed": seed,
            "outer_fold": fold,
            "fixed_epoch": fixed_epochs,
            "training_health_pass": healthy,
            "test_loaded": False,
            "test_rows_read": 0,
            "task_checkpoint_loaded": False,
        },
    )
    print(json.dumps({"status": "C66_OUTER_SOURCE_REFIT_COMPLETE", "fold": fold, "seed": seed, "fixed_epoch": fixed_epochs}))


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    if args.mode == "inner":
        inner(config, args.fold, args.seed)
    else:
        refit(config, args.fold, args.seed)


if __name__ == "__main__":
    main()
