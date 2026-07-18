#!/usr/bin/env python3
"""Train C66 Route F or Route E on a fold-local inner split only."""

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
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--route", choices=("F", "E"), required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 42, 3407))
    return parser.parse_args()


def completed(run_dir: Path) -> bool:
    path = run_dir / "run_status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") == "COMPLETE":
        print(json.dumps({"status": "C66_INNER_ROUTE_ALREADY_COMPLETE", "run_dir": str(run_dir)}))
        return True
    raise RuntimeError(f"C66 inner route contains a non-complete prior status: {path}")


def route_health(gradient: pd.DataFrame, updates: pd.DataFrame, route: str, selected_epoch: int) -> bool:
    return common.training_health_pass(gradient, updates, "route", int(selected_epoch), route)


def source_checkpoint(config: Mapping[str, Any], fold: int, seed: int) -> Path:
    path = common.inner_source_dir(config, fold, seed) / "checkpoints" / "source_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"C66 inner route requires source checkpoint: {path}")
    return path


def validate_source_payload(config: Mapping[str, Any], payload: Mapping[str, Any], fold: int, seed: int) -> None:
    if payload.get("stage") != "inner_source" or int(payload.get("outer_fold", -1)) != fold or int(payload.get("seed", -1)) != seed:
        raise RuntimeError("C66 inner route received a source checkpoint from another fold or seed")
    expected = common.generic_provenance(config, verify_hashes=False)
    for modality in ("image", "text"):
        if payload.get("generic_provenance", {}).get(modality, {}).get("expected_sha256") != expected[modality]["expected_sha256"]:
            raise RuntimeError(f"C66 inner source provenance mismatch for {modality}")
    if payload.get("task_checkpoint_loaded") is not False:
        raise RuntimeError("C66 inner source checkpoint violates task-checkpoint contract")


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    run_dir = common.inner_route_dir(config, args.fold, args.route, args.seed)
    if completed(run_dir):
        return
    checkpoint_path = source_checkpoint(config, args.fold, args.seed)
    payload = common.torch_load(checkpoint_path)
    validate_source_payload(config, payload, args.fold, args.seed)

    rows = common.fold_inner_rows(config, common.development_rows(config), args.fold)
    target_device = common.device()
    model = common.route_model(config, args.seed, target_device, args.route)
    model.load_fold_local_source(payload)
    model.configure_route(args.route)
    optimizer, inventory, optimizer_audit = common.optimizer_and_inventory(model, config, "route", args.route)
    before = common.initial_state(model)
    loaders = common.build_loaders(config, rows, args.seed + args.fold * 1000, ("train", "val"))
    section = dict(config["route_training"])
    result = common.early_stop_train(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        target_device,
        "route",
        int(section["max_epochs"]),
        int(section["patience"]),
    )
    updates = common.parameter_update_audit(model, before, "route")
    run_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(run_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(run_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(model, "route", str(checkpoint_path)).to_csv(
        run_dir / "initialization_inventory.csv", index=False
    )
    result["history"].to_csv(run_dir / "metrics_by_epoch.csv", index=False)
    result["gradient"].to_csv(run_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(run_dir / "parameter_update_audit.csv", index=False)
    metric = {
        "seed": args.seed,
        "outer_fold": args.fold,
        "route": args.route,
        "best_epoch": result["best_epoch"],
        "source_checkpoint": str(checkpoint_path),
        **result["val"]["metrics"],
    }
    pd.DataFrame([metric]).to_csv(run_dir / "metrics.csv", index=False)
    common.save_predictions(result["val"]["predictions"], run_dir / "predictions" / "inner_val_predictions.csv")
    common.torch_save(
        {
            "phase": "C66-LFFC",
            "stage": "inner_route",
            "route": args.route,
            "outer_fold": args.fold,
            "seed": args.seed,
            "best_epoch": result["best_epoch"],
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "source_checkpoint": str(checkpoint_path),
            "generic_provenance": common.generic_provenance(config, verify_hashes=False),
            "task_checkpoint_loaded": False,
            "test_loaded": False,
        },
        run_dir / "checkpoints" / "route_best.pt",
    )
    healthy = route_health(result["gradient"], updates, args.route, result["best_epoch"])
    common.save_run_status(
        run_dir / "run_status.json",
        {
            "phase": "C66-LFFC",
            "stage": "inner_route",
            "status": "COMPLETE",
            "route": args.route,
            "outer_fold": args.fold,
            "seed": args.seed,
            "best_epoch": result["best_epoch"],
            "training_health_pass": healthy,
            "test_loaded": False,
            "test_rows_read": 0,
            "task_checkpoint_loaded": False,
        },
    )
    print(json.dumps({"status": "C66_INNER_ROUTE_COMPLETE", "fold": args.fold, "route": args.route, "seed": args.seed, "best_epoch": result["best_epoch"]}))


if __name__ == "__main__":
    main()
