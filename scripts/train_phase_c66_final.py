#!/usr/bin/env python3
"""Train one frozen-contract C66 final checkpoint per scientific seed, without Test."""

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
    parser.add_argument("--seed", type=int, required=True, choices=(0, 42, 3407))
    return parser.parse_args()


def contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = protocol.resolve_path(config["project"]["final_output_dir"]) / "final_training_contract.json"
    if not path.exists():
        raise FileNotFoundError(f"C66 final training requires the OOF-authorized contract: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C66_FINAL_TRAINING_AUTHORIZED" or payload.get("test_loaded") is not False:
        raise RuntimeError("C66 final training contract is not authorized or already touched Test")
    return payload


def health(gradient: pd.DataFrame, updates: pd.DataFrame, stage: str, route: str | None, epoch: int) -> bool:
    return common.training_health_pass(gradient, updates, stage, int(epoch), route)


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    frozen = contract(config)
    seed_dir = common.final_seed_dir(config, args.seed)
    status_path = seed_dir / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE_TRAINED_TEST_LOCKED":
            print(json.dumps({"status": "C66_FINAL_ALREADY_TRAINED", "seed": args.seed}))
            return
        raise RuntimeError(f"C66 final seed directory contains a non-complete prior status: {status_path}")

    selected_route = str(frozen["selected_route"])
    source_epochs = int(frozen["source_fixed_epochs_by_seed"][str(args.seed)])
    route_epochs = int(frozen["route_fixed_epochs_by_seed"][str(args.seed)])
    rows = common.all_development_train_rows(common.development_rows(config))
    target_device = common.device()

    source = common.source_model(config, args.seed, target_device)
    source_optimizer, source_inventory, source_optimizer_audit = common.optimizer_and_inventory(source, config, "source")
    source_before = common.initial_state(source)
    source_loader = common.build_loaders(config, rows, args.seed + 50000, ("train",))["train"]
    source_result = common.fixed_epoch_train(source, source_loader, source_optimizer, target_device, "source", source_epochs)
    source_updates = common.parameter_update_audit(source, source_before, "source")
    source_checkpoint = {
        "phase": "C66-LFFC",
        "stage": "final_source",
        "seed": args.seed,
        "source_epoch": source_epochs,
        "sources": {name: value.detach().cpu() for name, value in source.sources.state_dict().items()},
        "source_evidence_stack": {name: value.detach().cpu() for name, value in source.source_evidence_stack.state_dict().items()},
        "generic_provenance": common.generic_provenance(config, verify_hashes=False),
        "task_checkpoint_loaded": False,
        "test_loaded": False,
    }

    route = common.route_model(config, args.seed, target_device, selected_route)
    route.load_fold_local_source(source_checkpoint)
    route.configure_route(selected_route)
    route_optimizer, route_inventory, route_optimizer_audit = common.optimizer_and_inventory(route, config, "route", selected_route)
    route_before = common.initial_state(route)
    route_loader = common.build_loaders(config, rows, args.seed + 50000, ("train",))["train"]
    route_result = common.fixed_epoch_train(route, route_loader, route_optimizer, target_device, "route", route_epochs)
    route_updates = common.parameter_update_audit(route, route_before, "route")

    source_dir = seed_dir / "source"
    route_dir = seed_dir / "route"
    source_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    source_inventory.to_csv(source_dir / "trainable_parameter_inventory.csv", index=False)
    source_optimizer_audit.to_csv(source_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(source, "source").to_csv(source_dir / "initialization_inventory.csv", index=False)
    source_result["history"].to_csv(source_dir / "metrics_by_epoch.csv", index=False)
    source_result["gradient"].to_csv(source_dir / "gradient_connectivity.csv", index=False)
    source_updates.to_csv(source_dir / "parameter_update_audit.csv", index=False)
    common.torch_save(source_checkpoint, source_dir / "checkpoints" / "final_source.pt")

    route_inventory.to_csv(route_dir / "trainable_parameter_inventory.csv", index=False)
    route_optimizer_audit.to_csv(route_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(route, "route", str(source_dir / "checkpoints" / "final_source.pt")).to_csv(route_dir / "initialization_inventory.csv", index=False)
    route_result["history"].to_csv(route_dir / "metrics_by_epoch.csv", index=False)
    route_result["gradient"].to_csv(route_dir / "gradient_connectivity.csv", index=False)
    route_updates.to_csv(route_dir / "parameter_update_audit.csv", index=False)
    common.torch_save(
        {
            "phase": "C66-LFFC",
            "stage": "final_route",
            "seed": args.seed,
            "route": selected_route,
            "source_fixed_epoch": source_epochs,
            "route_fixed_epoch": route_epochs,
            "model": {name: value.detach().cpu() for name, value in route.state_dict().items()},
            "generic_provenance": common.generic_provenance(config, verify_hashes=False),
            "task_checkpoint_loaded": False,
            "test_loaded": False,
        },
        route_dir / "checkpoints" / "final_model.pt",
    )
    source_health = health(source_result["gradient"], source_updates, "source", None, source_epochs)
    route_health = health(route_result["gradient"], route_updates, "route", selected_route, route_epochs)
    common.save_run_status(
        status_path,
        {
            "phase": "C66-LFFC",
            "stage": "final_fixed_epoch_training",
            "status": "COMPLETE_TRAINED_TEST_LOCKED",
            "seed": args.seed,
            "selected_route": selected_route,
            "source_fixed_epoch": source_epochs,
            "route_fixed_epoch": route_epochs,
            "source_training_health_pass": source_health,
            "route_training_health_pass": route_health,
            "test_loaded": False,
            "test_rows_read": 0,
            "task_checkpoint_loaded": False,
        },
    )
    print(json.dumps({"status": "C66_FINAL_SEED_TRAINED_TEST_LOCKED", "seed": args.seed, "route": selected_route, "test_loaded": False}))


if __name__ == "__main__":
    main()
