#!/usr/bin/env python3
"""Fixed-epoch C66 outer refit and its one permitted outer-validation evaluation."""

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
from scripts.train_phase_c66_source import refit as source_refit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 42, 3407))
    return parser.parse_args()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def route_health(gradient: pd.DataFrame, updates: pd.DataFrame, route: str, epoch: int) -> bool:
    expected = common.expected_groups("route", route)
    selected = gradient[gradient["epoch"].astype(int) == int(epoch)]
    gradient_ok = True
    for group in expected:
        rows = selected[selected["optimizer_group"].astype(str) == group]
        gradient_ok &= len(rows) > 0 and float(rows["max_norm"].max()) > 0.0
    summary = updates[updates["kind"].astype(str) == "module_summary"]
    update_ok = True
    for group in expected:
        rows = summary[summary["optimizer_group"].astype(str) == group]
        update_ok &= len(rows) == 1 and bool_value(rows.iloc[0]["updated"]) and bool_value(rows.iloc[0]["finite"])
    return bool(gradient_ok and update_ok)


def decision(config: Mapping[str, Any], fold: int) -> Dict[str, Any]:
    path = common.fold_decision_path(config, fold)
    if not path.exists():
        raise FileNotFoundError(f"C66 outer refit requires frozen route decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C66_INNER_ROUTE_FROZEN" or payload.get("outer_validation_read") is not False:
        raise RuntimeError("C66 outer refit route decision is not valid")
    return payload


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    frozen = decision(config, args.fold)
    root = common.outer_refit_dir(config, args.fold, args.seed)
    route_dir = root / "route"
    status_path = route_dir / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE":
            print(json.dumps({"status": "C66_OUTER_REFIT_ALREADY_COMPLETE", "run_dir": str(route_dir)}))
            return
        raise RuntimeError(f"C66 outer route contains a non-complete prior status: {status_path}")

    source_refit(config, args.fold, args.seed)
    source_path = root / "source" / "checkpoints" / "source_refit.pt"
    source_payload = common.torch_load(source_path)
    if source_payload.get("stage") != "outer_refit_source" or int(source_payload.get("seed", -1)) != args.seed or int(source_payload.get("outer_fold", -1)) != args.fold:
        raise RuntimeError("C66 outer route source checkpoint scope mismatch")

    selected_route = str(frozen["selected_route"])
    fixed_epochs = int(frozen["route_epochs_by_seed"][str(args.seed)])
    rows = common.fold_outer_rows(config, common.development_rows(config), args.fold)
    target_device = common.device()
    model = common.route_model(config, args.seed, target_device, selected_route)
    model.load_fold_local_source(source_payload)
    model.configure_route(selected_route)
    optimizer, inventory, optimizer_audit = common.optimizer_and_inventory(model, config, "route", selected_route)
    before = common.initial_state(model)
    loaders = common.build_loaders(config, rows, args.seed + args.fold * 1000, ("train", "val"))
    train_result = common.fixed_epoch_train(model, loaders["train"], optimizer, target_device, "route", fixed_epochs)
    # The frozen outer-validation partition is evaluated exactly once here, after fixed-epoch refitting.
    outer_val = common.run_epoch(model, loaders["val"], None, target_device, "route")
    updates = common.parameter_update_audit(model, before, "route")
    route_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(route_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(route_dir / "optimizer_parameter_groups.csv", index=False)
    common.initialization_inventory(model, "route", str(source_path)).to_csv(
        route_dir / "initialization_inventory.csv", index=False
    )
    train_result["history"].to_csv(route_dir / "metrics_by_epoch.csv", index=False)
    train_result["gradient"].to_csv(route_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(route_dir / "parameter_update_audit.csv", index=False)
    metric = {
        "seed": args.seed,
        "outer_fold": args.fold,
        "route": selected_route,
        "fixed_epoch": fixed_epochs,
        "source_fixed_epoch": int(frozen["source_epochs_by_seed"][str(args.seed)]),
        **outer_val["metrics"],
    }
    pd.DataFrame([metric]).to_csv(route_dir / "metrics.csv", index=False)
    common.save_predictions(outer_val["predictions"], route_dir / "predictions" / "outer_val_predictions.csv")
    common.torch_save(
        {
            "phase": "C66-LFFC",
            "stage": "outer_refit_route",
            "route": selected_route,
            "seed": args.seed,
            "outer_fold": args.fold,
            "fixed_epoch": fixed_epochs,
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "source_checkpoint": str(source_path),
            "generic_provenance": common.generic_provenance(config, verify_hashes=False),
            "task_checkpoint_loaded": False,
            "test_loaded": False,
        },
        route_dir / "checkpoints" / "outer_refit_route.pt",
    )
    healthy = route_health(train_result["gradient"], updates, selected_route, fixed_epochs)
    common.save_run_status(
        status_path,
        {
            "phase": "C66-LFFC",
            "stage": "outer_refit_route",
            "status": "COMPLETE",
            "route": selected_route,
            "seed": args.seed,
            "outer_fold": args.fold,
            "fixed_epoch": fixed_epochs,
            "outer_validation_evaluated_once": True,
            "training_health_pass": healthy,
            "test_loaded": False,
            "test_rows_read": 0,
            "task_checkpoint_loaded": False,
        },
    )
    print(json.dumps({"status": "C66_OUTER_REFIT_COMPLETE", "fold": args.fold, "seed": args.seed, "route": selected_route, "outer_val_auc": outer_val["metrics"]["AUC"]}))


if __name__ == "__main__":
    main()
