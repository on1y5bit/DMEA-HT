#!/usr/bin/env python3
"""Freeze C66's inner route and per-seed epochs before any outer validation is read."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as protocol  # noqa: E402
from scripts import c66_training_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    return parser.parse_args()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def complete_status(path: Path, expected_stage: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("stage") != expected_stage:
        raise RuntimeError(f"C66 expected completed {expected_stage} status: {path}")
    if not bool_value(status.get("training_health_pass")):
        raise RuntimeError(f"C66 inner training health failed: {path}")
    if status.get("test_loaded") is not False:
        raise RuntimeError(f"C66 inner run unexpectedly loaded Test: {path}")
    return status


def source_metric(config: Mapping[str, Any], fold: int, seed: int) -> Dict[str, Any]:
    run_dir = common.inner_source_dir(config, fold, seed)
    status = complete_status(run_dir / "run_status.json", "inner_source")
    frame = pd.read_csv(run_dir / "metrics.csv")
    if len(frame) != 1:
        raise RuntimeError(f"C66 inner source metric row missing: {run_dir}")
    row = frame.iloc[0].to_dict()
    if int(row["best_epoch"]) != int(status["best_epoch"]):
        raise RuntimeError("C66 inner source status/metric epoch mismatch")
    return row


def route_metric(config: Mapping[str, Any], fold: int, route: str, seed: int) -> Dict[str, Any]:
    run_dir = common.inner_route_dir(config, fold, route, seed)
    status = complete_status(run_dir / "run_status.json", "inner_route")
    frame = pd.read_csv(run_dir / "metrics.csv")
    inventory = pd.read_csv(run_dir / "trainable_parameter_inventory.csv")
    if len(frame) != 1:
        raise RuntimeError(f"C66 inner route metric row missing: {run_dir}")
    row = frame.iloc[0].to_dict()
    if int(row["best_epoch"]) != int(status["best_epoch"]):
        raise RuntimeError("C66 inner route status/metric epoch mismatch")
    row["trainable_parameter_count"] = int(
        inventory.loc[inventory["requires_grad"].map(bool_value), "parameter_count"].sum()
    )
    return row


def route_summary(rows: list[Dict[str, Any]], route: str) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    return {
        "route": route,
        "mean_inner_val_auc": float(frame["AUC"].mean()),
        "std_inner_val_auc": float(frame["AUC"].std(ddof=1)),
        "min_inner_val_auc": float(frame["AUC"].min()),
        "mean_positive_sensitivity_damage": float(frame["positive_sensitivity_damage"].mean()),
        "mean_trainable_parameter_count": float(frame["trainable_parameter_count"].mean()),
    }


def choose_route(summary_f: Mapping[str, Any], summary_e: Mapping[str, Any]) -> tuple[str, str]:
    mean_delta = float(summary_f["mean_inner_val_auc"]) - float(summary_e["mean_inner_val_auc"])
    if abs(mean_delta) >= 0.003:
        return ("F" if mean_delta > 0.0 else "E"), "higher_mean_inner_validation_auc"
    ranking = sorted(
        [summary_f, summary_e],
        key=lambda row: (
            float(row["std_inner_val_auc"]),
            -float(row["min_inner_val_auc"]),
            float(row["mean_positive_sensitivity_damage"]),
            float(row["mean_trainable_parameter_count"]),
            str(row["route"]),
        ),
    )
    return str(ranking[0]["route"]), "predeclared_tie_break_std_min_positive_damage_parameter_count"


def write_summary(config: Mapping[str, Any]) -> None:
    rows = []
    for fold in range(5):
        path = common.fold_decision_path(config, fold)
        if not path.exists():
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "outer_fold": fold,
                "selected_route": decision["selected_route"],
                "selection_rule": decision["selection_rule"],
                "route_f_mean_inner_auc": decision["route_summaries"]["F"]["mean_inner_val_auc"],
                "route_e_mean_inner_auc": decision["route_summaries"]["E"]["mean_inner_val_auc"],
            }
        )
    if rows:
        report_dir = protocol.report_dir(config)
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("outer_fold").to_csv(report_dir / "c66_inner_route_decisions.csv", index=False)


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    source_rows = [source_metric(config, args.fold, seed) for seed in protocol.SEEDS]
    route_rows: Dict[str, list[Dict[str, Any]]] = {
        route: [route_metric(config, args.fold, route, seed) for seed in protocol.SEEDS]
        for route in ("F", "E")
    }
    summaries = {route: route_summary(rows, route) for route, rows in route_rows.items()}
    selected_route, selection_rule = choose_route(summaries["F"], summaries["E"])
    route_epochs_by_seed_by_route = {
        route: {str(row["seed"]): int(row["best_epoch"]) for row in route_rows[route]}
        for route in ("F", "E")
    }
    route_by_seed = route_epochs_by_seed_by_route[selected_route]
    source_by_seed = {str(row["seed"]): int(row["best_epoch"]) for row in source_rows}
    decision = {
        "phase": "C66-LFFC",
        "stage": "inner_route_selection",
        "status": "C66_INNER_ROUTE_FROZEN",
        "outer_fold": args.fold,
        "selected_route": selected_route,
        "selection_rule": selection_rule,
        "route_summaries": summaries,
        "source_epochs_by_seed": source_by_seed,
        "route_epochs_by_seed": route_by_seed,
        "route_epochs_by_seed_by_route": route_epochs_by_seed_by_route,
        "source_inner_metrics": source_rows,
        "route_inner_metrics": route_rows,
        "outer_validation_read": False,
        "test_loaded": False,
        "test_rows_read": 0,
    }
    path = common.fold_decision_path(config, args.fold)
    protocol.write_json(path, decision)
    write_summary(config)
    print(json.dumps({"status": decision["status"], "fold": args.fold, "selected_route": selected_route, "test_loaded": False}))


if __name__ == "__main__":
    main()
