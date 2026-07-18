#!/usr/bin/env python3
"""Collect C66 leakage-free OOF predictions and apply the fixed final-training gate."""

from __future__ import annotations

import argparse
import itertools
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


def oof_report_dir(config: Mapping[str, Any]) -> Path:
    return protocol.resolve_path(config["project"]["oof_report_dir"])


def route_status(config: Mapping[str, Any], fold: int, seed: int) -> Dict[str, Any]:
    path = common.outer_refit_dir(config, fold, seed) / "route" / "run_status.json"
    if not path.exists():
        raise FileNotFoundError(path)
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("stage") != "outer_refit_route":
        raise RuntimeError(f"C66 outer refit did not complete: {path}")
    if status.get("test_loaded") is not False or not bool_value(status.get("outer_validation_evaluated_once")):
        raise RuntimeError(f"C66 outer refit status violates outer/Test protocol: {path}")
    return status


def read_fold_prediction(config: Mapping[str, Any], fold: int, seed: int) -> pd.DataFrame:
    run_dir = common.outer_refit_dir(config, fold, seed) / "route"
    route_status(config, fold, seed)
    frame = pd.read_csv(run_dir / "predictions" / "outer_val_predictions.csv", dtype={"patient_id": str})
    if frame["patient_id"].duplicated().any() or len(frame) == 0:
        raise RuntimeError(f"C66 invalid outer prediction file: {run_dir}")
    metric = pd.read_csv(run_dir / "metrics.csv")
    if len(metric) != 1:
        raise RuntimeError(f"C66 missing outer metric row: {run_dir}")
    observed = reporting.auc(frame["label"].to_numpy(dtype=int), frame["final_prob"].to_numpy(dtype=float))
    if not np.isclose(observed, float(metric.iloc[0]["AUC"]), atol=1e-12):
        raise RuntimeError(f"C66 outer prediction/metric AUC mismatch: {run_dir}")
    frame["outer_fold"] = fold
    frame["seed"] = seed
    return frame.sort_values("patient_id").reset_index(drop=True)


def fold_decisions(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    decisions = []
    for fold in range(5):
        path = common.fold_decision_path(config, fold)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "C66_INNER_ROUTE_FROZEN" or payload.get("outer_validation_read") is not False:
            raise RuntimeError(f"C66 inner decision is not frozen before OOF collection: {path}")
        decisions.append(payload)
    return decisions


def final_contract(config: Mapping[str, Any], decisions: list[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {route: int(sum(str(decision["selected_route"]) == route for decision in decisions)) for route in ("F", "E")}
    if counts["F"] >= int(config["final_selection"]["majority_route_fold_count"]):
        selected_route, reason = "F", "selected_in_at_least_three_outer_folds"
    elif counts["E"] >= int(config["final_selection"]["majority_route_fold_count"]):
        selected_route, reason = "E", "selected_in_at_least_three_outer_folds"
    else:
        mean_f = float(np.mean([decision["route_summaries"]["F"]["mean_inner_val_auc"] for decision in decisions]))
        mean_e = float(np.mean([decision["route_summaries"]["E"]["mean_inner_val_auc"] for decision in decisions]))
        if abs(mean_f - mean_e) < float(config["final_selection"]["pooled_inner_mean_tie_delta"]):
            selected_route, reason = "F", "pooled_inner_mean_tie_prefers_fewer_parameter_route_f"
        else:
            selected_route, reason = ("F" if mean_f > mean_e else "E"), "higher_pooled_inner_mean_auc"

    source_epochs = {
        str(seed): int(np.median([int(decision["source_epochs_by_seed"][str(seed)]) for decision in decisions]))
        for seed in protocol.SEEDS
    }
    route_epochs = {
        str(seed): int(
            np.median(
                [int(decision["route_epochs_by_seed_by_route"][selected_route][str(seed)]) for decision in decisions]
            )
        )
        for seed in protocol.SEEDS
    }
    return {
        "phase": "C66-LFFC",
        "status": "C66_FINAL_TRAINING_AUTHORIZED",
        "selected_route": selected_route,
        "selection_reason": reason,
        "outer_route_selection_counts": counts,
        "source_fixed_epochs_by_seed": source_epochs,
        "route_fixed_epochs_by_seed": route_epochs,
        "test_locked_until_final_checkpoints_complete": True,
        "test_loaded": False,
        "task_checkpoint_loaded": False,
        "historical_prediction_or_representation_input": False,
    }


def main() -> None:
    args = parse_args()
    config = protocol.load_c66_config(args.config)
    common.require_runtime_preflight(config)
    decisions = fold_decisions(config)
    inventory = protocol.read_c64_development_inventory(config)
    expected_ids = set(inventory["patient_id"].astype(str).tolist())
    metric_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    prediction_by_seed: Dict[int, pd.DataFrame] = {}
    health_pass = True
    for seed in protocol.SEEDS:
        parts = [read_fold_prediction(config, fold, seed) for fold in range(5)]
        frame = pd.concat(parts, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
        if len(frame) != protocol.DEVELOPMENT_PATIENT_COUNT or set(frame["patient_id"].astype(str)) != expected_ids or frame["patient_id"].duplicated().any():
            raise RuntimeError(f"C66 OOF coverage failed for seed {seed}")
        metrics = reporting.binary_counts(frame["label"].to_numpy(dtype=int), frame["final_prob"].to_numpy(dtype=float))
        oof_auc = reporting.auc(frame["label"].to_numpy(dtype=int), frame["final_prob"].to_numpy(dtype=float))
        fold_auc = []
        for fold, group in frame.groupby("outer_fold"):
            value = reporting.auc(group["label"].to_numpy(dtype=int), group["final_prob"].to_numpy(dtype=float))
            fold_auc.append(value)
            fold_rows.append({"seed": seed, "outer_fold": int(fold), "AUC": value, **reporting.binary_counts(group["label"].to_numpy(dtype=int), group["final_prob"].to_numpy(dtype=float))})
            status = route_status(config, int(fold), seed)
            health_pass &= bool_value(status.get("training_health_pass"))
        shortcut_rows.append(reporting.shortcut_row(frame, config, {"seed": seed, "split": "oof"}))
        metric_rows.append(
            {
                "seed": seed,
                "OOF_AUC": oof_auc,
                "min_outer_fold_AUC": float(min(fold_auc)),
                "mean_outer_fold_AUC": float(np.mean(fold_auc)),
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "Balanced_ACC": metrics["Balanced_ACC"],
                "TP": metrics["TP"],
                "FN": metrics["FN"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "positive_sensitivity_damage": 1.0 - metrics["Sensitivity"],
                "positive_negative_gap": float(frame.loc[frame["label"] == 1, "final_prob"].mean() - frame.loc[frame["label"] == 0, "final_prob"].mean()),
                "pairwise_inversion_count": int((frame.loc[frame["label"] == 1, "final_prob"].to_numpy()[:, None] < frame.loc[frame["label"] == 0, "final_prob"].to_numpy()[None, :]).sum()),
                "prediction_std": float(frame["final_prob"].std(ddof=1)),
            }
        )
        prediction_by_seed[seed] = frame

    oof_metrics = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["seed", "outer_fold"]).reset_index(drop=True)
    shortcut_metrics = pd.DataFrame(shortcut_rows).sort_values("seed").reset_index(drop=True)
    correlation_rows = []
    for left, right in itertools.combinations(protocol.SEEDS, 2):
        first = prediction_by_seed[left]
        second = prediction_by_seed[right]
        if not np.array_equal(first["patient_id"].to_numpy(dtype=str), second["patient_id"].to_numpy(dtype=str)):
            raise RuntimeError("C66 cross-seed OOF patient alignment failed")
        correlation_rows.append(
            {
                "seed_left": left,
                "seed_right": right,
                "prediction_spearman": float(first["final_prob"].corr(second["final_prob"], method="spearman")),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    mean_oof = float(oof_metrics["OOF_AUC"].mean())
    std_oof = float(oof_metrics["OOF_AUC"].std(ddof=1))
    oof_gate = dict(config["oof_gate"])
    checks = {
        "mean_oof_auc": mean_oof >= float(oof_gate["mean_auc_min"]),
        "two_seed_oof_auc": int((oof_metrics["OOF_AUC"] >= float(oof_gate["seed_auc_min"])).sum()) >= int(oof_gate["seed_auc_count_min"]),
        "oof_std": std_oof <= float(oof_gate["std_auc_max"]),
        "any_seed_oof_auc": bool((oof_metrics["OOF_AUC"] >= float(oof_gate["any_seed_auc_min"])).any()),
        "minimum_outer_fold_auc": float(fold_metrics["AUC"].min()) >= float(oof_gate["minimum_fold_auc_min"]),
        "mean_cross_seed_spearman": float(correlations["prediction_spearman"].mean()) >= float(oof_gate["mean_cross_seed_spearman_min"]),
        "any_seed_positive_sensitivity_damage": bool((oof_metrics["positive_sensitivity_damage"] <= float(oof_gate["any_seed_positive_sensitivity_damage_max"])).any()),
        "shortcut_safety": bool(shortcut_metrics["shortcut_safety_pass"].astype(bool).all()),
        "training_health": bool(health_pass),
        "finite_nonconstant_predictions": bool(np.isfinite(oof_metrics["prediction_std"]).all() and (oof_metrics["prediction_std"] > 0.0).all()),
        "test_unread": True,
    }
    passed = all(checks.values())
    status = "C66_FINAL_TRAINING_AUTHORIZED" if passed else "C66_LEAKAGE_FREE_COADAPTATION_FAIL"
    report_dir = oof_report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    oof_metrics.to_csv(report_dir / "c66_oof_metrics_by_seed.csv", index=False)
    fold_metrics.to_csv(report_dir / "c66_oof_fold_metrics.csv", index=False)
    shortcut_metrics.to_csv(report_dir / "c66_oof_shortcut_audit.csv", index=False)
    correlations.to_csv(report_dir / "c66_oof_cross_seed_spearman.csv", index=False)
    oof_metrics[["seed", "Sensitivity", "positive_sensitivity_damage", "TP", "FN"]].to_csv(report_dir / "c66_positive_preservation.csv", index=False)
    payload: Dict[str, Any] = {
        "phase": "C66-LFFC",
        "stage": "leakage_free_outer_oof_collection",
        "status": status,
        "checks": checks,
        "mean_oof_auc": mean_oof,
        "std_oof_auc": std_oof,
        "minimum_outer_fold_auc": float(fold_metrics["AUC"].min()),
        "mean_cross_seed_prediction_spearman": float(correlations["prediction_spearman"].mean()),
        "test_loaded": False,
        "test_rows_read": 0,
        "task_checkpoint_loaded": False,
        "historical_prediction_or_representation_input": False,
        "failure_action": "KEEP_DEMA_C17_STRICT_BEST" if not passed else "proceed_to_frozen_final_contract",
    }
    protocol.write_json(report_dir / "c66_oof_decision.json", payload)
    if passed:
        contract = final_contract(config, decisions)
        final_dir = protocol.resolve_path(config["project"]["final_output_dir"])
        protocol.write_json(final_dir / "final_training_contract.json", contract)
    print(json.dumps({"status": status, "mean_oof_auc": mean_oof, "std_oof_auc": std_oof, "test_loaded": False}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
