#!/usr/bin/env python3
"""Collect C64 Stage-A Validation evidence and select one safe route."""

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

from scripts import c64_common as common  # noqa: E402
from scripts import c64_reporting as reporting  # noqa: E402


POSITIVE_DAMAGE_MEAN_LIMIT = 0.10
POSITIVE_DAMAGE_SEED_LIMIT = 0.10
STAGE_A_STD_LIMIT = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-config", default="configs/dema_ht_c64_stage_a_head_only.yaml")
    parser.add_argument("--projector-config", default="configs/dema_ht_c64_stage_a_projector_cbpi.yaml")
    parser.add_argument("--full-config", default="configs/dema_ht_c64_stage_a_full_finetune.yaml")
    return parser.parse_args()


def run_path(config: Mapping[str, Any], candidate: str, seed: int) -> Path:
    return common.resolve_path(config["project"]["stage_a_output_dir"]) / candidate / "seed_runs" / f"seed_{seed}"


def select_candidate(summary: pd.DataFrame) -> Dict[str, Any] | None:
    safe = summary[summary["candidate_safety_pass"].astype(bool)].copy()
    if safe.empty:
        return None
    maximum = float(safe["validation_AUC_mean"].max())
    near = safe[(maximum - safe["validation_AUC_mean"]) < 0.003].copy()
    if len(near) > 1:
        near = near.sort_values(
            [
                "validation_AUC_std",
                "min_seed_AUC",
                "mean_positive_sensitivity_damage",
                "mean_inversion_delta",
                "trainable_parameter_count",
                "candidate",
            ],
            ascending=[True, False, True, True, True, True],
        )
        selected = near.iloc[0]
        rule = "mean_auc_within_0.003_then_lower_std_then_min_seed_auc_then_positive_damage_then_inversions_then_parameter_count"
    else:
        safe = safe.sort_values(
            ["validation_AUC_mean", "validation_AUC_std", "min_seed_AUC", "mean_positive_sensitivity_damage", "mean_inversion_delta", "trainable_parameter_count", "candidate"],
            ascending=[False, True, False, True, True, True, True],
        )
        selected = safe.iloc[0]
        rule = "highest_mean_validation_auc_then_lower_std_then_min_seed_auc_then_positive_damage_then_inversions_then_parameter_count"
    return {"candidate": str(selected["candidate"]), "rule": rule, "best_seed": int(selected["best_seed"])}


def main() -> None:
    args = parse_args()
    config_paths = [
        common.resolve_path(args.head_config),
        common.resolve_path(args.projector_config),
        common.resolve_path(args.full_config),
    ]
    configs = [common.load_c64_config(path) for path in config_paths]
    by_candidate = {str(config["candidate"]): config for config in configs}
    if set(by_candidate) != set(common.CANDIDATES):
        raise RuntimeError(f"C64 Stage-A candidate set mismatch: {sorted(by_candidate)}")
    base_config = configs[0]
    report_dir = common.resolve_path(base_config["project"]["report_dir"])
    output_root = common.resolve_path(base_config["project"]["stage_a_output_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    inventory_parts: List[pd.DataFrame] = []
    optimizer_parts: List[pd.DataFrame] = []
    initialization_parts: List[pd.DataFrame] = []
    for candidate in common.CANDIDATES:
        config = by_candidate[candidate]
        for seed in common.SEEDS:
            run_dir = run_path(config, candidate, seed)
            metric = reporting.metric_from_run(run_dir)
            prediction = reporting.read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
            if len(prediction) != 94 or int(prediction["label"].sum()) != 47:
                raise RuntimeError(f"C64 Stage-A Validation cardinality failed: {candidate}/{seed}")
            metric.update({"candidate": candidate, "seed": seed, "run_dir": str(run_dir)})
            metric_rows.append(metric)
            shortcut_rows.append(reporting.shortcut_row(prediction, config, {"candidate": candidate, "seed": seed}))
            baseline_path = reporting.historical_c61_path(config, seed, "val")
            baseline = reporting.read_prediction(baseline_path)
            positive_rows.append(reporting.positive_and_inversion_row(prediction, baseline, {"candidate": candidate, "seed": seed, "baseline": "C61"}))
            health = reporting.parameter_health(run_dir, candidate)
            health.update({"candidate": candidate, "seed": seed})
            health_rows.append(health)
            inventory = pd.read_csv(run_dir / "trainable_parameter_inventory.csv")
            inventory.insert(0, "seed", seed)
            inventory_parts.append(inventory)
            optimizer = pd.read_csv(run_dir / "optimizer_parameter_groups.csv")
            optimizer.insert(0, "seed", seed)
            optimizer_parts.append(optimizer)
            initialization_parts.append(pd.read_csv(run_dir / "initialization_inventory.csv"))

    metrics = pd.DataFrame(metric_rows).sort_values(["candidate", "seed"])
    shortcuts = pd.DataFrame(shortcut_rows).sort_values(["candidate", "seed"])
    positive = pd.DataFrame(positive_rows).sort_values(["candidate", "seed"])
    health = pd.DataFrame(health_rows).sort_values(["candidate", "seed"])
    metrics_by_seed = metrics.merge(shortcuts, on=["candidate", "seed"], how="left").merge(positive, on=["candidate", "seed"], how="left").merge(health, on=["candidate", "seed"], how="left", suffixes=("", "_health"))
    summaries: List[Dict[str, Any]] = []
    for candidate, frame in metrics_by_seed.groupby("candidate", sort=True):
        trainable_count = int(
            pd.concat([part for part in inventory_parts if str(part["candidate"].iloc[0]) == candidate], ignore_index=True)
            .query("requires_grad == True")["parameter_count"].sum()
        )
        best_row = frame.sort_values(["AUC", "seed"], ascending=[False, True]).iloc[0]
        auc_std = float(frame["AUC"].std(ddof=1))
        positive_pass = bool(
            float(frame["positive_sensitivity_damage"].mean()) <= POSITIVE_DAMAGE_MEAN_LIMIT
            and int((frame["positive_sensitivity_damage"] > POSITIVE_DAMAGE_SEED_LIMIT).sum()) <= 1
        )
        ranking_pass = bool(np.isfinite(frame["inversion_delta"].to_numpy(dtype=float)).all())
        shortcut_pass = bool(frame["shortcut_safety_pass"].astype(bool).all())
        health_pass = bool(frame["health_pass"].astype(bool).all())
        summary = {
            "candidate": candidate,
            "validation_AUC_mean": float(frame["AUC"].mean()),
            "validation_AUC_std": auc_std,
            "best_seed": int(best_row["seed"]),
            "best_seed_AUC": float(best_row["AUC"]),
            "min_seed_AUC": float(frame["AUC"].min()),
            "mean_positive_sensitivity_damage": float(frame["positive_sensitivity_damage"].mean()),
            "mean_inversion_delta": float(frame["inversion_delta"].mean()),
            "max_positive_sensitivity_damage": float(frame["positive_sensitivity_damage"].max()),
            "max_inversion_delta": int(frame["inversion_delta"].max()),
            "max_shortcut_only_AUC": float(frame["selected_structure_shortcut_only_label_AUC"].max()),
            "max_abs_prediction_shortcut_spearman": float(frame["max_abs_prediction_selected_structure_spearman"].max()),
            "trainable_parameter_count": trainable_count,
            "validation_std_pass": bool(np.isfinite(auc_std) and auc_std <= STAGE_A_STD_LIMIT),
            "positive_safety_pass": positive_pass,
            "ranking_safety_pass": ranking_pass,
            "shortcut_safety_pass": shortcut_pass,
            "training_health_pass": health_pass,
        }
        summary["candidate_safety_pass"] = bool(
            summary["validation_std_pass"]
            and positive_pass
            and ranking_pass
            and shortcut_pass
            and health_pass
            and np.isfinite(summary["validation_AUC_mean"])
        )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values("candidate").reset_index(drop=True)
    selection = select_candidate(summary_frame)
    authorized = selection is not None
    decision = {
        "phase": "C64-STCV",
        "status": "C64_STAGE_A_ROUTE_SELECTED" if authorized else "C64_STAGE_A_NO_SAFE_CANDIDATE",
        "selected_candidate": selection["candidate"] if selection else None,
        "selected_seed": selection["best_seed"] if selection else None,
        "selection_rule": selection["rule"] if selection else None,
        "candidate_summaries": summaries,
        "thresholds": {
            "stage_a_validation_std_max": STAGE_A_STD_LIMIT,
            "shortcut_only_auc_max": reporting.SHORTCUT_MAX_AUC,
            "prediction_shortcut_spearman_max": reporting.SHORTCUT_MAX_SPEARMAN,
            "positive_sensitivity_damage_mean_max": POSITIVE_DAMAGE_MEAN_LIMIT,
            "positive_sensitivity_damage_seed_max": POSITIVE_DAMAGE_SEED_LIMIT,
            "positive_damage_seed_count_max": 1,
            "inversion_delta": "audit_and_tie_break_only",
        },
        "test_loaded": False,
        "test_used_for_selection": False,
        "ensemble": False,
        "prediction_averaging": False,
        "initialization": "c61_validation_checkpoint_warm_start",
    }

    metrics_by_seed.to_csv(report_dir / "c64_stage_a_metrics_by_seed.csv", index=False)
    summary_frame.to_csv(report_dir / "c64_stage_a_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c64_stage_a_shortcut_audit.csv", index=False)
    positive.to_csv(report_dir / "c64_stage_a_positive_preservation.csv", index=False)
    positive.to_csv(report_dir / "c64_stage_a_pairwise_inversion_summary.csv", index=False)
    health.to_csv(report_dir / "c64_stage_a_training_health.csv", index=False)
    pd.concat(inventory_parts, ignore_index=True).to_csv(report_dir / "c64_stage_a_freeze_inventory.csv", index=False)
    pd.concat(optimizer_parts, ignore_index=True).to_csv(report_dir / "c64_stage_a_optimizer_parameter_groups.csv", index=False)
    pd.concat(initialization_parts, ignore_index=True).to_csv(report_dir / "c64_stage_a_initialization_inventory.csv", index=False)
    reporting.write_json(report_dir / "c64_stage_a_route_decision.json", decision)
    reporting.write_markdown(
        report_dir / "c64_stage_a_route_decision.md",
        [
            "# C64 Stage-A Route Decision",
            "",
            f"- Status: `{decision['status']}`.",
            f"- Selected candidate: `{decision['selected_candidate'] or 'none'}`.",
            f"- Validation/Test isolation: `test_loaded={decision['test_loaded']}`, `test_used_for_selection={decision['test_used_for_selection']}`.",
            "- Candidate selection is based only on fixed-split Validation AUC and predeclared safety tie-breaks.",
            "- All Stage-A runs were initialized from the matching C61 Validation-selected checkpoint; C63/C62/from-base inputs are excluded.",
        ],
    )
    print(json.dumps({"status": decision["status"], "selected_candidate": decision["selected_candidate"]}))
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
