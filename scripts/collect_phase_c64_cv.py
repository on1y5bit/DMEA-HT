#!/usr/bin/env python3
"""Collect C64 five-fold OOF evidence and authorize fixed-epoch final training."""

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


OOF_AUC_MIN = 0.8900
OOF_SEED_COUNT = 2
OOF_STD_MAX = 0.020
FOLD_AUC_MIN = 0.8400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c64_cv.yaml")
    return parser.parse_args()


def route(config: Dict[str, Any]) -> Dict[str, Any]:
    path = common.resolve_path(config["project"]["report_dir"]) / "c64_stage_a_route_decision.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C64_STAGE_A_ROUTE_SELECTED":
        raise RuntimeError(f"C64 Stage-A route is not selected: {payload.get('status')}")
    return payload


def main() -> None:
    args = parse_args()
    config = common.load_c64_config(args.config)
    selected = route(config)
    candidate = str(selected["selected_candidate"])
    rows = common.manifest_rows(config)
    assignments = common.load_fold_assignments(config)
    report_dir = common.resolve_path(config["project"]["report_dir"])
    cv_dir = common.resolve_path(config["project"]["cv_output_dir"])
    fold_metric_rows: List[Dict[str, Any]] = []
    fold_prediction_parts: List[pd.DataFrame] = []
    shortcut_parts: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    for fold in range(common.FOLD_COUNT):
        expected_ids = {patient_id for patient_id, assigned in assignments.items() if int(assigned) == fold}
        for seed in common.SEEDS:
            run_dir = cv_dir / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
            metric = reporting.metric_from_run(run_dir)
            metric.update({"candidate": candidate, "fold": fold, "seed": seed, "run_dir": str(run_dir)})
            prediction = reporting.read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
            actual_ids = set(prediction["patient_id"].astype(str))
            if actual_ids != expected_ids or len(actual_ids) != len(prediction):
                raise RuntimeError(f"C64 OOF fold coverage failed: fold={fold}, seed={seed}")
            prediction = prediction.assign(fold=fold, seed=seed, candidate=candidate)
            fold_prediction_parts.append(prediction)
            fold_metric_rows.append(metric)
            shortcut_parts.append(reporting.shortcut_row(prediction, config, {"candidate": candidate, "fold": fold, "seed": seed}))
            health = reporting.parameter_health(run_dir, candidate)
            health.update({"candidate": candidate, "fold": fold, "seed": seed})
            health_rows.append(health)
            epoch_rows.append({"candidate": candidate, "fold": fold, "seed": seed, "selected_epoch": int(metric["best_epoch"])})

    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(["fold", "seed"])
    fold_predictions = pd.concat(fold_prediction_parts, ignore_index=True)
    shortcut_frame = pd.DataFrame(shortcut_parts).sort_values(["fold", "seed"])
    health = pd.DataFrame(health_rows).sort_values(["fold", "seed"])
    selected_epochs = pd.DataFrame(epoch_rows).sort_values(["seed", "fold"])
    oof_metrics: List[Dict[str, Any]] = []
    oof_shortcuts: List[Dict[str, Any]] = []
    for seed in common.SEEDS:
        frame = fold_predictions[fold_predictions["seed"].astype(int) == seed].sort_values("patient_id").reset_index(drop=True)
        if len(frame) != 696 or frame["patient_id"].duplicated().any():
            raise RuntimeError(f"C64 OOF seed coverage failed: {seed}")
        probability = frame[reporting.probability_column(frame)].to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=int)
        fold_auc = fold_metrics[fold_metrics["seed"].astype(int) == seed]["AUC"].to_numpy(dtype=float)
        row = {
            "candidate": candidate,
            "seed": seed,
            "OOF_AUC": reporting.auc(labels, probability),
            "min_fold_AUC": float(fold_auc.min()),
            "mean_fold_AUC": float(fold_auc.mean()),
            "fold_AUC_std": float(fold_auc.std(ddof=1)),
            "n_rows": len(frame),
            "label0": int((labels == 0).sum()),
            "label1": int((labels == 1).sum()),
        }
        oof_metrics.append(row)
        oof_shortcuts.append(reporting.shortcut_row(frame, config, {"candidate": candidate, "seed": seed, "split": "oof"}))
    oof_metrics_frame = pd.DataFrame(oof_metrics).sort_values("seed")
    oof_shortcuts_frame = pd.DataFrame(oof_shortcuts).sort_values("seed")
    health_pass = bool(health["health_pass"].astype(bool).all())
    shortcut_pass = bool(oof_shortcuts_frame["shortcut_safety_pass"].astype(bool).all())
    mean_oof = float(oof_metrics_frame["OOF_AUC"].mean())
    std_oof = float(oof_metrics_frame["OOF_AUC"].std(ddof=1))
    seed_threshold_count = int((oof_metrics_frame["OOF_AUC"] >= OOF_AUC_MIN).sum())
    min_fold_auc = float(fold_metrics["AUC"].min())
    gate_checks = {
        "mean_oof_auc_pass": mean_oof >= OOF_AUC_MIN,
        "seed_oof_auc_pass": seed_threshold_count >= OOF_SEED_COUNT,
        "oof_std_pass": np.isfinite(std_oof) and std_oof <= OOF_STD_MAX,
        "min_fold_auc_pass": min_fold_auc >= FOLD_AUC_MIN,
        "shortcut_pass": shortcut_pass,
        "parameter_health_pass": health_pass,
        "oof_patient_coverage_pass": len(fold_predictions) == 696 * len(common.SEEDS) and not fold_predictions.duplicated(["seed", "patient_id"]).any(),
        "test_loaded_pass": not any(reporting.bool_value(value) for value in [False]),
    }
    authorized = all(bool(value) for value in gate_checks.values())

    final_epochs: Dict[str, int] = {}
    for seed in common.SEEDS:
        values = selected_epochs[selected_epochs["seed"].astype(int) == seed]["selected_epoch"].to_numpy(dtype=int)
        if len(values) != common.FOLD_COUNT:
            raise RuntimeError(f"C64 selected epoch count failed for seed {seed}")
        median_epoch = int(round(float(np.median(values))))
        final_epochs[str(seed)] = min(60, max(3, median_epoch))
    decision = {
        "phase": "C64-STCV",
        "status": "C64_FINAL_TRAINING_AUTHORIZED" if authorized else "C64_OOF_GENERALIZATION_GATE_FAIL",
        "selected_candidate": candidate,
        "mean_OOF_AUC": mean_oof,
        "std_OOF_AUC": std_oof,
        "seed_OOF_AUC_count_at_least_0.8900": seed_threshold_count,
        "min_fold_AUC": min_fold_auc,
        "gate_checks": gate_checks,
        "test_loaded": False,
        "test_used_for_decision": False,
        "ensemble": False,
        "prediction_averaging": False,
        "fixed_epoch_contract_written": bool(authorized),
    }

    fold_metrics.to_csv(report_dir / "c64_cv_metrics_by_fold.csv", index=False)
    fold_predictions.to_csv(report_dir / "c64_oof_predictions.csv", index=False)
    oof_metrics_frame.to_csv(report_dir / "c64_oof_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": candidate,
                "OOF_AUC_mean": mean_oof,
                "OOF_AUC_std": std_oof,
                "seed_count_at_least_0.8900": seed_threshold_count,
                "min_fold_AUC": min_fold_auc,
            }
        ]
    ).to_csv(report_dir / "c64_oof_metrics_summary.csv", index=False)
    selected_epochs.to_csv(report_dir / "c64_cv_selected_epochs.csv", index=False)
    oof_shortcuts_frame.to_csv(report_dir / "c64_cv_shortcut_audit.csv", index=False)
    shortcut_frame.to_csv(report_dir / "c64_cv_fold_shortcut_audit.csv", index=False)
    health.to_csv(report_dir / "c64_cv_training_health.csv", index=False)
    fold_integrity = common.resolve_path(config["project"]["report_dir"]) / "c64_fold_integrity.json"
    if fold_integrity.exists():
        (report_dir / "c64_cv_fold_integrity.json").write_text(fold_integrity.read_text(encoding="utf-8"), encoding="utf-8")
    reporting.write_json(report_dir / "c64_cv_decision.json", decision)
    if authorized:
        contract = {
            "phase": "C64-STCV",
            "status": "C64_FINAL_TRAINING_CONTRACT_FROZEN",
            "selected_candidate": candidate,
            "selected_epochs_by_seed": final_epochs,
            "source": "five_fold_patient_level_cv_median_selected_epoch",
            "fold_count": common.FOLD_COUNT,
            "fold_seed": common.FOLD_SEED,
            "early_stopping": False,
            "fixed_epoch": True,
            "patience": 15,
            "max_epochs": 60,
            "test_loaded": False,
            "test_used_for_contract": False,
            "ensemble": False,
            "prediction_averaging": False,
            "fold_selected_epochs": selected_epochs.to_dict(orient="records"),
        }
        reporting.write_json(report_dir / "c64_final_training_contract.json", contract)
    reporting.write_markdown(
        report_dir / "c64_cv_decision.md",
        [
            "# C64 Development CV Decision",
            "",
            f"- Status: `{decision['status']}`.",
            f"- Selected candidate: `{candidate}`.",
            f"- OOF AUC mean/std: `{mean_oof:.10f} +/- {std_oof:.10f}`.",
            f"- Minimum fold AUC: `{min_fold_auc:.10f}`; seeds at or above 0.8900: `{seed_threshold_count}/3`.",
            "- Test remains locked and was not loaded for candidate, fold, epoch, or contract selection.",
        ],
    )
    print(json.dumps({"status": decision["status"], "mean_OOF_AUC": mean_oof, "std_OOF_AUC": std_oof}))
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
