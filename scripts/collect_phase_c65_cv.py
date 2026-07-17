#!/usr/bin/env python3
"""Collect C65-B OOF evidence and authorize the fixed-epoch final contract."""

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

from scripts import c64_reporting as c64  # noqa: E402
from scripts import c64_common  # noqa: E402
from scripts import c65a_common as c65a  # noqa: E402
from scripts import c65b_common as common  # noqa: E402


OOF_AUC_MIN = 0.9000
OOF_SEED_COUNT = 2
OOF_STD_MAX = 0.0200
FOLD_AUC_MIN = 0.8400
PREDICTION_SPEARMAN_MIN = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65b.yaml")
    return parser.parse_args()


def probability_column(frame: pd.DataFrame) -> str:
    return c64.probability_column(frame)


def logit_column(frame: pd.DataFrame) -> str | None:
    for name in ("logit", "final_logit"):
        if name in frame.columns:
            return name
    return None


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return c65a.safe_spearman(left, right)


def cross_seed_stability(predictions: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    frames = {
        seed: predictions[predictions["seed"].astype(int) == seed].sort_values("patient_id").reset_index(drop=True)
        for seed in common.SEEDS
    }
    rows = []
    for index, left_seed in enumerate(common.SEEDS):
        for right_seed in common.SEEDS[index + 1 :]:
            left = frames[left_seed]
            right = frames[right_seed]
            if not np.array_equal(left["patient_id"].astype(str), right["patient_id"].astype(str)):
                raise RuntimeError("C65 cross-seed patient alignment failed")
            if not np.array_equal(left["label"].astype(int), right["label"].astype(int)):
                raise RuntimeError("C65 cross-seed label alignment failed")
            labels = left["label"].to_numpy(dtype=int)
            left_prob = left[probability_column(left)].to_numpy(dtype=float)
            right_prob = right[probability_column(right)].to_numpy(dtype=float)
            left_logit_name = logit_column(left)
            right_logit_name = logit_column(right)
            left_logit = left[left_logit_name].to_numpy(dtype=float) if left_logit_name else np.log(np.clip(left_prob, 1e-7, 1 - 1e-7) / np.clip(1 - left_prob, 1e-7, 1.0))
            right_logit = right[right_logit_name].to_numpy(dtype=float) if right_logit_name else np.log(np.clip(right_prob, 1e-7, 1 - 1e-7) / np.clip(1 - right_prob, 1e-7, 1.0))
            left_error = (left_prob >= 0.5) != labels
            right_error = (right_prob >= 0.5) != labels
            left_inversion = c64.inversion_vector(labels, left_prob)
            right_inversion = c64.inversion_vector(labels, right_prob)
            error_union = left_error | right_error
            inversion_union = left_inversion | right_inversion
            rows.append(
                {
                    "seed_a": left_seed,
                    "seed_b": right_seed,
                    "patient_count": int(len(left)),
                    "probability_spearman": safe_spearman(left_prob, right_prob),
                    "logit_spearman": safe_spearman(left_logit, right_logit),
                    "error_count_a": int(left_error.sum()),
                    "error_count_b": int(right_error.sum()),
                    "error_overlap_count": int((left_error & right_error).sum()),
                    "error_jaccard": float((left_error & right_error).sum() / max(int(error_union.sum()), 1)),
                    "inversion_count_a": int(left_inversion.sum()),
                    "inversion_count_b": int(right_inversion.sum()),
                    "inversion_overlap_count": int((left_inversion & right_inversion).sum()),
                    "inversion_jaccard": float((left_inversion & right_inversion).sum() / max(int(inversion_union.sum()), 1)),
                }
            )
    result = pd.DataFrame(rows)
    return result, {
        "mean_probability_spearman": float(result["probability_spearman"].mean()),
        "mean_logit_spearman": float(result["logit_spearman"].mean()),
        "mean_error_jaccard": float(result["error_jaccard"].mean()),
        "mean_inversion_jaccard": float(result["inversion_jaccard"].mean()),
    }


def main() -> None:
    args = parse_args()
    config = common.load_c65b_config(args.config)
    gate_path = common.report_dir(config) / "c65b_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C65B_COMMON_BACKBONE_CV_AUTHORIZED":
        raise RuntimeError(f"C65-B formal CV cannot be collected before gate: {gate.get('status')}")
    rows = common.development_rows(config)
    assignments = common.fold_assignments(config)
    expected_labels = {str(row["patient_id"]): int(row["label"]) for row in rows}
    cv_dir = common.cv_dir(config)
    report_dir = common.report_dir(config)
    fold_metrics_rows = []
    prediction_parts = []
    shortcut_rows = []
    health_rows = []
    for fold in range(common.FOLD_COUNT):
        expected_ids = {patient_id for patient_id, assigned in assignments.items() if int(assigned) == fold}
        for seed in common.SEEDS:
            run_dir = cv_dir / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
            metric = c64.metric_from_run(run_dir)
            metric.update({"candidate": common.CANDIDATE, "fold": fold, "seed": seed, "backbone_seed": 42, "run_dir": str(run_dir)})
            prediction = c64.read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
            actual_ids = set(prediction["patient_id"].astype(str))
            if actual_ids != expected_ids or len(actual_ids) != len(prediction):
                raise RuntimeError(f"C65 fold coverage failed: fold={fold}, seed={seed}")
            if any(expected_labels[patient_id] != int(label) for patient_id, label in zip(prediction["patient_id"].astype(str), prediction["label"].astype(int))):
                raise RuntimeError(f"C65 fold labels failed: fold={fold}, seed={seed}")
            prediction = prediction.assign(fold=fold, seed=seed, candidate=common.CANDIDATE, backbone_seed=42)
            prediction_parts.append(prediction)
            fold_metrics_rows.append(metric)
            shortcut_rows.append(c64.shortcut_row(prediction, config, {"candidate": common.CANDIDATE, "fold": fold, "seed": seed, "backbone_seed": 42}))
            health = c64.parameter_health(run_dir, common.CANDIDATE)
            health.update({"candidate": common.CANDIDATE, "fold": fold, "seed": seed, "backbone_seed": 42})
            health_rows.append(health)
    fold_metrics = pd.DataFrame(fold_metrics_rows).sort_values(["fold", "seed"])
    predictions = pd.concat(prediction_parts, ignore_index=True)
    shortcuts = pd.DataFrame(shortcut_rows).sort_values(["fold", "seed"])
    health = pd.DataFrame(health_rows).sort_values(["fold", "seed"])
    oof_metrics_rows = []
    for seed in common.SEEDS:
        frame = predictions[predictions["seed"].astype(int) == seed].sort_values("patient_id").reset_index(drop=True)
        if len(frame) != c65a.DEVELOPMENT_COUNT or frame["patient_id"].duplicated().any():
            raise RuntimeError(f"C65 OOF coverage failed for seed {seed}")
        probability = frame[probability_column(frame)].to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=int)
        seed_fold_auc = fold_metrics[fold_metrics["seed"].astype(int) == seed]["AUC"].to_numpy(dtype=float)
        oof_metrics_rows.append(
            {
                "candidate": common.CANDIDATE,
                "backbone_seed": 42,
                "head_seed": seed,
                "seed": seed,
                "OOF_AUC": c64.auc(labels, probability),
                "min_fold_AUC": float(seed_fold_auc.min()),
                "mean_fold_AUC": float(seed_fold_auc.mean()),
                "fold_AUC_std": float(seed_fold_auc.std(ddof=1)),
                "n_rows": len(frame),
                "label0": int((labels == 0).sum()),
                "label1": int((labels == 1).sum()),
            }
        )
    oof_metrics = pd.DataFrame(oof_metrics_rows).sort_values("seed")
    oof_shortcuts = []
    for seed in common.SEEDS:
        frame = predictions[predictions["seed"].astype(int) == seed].sort_values("patient_id").reset_index(drop=True)
        oof_shortcuts.append(c64.shortcut_row(frame, config, {"candidate": common.CANDIDATE, "head_seed": seed, "seed": seed, "split": "oof", "backbone_seed": 42}))
    oof_shortcuts_frame = pd.DataFrame(oof_shortcuts).sort_values("seed")
    cross_seed, cross_summary = cross_seed_stability(predictions)
    baseline_path = Path(str(config["audit"]["c61_run_dir"])) / "predictions" / "val_predictions_seed_42.csv"
    baseline = c64.read_prediction(baseline_path)
    positive_rows = []
    for seed in common.SEEDS:
        candidate = predictions[predictions["seed"].astype(int) == seed]
        candidate = candidate[candidate["patient_id"].astype(str).isin(set(baseline["patient_id"].astype(str)))].copy()
        positive_rows.append(c64.positive_and_inversion_row(candidate, baseline, {"backbone_seed": 42, "head_seed": seed, "baseline": "C61_seed42_val"}))
    positive = pd.DataFrame(positive_rows).sort_values("head_seed")
    mean_oof = float(oof_metrics["OOF_AUC"].mean())
    std_oof = float(oof_metrics["OOF_AUC"].std(ddof=1))
    seed_threshold_count = int((oof_metrics["OOF_AUC"] >= OOF_AUC_MIN).sum())
    min_fold_auc = float(fold_metrics["AUC"].min())
    health_pass = bool(health["health_pass"].astype(bool).all())
    shortcut_pass = bool(oof_shortcuts_frame["shortcut_safety_pass"].astype(bool).all())
    positive_pass = bool(float(positive["positive_sensitivity_damage"].mean()) <= float(config["audit"]["positive_damage_mean_max"]) and int((positive["positive_sensitivity_damage"] > float(config["audit"]["positive_damage_seed_max"])).sum()) <= int(config["audit"]["positive_damage_seed_count_max"]))
    gate_checks = {
        "mean_oof_auc_pass": mean_oof >= OOF_AUC_MIN,
        "seed_oof_auc_pass": seed_threshold_count >= OOF_SEED_COUNT,
        "oof_std_pass": np.isfinite(std_oof) and std_oof <= OOF_STD_MAX,
        "oof_std_lower_than_c64_pass": np.isfinite(std_oof) and std_oof < float(config["audit"]["c64_oof_std"]),
        "min_fold_auc_pass": min_fold_auc >= FOLD_AUC_MIN,
        "cross_seed_probability_spearman_pass": cross_summary["mean_probability_spearman"] >= PREDICTION_SPEARMAN_MIN,
        "shortcut_pass": shortcut_pass,
        "positive_safety_pass": positive_pass,
        "parameter_health_pass": health_pass,
        "oof_patient_coverage_pass": len(predictions) == c65a.DEVELOPMENT_COUNT * len(common.SEEDS) and not predictions.duplicated(["seed", "patient_id"]).any(),
        "test_loaded_pass": False,
    }
    authorized = all(bool(item) for item in gate_checks.values())
    final_epochs = {}
    selected_epochs = []
    for _, row in fold_metrics.iterrows():
        selected_epochs.append({"fold": int(row["fold"]), "seed": int(row["seed"]), "selected_epoch": int(row["best_epoch"])})
    for seed in common.SEEDS:
        values = [item["selected_epoch"] for item in selected_epochs if item["seed"] == seed]
        final_epochs[str(seed)] = min(60, max(3, int(round(float(np.median(values))))))
    decision = {
        "phase": "C65-VACS",
        "status": "C65_FINAL_TRAINING_AUTHORIZED" if authorized else "C65_COMMON_BACKBONE_STABILITY_FAIL",
        "candidate": common.CANDIDATE,
        "backbone_seed": 42,
        "mean_OOF_AUC": mean_oof,
        "std_OOF_AUC": std_oof,
        "seed_OOF_AUC_count_at_least_0.9000": seed_threshold_count,
        "min_fold_AUC": min_fold_auc,
        "mean_cross_seed_probability_spearman": cross_summary["mean_probability_spearman"],
        "gate_checks": gate_checks,
        "selected_epochs_by_seed": final_epochs,
        "test_loaded": False,
        "test_used_for_decision": False,
        "fixed_epoch_contract_written": bool(authorized),
        "ensemble": False,
        "prediction_averaging": False,
    }
    fold_metrics.to_csv(report_dir / "c65_cv_metrics_by_fold.csv", index=False)
    predictions.to_csv(report_dir / "c65_oof_predictions.csv", index=False)
    oof_metrics.to_csv(report_dir / "c65_oof_metrics_by_seed.csv", index=False)
    pd.DataFrame([{"candidate": common.CANDIDATE, "backbone_seed": 42, "OOF_AUC_mean": mean_oof, "OOF_AUC_std": std_oof, "seed_count_at_least_0.9000": seed_threshold_count, "min_fold_AUC": min_fold_auc}]).to_csv(report_dir / "c65_oof_metrics_summary.csv", index=False)
    cross_seed.to_csv(report_dir / "c65_cross_seed_prediction_stability.csv", index=False)
    positive.to_csv(report_dir / "c65_positive_preservation.csv", index=False)
    oof_shortcuts_frame.to_csv(report_dir / "c65_shortcut_audit.csv", index=False)
    health.to_csv(report_dir / "c65_parameter_update_audit.csv", index=False)
    pd.DataFrame(selected_epochs).sort_values(["seed", "fold"]).to_csv(report_dir / "c65_cv_selected_epochs.csv", index=False)
    common.c65a.write_json(report_dir / "c65_cv_decision.json", decision)
    common.c65a.write_markdown(report_dir / "c65_cv_decision.md", [
        "# C65-B Common-Backbone CV Decision",
        "",
        f"- Status: `{decision['status']}`.",
        f"- Common frozen backbone: `C61 seed 42`; head Seeds: `{list(common.SEEDS)}`.",
        f"- OOF AUC mean/std: `{mean_oof:.10f} +/- {std_oof:.10f}`; minimum fold AUC: `{min_fold_auc:.10f}`.",
        f"- Seeds at or above 0.9000: `{seed_threshold_count}/3`; mean cross-seed probability Spearman: `{cross_summary['mean_probability_spearman']:.10f}`.",
        "- Test remained locked and was not loaded for route, epoch, contract, or selection decisions.",
    ])
    print(json.dumps({"status": decision["status"], "mean_OOF_AUC": mean_oof, "std_OOF_AUC": std_oof, "test_loaded": False}, sort_keys=True))
    if authorized:
        contract = {
            "phase": "C65-VACS",
            "status": "C65_FINAL_TRAINING_CONTRACT_FROZEN",
            "candidate": common.CANDIDATE,
            "backbone_seed": 42,
            "selected_epochs_by_seed": final_epochs,
            "source": "c65b_five_fold_patient_level_cv_median_selected_epoch",
            "fold_count": common.FOLD_COUNT,
            "fold_seed": c65a.FOLD_SEED,
            "early_stopping": False,
            "fixed_epoch": True,
            "patience": 15,
            "max_epochs": 60,
            "test_loaded": False,
            "test_used_for_contract": False,
            "ensemble": False,
            "prediction_averaging": False,
        }
        common.c65a.write_json(report_dir / "c65_final_training_contract.json", contract)
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
