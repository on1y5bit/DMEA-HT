#!/usr/bin/env python3
"""Collect C62 full-training audits and freeze Validation before Test reporting."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402
from scripts import c62_common as common  # noqa: E402
from scripts import collect_phase_c41_report as base  # noqa: E402


SEEDS = common.SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c62_e2e_cbpi_multiseed.yaml")
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def training_health(config: Mapping[str, Any], run_dir: Path, report_dir: Path) -> Tuple[pd.DataFrame, bool, bool]:
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    updates = pd.read_csv(run_dir / "reports" / "parameter_update_audit.csv")
    inventory = pd.read_csv(run_dir / "reports" / "trainable_parameter_inventory.csv")
    gradient_path = report_dir / "c62_gradient_connectivity_audit.csv"
    gradients = pd.read_csv(gradient_path) if gradient_path.exists() else pd.DataFrame()
    rows: list[Dict[str, Any]] = []
    health_pass = True
    compliance_pass = True
    for seed in SEEDS:
        selected = epoch[(epoch["seed"].astype(int) == seed) & epoch["selected_by_val_auc"].astype(str).str.lower().isin(["true", "1"])]
        diagnostics = pd.read_csv(run_dir / "reports" / "patient_diagnostics_val.csv")
        diagnostics = diagnostics[diagnostics["seed"].astype(int) == seed]
        probabilities = diagnostics["final_prob"].to_numpy(dtype=float)
        prediction_ok = len(diagnostics) == 94 and np.isfinite(probabilities).all() and float(probabilities.std()) > 0.0
        state_ok = len(diagnostics) == 94 and np.isfinite(diagnostics["patient_state_norm"].to_numpy(dtype=float)).all() and float(diagnostics["patient_state_norm"].std()) > 0.0
        selected_ok = len(selected) == 1
        rows.append({"seed": seed, "category": "prediction_health", "state_std": float(probabilities.std()) if len(probabilities) else 0.0, "health_pass": prediction_ok})
        rows.append({"seed": seed, "category": "patient_state_health", "state_std": float(diagnostics["patient_state_component_std"].to_numpy(dtype=float).std()) if len(diagnostics) else 0.0, "health_pass": state_ok})
        health_pass &= prediction_ok and state_ok and selected_ok
        for group in common.GROUPS:
            gradient_rows = gradients[(gradients.get("seed", pd.Series(dtype=int)).astype(int) == seed) & (gradients.get("group", pd.Series(dtype=str)).astype(str) == group)] if not gradients.empty else pd.DataFrame()
            selected_gradient = selected[f"{group}_grad_norm"].iloc[0] if selected_ok and f"{group}_grad_norm" in selected.columns else np.nan
            gradient_ok = bool(len(gradient_rows) == 1 and gradient_rows["pass"].astype(str).str.lower().eq("true").all() and np.isfinite(float(selected_gradient)) and float(selected_gradient) > 0.0)
            rows.append({"seed": seed, "category": f"gradient_connectivity_{group}", "selected_epoch_gradient_norm": float(selected_gradient) if np.isfinite(float(selected_gradient)) else np.nan, "health_pass": gradient_ok})
            health_pass &= gradient_ok
            group_update = updates[(updates["seed"].astype(int) == seed) & (updates["kind"].astype(str) == "group_summary") & (updates["group"].astype(str) == group)]
            update_ok = bool(len(group_update) == 1 and group_update["updated"].astype(str).str.lower().eq("true").all() and group_update["relative_parameter_change"].astype(float).gt(0.0).all() and group_update["finite"].astype(str).str.lower().eq("true").all())
            rows.append({"seed": seed, "category": f"parameter_update_{group}", "relative_parameter_change": float(group_update["relative_parameter_change"].iloc[0]) if len(group_update) == 1 else np.nan, "health_pass": update_ok})
            compliance_pass &= update_ok
            health_pass &= update_ok
    frozen_count = int((~inventory["requires_grad"].astype(bool)).sum()) if not inventory.empty else 1
    compliance_pass &= frozen_count == 0 and len(gradients) == len(SEEDS) * len(common.GROUPS)
    return pd.DataFrame(rows), bool(health_pass), bool(compliance_pass)


def add_c61_comparison(config: Mapping[str, Any], run_dir: Path, comparisons: pd.DataFrame) -> pd.DataFrame:
    result = comparisons.copy()
    c61_values = []
    for seed in SEEDS:
        c62 = base.read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
        c61 = base.read_prediction(resolve_path(config["c61"]["run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv")
        if not np.array_equal(c62["patient_id"].to_numpy(dtype=str), c61["patient_id"].to_numpy(dtype=str)) or not np.array_equal(c62["label"].to_numpy(dtype=int), c61["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"C61/C62 patient or label alignment failed for seed {seed}")
        c61_values.append(base.auc(c61["label"], c61[base.probability_column(c61)]))
    result["C61_AUC"] = c61_values
    result["C62_minus_C61_AUC"] = result["AUC"].astype(float) - result["C61_AUC"].astype(float)
    return result


def freeze_validation_decision(config: Mapping[str, Any], run_dir: Path, report_dir: Path) -> Dict[str, Any]:
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"].astype(str)) != {"val"}:
        raise RuntimeError("C62 Validation decision requires Validation-only metrics")
    comparisons, positive, inversions = base.validation_comparisons(config, run_dir, metrics)
    comparisons = comparisons.rename(columns={column: column.replace("C41", "C62") for column in comparisons.columns})
    positive = positive.rename(columns={column: column.replace("c41", "c62") for column in positive.columns})
    inversions = inversions.rename(columns={column: column.replace("C41", "C62") for column in inversions.columns})
    comparisons = add_c61_comparison(config, run_dir, comparisons)
    shortcuts = base.shortcut_audit(config, run_dir)
    shortcuts = shortcuts.rename(columns={column: column.replace("C41", "C62") for column in shortcuts.columns})
    shortcuts["combination"] = "C62-E2E-CBPI"
    health, health_pass, compliance_audit_pass = training_health(config, run_dir, report_dir)
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    gate = json.loads((report_dir / "c62_gate.json").read_text(encoding="utf-8"))
    gate_pass = gate.get("status") == "C62_E2E_CBPI_DIRECT_MULTI_SEED_AUTHORIZED" and int(gate.get("passed", 0)) == int(gate.get("total", 0)) == 20
    inventory = pd.read_csv(run_dir / "reports" / "trainable_parameter_inventory.csv")
    per_seed_parameter_count = inventory.groupby("seed")["parameter_count"].sum() if "seed" in inventory.columns else pd.Series(dtype=float)
    capacity_pass = bool(
        not inventory.empty
        and not per_seed_parameter_count.empty
        and int(per_seed_parameter_count.max()) <= int(config["c62"]["trainable_parameter_limit"])
        and inventory["requires_grad"].astype(bool).all()
    )
    auc_values = comparisons["AUC"].to_numpy(dtype=float)
    mean_auc = float(auc_values.mean())
    std_auc = float(auc_values.std(ddof=1))
    auc_pass = bool(mean_auc >= 0.9000 and int((auc_values >= 0.9000).sum()) >= 2 and std_auc <= 0.025)
    positive_pass = bool(float(positive["c62_minus_c17_sensitivity"].min()) >= -0.10 and int(positive["c17_tp_to_c62_fn"].sum()) <= int(positive["c17_fn_to_c62_tp"].sum()) + 3)
    mean_c27_inversions = float(inversions["C27_inversions"].mean())
    mean_c62_inversions = float(inversions["C62_inversions"].mean())
    ranking_pass = bool((mean_c62_inversions - mean_c27_inversions) / max(mean_c27_inversions, 1.0) <= 0.10 and int(inversions["C62_minus_C27_inversions"].max()) <= 20)
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    full_training_pass = bool(gate_pass and compliance_audit_pass and capacity_pass and health_pass)
    if not full_training_pass:
        label = "DEMA_C62_PARTIAL_FREEZE_CONTRACT_FAIL" if (not capacity_pass or not compliance_audit_pass) else "DEMA_C62_TRAINING_INVALID"
    elif not shortcut_pass:
        label = "DEMA_C62_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "DEMA_C62_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "DEMA_C62_RANKING_DAMAGE"
    elif not auc_pass:
        label = "DEMA_C62_E2E_AUC_TARGET_NOT_REACHED"
    elif mean_auc > float(config["c61"].get("mean_validation_auc", 0.9085559076505206)):
        label = "PROMOTE_DEMA_C62_E2E_CBPI_NEW_STRICT_BEST"
    else:
        label = "PROMOTE_DEMA_C62_E2E_CBPI_FULL_TRAINING_COMPLIANT"
    promoted = label.startswith("PROMOTE_")
    median_index = int(np.argsort(auc_values)[len(auc_values) // 2])
    deployment_seed = SEEDS[median_index] if promoted else None
    decision = {
        "phase": "C62-E2E-CBPI",
        "decision_label": label,
        "goal_reached": bool(auc_pass and full_training_pass),
        "official_full_training_compliant": full_training_pass,
        "historical_reference": "HISTORICAL_PARTIALLY_FROZEN_REFERENCE",
        "strict_best": "C62_E2E_CBPI" if promoted else "HISTORICAL_PARTIALLY_FROZEN_REFERENCE",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C61": mean_auc - float(config["c61"].get("mean_validation_auc", 0.9085559076505206)),
        "mean_AUC_gain_vs_C17": mean_auc - float(config["c17"]["mean_validation_auc"]),
        "mean_AUC_gain_vs_C27": mean_auc - float(config["c27"]["mean_validation_auc"]),
        "auc_gate_pass": auc_pass,
        "full_training_gate_pass": full_training_pass,
        "positive_safety_pass": positive_pass,
        "ranking_safety_pass": ranking_pass,
        "shortcut_safety_pass": shortcut_pass,
        "training_health_pass": health_pass,
        "capacity_gate_pass": capacity_pass,
        "deployment_seed": deployment_seed,
        "deployment_checkpoint": str(run_dir / "checkpoints" / f"seed_{deployment_seed}_best.pt") if promoted else None,
        "validation_decision_frozen_before_test": True,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "threshold_tuned": False,
        "no_smoke_or_pilot": True,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(report_dir / "c62_metrics_by_seed.csv", index=False)
    pd.DataFrame([
        {
            "split": "val",
            "AUC_mean": mean_auc,
            "AUC_std": std_auc,
            "C61_AUC_mean": float(comparisons["C61_AUC"].mean()),
            "C27_AUC_mean": float(comparisons["C27_AUC"].mean()),
            "C17_AUC_mean": float(comparisons["C17_AUC"].mean()),
            "C62_minus_C61_AUC_mean": float(comparisons["C62_minus_C61_AUC"].mean()),
            "C62_minus_C27_AUC_mean": float(comparisons["C62_minus_C27_AUC"].mean()),
            "C62_minus_C17_AUC_mean": float(comparisons["C62_minus_C17_AUC"].mean()),
        }
    ]).to_csv(report_dir / "c62_metrics_summary.csv", index=False)
    epoch.to_csv(report_dir / "c62_metrics_by_epoch.csv", index=False)
    for name in ("parameter_update_audit", "trainable_parameter_inventory", "optimizer_parameter_groups"):
        source = run_dir / "reports" / f"{name}.csv"
        if source.exists():
            shutil.copy2(source, report_dir / f"c62_{name}.csv")
    positive.to_csv(report_dir / "c62_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c62_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c62_shortcut_audit.csv", index=False)
    health.to_csv(report_dir / "c62_training_health.csv", index=False)
    (report_dir / "c62_validation_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (report_dir / "c62_route_decision.md").write_text(
        "\n".join([
            "# C62-E2E-CBPI Validation Decision",
            "",
            f"- Decision: `{label}`.",
            f"- Full-training compliance: `{full_training_pass}`; Gate: `{gate.get('passed', 0)}/{gate.get('total', 0)}`.",
            f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
            f"- C62 minus C61 mean AUC: `{decision['mean_AUC_gain_vs_C61']:.10f}`.",
            f"- AUC/positive/ranking/shortcut/health gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{health_pass}`.",
            "- C61 remains a historical partially frozen reference; it is not relabeled as the full-training final model.",
            "- Validation was frozen before reporting-only Test.",
        ]) + "\n",
        encoding="utf-8",
    )
    return decision


def write_final_report(config: Mapping[str, Any], run_dir: Path, report_dir: Path) -> Dict[str, Any]:
    decision = json.loads((report_dir / "c62_validation_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"].astype(str)) != {"val", "test"}:
        raise RuntimeError("C62 final report requires Validation and reporting-only Test rows")
    test = metrics[metrics["split"].astype(str) == "test"]
    summary = pd.read_csv(report_dir / "c62_metrics_summary.csv")
    summary = pd.concat([
        summary,
        pd.DataFrame([{
            "split": "test",
            "AUC_mean": float(test["AUC"].mean()),
            "AUC_std": float(test["AUC"].std(ddof=1)),
            "Sensitivity_mean": float(test["Sensitivity"].mean()),
            "Specificity_mean": float(test["Specificity"].mean()),
            "Balanced_ACC_mean": float(test["Balanced_ACC"].mean()),
        }]),
    ], ignore_index=True, sort=False)
    summary.to_csv(report_dir / "c62_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c62_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c62_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c62_training_health.csv")
    shortcuts = pd.read_csv(report_dir / "c62_shortcut_audit.csv")
    lines = [
        "# DMEA-HT Phase C62-E2E-CBPI Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Full-training compliance: `{decision['official_full_training_compliant']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- C62 minus C61/C27/C17 mean AUC: `{decision['mean_AUC_gain_vs_C61']:.10f}` / `{decision['mean_AUC_gain_vs_C27']:.10f}` / `{decision['mean_AUC_gain_vs_C17']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C62 FN / FN-to-C62 TP: `{int(positive['c17_tp_to_c62_fn'].sum())}` / `{int(positive['c17_fn_to_c62_tp'].sum())}`.",
        f"- C27-to-C62 repaired/introduced pairs: `{int(inversions['C27_to_C62_repaired'].sum())}`/{int(inversions['C27_to_C62_introduced'].sum())}`.",
        f"- Training health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcuts['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        "- C61 is retained as `HISTORICAL_PARTIALLY_FROZEN_REFERENCE` and is not relabeled as the C62 full-training model.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Test was reporting-only and did not alter Validation selection or promotion.",
        "- Deployment contract is one checkpoint, one model, one forward, with no ensemble or averaging.",
    ]
    (report_dir / "phase_c62_dema_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    config = common.load_c62_config(args.config)
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        print(json.dumps({"status": "C62_VALIDATION_DECISION_FROZEN", "decision": decision["decision_label"]}))
    else:
        decision = write_final_report(config, run_dir, report_dir)
        print(json.dumps({"status": "C62_FINAL_REPORT_COMPLETE", "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
