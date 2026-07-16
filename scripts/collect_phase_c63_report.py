#!/usr/bin/env python3
"""Collect C63 from-base audits and freeze Validation before Test reporting."""

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

from scripts import c63_common as common  # noqa: E402
from scripts import collect_phase_c41_report as base  # noqa: E402


SEEDS = common.SEEDS
HISTORICAL_RUN_DIRS = {
    "C17": Path("/home/linruixin/chen/project/DMEA-HT/runs/dema_ht_c17_formal_multiseed"),
    "C27": Path("/home/linruixin/chen/project/DMEA-HT/runs/dema_ht_c27_vtme_multiseed"),
    "C61": Path("/home/linruixin/chen/project/DMEA-HT/runs/dema_ht_c61_cbpi_multiseed"),
    "C62": Path("/home/linruixin/chen/project/DMEA-HT/runs/dema_ht_c62_e2e_cbpi_multiseed"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c63_from_base_e2e_cbpi_multiseed.yaml")
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def historical_prediction(name: str, seed: int) -> pd.DataFrame:
    return base.read_prediction(HISTORICAL_RUN_DIRS[name] / "predictions" / f"val_predictions_seed_{seed}.csv")


def historical_validation(
    run_dir: Path, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[Dict[str, Any]] = []
    positive_rows: list[Dict[str, Any]] = []
    inversion_rows: list[Dict[str, Any]] = []
    for seed in SEEDS:
        c63 = base.read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
        labels = c63["label"].to_numpy(dtype=int)
        if len(c63) != 94 or int((labels == 1).sum()) != 47:
            raise RuntimeError(f"C63 Validation balance failed for seed {seed}")
        baselines = {name: historical_prediction(name, seed) for name in ("C17", "C27", "C61", "C62")}
        ids = c63["patient_id"].to_numpy(dtype=str)
        for name, frame in baselines.items():
            if not np.array_equal(ids, frame["patient_id"].to_numpy(dtype=str)):
                raise RuntimeError(f"C63/{name} patient alignment failed for seed {seed}")
            if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
                raise RuntimeError(f"C63/{name} label alignment failed for seed {seed}")
        probabilities = {
            "C63": c63[base.probability_column(c63)].to_numpy(dtype=float),
            **{name: frame[base.probability_column(frame)].to_numpy(dtype=float) for name, frame in baselines.items()},
        }
        metric = metrics[(metrics["seed"].astype(int) == seed) & (metrics["split"].astype(str) == "val")]
        if len(metric) != 1:
            raise RuntimeError(f"C63 Validation metric row missing for seed {seed}")
        row = metric.iloc[0].to_dict()
        metric_rows.append(
            {
                **row,
                **{f"{name}_AUC": base.auc(labels, value) for name, value in probabilities.items()},
                **{
                    f"C63_minus_{name}_AUC": float(row["AUC"]) - base.auc(labels, value)
                    for name, value in probabilities.items()
                    if name != "C63"
                },
            }
        )
        c17_counts = base.binary_counts(labels, probabilities["C17"])
        c63_counts = base.binary_counts(labels, probabilities["C63"])
        positive = labels == 1
        c17_class = probabilities["C17"] >= 0.5
        c63_class = probabilities["C63"] >= 0.5
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c63_fn": int((positive & c17_class & ~c63_class).sum()),
                "c17_fn_to_c63_tp": int((positive & ~c17_class & c63_class).sum()),
                "c17_sensitivity": c17_counts["Sensitivity"],
                "c63_sensitivity": c63_counts["Sensitivity"],
                "c63_minus_c17_sensitivity": c63_counts["Sensitivity"] - c17_counts["Sensitivity"],
            }
        )
        c27_inversions = base.inversion_vector(labels, probabilities["C27"])
        c63_inversions = base.inversion_vector(labels, probabilities["C63"])
        inversion_rows.append(
            {
                "seed": seed,
                "C27_inversions": int(c27_inversions.sum()),
                "C63_inversions": int(c63_inversions.sum()),
                "C63_minus_C27_inversions": int(c63_inversions.sum() - c27_inversions.sum()),
                "C63_inversion_ratio_vs_C27": float((c63_inversions.sum() - c27_inversions.sum()) / max(c27_inversions.sum(), 1)),
                "C27_to_C63_repaired": int((c27_inversions & ~c63_inversions).sum()),
                "C27_to_C63_introduced": int((~c27_inversions & c63_inversions).sum()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(positive_rows), pd.DataFrame(inversion_rows)


def training_health(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Tuple[pd.DataFrame, bool, bool]:
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    updates = pd.read_csv(run_dir / "reports" / "parameter_update_audit.csv")
    inventory = pd.read_csv(run_dir / "reports" / "trainable_parameter_inventory.csv")
    initialization = pd.read_csv(run_dir / "reports" / "initialization_inventory.csv")
    gradients = pd.read_csv(report_dir / "c63_gradient_connectivity_audit.csv")
    rows: list[Dict[str, Any]] = []
    health_pass = True
    compliance_pass = True
    for seed in SEEDS:
        selected = epoch[
            (epoch["seed"].astype(int) == seed)
            & epoch["selected_by_val_auc"].astype(str).str.lower().isin(["true", "1"])
        ]
        diagnostics = pd.read_csv(run_dir / "reports" / "patient_diagnostics_val.csv")
        diagnostics = diagnostics[diagnostics["seed"].astype(int) == seed]
        probabilities = diagnostics["final_prob"].to_numpy(dtype=float)
        prediction_ok = len(diagnostics) == 94 and np.isfinite(probabilities).all() and float(probabilities.std()) > 0.0
        state_ok = len(diagnostics) == 94 and np.isfinite(diagnostics["patient_state_norm"].to_numpy(dtype=float)).all() and float(diagnostics["patient_state_norm"].std()) > 0.0
        selected_ok = len(selected) == 1
        health_pass &= prediction_ok and state_ok and selected_ok
        rows.extend(
            [
                {"seed": seed, "category": "prediction_health", "state_std": float(probabilities.std()) if len(probabilities) else 0.0, "health_pass": prediction_ok},
                {"seed": seed, "category": "patient_state_health", "state_std": float(diagnostics["patient_state_component_std"].to_numpy(dtype=float).std()) if len(diagnostics) else 0.0, "health_pass": state_ok},
            ]
        )
        for group in common.MODULE_GROUPS:
            gradient_rows = gradients[(gradients["seed"].astype(int) == seed) & (gradients["module_group"].astype(str) == group)]
            selected_gradient = selected[f"{group}_grad_norm"].iloc[0] if selected_ok and f"{group}_grad_norm" in selected.columns else np.nan
            gradient_ok = bool(
                len(gradient_rows) == 1
                and gradient_rows["pass"].astype(str).str.lower().eq("true").all()
                and np.isfinite(float(selected_gradient))
                and float(selected_gradient) > 0.0
            )
            update_rows = updates[(updates["seed"].astype(int) == seed) & (updates["kind"].astype(str) == "module_summary") & (updates["module_group"].astype(str) == group)]
            update_ok = bool(
                len(update_rows) == 1
                and update_rows["updated"].astype(str).str.lower().eq("true").all()
                and update_rows["relative_parameter_change"].astype(float).gt(0.0).all()
                and update_rows["finite"].astype(str).str.lower().eq("true").all()
            )
            health_pass &= gradient_ok and update_ok
            compliance_pass &= gradient_ok and update_ok
            rows.extend(
                [
                    {"seed": seed, "category": f"gradient_connectivity_{group}", "selected_epoch_gradient_norm": float(selected_gradient) if np.isfinite(float(selected_gradient)) else np.nan, "health_pass": gradient_ok},
                    {"seed": seed, "category": f"parameter_update_{group}", "relative_parameter_change": float(update_rows["relative_parameter_change"].iloc[0]) if len(update_rows) == 1 else np.nan, "health_pass": update_ok},
                ]
            )
    frozen_count = int((~inventory["requires_grad"].astype(bool)).sum()) if not inventory.empty else 1
    contamination = bool(initialization["task_trained_checkpoint_used"].astype(bool).any()) if not initialization.empty else True
    compliance_pass &= frozen_count == 0 and not contamination
    return pd.DataFrame(rows), bool(health_pass), bool(compliance_pass)


def freeze_validation_decision(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"].astype(str)) != {"val"}:
        raise RuntimeError("C63 Validation decision requires Validation-only metrics")
    comparisons, positive, inversions = historical_validation(run_dir, metrics)
    shortcuts = base.shortcut_audit(config, run_dir)
    shortcuts = shortcuts.rename(columns={column: column.replace("C41", "C63") for column in shortcuts.columns})
    shortcuts["combination"] = "C63-FS-CBPI"
    health, health_pass, compliance_audit_pass = training_health(config, run_dir, report_dir)
    gate = json.loads((report_dir / "c63_gate.json").read_text(encoding="utf-8"))
    gate_pass = gate.get("status") == "C63_FROM_BASE_E2E_DIRECT_MULTI_SEED_AUTHORIZED" and int(gate.get("passed", 0)) == int(gate.get("total", 0)) == 24
    inventory = pd.read_csv(run_dir / "reports" / "trainable_parameter_inventory.csv")
    per_seed_parameter_count = inventory.groupby("seed")["parameter_count"].sum()
    capacity_pass = bool(not per_seed_parameter_count.empty and int(per_seed_parameter_count.max()) <= int(config["initialization"].get("trainable_parameter_limit", 100000000)) and inventory["requires_grad"].astype(bool).all())
    initialization = pd.read_csv(run_dir / "reports" / "initialization_inventory.csv")
    authenticity_pass = bool(not initialization["task_trained_checkpoint_used"].astype(bool).any() and initialization["initialization_type"].eq("random_task_specific").all())
    auc_values = comparisons["AUC"].to_numpy(dtype=float)
    mean_auc = float(auc_values.mean())
    std_auc = float(auc_values.std(ddof=1))
    auc_pass = bool(mean_auc >= 0.9000 and int((auc_values >= 0.9000).sum()) >= 2 and std_auc <= 0.025)
    positive_pass = bool(float(positive["c63_minus_c17_sensitivity"].min()) >= -0.10 and int(positive["c17_tp_to_c63_fn"].sum()) <= int(positive["c17_fn_to_c63_tp"].sum()) + 3)
    mean_c27_inversions = float(inversions["C27_inversions"].mean())
    mean_c63_inversions = float(inversions["C63_inversions"].mean())
    ranking_pass = bool((mean_c63_inversions - mean_c27_inversions) / max(mean_c27_inversions, 1.0) <= 0.10 and int(inversions["C63_minus_C27_inversions"].max()) <= 20)
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    full_training_pass = bool(gate_pass and authenticity_pass and compliance_audit_pass and capacity_pass and health_pass)
    if not full_training_pass:
        label = "DEMA_C63_TASK_CHECKPOINT_CONTAMINATION" if not authenticity_pass else "DEMA_C63_FULL_TRAINING_CONTRACT_FAIL"
    elif not shortcut_pass:
        label = "DEMA_C63_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "DEMA_C63_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "DEMA_C63_RANKING_DAMAGE"
    elif not auc_pass:
        label = "DEMA_C63_FROM_BASE_AUC_TARGET_NOT_REACHED"
    elif mean_auc > float(config["historical"]["c61_mean_validation_auc"]):
        label = "PROMOTE_DEMA_C63_FROM_BASE_NEW_STRICT_BEST"
    else:
        label = "PROMOTE_DEMA_C63_FROM_BASE_REPRODUCIBLE_FULL_TRAINING"
    promoted = label.startswith("PROMOTE_")
    median_index = int(np.argsort(auc_values)[len(auc_values) // 2])
    deployment_seed = SEEDS[median_index] if promoted else None
    decision = {
        "phase": "C63-FS-CBPI",
        "decision_label": label,
        "goal_reached": bool(auc_pass and full_training_pass),
        "official_from_base_model": full_training_pass and promoted,
        "initialization_authenticity": "FROM_BASE_NO_TASK_CHECKPOINT",
        "historical_c61_status": "HISTORICAL_PARTIALLY_FROZEN_REFERENCE",
        "historical_c62_status": "HISTORICAL_FULL_PARAMETER_WARM_START_REFERENCE",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C61": mean_auc - float(config["historical"]["c61_mean_validation_auc"]),
        "mean_AUC_gain_vs_C62": mean_auc - float(config["historical"]["c62_mean_validation_auc"]),
        "mean_AUC_gain_vs_C27": mean_auc - float(config["historical"]["c27_mean_validation_auc"]),
        "mean_AUC_gain_vs_C17": mean_auc - float(config["historical"]["c17_mean_validation_auc"]),
        "auc_gate_pass": auc_pass,
        "full_training_gate_pass": full_training_pass,
        "positive_safety_pass": positive_pass,
        "ranking_safety_pass": ranking_pass,
        "shortcut_safety_pass": shortcut_pass,
        "training_health_pass": health_pass,
        "capacity_gate_pass": capacity_pass,
        "authenticity_gate_pass": authenticity_pass,
        "deployment_seed": deployment_seed,
        "deployment_checkpoint": str(run_dir / "checkpoints" / f"seed_{deployment_seed}_best.pt") if promoted else None,
        "validation_decision_frozen_before_test": True,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "threshold_tuned": False,
        "no_smoke_or_pilot": True,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(report_dir / "c63_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
                "C61_AUC_mean": float(comparisons["C61_AUC"].mean()),
                "C62_AUC_mean": float(comparisons["C62_AUC"].mean()),
                "C27_AUC_mean": float(comparisons["C27_AUC"].mean()),
                "C17_AUC_mean": float(comparisons["C17_AUC"].mean()),
                "C63_minus_C61_AUC_mean": float(comparisons["C63_minus_C61_AUC"].mean()),
                "C63_minus_C62_AUC_mean": float(comparisons["C63_minus_C62_AUC"].mean()),
                "C63_minus_C27_AUC_mean": float(comparisons["C63_minus_C27_AUC"].mean()),
                "C63_minus_C17_AUC_mean": float(comparisons["C63_minus_C17_AUC"].mean()),
            }
        ]
    ).to_csv(report_dir / "c63_metrics_summary.csv", index=False)
    pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv").to_csv(report_dir / "c63_metrics_by_epoch.csv", index=False)
    for name in ("parameter_update_audit", "trainable_parameter_inventory", "initialization_inventory", "initial_parameter_hash_by_seed", "optimizer_parameter_groups"):
        source = run_dir / "reports" / f"{name}.csv"
        if source.exists():
            shutil.copy2(source, report_dir / f"c63_{name}.csv")
    positive.to_csv(report_dir / "c63_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c63_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c63_shortcut_audit.csv", index=False)
    health.to_csv(report_dir / "c63_training_health.csv", index=False)
    (report_dir / "c63_validation_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (report_dir / "c63_route_decision.md").write_text(
        "\n".join(
            [
                "# C63-FS-CBPI Validation Decision",
                "",
                f"- Decision: `{label}`.",
                f"- From-base authenticity: `{authenticity_pass}`; no C13-C62 task checkpoint was used for initialization.",
                f"- Full-training compliance: `{full_training_pass}`; Gate: `{gate.get('passed', 0)}/{gate.get('total', 0)}`.",
                f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
                f"- C63 minus C61/C62 mean AUC: `{decision['mean_AUC_gain_vs_C61']:.10f}` / `{decision['mean_AUC_gain_vs_C62']:.10f}`.",
                f"- AUC/positive/ranking/shortcut/health gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{health_pass}`.",
                "- C61 is a historical partially frozen reference; C62 is a historical full-parameter task-checkpoint warm-start reference.",
                "- Historical predictions are used only for audit-only comparisons, never as C63 model inputs.",
                "- Validation was frozen before reporting-only Test.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


def write_final_report(config: Mapping[str, Any], run_dir: Path, report_dir: Path) -> Dict[str, Any]:
    decision = json.loads((report_dir / "c63_validation_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"].astype(str)) != {"val", "test"}:
        raise RuntimeError("C63 final report requires Validation and reporting-only Test rows")
    test = metrics[metrics["split"].astype(str) == "test"]
    summary = pd.read_csv(report_dir / "c63_metrics_summary.csv")
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "split": "test",
                        "AUC_mean": float(test["AUC"].mean()),
                        "AUC_std": float(test["AUC"].std(ddof=1)),
                        "Sensitivity_mean": float(test["Sensitivity"].mean()),
                        "Specificity_mean": float(test["Specificity"].mean()),
                        "Balanced_ACC_mean": float(test["Balanced_ACC"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(report_dir / "c63_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c63_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c63_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c63_training_health.csv")
    shortcuts = pd.read_csv(report_dir / "c63_shortcut_audit.csv")
    lines = [
        "# DMEA-HT Phase C63-FS-CBPI Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Official from-base model: `{decision['official_from_base_model']}`.",
        f"- Initialization authenticity: `{decision['initialization_authenticity']}`.",
        f"- Frozen predictive parameters: `0`; all optimizer groups positive and updated.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- C63 minus C61/C62/C27/C17 mean AUC: `{decision['mean_AUC_gain_vs_C61']:.10f}` / `{decision['mean_AUC_gain_vs_C62']:.10f}` / `{decision['mean_AUC_gain_vs_C27']:.10f}` / `{decision['mean_AUC_gain_vs_C17']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C63 FN / FN-to-C63 TP: `{int(positive['c17_tp_to_c63_fn'].sum())}` / `{int(positive['c17_fn_to_c63_tp'].sum())}`.",
        f"- C27-to-C63 repaired/introduced pairs: `{int(inversions['C27_to_C63_repaired'].sum())}` / `{int(inversions['C27_to_C63_introduced'].sum())}`.",
        f"- Training health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcuts['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        "- C61 remains `HISTORICAL_PARTIALLY_FROZEN_REFERENCE`.",
        "- C62 remains `HISTORICAL_FULL_PARAMETER_WARM_START_REFERENCE` and is not an independent from-base result.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Historical predictions were used only for audit comparisons; no saved prediction or representation was used as a C63 model input.",
        "- Test was reporting-only and did not alter Validation selection or promotion.",
        "- Deployment contract is one checkpoint, one model, one forward, with no ensemble or averaging.",
    ]
    (report_dir / "phase_c63_dema_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    config = common.load_c63_config(args.config)
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        print(json.dumps({"status": "C63_VALIDATION_DECISION_FROZEN", "decision": decision["decision_label"]}))
    else:
        decision = write_final_report(config, run_dir, report_dir)
        print(json.dumps({"status": "C63_FINAL_REPORT_COMPLETE", "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
