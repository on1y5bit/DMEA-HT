#!/usr/bin/env python3
"""Freeze and report C46-MCFS Validation and reporting-only Test evidence."""

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

from dmea_ht.config import load_config  # noqa: E402
from scripts import collect_phase_c41_report as base  # noqa: E402


SEEDS = base.SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c46_mcfs_multiseed.yaml")
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def rename_c41_columns(frame: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    return frame.rename(columns={column: column.replace(old, new) for column in frame.columns})


def freeze_validation_decision(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C46 Validation decision requires Validation-only metrics")
    comparisons, positive, inversions = base.validation_comparisons(config, run_dir, metrics)
    comparisons = rename_c41_columns(comparisons, "C41", "C46")
    positive = rename_c41_columns(positive, "c41", "c46")
    inversions = rename_c41_columns(inversions, "C41", "C46")
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    health, health_pass = base.training_health(run_dir, epoch)
    shortcuts = base.shortcut_audit(config, run_dir)
    shortcuts["combination"] = "C46-MCFS"
    auc_values = comparisons["AUC"].to_numpy(dtype=float)
    mean_auc = float(auc_values.mean())
    std_auc = float(auc_values.std(ddof=1))
    auc_pass = bool(
        mean_auc >= 0.9000
        and int((auc_values >= 0.9000).sum()) >= 2
        and std_auc <= 0.025
    )
    positive_pass = bool(
        float(positive["c46_minus_c17_sensitivity"].min()) >= -0.10
        and int(positive["c17_tp_to_c46_fn"].sum())
        <= int(positive["c17_fn_to_c46_tp"].sum()) + 3
    )
    mean_c27_inversions = float(inversions["C27_inversions"].mean())
    mean_c46_inversions = float(inversions["C46_inversions"].mean())
    ranking_pass = bool(
        (mean_c46_inversions - mean_c27_inversions) / max(mean_c27_inversions, 1.0) <= 0.10
        and int(inversions["C46_minus_C27_inversions"].max()) <= 20
    )
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    capacity_pass = True
    for path in sorted((run_dir / "seed_runs").glob("seed_*/reports/run_config.json")):
        runtime = json.loads(path.read_text(encoding="utf-8"))
        capacity_pass &= int(runtime["trainable_parameter_count"]) <= int(config["c46"]["trainable_parameter_limit"])
    if not capacity_pass or not health_pass:
        label = "C46_TRAINING_INVALID"
    elif not shortcut_pass:
        label = "C46_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "C46_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "C46_RANKING_DAMAGE"
    elif not auc_pass:
        label = "C46_NO_AUC_GAIN"
    else:
        label = "GOAL_REACHED_DEMA_HT_AUC_090_PLUS"
    promoted = label == "GOAL_REACHED_DEMA_HT_AUC_090_PLUS"
    median_index = int(np.argsort(auc_values)[len(auc_values) // 2])
    deployment_seed = SEEDS[median_index] if promoted else None
    decision = {
        "phase": "C46-MCFS",
        "decision_label": label,
        "goal_reached": promoted,
        "strict_best": "C46_MCFS" if promoted else "KEEP_DEMA_C17_STRICT_BEST",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C17": mean_auc - float(config["c17"]["mean_validation_auc"]),
        "mean_AUC_gain_vs_C27": mean_auc - float(config["c27"]["mean_validation_auc"]),
        "auc_gate_pass": auc_pass,
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
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(report_dir / "c46_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
                "C17_AUC_mean": float(comparisons["C17_AUC"].mean()),
                "C27_AUC_mean": float(comparisons["C27_AUC"].mean()),
                "C46_minus_C27_AUC_mean": float(comparisons["C46_minus_C27_AUC"].mean()),
            }
        ]
    ).to_csv(report_dir / "c46_metrics_summary.csv", index=False)
    epoch.to_csv(report_dir / "c46_metrics_by_epoch.csv", index=False)
    pd.read_csv(run_dir / "reports" / "parameter_drift.csv").to_csv(report_dir / "c46_parameter_drift.csv", index=False)
    pd.read_csv(run_dir / "reports" / "patient_diagnostics_val.csv").to_csv(report_dir / "c46_patient_diagnostics_val.csv", index=False)
    health.to_csv(report_dir / "c46_training_health.csv", index=False)
    positive.to_csv(report_dir / "c46_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c46_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c46_shortcut_audit.csv", index=False)
    (report_dir / "c46_validation_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (report_dir / "c46_route_decision.md").write_text(
        "\n".join(
            [
                "# C46-MCFS Validation Decision",
                "",
                f"- Decision: `{label}`.",
                f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
                f"- AUC/positive/ranking/shortcut/health gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{health_pass}`.",
                f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
                "- Validation decision was frozen before reporting-only evaluation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


def write_final_report(config: Mapping[str, Any], run_dir: Path, report_dir: Path) -> Dict[str, Any]:
    decision = json.loads((report_dir / "c46_validation_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val", "test"}:
        raise RuntimeError("C46 final report requires Validation and reporting-only Test rows")
    test = metrics[metrics["split"] == "test"]
    summary = pd.read_csv(report_dir / "c46_metrics_summary.csv")
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
    summary.to_csv(report_dir / "c46_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c46_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c46_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c46_training_health.csv")
    shortcut = pd.read_csv(report_dir / "c46_shortcut_audit.csv")
    lines = [
        "# DMEA-HT Phase C46-MCFS Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- Mean Validation gain versus C17/C27: `{decision['mean_AUC_gain_vs_C17']:.10f}` / `{decision['mean_AUC_gain_vs_C27']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C46 FN / FN-to-C46 TP: `{int(positive['c17_tp_to_c46_fn'].sum())}`/`{int(positive['c17_fn_to_c46_tp'].sum())}`.",
        f"- C27-to-C46 repaired/introduced pairs: `{int(inversions['C27_to_C46_repaired'].sum())}`/`{int(inversions['C27_to_C46_introduced'].sum())}`.",
        f"- Training health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Test was reporting-only and did not alter Validation selection or the decision.",
        "- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.",
    ]
    (report_dir / "phase_c46_dema_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c46":
        raise RuntimeError("C46 report requires the formal C46 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        print(json.dumps({"status": "C46_VALIDATION_DECISION_FROZEN", "decision": decision["decision_label"]}))
    else:
        decision = write_final_report(config, run_dir, report_dir)
        print(json.dumps({"status": "C46_FINAL_REPORT_COMPLETE", "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
