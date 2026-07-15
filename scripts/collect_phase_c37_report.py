#!/usr/bin/env python3
"""Freeze C37 validation and collect reporting-only Test results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402
from scripts.collect_phase_c31a_report import (  # noqa: E402
    RAW_SHORTCUT_FIELDS,
    SELECTED_SHORTCUT_FIELDS,
    safe_spearman,
    shortcut_only_auc,
)
from scripts.train_phase_c37 import MODULE_CATEGORIES, SEEDS  # noqa: E402


C17_MEAN = 0.8696242643730194
C27_MEAN = 0.8822996831145314
PROMOTION_GAIN = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c37_e2e_vrl_multiseed.yaml")
    parser.add_argument("--stage", required=True, choices=("validation", "final"))
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    return float(roc_auc_score(y, p))


def binary_counts(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((negative & ~predicted).sum())
    fp = int((negative & predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_ACC": 0.5 * (sensitivity + specificity),
    }


def inversion_vector(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    positive_ids = np.where(labels == 1)[0]
    negative_ids = np.where(labels == 0)[0]
    return (
        probabilities[positive_ids, None] < probabilities[negative_ids][None, :]
    ).reshape(-1)


def aligned_validation(
    config: Mapping[str, Any], run_dir: Path, seed: int
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]:
    c37 = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
    baseline_paths = {
        "C17": Path(config["c17_run_dir"]),
        "C27": Path(config["c27"]["c27_run_dir"]),
        "C32": Path(config["baselines"]["c32_run_dir"]),
        "C33": Path(config["baselines"]["c33_run_dir"]),
        "C36": Path(config["baselines"]["c36_run_dir"]),
    }
    baselines = {
        name: read_prediction(path / "predictions" / f"val_predictions_seed_{seed}.csv")
        for name, path in baseline_paths.items()
    }
    ids = c37["patient_id"].to_numpy(dtype=str)
    labels = c37["label"].to_numpy(dtype=int)
    for name, frame in baselines.items():
        if not np.array_equal(ids, frame["patient_id"].to_numpy(dtype=str)):
            raise RuntimeError(f"C37 {name} patient alignment failed for seed {seed}")
        if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"C37 {name} label alignment failed for seed {seed}")
    if len(c37) != 94 or int((labels == 1).sum()) != 47:
        raise RuntimeError(f"C37 validation balance failed for seed {seed}")
    return c37, baselines, labels


def validation_comparisons(
    config: Mapping[str, Any], run_dir: Path, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        c37, baselines, labels = aligned_validation(config, run_dir, seed)
        probabilities = {
            "C37": c37[probability_column(c37)].to_numpy(dtype=float),
            **{
                name: frame[probability_column(frame)].to_numpy(dtype=float)
                for name, frame in baselines.items()
            },
        }
        metric = metrics[
            (metrics["seed"].astype(int) == seed) & (metrics["split"] == "val")
        ]
        if len(metric) != 1:
            raise RuntimeError(f"C37 validation metric row missing for seed {seed}")
        metric_row = metric.iloc[0].to_dict()
        metric_rows.append(
            {
                **metric_row,
                **{f"{name}_AUC": auc(labels, value) for name, value in probabilities.items()},
                **{
                    f"C37_minus_{name}_AUC": float(metric_row["AUC"]) - auc(labels, value)
                    for name, value in probabilities.items()
                    if name != "C37"
                },
            }
        )

        positive = labels == 1
        c17_class = probabilities["C17"] >= 0.5
        c27_class = probabilities["C27"] >= 0.5
        c37_class = probabilities["C37"] >= 0.5
        c27_damage = positive & (
            (c17_class & ~c27_class)
            | ((probabilities["C27"] - probabilities["C17"]) <= -0.05)
        )
        c37_damage = positive & (
            (c17_class & ~c37_class)
            | ((probabilities["C37"] - probabilities["C17"]) <= -0.05)
        )
        c17_counts = binary_counts(labels, probabilities["C17"])
        c27_counts = binary_counts(labels, probabilities["C27"])
        c37_counts = binary_counts(labels, probabilities["C37"])
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c37_fn": int((positive & c17_class & ~c37_class).sum()),
                "c17_fn_to_c37_tp": int((positive & ~c17_class & c37_class).sum()),
                "c37_sensitivity": c37_counts["Sensitivity"],
                "c17_sensitivity": c17_counts["Sensitivity"],
                "c27_sensitivity": c27_counts["Sensitivity"],
                "c37_minus_c17_sensitivity": c37_counts["Sensitivity"] - c17_counts["Sensitivity"],
                "c27_material_positive_damage_count": int(c27_damage.sum()),
                "c37_material_positive_damage_count": int(c37_damage.sum()),
                "c37_minus_c27_material_damage": int(c37_damage.sum() - c27_damage.sum()),
            }
        )
        inv27 = inversion_vector(labels, probabilities["C27"])
        inv37 = inversion_vector(labels, probabilities["C37"])
        inversion_rows.append(
            {
                "seed": seed,
                "total_pairs": int(len(inv37)),
                "C17_inversions": int(inversion_vector(labels, probabilities["C17"]).sum()),
                "C27_inversions": int(inv27.sum()),
                "C32_inversions": int(inversion_vector(labels, probabilities["C32"]).sum()),
                "C33_inversions": int(inversion_vector(labels, probabilities["C33"]).sum()),
                "C36_inversions": int(inversion_vector(labels, probabilities["C36"]).sum()),
                "C37_inversions": int(inv37.sum()),
                "C37_minus_C27_inversions": int(inv37.sum() - inv27.sum()),
                "C27_to_C37_repaired": int((inv27 & ~inv37).sum()),
                "C27_to_C37_introduced": int((~inv27 & inv37).sum()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(positive_rows), pd.DataFrame(inversion_rows)


def training_health(
    run_dir: Path, metrics_by_epoch: pd.DataFrame
) -> Tuple[pd.DataFrame, bool]:
    drift = pd.read_csv(run_dir / "reports" / "parameter_drift.csv")
    diagnostics = pd.read_csv(
        run_dir / "reports" / "patient_diagnostics_val.csv", dtype={"patient_id": str}
    )
    selected = metrics_by_epoch[
        metrics_by_epoch["selected_by_val_auc"].astype(str).str.lower().eq("true")
    ]
    rows: List[Dict[str, Any]] = []
    passed = len(selected) == len(SEEDS)
    for seed in SEEDS:
        selected_row = selected[selected["seed"].astype(int) == seed]
        diag = diagnostics[diagnostics["seed"].astype(int) == seed]
        probabilities = diag["final_prob"].to_numpy(dtype=float)
        state_norms = diag["patient_state_norm"].to_numpy(dtype=float)
        prediction_ok = (
            len(diag) == 94
            and np.isfinite(probabilities).all()
            and float(probabilities.std()) > 0.0
            and float(probabilities.max() - probabilities.min()) > 1e-4
            and not (float(probabilities.max()) < 0.001 or float(probabilities.min()) > 0.999)
        )
        state_ok = (
            len(diag) == 94
            and np.isfinite(state_norms).all()
            and float(state_norms.std()) > 0.0
        )
        passed &= prediction_ok and state_ok
        rows.extend(
            [
                {
                    "seed": seed,
                    "category": "prediction_health",
                    "state_std": float(probabilities.std()) if len(probabilities) else np.nan,
                    "health_pass": prediction_ok,
                },
                {
                    "seed": seed,
                    "category": "patient_state_health",
                    "state_std": float(state_norms.std()) if len(state_norms) else np.nan,
                    "health_pass": state_ok,
                },
            ]
        )
        for category in MODULE_CATEGORIES:
            drift_values = drift.loc[
                (drift["seed"].astype(int) == seed) & (drift["category"] == category),
                "relative_parameter_drift",
            ].to_numpy(dtype=float)
            gradient = (
                float(selected_row[f"{category}_grad_norm"].iloc[0])
                if len(selected_row)
                else np.nan
            )
            row_pass = bool(
                len(drift_values)
                and np.isfinite(drift_values).all()
                and float(drift_values.max()) > 0.0
                and np.isfinite(gradient)
                and gradient > 0.0
            )
            passed &= row_pass
            rows.append(
                {
                    "seed": seed,
                    "category": category,
                    "relative_drift_mean": float(drift_values.mean()) if len(drift_values) else np.nan,
                    "relative_drift_maximum": float(drift_values.max()) if len(drift_values) else np.nan,
                    "selected_epoch_gradient_norm": gradient,
                    "health_pass": row_pass,
                }
            )
    return pd.DataFrame(rows), bool(passed)


def shortcut_audit(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        frame = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
        probability = frame[probability_column(frame)].to_numpy(dtype=float)
        selected_auc = shortcut_only_auc(frame)
        correlations = {
            field: safe_spearman(
                probability,
                pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float),
            )
            for field in SELECTED_SHORTCUT_FIELDS
        }
        maximum = max(abs(value) for value in correlations.values())
        raw_warnings = {}
        for field in RAW_SHORTCUT_FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
            raw_auc = auc(frame["label"], values)
            raw_warnings[field] = max(raw_auc, 1.0 - raw_auc)
        rows.append(
            {
                "seed": seed,
                "combination": "C37-E2E-VRL",
                "selected_structure_shortcut_only_label_AUC": selected_auc,
                "max_abs_prediction_selected_structure_spearman": maximum,
                "shortcut_safety_pass": selected_auc <= 0.55 and maximum <= 0.35,
                "shortcut_fields_used_as_model_inputs": False,
                **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
                **{
                    f"{field}_orientation_invariant_label_AUC_warning": value
                    for field, value in raw_warnings.items()
                },
            }
        )
    return pd.DataFrame(rows)


def freeze_validation_decision(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C37 validation decision requires validation-only metrics")
    comparisons, positive, inversions = validation_comparisons(config, run_dir, metrics)
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    health, health_pass = training_health(run_dir, epoch)
    shortcuts = shortcut_audit(run_dir)
    c37_auc = comparisons["AUC"].to_numpy(dtype=float)
    c27_auc = comparisons["C27_AUC"].to_numpy(dtype=float)
    mean_auc = float(c37_auc.mean())
    std_auc = float(c37_auc.std(ddof=1))
    c27_mean = float(config["c27"]["mean_validation_auc"])
    auc_pass = bool(
        mean_auc >= c27_mean + PROMOTION_GAIN
        and int((c37_auc > c27_auc).sum()) >= 2
        and float((c37_auc - c27_auc).min()) >= -0.01
        and std_auc <= 0.025
    )
    positive_pass = bool(
        int(positive["c17_tp_to_c37_fn"].sum()) <= int(positive["c17_fn_to_c37_tp"].sum())
        and float(positive["c37_minus_c17_sensitivity"].min()) >= -0.05
        and int(positive["c37_material_positive_damage_count"].sum())
        <= int(positive["c27_material_positive_damage_count"].sum())
    )
    ranking_pass = bool(
        float(inversions["C37_inversions"].mean()) <= float(inversions["C27_inversions"].mean())
        and int(inversions["C27_to_C37_repaired"].sum()) >= int(inversions["C27_to_C37_introduced"].sum())
        and int(inversions["C37_minus_C27_inversions"].max()) <= 10
    )
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    audit = pd.read_csv(report_dir / "c37_trainable_parameter_audit.csv")
    capacity_pass = int(audit.loc[audit["trainable"].astype(bool), "parameter_count"].sum()) <= int(
        config["c37"]["trainable_parameter_limit"]
    )
    if not capacity_pass or not health_pass:
        label = "DEMA_C37_TRAINING_INVALID"
    elif not shortcut_pass:
        label = "DEMA_C37_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "DEMA_C37_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "DEMA_C37_RANKING_WORSENING"
    elif not auc_pass:
        label = "DEMA_C37_NO_AUC_GAIN"
    else:
        label = "PROMOTE_DEMA_C37_E2E_VRL"
    promoted = label == "PROMOTE_DEMA_C37_E2E_VRL"
    median_seed = int(np.argsort(c37_auc)[len(c37_auc) // 2])
    deployment_seed = SEEDS[median_seed] if promoted else None
    decision = {
        "phase": "C37-E2E-VRL",
        "decision_label": label,
        "promoted": promoted,
        "strict_best": "DEMA_C37_E2E_VRL" if promoted else "KEEP_DEMA_C17_STRICT_BEST",
        "stop_label": None if promoted else "STOP_VISIT_LEVEL_MODEL_ROUTE",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C17": mean_auc - C17_MEAN,
        "mean_AUC_gain_vs_C27": mean_auc - c27_mean,
        "mean_AUC_gain_vs_C32": mean_auc - float(config["baselines"]["c32_mean_validation_auc"]),
        "mean_AUC_gain_vs_C33": mean_auc - float(config["baselines"]["c33_mean_validation_auc"]),
        "mean_AUC_gain_vs_C36": mean_auc - float(config["baselines"]["c36_mean_validation_auc"]),
        "mean_0_90_reached": mean_auc >= 0.90,
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
    comparisons.to_csv(report_dir / "c37_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
                "Sensitivity_mean": float(comparisons["Sensitivity"].mean()),
                "Specificity_mean": float(comparisons["Specificity"].mean()),
                "Balanced_ACC_mean": float(comparisons["Balanced_ACC"].mean()),
                "C17_AUC_mean": float(comparisons["C17_AUC"].mean()),
                "C27_AUC_mean": float(comparisons["C27_AUC"].mean()),
                "C32_AUC_mean": float(comparisons["C32_AUC"].mean()),
                "C33_AUC_mean": float(comparisons["C33_AUC"].mean()),
                "C36_AUC_mean": float(comparisons["C36_AUC"].mean()),
                "C37_minus_C27_AUC_mean": float(comparisons["C37_minus_C27_AUC"].mean()),
            }
        ]
    ).to_csv(report_dir / "c37_metrics_summary.csv", index=False)
    epoch.to_csv(report_dir / "c37_metrics_by_epoch.csv", index=False)
    pd.read_csv(run_dir / "reports" / "parameter_drift.csv").to_csv(
        report_dir / "c37_parameter_drift.csv", index=False
    )
    pd.read_csv(run_dir / "reports" / "patient_diagnostics_val.csv").to_csv(
        report_dir / "c37_patient_diagnostics_val.csv", index=False
    )
    health.to_csv(report_dir / "c37_training_health.csv", index=False)
    positive.to_csv(report_dir / "c37_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c37_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c37_shortcut_audit.csv", index=False)
    (report_dir / "c37_validation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    route_lines = [
        "# C37-E2E-VRL Validation Route Decision",
        "",
        f"- Decision: `{label}`.",
        f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
        f"- Mean gain versus C27: `{mean_auc - c27_mean:.10f}`.",
        f"- AUC/positive/ranking/shortcut/training gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{health_pass}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Validation decision was frozen before reporting-only evaluation.",
    ]
    (report_dir / "c37_route_decision.md").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )
    return decision


def write_final_report(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    decision = json.loads(
        (report_dir / "c37_validation_decision.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val", "test"}:
        raise RuntimeError("C37 final report requires validation and reporting-only rows")
    val = pd.read_csv(report_dir / "c37_metrics_by_seed.csv")
    test = metrics[metrics["split"] == "test"].sort_values("seed").copy()
    for column in (
        "C17_AUC",
        "C27_AUC",
        "C32_AUC",
        "C33_AUC",
        "C36_AUC",
        "C37_minus_C17_AUC",
        "C37_minus_C27_AUC",
        "C37_minus_C32_AUC",
        "C37_minus_C33_AUC",
        "C37_minus_C36_AUC",
    ):
        test[column] = np.nan
    pd.concat([val, test], ignore_index=True, sort=False).to_csv(
        report_dir / "c37_metrics_by_seed.csv", index=False
    )
    summary = pd.read_csv(report_dir / "c37_metrics_summary.csv")
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
    )
    summary.to_csv(report_dir / "c37_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c37_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c37_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c37_training_health.csv")
    shortcut = pd.read_csv(report_dir / "c37_shortcut_audit.csv")
    lines = [
        "# DMEA-HT Phase C37-E2E-VRL Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- Mean Validation gain versus C17: `{decision['mean_AUC_gain_vs_C17']:.10f}`.",
        f"- Mean Validation gain versus C27: `{decision['mean_AUC_gain_vs_C27']:.10f}`.",
        f"- Mean Validation gain versus C32/C33/C36: `{decision['mean_AUC_gain_vs_C32']:.10f}` / `{decision['mean_AUC_gain_vs_C33']:.10f}` / `{decision['mean_AUC_gain_vs_C36']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C37 FN / FN-to-C37 TP: `{int(positive['c17_tp_to_c37_fn'].sum())}`/`{int(positive['c17_fn_to_c37_tp'].sum())}`.",
        f"- Aggregate C27-to-C37 repaired/introduced pairs: `{int(inversions['C27_to_C37_repaired'].sum())}`/`{int(inversions['C27_to_C37_introduced'].sum())}`.",
        f"- Training health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Reporting-only results did not alter architecture, checkpoints, threshold, promotion, or deployment seed.",
        "- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.",
    ]
    (report_dir / "phase_c37_dema_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c37":
        raise RuntimeError("C37 report requires the formal C37 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        status = "C37_VALIDATION_DECISION_FROZEN"
    else:
        decision = write_final_report(config, run_dir, report_dir)
        status = "C37_FINAL_REPORT_COMPLETE"
    print(json.dumps({"status": status, "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
