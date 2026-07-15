#!/usr/bin/env python3
"""Freeze C34 validation, then collect reporting-only Test results."""

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


SEEDS = (0, 42, 3407)
C17_MEAN = 0.8696242643730194
PROMOTION_GAIN = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c34_msct_multiseed.yaml")
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    c34 = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
    c27 = read_prediction(
        resolve_path(config["c27"]["run_dir"])
        / "predictions"
        / f"val_predictions_seed_{seed}.csv"
    )
    c17 = read_prediction(
        resolve_path(config["c17"]["run_dir"])
        / "predictions"
        / f"val_predictions_seed_{seed}.csv"
    )
    ids = c34["patient_id"].to_numpy(dtype=str)
    labels = c34["label"].to_numpy(dtype=int)
    for name, frame in (("C27", c27), ("C17", c17)):
        if not np.array_equal(ids, frame["patient_id"].to_numpy(dtype=str)):
            raise RuntimeError(f"C34 {name} patient alignment failed for seed {seed}")
        if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"C34 {name} label alignment failed for seed {seed}")
    if len(c34) != 94 or int((labels == 1).sum()) != 47:
        raise RuntimeError(f"C34 validation balance failed for seed {seed}")
    return c34, c27, c17, labels


def validation_comparisons(
    config: Mapping[str, Any], run_dir: Path, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        c34, c27, c17, labels = aligned_validation(config, run_dir, seed)
        c34_prob = c34[probability_column(c34)].to_numpy(dtype=float)
        c27_prob = c27[probability_column(c27)].to_numpy(dtype=float)
        c17_prob = c17[probability_column(c17)].to_numpy(dtype=float)
        c34_counts = binary_counts(labels, c34_prob)
        c27_counts = binary_counts(labels, c27_prob)
        c17_counts = binary_counts(labels, c17_prob)
        metric = metrics[
            (metrics["seed"].astype(int) == seed) & (metrics["split"] == "val")
        ]
        if len(metric) != 1:
            raise RuntimeError(f"C34 validation metric row missing for seed {seed}")
        metric_row = metric.iloc[0].to_dict()
        c34_auc = float(metric_row["AUC"])
        c27_auc = auc(labels, c27_prob)
        c17_auc = auc(labels, c17_prob)
        positive = labels == 1
        c17_class = c17_prob >= 0.5
        c27_class = c27_prob >= 0.5
        c34_class = c34_prob >= 0.5
        c27_damage = positive & (
            (c17_class & ~c27_class) | ((c27_prob - c17_prob) <= -0.05)
        )
        c34_damage = positive & (
            (c17_class & ~c34_class) | ((c34_prob - c17_prob) <= -0.05)
        )
        metric_rows.append(
            {
                **metric_row,
                "C17_AUC": c17_auc,
                "C27_AUC": c27_auc,
                "C34_minus_C17_AUC": c34_auc - c17_auc,
                "C34_minus_C27_AUC": c34_auc - c27_auc,
                "minor_AUC_variation_vs_C27": abs(c34_auc - c27_auc) < 0.003,
            }
        )
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c34_fn": int((positive & c17_class & ~c34_class).sum()),
                "c17_fn_to_c34_tp": int((positive & ~c17_class & c34_class).sum()),
                "c34_sensitivity": c34_counts["Sensitivity"],
                "c17_sensitivity": c17_counts["Sensitivity"],
                "c27_sensitivity": c27_counts["Sensitivity"],
                "c34_minus_c17_sensitivity": c34_counts["Sensitivity"]
                - c17_counts["Sensitivity"],
                "c27_material_positive_damage_count": int(c27_damage.sum()),
                "c34_material_positive_damage_count": int(c34_damage.sum()),
                "c34_minus_c27_material_damage": int(
                    c34_damage.sum() - c27_damage.sum()
                ),
            }
        )
        inv17 = inversion_vector(labels, c17_prob)
        inv27 = inversion_vector(labels, c27_prob)
        inv34 = inversion_vector(labels, c34_prob)
        inversion_rows.append(
            {
                "seed": seed,
                "total_pairs": int(len(inv34)),
                "C17_inversions": int(inv17.sum()),
                "C27_inversions": int(inv27.sum()),
                "C34_inversions": int(inv34.sum()),
                "C34_minus_C27_inversions": int(inv34.sum() - inv27.sum()),
                "C27_to_C34_repaired": int((inv27 & ~inv34).sum()),
                "C27_to_C34_introduced": int((~inv27 & inv34).sum()),
                "minor_ranking_variation_vs_C27": abs(
                    int(inv34.sum() - inv27.sum())
                )
                <= 3,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(positive_rows), pd.DataFrame(inversion_rows)


def state_health(
    run_dir: Path, metrics_by_epoch: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, bool]:
    source = pd.read_csv(run_dir / "reports" / "source_state_audit_val.csv")
    trajectory = pd.read_csv(run_dir / "reports" / "trajectory_state_audit_val.csv")
    diagnostics = pd.read_csv(
        run_dir / "reports" / "patient_diagnostics_val.csv", dtype={"patient_id": str}
    )
    drift = pd.read_csv(run_dir / "reports" / "parameter_drift.csv")
    health_rows: List[Dict[str, Any]] = []
    state_pass = True
    for seed in SEEDS:
        for source_name in sorted(source["source"].unique()):
            values = source.loc[
                (source["seed"].astype(int) == seed)
                & (source["source"] == source_name)
                & source["observed"].astype(str).str.lower().eq("true"),
                "state_mean",
            ].to_numpy(dtype=float)
            finite = len(values) > 0 and np.isfinite(values).all()
            varying = len(values) > 1 and float(np.std(values)) > 0.0
            unsaturated = finite and float(values.min()) > 0.0 and float(values.max()) < 1.0
            row_pass = bool(finite and varying and unsaturated)
            state_pass &= row_pass
            health_rows.append(
                {
                    "seed": seed,
                    "category": f"source_{source_name}",
                    "parameter_count": 0,
                    "relative_drift_mean": np.nan,
                    "relative_drift_maximum": np.nan,
                    "state_mean": float(values.mean()) if len(values) else np.nan,
                    "state_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "state_min": float(values.min()) if len(values) else np.nan,
                    "state_max": float(values.max()) if len(values) else np.nan,
                    "health_pass": row_pass,
                }
            )
        traj = trajectory[trajectory["seed"].astype(int) == seed]
        finite = bool(
            len(traj)
            and np.isfinite(traj[list(("latest_state", "history_state", "state_delta", "history_dispersion", "latest_source_disagreement"))].to_numpy(dtype=float)).all()
        )
        multi = traj[~traj["single_visit"].astype(str).str.lower().eq("true")]
        delta_nonconstant = bool(len(multi) and float(multi["state_delta"].std()) > 0.0)
        trajectory_pass = finite and float(traj["latest_state"].std()) > 0.0 and delta_nonconstant
        state_pass &= trajectory_pass
        health_rows.append(
            {
                "seed": seed,
                "category": "trajectory",
                "parameter_count": 0,
                "relative_drift_mean": np.nan,
                "relative_drift_maximum": np.nan,
                "state_mean": float(traj["latest_state"].mean()) if len(traj) else np.nan,
                "state_std": float(traj["latest_state"].std()) if len(traj) > 1 else np.nan,
                "state_min": float(traj["latest_state"].min()) if len(traj) else np.nan,
                "state_max": float(traj["latest_state"].max()) if len(traj) else np.nan,
                "health_pass": trajectory_pass,
            }
        )

    selected = metrics_by_epoch[
        metrics_by_epoch["selected_by_val_auc"].astype(str).str.lower().eq("true")
    ]
    training_pass = len(selected) == len(SEEDS)
    for category in ("source_heads", "fallback_tokens", "empty_visit_state", "trajectory_norm", "classifier"):
        for seed in SEEDS:
            drift_values = drift.loc[
                (drift["seed"].astype(int) == seed) & (drift["category"] == category),
                "relative_parameter_drift",
            ].to_numpy(dtype=float)
            selected_row = selected[selected["seed"].astype(int) == seed]
            grad_col = f"{category}_grad_norm"
            gradient = float(selected_row[grad_col].iloc[0]) if len(selected_row) else np.nan
            row_pass = bool(
                len(drift_values)
                and np.isfinite(drift_values).all()
                and float(drift_values.max()) > 0.0
                and np.isfinite(gradient)
                and gradient > 0.0
            )
            training_pass &= row_pass
            health_rows.append(
                {
                    "seed": seed,
                    "category": category,
                    "parameter_count": int(
                        drift.loc[
                            (drift["seed"].astype(int) == seed)
                            & (drift["category"] == category),
                            "parameter_count",
                        ].sum()
                    ),
                    "relative_drift_mean": float(drift_values.mean()) if len(drift_values) else np.nan,
                    "relative_drift_maximum": float(drift_values.max()) if len(drift_values) else np.nan,
                    "selected_epoch_gradient_norm": gradient,
                    "health_pass": row_pass,
                }
            )
    prediction_std_pass = bool(
        len(diagnostics)
        and diagnostics.groupby("seed")["final_prob"].std().gt(0.0).all()
    )
    training_pass &= prediction_std_pass
    return (
        pd.DataFrame(health_rows),
        source,
        trajectory,
        diagnostics,
        bool(state_pass),
        bool(training_pass),
    )


def shortcut_audit(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        frame = read_prediction(
            run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"
        )
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
        raw_warnings: Dict[str, float] = {}
        for field in RAW_SHORTCUT_FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
            raw_auc = auc(frame["label"], values)
            raw_warnings[field] = max(raw_auc, 1.0 - raw_auc)
        rows.append(
            {
                "seed": seed,
                "combination": "C34",
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
        raise RuntimeError("C34 validation decision requires validation-only metrics")
    comparison, positive, inversions = validation_comparisons(config, run_dir, metrics)
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    health, source, trajectory, diagnostics, state_pass, training_pass = state_health(
        run_dir, epoch
    )
    shortcuts = shortcut_audit(run_dir)
    c34_auc = comparison["AUC"].to_numpy(dtype=float)
    c27_auc = comparison["C27_AUC"].to_numpy(dtype=float)
    c27_mean = float(config["c27"]["mean_validation_auc"])
    mean_auc = float(c34_auc.mean())
    std_auc = float(c34_auc.std(ddof=1))
    auc_pass = bool(
        mean_auc >= c27_mean + PROMOTION_GAIN
        and int((c34_auc > c27_auc).sum()) >= 2
        and float((c34_auc - c27_auc).min()) >= -0.01
        and std_auc <= 0.025
    )
    positive_pass = bool(
        int(positive["c17_tp_to_c34_fn"].sum())
        <= int(positive["c17_fn_to_c34_tp"].sum())
        and float(positive["c34_minus_c17_sensitivity"].min()) >= -0.05
        and int(positive["c34_material_positive_damage_count"].sum())
        <= int(positive["c27_material_positive_damage_count"].sum())
    )
    ranking_pass = bool(
        float(inversions["C34_inversions"].mean())
        <= float(inversions["C27_inversions"].mean())
        and int(inversions["C27_to_C34_repaired"].sum())
        >= int(inversions["C27_to_C34_introduced"].sum())
        and int(inversions["C34_minus_C27_inversions"].max()) <= 10
    )
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    parameter_count = int(pd.read_csv(resolve_path(config["project"]["report_dir"]) / "c34_trainable_parameter_audit.csv").query("trainable == True")["parameter_count"].sum())
    capacity_pass = parameter_count <= int(config["c34"]["trainable_parameter_limit"])
    if not capacity_pass:
        label = "DEMA_C34_CAPACITY_CONTRACT_FAIL"
    elif not training_pass:
        label = "DEMA_C34_TRAINING_INVALID"
    elif not state_pass:
        label = "DEMA_C34_STATE_COLLAPSE"
    elif not shortcut_pass:
        label = "DEMA_C34_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "DEMA_C34_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "DEMA_C34_RANKING_WORSENING"
    elif not auc_pass:
        label = "DEMA_C34_NO_AUC_GAIN"
    else:
        label = "PROMOTE_DEMA_C34_MSCT"
    promoted = label == "PROMOTE_DEMA_C34_MSCT"
    median_seed = int(np.argsort(c34_auc)[len(c34_auc) // 2])
    deployment_seed = SEEDS[median_seed] if promoted else None
    decision = {
        "phase": "C34-MSCT",
        "decision_label": label,
        "promoted": promoted,
        "strict_best": "DEMA_C34_MSCT" if promoted else "KEEP_DEMA_C17_STRICT_BEST",
        "stop_label": None if promoted else "STOP_C34_MSCT_TUNING",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C17": mean_auc - C17_MEAN,
        "mean_AUC_gain_vs_C27": mean_auc - c27_mean,
        "mean_0_90_reached": mean_auc >= float(config["c34"]["auc_target"]),
        "auc_gate_pass": auc_pass,
        "positive_safety_pass": positive_pass,
        "ranking_safety_pass": ranking_pass,
        "shortcut_safety_pass": shortcut_pass,
        "state_health_pass": state_pass,
        "training_health_pass": training_pass,
        "capacity_gate_pass": capacity_pass,
        "deployment_seed": deployment_seed,
        "deployment_checkpoint": (
            str(run_dir / "checkpoints" / f"seed_{deployment_seed}_best.pt")
            if promoted
            else None
        ),
        "validation_decision_frozen_before_test": True,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "threshold_tuned": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "c34_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
                "Sensitivity_mean": float(comparison["Sensitivity"].mean()),
                "Specificity_mean": float(comparison["Specificity"].mean()),
                "Balanced_ACC_mean": float(comparison["Balanced_ACC"].mean()),
                "C17_AUC_mean": float(comparison["C17_AUC"].mean()),
                "C27_AUC_mean": float(comparison["C27_AUC"].mean()),
                "C34_minus_C17_AUC_mean": float(comparison["C34_minus_C17_AUC"].mean()),
                "C34_minus_C27_AUC_mean": float(comparison["C34_minus_C27_AUC"].mean()),
            }
        ]
    ).to_csv(report_dir / "c34_metrics_summary.csv", index=False)
    epoch.to_csv(report_dir / "c34_metrics_by_epoch.csv", index=False)
    health.to_csv(report_dir / "c34_parameter_drift.csv", index=False)
    source_summary = (
        source.groupby(["seed", "source"], as_index=False)
        .agg(
            observed_patients=("observed", "sum"),
            state_mean_mean=("state_mean", "mean"),
            state_mean_std=("state_mean", "std"),
            state_min=("state_mean", "min"),
            state_max=("state_mean", "max"),
        )
    )
    source_summary.to_csv(report_dir / "c34_source_state_audit.csv", index=False)
    trajectory_summary = (
        trajectory.groupby(["seed", "single_visit"], as_index=False)[
            ["latest_state", "history_state", "state_delta", "history_dispersion", "latest_source_disagreement"]
        ]
        .agg(["mean", "std"])
    )
    trajectory_summary.to_csv(report_dir / "c34_trajectory_state_audit.csv", index=False)
    diagnostics.to_csv(report_dir / "c34_patient_diagnostics_val.csv", index=False)
    positive.to_csv(report_dir / "c34_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c34_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c34_shortcut_audit.csv", index=False)
    (report_dir / "c34_validation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    route_lines = [
        "# C34-MSCT Validation Route Decision",
        "",
        f"- Decision: `{label}`.",
        f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
        f"- Mean gain versus C27: `{mean_auc - c27_mean:.10f}`.",
        f"- AUC/positive/ranking/shortcut/state/training gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{state_pass}`/`{training_pass}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Validation decision was frozen before reporting-only evaluation.",
    ]
    (report_dir / "c34_route_decision.md").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )
    return decision


def write_final_report(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    decision = json.loads(
        (report_dir / "c34_validation_decision.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val", "test"}:
        raise RuntimeError("C34 final report requires validation and reporting-only rows")
    val = pd.read_csv(report_dir / "c34_metrics_by_seed.csv")
    test = metrics[metrics["split"] == "test"].sort_values("seed")
    combined = pd.concat([val, test], ignore_index=True, sort=False)
    combined.to_csv(report_dir / "c34_metrics_by_seed.csv", index=False)
    summary = pd.read_csv(report_dir / "c34_metrics_summary.csv")
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
    summary.to_csv(report_dir / "c34_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c34_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c34_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c34_parameter_drift.csv")
    shortcut = pd.read_csv(report_dir / "c34_shortcut_audit.csv")
    source = pd.read_csv(report_dir / "c34_source_state_audit.csv")
    trajectory = pd.read_csv(report_dir / "c34_trajectory_state_audit.csv")
    lines = [
        "# DEMA-HT Phase C34-MSCT Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- Mean Validation gain versus C27: `{decision['mean_AUC_gain_vs_C27']:.10f}`.",
        f"- Mean Validation gain versus C17: `{decision['mean_AUC_gain_vs_C17']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C34 FN / FN-to-C34 TP: `{int(positive['c17_tp_to_c34_fn'].sum())}`/`{int(positive['c17_fn_to_c34_tp'].sum())}`.",
        f"- Aggregate C27-to-C34 repaired/introduced pairs: `{int(inversions['C27_to_C34_repaired'].sum())}`/`{int(inversions['C27_to_C34_introduced'].sum())}`.",
        f"- State audit rows: `{len(source)}` source summaries and `{len(trajectory)}` trajectory summaries.",
        f"- Parameter/state health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Reporting-only results did not alter architecture, checkpoints, threshold, promotion, or deployment seed.",
        "- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.",
    ]
    (report_dir / "phase_c34_dema_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c34":
        raise RuntimeError("C34 report requires the formal C34 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        status = "C34_VALIDATION_DECISION_FROZEN"
    else:
        decision = write_final_report(config, run_dir, report_dir)
        status = "C34_FINAL_REPORT_COMPLETE"
    print(json.dumps({"status": status, "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
