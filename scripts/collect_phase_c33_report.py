#!/usr/bin/env python3
"""Freeze the C33-JERA validation decision and collect reporting-only Test results."""

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
C27_MEAN = 0.8822996831145314
C32_MEAN = 0.8743020974800061
PROMOTION_GAIN = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dema_ht_c33_jera_multiseed.yaml"
    )
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


def aligned_validation(
    config: Mapping[str, Any], run_dir: Path, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c33 = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
    c27 = read_prediction(
        resolve_path(config["c33"]["c27_run_dir"])
        / "predictions"
        / f"val_predictions_seed_{seed}.csv"
    )
    c17 = read_prediction(
        resolve_path(config["c33"]["c17_run_dir"])
        / "predictions"
        / f"val_predictions_seed_{seed}.csv"
    )
    c32 = read_prediction(
        resolve_path(config["c33"]["c32_run_dir"])
        / "predictions"
        / f"val_predictions_seed_{seed}.csv"
    )
    ids = c33["patient_id"].to_numpy(dtype=str)
    labels = c33["label"].to_numpy(dtype=int)
    for name, frame in (("C27", c27), ("C17", c17), ("C32", c32)):
        if not np.array_equal(ids, frame["patient_id"].to_numpy(dtype=str)):
            raise RuntimeError(f"C33 {name} patient alignment failed for seed {seed}")
        if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"C33 {name} label alignment failed for seed {seed}")
    if len(c33) != 94 or int((labels == 1).sum()) != 47:
        raise RuntimeError(f"C33 validation balance failed for seed {seed}")
    return c33, c27, c17, c32, labels


def validation_comparisons(
    config: Mapping[str, Any], run_dir: Path, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        c33, c27, c17, c32, labels = aligned_validation(config, run_dir, seed)
        c33_prob = c33[probability_column(c33)].to_numpy(dtype=float)
        c27_prob = c27[probability_column(c27)].to_numpy(dtype=float)
        c17_prob = c17[probability_column(c17)].to_numpy(dtype=float)
        c32_prob = c32[probability_column(c32)].to_numpy(dtype=float)
        c33_counts = binary_counts(labels, c33_prob)
        c27_counts = binary_counts(labels, c27_prob)
        c17_counts = binary_counts(labels, c17_prob)
        c32_counts = binary_counts(labels, c32_prob)
        metric = metrics[
            (metrics["seed"].astype(int) == seed) & (metrics["split"] == "val")
        ]
        if len(metric) != 1:
            raise RuntimeError(f"C33 validation metric row missing for seed {seed}")
        metric_row = metric.iloc[0].to_dict()
        c33_auc = float(metric_row["AUC"])
        c27_auc = auc(labels, c27_prob)
        c17_auc = auc(labels, c17_prob)
        c32_auc = auc(labels, c32_prob)
        metric_rows.append(
            {
                **metric_row,
                "C17_AUC": c17_auc,
                "C27_AUC": c27_auc,
                "C32_AUC": c32_auc,
                "C33_minus_C17_AUC": c33_auc - c17_auc,
                "C33_minus_C27_AUC": c33_auc - c27_auc,
                "C33_minus_C32_AUC": c33_auc - c32_auc,
                "minor_AUC_variation_vs_C27": abs(c33_auc - c27_auc) < 0.003,
            }
        )

        positive = labels == 1
        c17_class = c17_prob >= 0.5
        c27_class = c27_prob >= 0.5
        c32_class = c32_prob >= 0.5
        c33_class = c33_prob >= 0.5
        c27_damage = positive & (
            (c17_class & ~c27_class) | ((c27_prob - c17_prob) <= -0.05)
        )
        c32_damage = positive & (
            (c17_class & ~c32_class) | ((c32_prob - c17_prob) <= -0.05)
        )
        c33_damage = positive & (
            (c17_class & ~c33_class) | ((c33_prob - c17_prob) <= -0.05)
        )
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c33_fn": int((positive & c17_class & ~c33_class).sum()),
                "c17_fn_to_c33_tp": int((positive & ~c17_class & c33_class).sum()),
                "c33_sensitivity": c33_counts["Sensitivity"],
                "c17_sensitivity": c17_counts["Sensitivity"],
                "c27_sensitivity": c27_counts["Sensitivity"],
                "c32_sensitivity": c32_counts["Sensitivity"],
                "c33_minus_c17_sensitivity": (
                    c33_counts["Sensitivity"] - c17_counts["Sensitivity"]
                ),
                "c27_material_positive_damage_count": int(c27_damage.sum()),
                "c32_material_positive_damage_count": int(c32_damage.sum()),
                "c33_material_positive_damage_count": int(c33_damage.sum()),
                "c33_minus_c27_material_damage": int(
                    c33_damage.sum() - c27_damage.sum()
                ),
                "c33_minus_c32_material_damage": int(
                    c33_damage.sum() - c32_damage.sum()
                ),
            }
        )

        positive_ids = np.where(positive)[0]
        negative_ids = np.where(~positive)[0]

        def inversion(probabilities: np.ndarray) -> np.ndarray:
            return (
                probabilities[positive_ids, None]
                < probabilities[negative_ids][None, :]
            ).reshape(-1)

        inv17 = inversion(c17_prob)
        inv27 = inversion(c27_prob)
        inv32 = inversion(c32_prob)
        inv33 = inversion(c33_prob)
        inversion_rows.append(
            {
                "seed": seed,
                "total_pairs": int(len(inv33)),
                "C17_inversions": int(inv17.sum()),
                "C27_inversions": int(inv27.sum()),
                "C32_inversions": int(inv32.sum()),
                "C33_inversions": int(inv33.sum()),
                "C33_minus_C27_inversions": int(inv33.sum() - inv27.sum()),
                "C33_minus_C32_inversions": int(inv33.sum() - inv32.sum()),
                "C27_to_C33_repaired": int((inv27 & ~inv33).sum()),
                "C27_to_C33_introduced": int((~inv27 & inv33).sum()),
                "C32_to_C33_repaired": int((inv32 & ~inv33).sum()),
                "C32_to_C33_introduced": int((~inv32 & inv33).sum()),
                "minor_ranking_variation_vs_C27": abs(int(inv33.sum() - inv27.sum())) <= 3,
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(positive_rows),
        pd.DataFrame(inversion_rows),
    )


def parameter_health(
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    drift = pd.read_csv(run_dir / "reports" / "parameter_drift.csv")
    diagnostics = pd.read_csv(
        run_dir / "reports" / "patient_state_change_val.csv",
        dtype={"patient_id": str},
    )
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    selected = epoch[
        epoch["selected_by_val_auc"].astype(str).str.lower().eq("true")
    ]
    categories = ("image", "text", "bio", "patient_projection", "classifier")
    rows: List[Dict[str, Any]] = []
    valid = len(selected) == len(SEEDS)
    for seed in SEEDS:
        for category in categories:
            frame = drift[
                (drift["seed"].astype(int) == seed)
                & (drift["category"] == category)
            ]
            values = frame["relative_parameter_drift"].to_numpy(dtype=float)
            gradient_column = f"{category}_grad_norm"
            selected_row = selected[selected["seed"].astype(int) == seed]
            gradient = float(selected_row[gradient_column].iloc[0]) if len(selected_row) else float("nan")
            row_valid = bool(
                len(values)
                and np.isfinite(values).all()
                and float(values.max()) > 0.0
                and np.isfinite(gradient)
                and gradient > 0.0
            )
            valid &= row_valid
            rows.append(
                {
                    "seed": seed,
                    "category": category,
                    "parameter_count": int(frame["parameter_count"].sum()),
                    "relative_drift_mean": float(values.mean()),
                    "relative_drift_median": float(np.median(values)),
                    "relative_drift_maximum": float(values.max()),
                    "selected_epoch_gradient_norm": gradient,
                    "health_pass": row_valid,
                }
            )
    l2 = diagnostics["original_vs_adapted_l2_delta"].to_numpy(dtype=float)
    cosine = diagnostics["original_vs_adapted_cosine"].to_numpy(dtype=float)
    state_valid = bool(
        len(diagnostics)
        and np.isfinite(l2).all()
        and np.isfinite(cosine).all()
        and float(l2.max()) > 0.0
        and float(np.std(l2)) > 0.0
    )
    valid &= state_valid
    rows.append(
        {
            "seed": "all",
            "category": "patient_state",
            "parameter_count": 0,
            "relative_drift_mean": float("nan"),
            "relative_drift_median": float("nan"),
            "relative_drift_maximum": float("nan"),
            "selected_epoch_gradient_norm": float("nan"),
            "patient_state_cosine_mean": float(cosine.mean()),
            "patient_state_cosine_minimum": float(cosine.min()),
            "patient_state_l2_delta_mean": float(l2.mean()),
            "patient_state_l2_delta_maximum": float(l2.max()),
            "health_pass": state_valid,
        }
    )
    return pd.DataFrame(rows), diagnostics, valid


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
                "combination": "C33",
                "selected_structure_shortcut_only_label_AUC": selected_auc,
                "max_abs_prediction_selected_structure_spearman": maximum,
                "shortcut_safety_pass": selected_auc <= 0.55 and maximum <= 0.35,
                "shortcut_fields_used_as_model_inputs": False,
                **{
                    f"prediction_spearman_{field}": value
                    for field, value in correlations.items()
                },
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
        raise RuntimeError("C33 validation decision requires validation-only metrics")
    comparison, positive, inversions = validation_comparisons(config, run_dir, metrics)
    health_summary, diagnostics, health_pass = parameter_health(run_dir)
    shortcuts = shortcut_audit(run_dir)

    c33_auc = comparison["AUC"].to_numpy(dtype=float)
    c27_auc = comparison["C27_AUC"].to_numpy(dtype=float)
    mean_auc = float(c33_auc.mean())
    std_auc = float(c33_auc.std(ddof=1))
    auc_pass = bool(
        mean_auc >= C27_MEAN + PROMOTION_GAIN
        and int((c33_auc > c27_auc).sum()) >= 2
        and float((c33_auc - c27_auc).min()) >= -0.01
        and std_auc <= 0.025
    )
    positive_pass = bool(
        int(positive["c17_tp_to_c33_fn"].sum())
        <= int(positive["c17_fn_to_c33_tp"].sum())
        and float(positive["c33_minus_c17_sensitivity"].min()) >= -0.05
        and int(positive["c33_material_positive_damage_count"].sum())
        <= int(positive["c27_material_positive_damage_count"].sum())
    )
    ranking_pass = bool(
        float(inversions["C33_inversions"].mean())
        <= float(inversions["C27_inversions"].mean())
        and int(inversions["C27_to_C33_repaired"].sum())
        >= int(inversions["C27_to_C33_introduced"].sum())
        and int(inversions["C33_minus_C27_inversions"].max()) <= 10
    )
    shortcut_pass = bool(
        shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all()
    )
    if not health_pass:
        label = "DEMA_C33_TRAINING_INVALID"
    elif not shortcut_pass:
        label = "DEMA_C33_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "DEMA_C33_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "DEMA_C33_RANKING_WORSENING"
    elif not auc_pass:
        label = "DEMA_C33_NO_AUC_GAIN"
    else:
        label = "PROMOTE_DEMA_C33_JERA"
    promoted = label == "PROMOTE_DEMA_C33_JERA"
    median_seed = int(
        comparison.iloc[np.argsort(comparison["AUC"].to_numpy(dtype=float))[1]]["seed"]
    )
    decision = {
        "phase": "C33-JERA",
        "decision_label": label,
        "promoted": promoted,
        "strict_best": "DEMA_C33_JERA" if promoted else "KEEP_DEMA_C17_STRICT_BEST",
        "stop_label": None if promoted else "STOP_C33_JERA_TUNING",
        "validation_mean_AUC": mean_auc,
        "validation_std_AUC": std_auc,
        "mean_AUC_gain_vs_C17": mean_auc - C17_MEAN,
        "mean_AUC_gain_vs_C27": mean_auc - C27_MEAN,
        "mean_AUC_gain_vs_C32": mean_auc - C32_MEAN,
        "mean_0_90_reached": mean_auc >= 0.90,
        "auc_gate_pass": auc_pass,
        "positive_safety_pass": positive_pass,
        "ranking_safety_pass": ranking_pass,
        "shortcut_safety_pass": shortcut_pass,
        "training_health_pass": health_pass,
        "deployment_seed": median_seed if promoted else None,
        "deployment_checkpoint": (
            str(run_dir / "checkpoints" / f"seed_{median_seed}_best.pt")
            if promoted
            else None
        ),
        "validation_decision_frozen_before_test": True,
        "test_used_for_decision": False,
        "ensemble_used": False,
        "threshold_tuned": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "c33_metrics_by_seed.csv", index=False)
    summary = pd.DataFrame(
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
                "C32_AUC_mean": float(comparison["C32_AUC"].mean()),
                "C33_minus_C17_AUC_mean": float(comparison["C33_minus_C17_AUC"].mean()),
                "C33_minus_C27_AUC_mean": float(comparison["C33_minus_C27_AUC"].mean()),
                "C33_minus_C32_AUC_mean": float(comparison["C33_minus_C32_AUC"].mean()),
            }
        ]
    )
    summary.to_csv(report_dir / "c33_metrics_summary.csv", index=False)
    pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv").to_csv(
        report_dir / "c33_metrics_by_epoch.csv", index=False
    )
    health_summary.to_csv(report_dir / "c33_parameter_drift.csv", index=False)
    diagnostics.to_csv(report_dir / "c33_patient_state_change.csv", index=False)
    positive.to_csv(report_dir / "c33_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c33_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c33_shortcut_audit.csv", index=False)
    (report_dir / "c33_validation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    route_lines = [
        "# C33-JERA Validation Route Decision",
        "",
        f"- Decision: `{label}`.",
        f"- Validation AUC mean/std: `{mean_auc:.10f} +/- {std_auc:.10f}`.",
        f"- Mean gain versus C27: `{mean_auc - C27_MEAN:.10f}`.",
        f"- AUC/positive/ranking/shortcut/health gates: `{auc_pass}`/`{positive_pass}`/`{ranking_pass}`/`{shortcut_pass}`/`{health_pass}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Validation decision was frozen before reporting-only evaluation.",
    ]
    (report_dir / "c33_route_decision.md").write_text(
        "\n".join(route_lines) + "\n", encoding="utf-8"
    )
    return decision


def write_final_report(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    decision = json.loads(
        (report_dir / "c33_validation_decision.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val", "test"}:
        raise RuntimeError("C33 final report requires validation and reporting-only rows")
    val = pd.read_csv(report_dir / "c33_metrics_by_seed.csv")
    test = metrics[metrics["split"] == "test"].sort_values("seed")
    combined = pd.concat([val, test], ignore_index=True, sort=False)
    combined.to_csv(report_dir / "c33_metrics_by_seed.csv", index=False)
    summary = pd.read_csv(report_dir / "c33_metrics_summary.csv")
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
    summary.to_csv(report_dir / "c33_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c33_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c33_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c33_parameter_drift.csv")
    shortcut = pd.read_csv(report_dir / "c33_shortcut_audit.csv")
    lines = [
        "# DEMA-HT Phase C33-JERA Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- Mean Validation gain versus C27: `{decision['mean_AUC_gain_vs_C27']:.10f}`.",
        f"- Mean Validation gain versus C32: `{decision['mean_AUC_gain_vs_C32']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP->C33 FN / FN->C33 TP: `{int(positive['c17_tp_to_c33_fn'].sum())}`/`{int(positive['c17_fn_to_c33_tp'].sum())}`.",
        f"- Aggregate C27->C33 repaired/introduced pairs: `{int(inversions['C27_to_C33_repaired'].sum())}`/`{int(inversions['C27_to_C33_introduced'].sum())}`.",
        f"- Parameter health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Reporting-only results did not alter architecture, checkpoints, threshold, promotion, or deployment seed.",
        "- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.",
    ]
    (report_dir / "phase_c33_dema_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c33":
        raise RuntimeError("C33 report requires the formal C33 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        status = "C33_VALIDATION_DECISION_FROZEN"
    else:
        decision = write_final_report(config, run_dir, report_dir)
        status = "C33_FINAL_REPORT_COMPLETE"
    print(json.dumps({"status": status, "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
