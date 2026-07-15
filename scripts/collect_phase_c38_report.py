#!/usr/bin/env python3
"""Freeze and report C38-MPES Validation and reporting-only Test evidence."""

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
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.collect_phase_c31a_report import (  # noqa: E402
    RAW_SHORTCUT_FIELDS,
    SELECTED_SHORTCUT_FIELDS,
    safe_spearman,
    shortcut_only_auc,
)
from scripts.train_phase_c38 import SEEDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c38_mpes_multiseed.yaml")
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def ensure_audit_shortcut_columns(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    missing = [field for field in SELECTED_SHORTCUT_FIELDS if field not in frame.columns]
    if not missing:
        return frame
    derived: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(config["project"]["manifest"]):
        patient_id = str(row["patient_id"])
        visits = list(row.get("visits") or [])
        report_present = [bool(str(visit.get("report_text", "") or "").strip()) for visit in visits]
        derived[patient_id] = {
            "reconstructable_visit_count": int(sum(report_present)),
            "visit_report_coverage": float(sum(report_present) / len(visits)) if visits else 0.0,
            "dated_bio_visit_count": int(
                sum(visit.get("dated_bio_row_id") is not None for visit in visits)
            ),
        }
    result = frame.copy()
    for field in missing:
        result[field] = result["patient_id"].map(
            {patient_id: values.get(field, np.nan) for patient_id, values in derived.items()}
        )
    if any(field not in result.columns or result[field].isna().all() for field in missing):
        raise RuntimeError(f"C38 audit-only shortcut columns unavailable: {missing}")
    return result


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


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
    }


def inversion_vector(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    return (probabilities[positive, None] < probabilities[negative][None, :]).reshape(-1)


def aligned_validation(
    config: Mapping[str, Any], run_dir: Path, seed: int
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]:
    c38 = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
    baselines = {
        "C17": read_prediction(
            Path(config["c17"]["c17_run_dir"])
            / "predictions"
            / f"val_predictions_seed_{seed}.csv"
        ),
        "C27": read_prediction(
            Path(config["c27"]["c27_run_dir"])
            / "predictions"
            / f"val_predictions_seed_{seed}.csv"
        ),
    }
    ids = c38["patient_id"].to_numpy(dtype=str)
    labels = c38["label"].to_numpy(dtype=int)
    if len(c38) != 94 or int((labels == 1).sum()) != 47:
        raise RuntimeError(f"C38 validation balance failed for seed {seed}")
    for name, frame in baselines.items():
        if not np.array_equal(ids, frame["patient_id"].to_numpy(dtype=str)):
            raise RuntimeError(f"C38 {name} patient alignment failed for seed {seed}")
        if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"C38 {name} label alignment failed for seed {seed}")
    return c38, baselines, labels


def validation_comparisons(
    config: Mapping[str, Any], run_dir: Path, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    inversion_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        c38, baselines, labels = aligned_validation(config, run_dir, seed)
        probabilities = {
            "C38": c38[probability_column(c38)].to_numpy(dtype=float),
            **{
                name: frame[probability_column(frame)].to_numpy(dtype=float)
                for name, frame in baselines.items()
            },
        }
        metric = metrics[(metrics["seed"].astype(int) == seed) & (metrics["split"] == "val")]
        if len(metric) != 1:
            raise RuntimeError(f"C38 validation metric row missing for seed {seed}")
        row = metric.iloc[0].to_dict()
        metric_rows.append(
            {
                **row,
                **{f"{name}_AUC": auc(labels, value) for name, value in probabilities.items()},
                **{
                    f"C38_minus_{name}_AUC": float(row["AUC"]) - auc(labels, value)
                    for name, value in probabilities.items()
                    if name != "C38"
                },
            }
        )
        c17_class = probabilities["C17"] >= 0.5
        c38_class = probabilities["C38"] >= 0.5
        c17_counts = binary_counts(labels, probabilities["C17"])
        c38_counts = binary_counts(labels, probabilities["C38"])
        positive = labels == 1
        positive_rows.append(
            {
                "seed": seed,
                "c17_tp_to_c38_fn": int((positive & c17_class & ~c38_class).sum()),
                "c17_fn_to_c38_tp": int((positive & ~c17_class & c38_class).sum()),
                "c17_sensitivity": c17_counts["Sensitivity"],
                "c38_sensitivity": c38_counts["Sensitivity"],
                "c38_minus_c17_sensitivity": c38_counts["Sensitivity"] - c17_counts["Sensitivity"],
            }
        )
        c27_inversions = inversion_vector(labels, probabilities["C27"])
        c38_inversions = inversion_vector(labels, probabilities["C38"])
        inversion_rows.append(
            {
                "seed": seed,
                "C17_inversions": int(inversion_vector(labels, probabilities["C17"]).sum()),
                "C27_inversions": int(c27_inversions.sum()),
                "C38_inversions": int(c38_inversions.sum()),
                "C38_minus_C27_inversions": int(c38_inversions.sum() - c27_inversions.sum()),
                "C38_inversion_ratio_vs_C27": float(
                    (c38_inversions.sum() - c27_inversions.sum()) / max(c27_inversions.sum(), 1)
                ),
                "C27_to_C38_repaired": int((c27_inversions & ~c38_inversions).sum()),
                "C27_to_C38_introduced": int((~c27_inversions & c38_inversions).sum()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(positive_rows), pd.DataFrame(inversion_rows)


def training_health(run_dir: Path, epoch: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    drift = pd.read_csv(run_dir / "reports" / "parameter_drift.csv")
    diagnostics = pd.read_csv(
        run_dir / "reports" / "patient_diagnostics_val.csv", dtype={"patient_id": str}
    )
    selected = epoch[epoch["selected_by_val_auc"].astype(str).str.lower().eq("true")]
    rows: List[Dict[str, Any]] = []
    passed = len(selected) == len(SEEDS)
    for seed in SEEDS:
        selected_row = selected[selected["seed"].astype(int) == seed]
        diag = diagnostics[diagnostics["seed"].astype(int) == seed]
        probabilities = diag["final_prob"].to_numpy(dtype=float)
        states = diag["patient_state_norm"].to_numpy(dtype=float)
        prediction_ok = (
            len(diag) == 94
            and np.isfinite(probabilities).all()
            and float(probabilities.std()) > 0.0
            and float(probabilities.max() - probabilities.min()) > 1e-4
        )
        state_ok = len(diag) == 94 and np.isfinite(states).all() and float(states.std()) > 0.0
        head_gradient = float(selected_row["head_grad_norm"].iloc[0]) if len(selected_row) else np.nan
        drift_values = drift[drift["seed"].astype(int) == seed]["relative_parameter_drift"].to_numpy(dtype=float)
        head_ok = bool(np.isfinite(head_gradient) and head_gradient > 0.0 and len(drift_values) > 0 and np.isfinite(drift_values).all())
        passed &= prediction_ok and state_ok and head_ok
        rows.extend(
            [
                {"seed": seed, "category": "prediction_health", "state_std": float(probabilities.std()), "health_pass": prediction_ok},
                {"seed": seed, "category": "patient_state_health", "state_std": float(states.std()), "health_pass": state_ok},
                {"seed": seed, "category": "new_head_health", "selected_epoch_gradient_norm": head_gradient, "relative_drift_maximum": float(drift_values.max()) if len(drift_values) else np.nan, "health_pass": head_ok},
            ]
        )
    return pd.DataFrame(rows), bool(passed)


def shortcut_audit(config: Mapping[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        frame = ensure_audit_shortcut_columns(
            read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
            config,
        )
        probability = frame[probability_column(frame)].to_numpy(dtype=float)
        correlations = {
            field: safe_spearman(
                probability,
                pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float),
            )
            for field in SELECTED_SHORTCUT_FIELDS
        }
        maximum = max(abs(value) for value in correlations.values())
        raw_warnings = {
            field: max(
                auc(frame["label"], pd.to_numeric(frame[field], errors="coerce").fillna(0.0)),
                1.0 - auc(frame["label"], pd.to_numeric(frame[field], errors="coerce").fillna(0.0)),
            )
            for field in RAW_SHORTCUT_FIELDS
        }
        selected_auc = shortcut_only_auc(frame)
        rows.append(
            {
                "seed": seed,
                "combination": "C38-MPES",
                "selected_structure_shortcut_only_label_AUC": selected_auc,
                "max_abs_prediction_selected_structure_spearman": maximum,
                "shortcut_safety_pass": selected_auc <= 0.55 and maximum <= 0.35,
                "shortcut_fields_used_as_model_inputs": False,
                **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
                **{f"{field}_orientation_invariant_label_AUC_warning": value for field, value in raw_warnings.items()},
            }
        )
    return pd.DataFrame(rows)


def freeze_validation_decision(
    config: Mapping[str, Any], run_dir: Path, report_dir: Path
) -> Dict[str, Any]:
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C38 validation decision requires validation-only metrics")
    comparisons, positive, inversions = validation_comparisons(config, run_dir, metrics)
    epoch = pd.read_csv(run_dir / "reports" / "metrics_by_epoch.csv")
    health, health_pass = training_health(run_dir, epoch)
    shortcuts = shortcut_audit(config, run_dir)
    auc_values = comparisons["AUC"].to_numpy(dtype=float)
    c27_values = comparisons["C27_AUC"].to_numpy(dtype=float)
    mean_auc = float(auc_values.mean())
    std_auc = float(auc_values.std(ddof=1))
    auc_pass = bool(
        mean_auc >= 0.9000
        and int((auc_values >= 0.9000).sum()) >= 2
        and std_auc <= 0.025
    )
    positive_pass = bool(
        float(positive["c38_minus_c17_sensitivity"].min()) >= -0.10
        and int(positive["c17_tp_to_c38_fn"].sum())
        <= int(positive["c17_fn_to_c38_tp"].sum()) + 3
    )
    mean_c27_inversions = float(inversions["C27_inversions"].mean())
    mean_c38_inversions = float(inversions["C38_inversions"].mean())
    ranking_pass = bool(
        (mean_c38_inversions - mean_c27_inversions) / max(mean_c27_inversions, 1.0) <= 0.10
        and int(inversions["C38_minus_C27_inversions"].max()) <= 20
    )
    shortcut_pass = bool(shortcuts["shortcut_safety_pass"].astype(str).str.lower().eq("true").all())
    capacity_pass = True
    for path in sorted((run_dir / "seed_runs").glob("seed_*/reports/run_config.json")):
        runtime = json.loads(path.read_text(encoding="utf-8"))
        capacity_pass &= int(runtime["trainable_parameter_count"]) <= int(config["c38"]["trainable_parameter_limit"])

    if not capacity_pass or not health_pass:
        label = "C38_TRAINING_INVALID"
    elif not shortcut_pass:
        label = "C38_SHORTCUT_CONCERN"
    elif not positive_pass:
        label = "C38_POSITIVE_DAMAGE"
    elif not ranking_pass:
        label = "C38_RANKING_DAMAGE"
    elif not auc_pass:
        label = "C38_NO_AUC_GAIN"
    else:
        label = "GOAL_REACHED_DEMA_HT_AUC_090_PLUS"
    promoted = label == "GOAL_REACHED_DEMA_HT_AUC_090_PLUS"
    median_index = int(np.argsort(auc_values)[len(auc_values) // 2])
    deployment_seed = SEEDS[median_index] if promoted else None
    decision = {
        "phase": "C38-MPES",
        "decision_label": label,
        "goal_reached": promoted,
        "strict_best": "C38_MPES" if promoted else "KEEP_DEMA_C17_STRICT_BEST",
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
    comparisons.to_csv(report_dir / "c38_metrics_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
                "C17_AUC_mean": float(comparisons["C17_AUC"].mean()),
                "C27_AUC_mean": float(comparisons["C27_AUC"].mean()),
                "C38_minus_C27_AUC_mean": float(comparisons["C38_minus_C27_AUC"].mean()),
            }
        ]
    ).to_csv(report_dir / "c38_metrics_summary.csv", index=False)
    epoch.to_csv(report_dir / "c38_metrics_by_epoch.csv", index=False)
    pd.read_csv(run_dir / "reports" / "parameter_drift.csv").to_csv(report_dir / "c38_parameter_drift.csv", index=False)
    pd.read_csv(run_dir / "reports" / "patient_diagnostics_val.csv").to_csv(report_dir / "c38_patient_diagnostics_val.csv", index=False)
    health.to_csv(report_dir / "c38_training_health.csv", index=False)
    positive.to_csv(report_dir / "c38_positive_preservation.csv", index=False)
    inversions.to_csv(report_dir / "c38_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(report_dir / "c38_shortcut_audit.csv", index=False)
    (report_dir / "c38_validation_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (report_dir / "c38_route_decision.md").write_text(
        "\n".join(
            [
                "# C38-MPES Validation Decision",
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
    decision = json.loads((report_dir / "c38_validation_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")
    if set(metrics["split"]) != {"val", "test"}:
        raise RuntimeError("C38 final report requires validation and reporting-only rows")
    test = metrics[metrics["split"] == "test"]
    summary = pd.read_csv(report_dir / "c38_metrics_summary.csv")
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
    summary.to_csv(report_dir / "c38_metrics_summary.csv", index=False)
    positive = pd.read_csv(report_dir / "c38_positive_preservation.csv")
    inversions = pd.read_csv(report_dir / "c38_pairwise_inversion_summary.csv")
    health = pd.read_csv(report_dir / "c38_training_health.csv")
    shortcut = pd.read_csv(report_dir / "c38_shortcut_audit.csv")
    lines = [
        "# DMEA-HT Phase C38-MPES Final Report",
        "",
        f"- Decision: `{decision['decision_label']}`.",
        f"- Validation AUC mean/std: `{decision['validation_mean_AUC']:.10f} +/- {decision['validation_std_AUC']:.10f}`.",
        f"- Mean Validation gain versus C17/C27: `{decision['mean_AUC_gain_vs_C17']:.10f}` / `{decision['mean_AUC_gain_vs_C27']:.10f}`.",
        f"- Reporting-only Test AUC mean/std: `{test['AUC'].mean():.10f} +/- {test['AUC'].std(ddof=1):.10f}`.",
        f"- Aggregate C17 TP-to-C38 FN / FN-to-C38 TP: `{int(positive['c17_tp_to_c38_fn'].sum())}`/`{int(positive['c17_fn_to_c38_tp'].sum())}`.",
        f"- C27-to-C38 repaired/introduced pairs: `{int(inversions['C27_to_C38_repaired'].sum())}`/`{int(inversions['C27_to_C38_introduced'].sum())}`.",
        f"- Training health rows passed: `{int(health['health_pass'].astype(str).str.lower().eq('true').sum())}/{len(health)}`.",
        f"- Shortcut-only label AUC max: `{shortcut['selected_structure_shortcut_only_label_AUC'].max():.10f}`.",
        f"- Deployment checkpoint: `{decision['deployment_checkpoint'] or 'none'}`.",
        "- Test was reporting-only and did not alter Validation selection or the decision.",
        "- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.",
    ]
    (report_dir / "phase_c38_dema_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c38":
        raise RuntimeError("C38 report requires the formal C38 config")
    run_dir = resolve_path(config["project"]["output_dir"])
    report_dir = resolve_path(config["project"]["report_dir"])
    if args.stage == "validation":
        decision = freeze_validation_decision(config, run_dir, report_dir)
        print(json.dumps({"status": "C38_VALIDATION_DECISION_FROZEN", "decision": decision["decision_label"]}))
    else:
        decision = write_final_report(config, run_dir, report_dir)
        print(json.dumps({"status": "C38_FINAL_REPORT_COMPLETE", "decision": decision["decision_label"]}))


if __name__ == "__main__":
    main()
