#!/usr/bin/env python3
"""Reporting and audit helpers for Phase C64-STCV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts import c64_common as common
from scripts import collect_phase_c41_report as base
from scripts import train_phase_c40 as core


SEEDS = common.SEEDS
SHORTCUT_MAX_AUC = 0.55
SHORTCUT_MAX_SPEARMAN = 0.35


def read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if "patient_id" not in frame or "label" not in frame:
        raise RuntimeError(f"Prediction file lacks patient_id/label: {path}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No prediction probability column in {list(frame.columns)}")


def auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, p))


def binary_counts(labels: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= 0.5
    positive = y == 1
    negative = y == 0
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


def inversion_vector(labels: Sequence[int], probabilities: Sequence[float]) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    return (p[positive, None] < p[negative][None, :]).reshape(-1)


def ensure_shortcuts(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    return base.ensure_audit_shortcut_columns(frame.copy(), config)


def shortcut_row(frame: pd.DataFrame, config: Mapping[str, Any], identity: Mapping[str, Any]) -> Dict[str, Any]:
    frame = ensure_shortcuts(frame, config)
    probability = frame[probability_column(frame)].to_numpy(dtype=float)
    correlations = {
        field: base.safe_spearman(
            probability,
            pd.to_numeric(frame[field], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        )
        for field in base.SELECTED_SHORTCUT_FIELDS
    }
    selected_auc = float(base.shortcut_only_auc(frame))
    maximum = max(abs(value) for value in correlations.values())
    return {
        **dict(identity),
        "selected_structure_shortcut_only_label_AUC": selected_auc,
        "max_abs_prediction_selected_structure_spearman": float(maximum),
        "shortcut_safety_pass": bool(selected_auc <= SHORTCUT_MAX_AUC and maximum <= SHORTCUT_MAX_SPEARMAN),
        "shortcut_fields_used_as_model_inputs": False,
        **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
    }


def historical_c61_path(config: Mapping[str, Any], seed: int, split: str = "val") -> Path:
    audit = config.get("audit", {})
    root = Path(str(audit.get("c61_run_dir", "/home/linruixin/chen/project/DMEA-HT/runs/dema_ht_c61_cbpi_multiseed")))
    return root / "predictions" / f"{split}_predictions_seed_{seed}.csv"


def positive_and_inversion_row(
    candidate_frame: pd.DataFrame, baseline_frame: pd.DataFrame, identity: Mapping[str, Any]
) -> Dict[str, Any]:
    candidate = candidate_frame.sort_values("patient_id").reset_index(drop=True)
    baseline = baseline_frame.sort_values("patient_id").reset_index(drop=True)
    if not np.array_equal(candidate["patient_id"].to_numpy(dtype=str), baseline["patient_id"].to_numpy(dtype=str)):
        raise RuntimeError("C64 audit prediction patient alignment failed")
    if not np.array_equal(candidate["label"].to_numpy(dtype=int), baseline["label"].to_numpy(dtype=int)):
        raise RuntimeError("C64 audit prediction label alignment failed")
    labels = candidate["label"].to_numpy(dtype=int)
    candidate_probability = candidate[probability_column(candidate)].to_numpy(dtype=float)
    baseline_probability = baseline[probability_column(baseline)].to_numpy(dtype=float)
    candidate_counts = binary_counts(labels, candidate_probability)
    baseline_counts = binary_counts(labels, baseline_probability)
    candidate_inversions = inversion_vector(labels, candidate_probability)
    baseline_inversions = inversion_vector(labels, baseline_probability)
    return {
        **dict(identity),
        "baseline_sensitivity": baseline_counts["Sensitivity"],
        "candidate_sensitivity": candidate_counts["Sensitivity"],
        "positive_sensitivity_damage": max(0.0, baseline_counts["Sensitivity"] - candidate_counts["Sensitivity"]),
        "baseline_inversions": int(baseline_inversions.sum()),
        "candidate_inversions": int(candidate_inversions.sum()),
        "inversion_delta": int(candidate_inversions.sum() - baseline_inversions.sum()),
        "inversions_repaired": int((baseline_inversions & ~candidate_inversions).sum()),
        "inversions_introduced": int((~baseline_inversions & candidate_inversions).sum()),
    }


def parameter_health(run_dir: Path, candidate: str) -> Dict[str, Any]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return {"health_pass": False, "health_reason": "run_status_missing"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("test_loaded", True):
        return {"health_pass": False, "health_reason": "run_not_complete_or_test_loaded"}
    inventory = pd.read_csv(run_dir / "trainable_parameter_inventory.csv")
    expected_groups = set(common.expected_trainable_groups(candidate))
    active_inventory = inventory[inventory["requires_grad"].astype(bool)]
    actual_groups = set(active_inventory["optimizer_group"].astype(str))
    scope_ok = actual_groups == expected_groups and bool(active_inventory["parameter_count"].sum() > 0)
    gradient = pd.read_csv(run_dir / "gradient_connectivity.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    if len(metrics) != 1:
        return {"health_pass": False, "health_reason": "metric_row_missing"}
    selected_epoch = int(metrics.iloc[0]["best_epoch"])
    selected = gradient[gradient["epoch"].astype(int) == selected_epoch]
    expected_module_groups = set(
        inventory.loc[inventory["requires_grad"].astype(bool), "module_group"].astype(str)
    )
    gradient_ok = True
    for group in expected_module_groups:
        rows = selected[selected["module_group"].astype(str) == group]
        gradient_ok &= len(rows) == 1 and float(rows.iloc[0]["gradient_norm"]) > 0.0 and int(rows.iloc[0]["active_batch_count"]) > 0
    updates = pd.read_csv(run_dir / "parameter_update_audit.csv")
    summaries = updates[updates["kind"].astype(str) == "module_summary"]
    update_ok = True
    for group in expected_groups:
        rows = summaries[summaries["optimizer_group"].astype(str) == group]
        update_ok &= (
            len(rows) > 0
            and rows["updated"].map(bool_value).all()
            and rows["finite"].map(bool_value).all()
        )
    diagnostics_path = run_dir / "patient_diagnostics_val.csv"
    prediction_ok = False
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        probability = pd.to_numeric(diagnostics["final_prob"], errors="coerce").to_numpy(dtype=float)
        prediction_ok = len(diagnostics) > 0 and np.isfinite(probability).all() and float(probability.std()) > 0.0
    passed = bool(scope_ok and gradient_ok and update_ok and prediction_ok)
    return {
        "candidate": candidate,
        "selected_epoch": selected_epoch,
        "expected_optimizer_groups": sorted(expected_groups),
        "actual_optimizer_groups": sorted(actual_groups),
        "scope_pass": bool(scope_ok),
        "gradient_connectivity_pass": bool(gradient_ok),
        "parameter_update_pass": bool(update_ok),
        "prediction_health_pass": bool(prediction_ok),
        "health_pass": passed,
        "health_reason": "ok" if passed else "parameter_or_prediction_health_failed",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric_from_run(run_dir: Path) -> Dict[str, Any]:
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise RuntimeError(f"C64 run is not complete: {run_dir}")
    frame = pd.read_csv(run_dir / "metrics.csv")
    if len(frame) != 1:
        raise RuntimeError(f"C64 run metrics must have exactly one row: {run_dir}")
    return frame.iloc[0].to_dict()


def finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    return all(field in frame.columns and np.isfinite(pd.to_numeric(frame[field], errors="coerce")).all() for field in columns)
