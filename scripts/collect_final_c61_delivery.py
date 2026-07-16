#!/usr/bin/env python3
"""Validate and materialize the frozen C61 reproducible delivery package.

This tool reads frozen data, checkpoints, predictions, and reports. It does not
update model parameters or create any new experiment artifacts outside the
requested delivery directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402


SEEDS = (0, 42, 3407)
MODEL_IMPLEMENTATION_COMMIT = "ec0aec77ebeaf299f928874ab6e57ce991d65dbe"
EXPECTED_MANIFEST_SHA256 = "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
EXPECTED_CONFIG_SHA256 = "c784dd0bfd7269683f4b1cd04c0636538c77cc6e4a18b678b16a533ecd813317"
EXPECTED_DEPLOYMENT_SHA256 = "57e00f284b3334936e2c9eccf02d9035cfa64173a663c0a1339b141ca48c5bd6"
EXPECTED_C61_VALIDATION = {
    0: 0.8981439565414215,
    42: 0.9013128112267994,
    3407: 0.9262109551833408,
}
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c61_cbpi_multiseed.yaml")
    parser.add_argument("--output-dir", default="analysis_reports/final_c61_delivery")
    parser.add_argument("--reproduction-csv", required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def add_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def normalized_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def probability_column(frame: pd.DataFrame) -> str:
    for column in ("final_prob", "prob", "probability"):
        if column in frame.columns:
            return column
    raise RuntimeError("Prediction CSV has no probability column")


def class_values(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_class" in frame.columns:
        return frame["predicted_class"].to_numpy(dtype=np.int64)
    return (frame[probability_column(frame)].to_numpy(dtype=float) >= 0.5).astype(np.int64)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).copy()
    if "patient_id" not in frame.columns or "label" not in frame.columns:
        raise RuntimeError(f"Prediction CSV is missing patient_id/label: {path}")
    frame["patient_id"] = frame["patient_id"].map(normalized_id)
    frame["label"] = frame["label"].astype(int)
    return frame


def binary_metrics(frame: pd.DataFrame, seed: int, split: str, best_epoch: int) -> Dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame[probability_column(frame)].to_numpy(dtype=np.float64)
    predicted = class_values(frame)
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & (predicted == 1)).sum())
    fn = int((positive & (predicted == 0)).sum())
    tn = int((negative & (predicted == 0)).sum())
    fp = int((negative & (predicted == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "seed": int(seed),
        "split": split,
        "best_epoch": int(best_epoch),
        "AUC": float(roc_auc_score(labels, probabilities)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_ACC": float(0.5 * (sensitivity + specificity)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "n_rows": int(len(frame)),
    }


def expected_manifest(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    expected: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for row in rows:
        split = str(row.get("split", ""))
        patient_id = normalized_id(row.get("patient_id"))
        if split not in expected or not patient_id:
            raise RuntimeError(f"Invalid manifest row split/patient_id: {row}")
        if patient_id in expected[split]:
            raise RuntimeError(f"Duplicate patient_id in split {split}: {patient_id}")
        expected[split][patient_id] = int(row["label"])
    return expected


def validate_prediction_frame(
    frame: pd.DataFrame, expected: Mapping[str, int], split: str
) -> Tuple[bool, str]:
    actual_ids = frame["patient_id"].tolist()
    expected_ids = sorted(expected)
    if actual_ids != sorted(actual_ids):
        return False, "patient IDs are not sorted"
    if actual_ids != expected_ids:
        return False, f"patient ID mismatch expected={len(expected_ids)} actual={len(actual_ids)}"
    actual_labels = frame["label"].to_numpy(dtype=np.int64)
    expected_labels = np.asarray([expected[patient_id] for patient_id in expected_ids], dtype=np.int64)
    if not np.array_equal(actual_labels, expected_labels):
        return False, f"{split} labels differ from frozen manifest"
    if "split" in frame.columns and not frame["split"].astype(str).eq(split).all():
        return False, f"prediction split column is not {split}"
    return True, f"{split} rows={len(frame)} patient IDs and labels exact"


def aligned_frames(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_index = left.set_index("patient_id")
    right_index = right.set_index("patient_id")
    common = sorted(set(left_index.index) & set(right_index.index))
    if common != sorted(left_index.index) or common != sorted(right_index.index):
        raise RuntimeError("Prediction patient IDs do not align exactly")
    labels = right_index.loc[common]["label"].to_numpy(dtype=np.int64)
    left_prob = left_index.loc[common][probability_column(left)].to_numpy(dtype=np.float64)
    right_prob = right_index.loc[common][probability_column(right)].to_numpy(dtype=np.float64)
    if not np.array_equal(labels, left_index.loc[common]["label"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Prediction labels do not align exactly")
    return labels, left_prob, right_prob


def positive_audit(
    c61_frames: Mapping[int, pd.DataFrame], c17_run: Path, official: pd.DataFrame
) -> Tuple[bool, str]:
    if official.empty:
        return False, "positive-preservation report is empty"
    observed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        path = c17_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        if not path.exists():
            return False, f"missing C17 prediction: {path}"
        c17 = load_frame(path)
        c61 = c61_frames[seed]
        labels, c17_prob, c61_prob = aligned_frames(c17, c61)
        c17_class = (c17_prob >= 0.5).astype(np.int64)
        c61_class = (c61_prob >= 0.5).astype(np.int64)
        positive = labels == 1
        c17_tp_to_c61_fn = int((positive & (c17_class == 1) & (c61_class == 0)).sum())
        c17_fn_to_c61_tp = int((positive & (c17_class == 0) & (c61_class == 1)).sum())
        c17_sensitivity = float(((positive) & (c17_class == 1)).sum() / max(positive.sum(), 1))
        c61_sensitivity = float(((positive) & (c61_class == 1)).sum() / max(positive.sum(), 1))
        observed.append(
            {
                "seed": seed,
                "c17_tp_to_c61_fn": c17_tp_to_c61_fn,
                "c17_fn_to_c61_tp": c17_fn_to_c61_tp,
                "c17_sensitivity": c17_sensitivity,
                "c61_sensitivity": c61_sensitivity,
                "c61_minus_c17_sensitivity": c61_sensitivity - c17_sensitivity,
            }
        )
    observed_frame = pd.DataFrame(observed).sort_values("seed").reset_index(drop=True)
    expected_frame = official.copy()
    expected_frame["seed"] = expected_frame["seed"].astype(int)
    expected_frame = expected_frame.sort_values("seed").reset_index(drop=True)
    columns = [
        "seed",
        "c17_tp_to_c61_fn",
        "c17_fn_to_c61_tp",
        "c17_sensitivity",
        "c61_sensitivity",
        "c61_minus_c17_sensitivity",
    ]
    if not set(columns).issubset(expected_frame.columns):
        return False, "positive-preservation report columns are incomplete"
    matches = np.allclose(
        observed_frame[columns[3:]].to_numpy(dtype=float),
        expected_frame[columns[3:]].to_numpy(dtype=float),
        atol=1e-6,
        rtol=0.0,
    ) and np.array_equal(
        observed_frame[columns[1:3]].to_numpy(dtype=int),
        expected_frame[columns[1:3]].to_numpy(dtype=int),
    )
    return bool(matches), f"recomputed C17 TP->C61 FN / FN->C61 TP={int(observed_frame[columns[1]].sum())}/{int(observed_frame[columns[2]].sum())}"


def ranking_audit(
    c61_frames: Mapping[int, pd.DataFrame], c27_run: Path, official: pd.DataFrame
) -> Tuple[bool, str]:
    if official.empty:
        return False, "pairwise inversion report is empty"
    observed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        path = c27_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        if not path.exists():
            return False, f"missing C27 prediction: {path}"
        c27 = load_frame(path)
        c61 = c61_frames[seed]
        labels, c27_prob, c61_prob = aligned_frames(c27, c61)
        positive = labels == 1
        negative = labels == 0
        c27_pairs = c27_prob[positive][:, None] < c27_prob[negative][None, :]
        c61_pairs = c61_prob[positive][:, None] < c61_prob[negative][None, :]
        observed.append(
            {
                "seed": seed,
                "C61_inversions": int(c61_pairs.sum()),
                "C61_minus_C27_inversions": int(c61_pairs.sum() - c27_pairs.sum()),
                "C27_to_C61_repaired": int((c27_pairs & ~c61_pairs).sum()),
                "C27_to_C61_introduced": int((~c27_pairs & c61_pairs).sum()),
            }
        )
    observed_frame = pd.DataFrame(observed).sort_values("seed").reset_index(drop=True)
    expected_frame = official.copy()
    expected_frame["seed"] = expected_frame["seed"].astype(int)
    expected_frame = expected_frame.sort_values("seed").reset_index(drop=True)
    columns = [
        "seed",
        "C61_inversions",
        "C61_minus_C27_inversions",
        "C27_to_C61_repaired",
        "C27_to_C61_introduced",
    ]
    if not set(columns).issubset(expected_frame.columns):
        return False, "pairwise inversion report columns are incomplete"
    matches = np.array_equal(
        observed_frame[columns].to_numpy(dtype=int), expected_frame[columns].to_numpy(dtype=int)
    )
    return bool(matches), f"recomputed repaired/introduced={int(observed_frame[columns[3]].sum())}/{int(observed_frame[columns[4]].sum())}"


def shortcut_probe_auc(frame: pd.DataFrame) -> float:
    matrix = pd.DataFrame(index=frame.index)
    for field in SELECTED_SHORTCUT_FIELDS:
        if field not in frame.columns:
            raise RuntimeError(f"Prediction CSV missing shortcut audit field: {field}")
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if values.notna().any() else 0.0)
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        matrix.to_numpy(dtype=np.float64),
        labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, probabilities))


def safe_spearman(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_values = pd.to_numeric(pd.Series(list(left)), errors="coerce")
    right_values = pd.to_numeric(pd.Series(list(right)), errors="coerce")
    valid = left_values.notna() & right_values.notna()
    if int(valid.sum()) < 3 or int(left_values[valid].nunique()) < 2 or int(right_values[valid].nunique()) < 2:
        return 0.0
    result = left_values[valid].rank().corr(right_values[valid].rank())
    return float(result) if result is not None and math.isfinite(float(result)) else 0.0


def shortcut_audit(c61_frames: Mapping[int, pd.DataFrame], official: pd.DataFrame) -> Tuple[bool, str]:
    if official.empty:
        return False, "shortcut report is empty"
    observed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        frame = c61_frames[seed]
        auc = shortcut_probe_auc(frame)
        probability = frame[probability_column(frame)].to_numpy(dtype=float)
        correlations = [safe_spearman(probability, frame[field]) for field in SELECTED_SHORTCUT_FIELDS]
        observed.append(
            {
                "seed": seed,
                "selected_structure_shortcut_only_label_AUC": auc,
                "max_abs_prediction_selected_structure_spearman": max(abs(value) for value in correlations),
            }
        )
    observed_frame = pd.DataFrame(observed).sort_values("seed").reset_index(drop=True)
    expected_frame = official.copy()
    expected_frame["seed"] = expected_frame["seed"].astype(int)
    expected_frame = expected_frame.sort_values("seed").reset_index(drop=True)
    auc_matches = np.allclose(
        observed_frame["selected_structure_shortcut_only_label_AUC"].to_numpy(dtype=float),
        expected_frame["selected_structure_shortcut_only_label_AUC"].to_numpy(dtype=float),
        atol=1e-6,
        rtol=0.0,
    )
    safety = bool(
        auc_matches
        and np.isfinite(observed_frame["max_abs_prediction_selected_structure_spearman"]).all()
        and (observed_frame["selected_structure_shortcut_only_label_AUC"] <= 0.55).all()
        and (observed_frame["max_abs_prediction_selected_structure_spearman"] <= 0.35).all()
        and expected_frame.get("shortcut_safety_pass", pd.Series([False] * len(expected_frame))).map(bool_text).all()
    )
    return safety, f"recomputed shortcut-only AUC max={observed_frame['selected_structure_shortcut_only_label_AUC'].max():.10f}"


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Checkpoint payload is not a mapping: {path}")
    return dict(payload)


def runtime_text(config_path: Path, manifest_path: Path, deployment: Path, current_commit: str) -> str:
    cuda_version = torch.version.cuda or "none"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    lines = [
        f"project_root={REPO_ROOT}",
        f"config={config_path}",
        f"manifest={manifest_path}",
        f"deployment_checkpoint={deployment}",
        f"verification_git_commit={current_commit}",
        f"model_implementation_commit={MODEL_IMPLEMENTATION_COMMIT}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch_version={torch.__version__}",
        f"cuda_version={cuda_version}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu={gpu}",
        "inference_contract=one checkpoint, one model, one forward",
        "test_policy=reporting-only; no selection or tuning",
    ]
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def artifact_inventory(required: Mapping[str, Path], output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for role, path in required.items():
        path = path.resolve()
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        rows.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(output_dir.iterdir()):
        path = path.resolve()
        if path.name == "c61_artifact_sha256.csv" or path in seen or not path.is_file():
            continue
        rows.append({"role": f"final_delivery/{path.name}", "path": str(path), "sha256": sha256_file(path)})
    return pd.DataFrame(rows, columns=["role", "path", "sha256"])


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    reproduction_path = resolve_path(args.reproduction_csv)
    config = json.loads(json.dumps(load_config(config_path)))
    if str(config.get("phase", "")).lower() != "c61":
        raise RuntimeError("Final delivery requires the C61 config")
    output_dir.mkdir(parents=True, exist_ok=True)

    project = config["project"]
    run_dir = resolve_path(project["output_dir"])
    report_dir = resolve_path(project["report_dir"])
    manifest_path = resolve_path(project["manifest"])
    decision_path = report_dir / "c61_validation_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    rows = read_jsonl(manifest_path)
    manifest = expected_manifest(rows)

    required: Dict[str, Path] = {
        "config": config_path,
        "manifest": manifest_path,
        "route_decision": decision_path,
        "metrics_by_seed": run_dir / "reports" / "metrics_by_seed.csv",
        "metrics_summary": run_dir / "reports" / "metrics_summary.csv",
        "validation_metrics_report": report_dir / "c61_metrics_by_seed.csv",
        "positive_preservation_report": report_dir / "c61_positive_preservation.csv",
        "pairwise_ranking_report": report_dir / "c61_pairwise_inversion_summary.csv",
        "training_health_report": report_dir / "c61_training_health.csv",
        "shortcut_report": report_dir / "c61_shortcut_audit.csv",
        "final_phase_report": report_dir / "phase_c61_dema_final_report.md",
        "c61_model_source": REPO_ROOT / "dmea_ht" / "c61_cbpi.py",
        "c59_model_source": REPO_ROOT / "dmea_ht" / "c59_pmese.py",
        "c47_model_source": REPO_ROOT / "dmea_ht" / "c47_drfe.py",
        "predict_entrypoint": REPO_ROOT / "scripts" / "predict_c61.py",
    }
    checkpoint_paths: Dict[int, Path] = {}
    val_frames: Dict[int, pd.DataFrame] = {}
    test_frames: Dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        checkpoint = run_dir / "checkpoints" / f"seed_{seed}_best.pt"
        val_path = run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"
        test_path = run_dir / "predictions" / f"test_predictions_seed_{seed}.csv"
        checkpoint_paths[seed] = checkpoint
        required[f"checkpoint_seed_{seed}"] = checkpoint
        required[f"validation_prediction_seed_{seed}"] = val_path
        required[f"test_prediction_seed_{seed}"] = test_path
        if val_path.exists():
            val_frames[seed] = load_frame(val_path)
        if test_path.exists():
            test_frames[seed] = load_frame(test_path)

    checks: List[Dict[str, Any]] = []
    decision_flags = (
        decision.get("decision_label") == "GOAL_REACHED_DEMA_HT_AUC_090_PLUS"
        and bool(decision.get("goal_reached"))
        and bool(decision.get("auc_gate_pass"))
    )
    add_check(checks, "goal_decision", decision_flags, str(decision.get("decision_label")))
    add_check(
        checks,
        "model_route",
        config.get("model", {}).get("variant") == "c61_cbpi" and config.get("phase") == "c61",
        "phase=c61 variant=c61_cbpi",
    )

    config_hash = sha256_file(config_path) if config_path.exists() else "missing"
    manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else "missing"
    add_check(
        checks,
        "freeze_hashes",
        config_hash == EXPECTED_CONFIG_SHA256 and manifest_hash == EXPECTED_MANIFEST_SHA256,
        f"config={config_hash} manifest={manifest_hash}",
    )

    missing_required = [str(path) for path in required.values() if not path.exists()]
    split_sets = [set(manifest[split]) for split in ("train", "val", "test")]
    split_overlap = bool(split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
    add_check(
        checks,
        "patient_level_split_isolation",
        not split_overlap,
        "train/val/test patient IDs are disjoint" if not split_overlap else "patient ID appears in multiple splits",
    )
    add_check(
        checks,
        "artifact_inventory",
        not missing_required,
        "all required frozen artifacts present" if not missing_required else "missing: " + "; ".join(missing_required),
    )
    add_check(
        checks,
        "three_checkpoints",
        len(checkpoint_paths) == 3 and all(path.exists() for path in checkpoint_paths.values()),
        ", ".join(f"seed {seed}: {path.exists()}" for seed, path in checkpoint_paths.items()),
    )

    prediction_checks = []
    for seed in SEEDS:
        if seed not in val_frames or seed not in test_frames:
            prediction_checks.append(f"seed {seed}: missing val/test")
            continue
        val_ok, val_detail = validate_prediction_frame(val_frames[seed], manifest["val"], "val")
        test_ok, test_detail = validate_prediction_frame(test_frames[seed], manifest["test"], "test")
        prediction_checks.append(f"seed {seed}: val={val_ok}, test={test_ok}")
        if not val_ok or not test_ok:
            prediction_checks.append(val_detail + "; " + test_detail)
    add_check(
        checks,
        "validation_and_test_predictions",
        len(val_frames) == 3
        and len(test_frames) == 3
        and all(validate_prediction_frame(val_frames[seed], manifest["val"], "val")[0] for seed in SEEDS)
        and all(validate_prediction_frame(test_frames[seed], manifest["test"], "test")[0] for seed in SEEDS),
        " | ".join(prediction_checks),
    )

    official_by_seed = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv") if (run_dir / "reports" / "metrics_by_seed.csv").exists() else pd.DataFrame()
    official_val_report = pd.read_csv(report_dir / "c61_metrics_by_seed.csv") if (report_dir / "c61_metrics_by_seed.csv").exists() else pd.DataFrame()
    if not official_val_report.empty and "split" in official_val_report.columns:
        official_val_report = official_val_report[official_val_report["split"].astype(str) == "val"].copy()
    best_epochs = {
        int(row.seed): int(row.best_epoch)
        for row in official_val_report.itertuples()
        if hasattr(row, "best_epoch")
    }
    computed_val = [binary_metrics(val_frames[seed], seed, "val", best_epochs.get(seed, -1)) for seed in SEEDS if seed in val_frames]
    computed_test = [binary_metrics(test_frames[seed], seed, "test", best_epochs.get(seed, -1)) for seed in SEEDS if seed in test_frames]
    computed_by_seed = pd.DataFrame(computed_val + computed_test)
    metrics_match = True
    metric_details = []
    if official_by_seed.empty:
        metrics_match = False
        metric_details.append("missing reports/metrics_by_seed.csv")
    else:
        for observed in computed_val + computed_test:
            reference = official_by_seed[
                (official_by_seed["seed"].astype(int) == observed["seed"])
                & (official_by_seed["split"].astype(str) == observed["split"])
            ]
            if reference.empty:
                metrics_match = False
                metric_details.append(f"missing official row seed={observed['seed']} split={observed['split']}")
                continue
            row = reference.iloc[0]
            for field in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC"):
                if not math.isclose(float(row[field]), float(observed[field]), abs_tol=1e-6, rel_tol=0.0):
                    metrics_match = False
                    metric_details.append(f"{field} mismatch seed={observed['seed']} split={observed['split']}")
            for field in ("TN", "FP", "FN", "TP", "n_rows"):
                if int(row[field]) != int(observed[field]):
                    metrics_match = False
                    metric_details.append(f"{field} mismatch seed={observed['seed']} split={observed['split']}")
    val_auc = np.asarray([row["AUC"] for row in computed_val], dtype=float)
    test_auc = np.asarray([row["AUC"] for row in computed_test], dtype=float)
    val_mean = float(val_auc.mean()) if len(val_auc) else float("nan")
    val_std = float(val_auc.std(ddof=1)) if len(val_auc) > 1 else float("nan")
    test_mean = float(test_auc.mean()) if len(test_auc) else float("nan")
    test_std = float(test_auc.std(ddof=1)) if len(test_auc) > 1 else float("nan")
    decision_metrics_match = (
        len(val_auc) == 3
        and math.isclose(val_mean, float(decision.get("validation_mean_AUC", float("nan"))), abs_tol=1e-6, rel_tol=0.0)
        and math.isclose(val_std, float(decision.get("validation_std_AUC", float("nan"))), abs_tol=1e-6, rel_tol=0.0)
        and all(math.isclose(val_auc[index], EXPECTED_C61_VALIDATION[seed], abs_tol=1e-6, rel_tol=0.0) for index, seed in enumerate(SEEDS))
    )
    add_check(
        checks,
        "metrics_recomputed",
        metrics_match and decision_metrics_match and len(computed_val) == 3 and len(computed_test) == 3,
        f"val mean/std={val_mean:.10f}/{val_std:.10f}; test mean/std={test_mean:.10f}/{test_std:.10f}; "
        + ("exact against frozen reports" if not metric_details else "; ".join(metric_details)),
    )

    c17_run = resolve_path(config["c17"]["c17_run_dir"])
    c27_run = resolve_path(config["c27"]["c27_run_dir"])
    positive_report_path = report_dir / "c61_positive_preservation.csv"
    ranking_report_path = report_dir / "c61_pairwise_inversion_summary.csv"
    shortcut_report_path = report_dir / "c61_shortcut_audit.csv"
    positive_ok, positive_detail = positive_audit(
        val_frames, c17_run, pd.read_csv(positive_report_path) if positive_report_path.exists() else pd.DataFrame()
    ) if len(val_frames) == 3 else (False, "Validation predictions incomplete")
    ranking_ok, ranking_detail = ranking_audit(
        val_frames, c27_run, pd.read_csv(ranking_report_path) if ranking_report_path.exists() else pd.DataFrame()
    ) if len(val_frames) == 3 else (False, "Validation predictions incomplete")
    shortcut_ok, shortcut_detail = shortcut_audit(
        val_frames, pd.read_csv(shortcut_report_path) if shortcut_report_path.exists() else pd.DataFrame()
    ) if len(val_frames) == 3 else (False, "Validation predictions incomplete")
    add_check(checks, "positive_preservation", positive_ok and bool(decision.get("positive_safety_pass")), positive_detail)
    add_check(checks, "pairwise_ranking", ranking_ok and bool(decision.get("ranking_safety_pass")), ranking_detail)
    add_check(checks, "shortcut_audit", shortcut_ok and bool(decision.get("shortcut_safety_pass")), shortcut_detail)

    health_path = report_dir / "c61_training_health.csv"
    health = pd.read_csv(health_path) if health_path.exists() else pd.DataFrame()
    health_ok = (
        len(health) == 9
        and "health_pass" in health.columns
        and health["health_pass"].map(bool_text).all()
        and bool(decision.get("training_health_pass"))
    )
    add_check(checks, "training_health", health_ok, f"health rows passed={int(health.get('health_pass', pd.Series(dtype=bool)).map(bool_text).sum())}/{len(health)}")

    metadata_ok = True
    metadata_details = []
    for seed, path in checkpoint_paths.items():
        if not path.exists():
            metadata_ok = False
            metadata_details.append(f"seed {seed} missing")
            continue
        payload = checkpoint_metadata(path)
        seed_ok = int(payload.get("seed", -1)) == seed
        epoch_ok = int(payload.get("best_epoch", -1)) == best_epochs.get(seed, -2)
        metric_ok = str(payload.get("selection_metric", "")) == "validation_auc_only"
        model_ok = isinstance(payload.get("model"), Mapping)
        metadata_ok &= seed_ok and epoch_ok and metric_ok and model_ok
        metadata_details.append(f"seed {seed}: seed={seed_ok}, epoch={epoch_ok}, metric={metric_ok}, model={model_ok}")
    add_check(checks, "checkpoint_metadata", metadata_ok, " | ".join(metadata_details))

    isolation_ok = (
        bool(decision.get("validation_decision_frozen_before_test"))
        and not bool(decision.get("test_used_for_decision", True))
        and not bool(decision.get("ensemble_used", True))
        and str(config.get("training", {}).get("primary_metric")) == "val_AUC"
    )
    add_check(checks, "validation_test_isolation", isolation_ok, "Validation frozen before Test; Test is reporting-only")
    deployment = checkpoint_paths[42]
    deployment_hash = sha256_file(deployment) if deployment.exists() else "missing"
    deployment_ok = (
        int(decision.get("deployment_seed", -1)) == 42
        and deployment_hash == EXPECTED_DEPLOYMENT_SHA256
        and bool(config.get("deployment", {}).get("one_checkpoint"))
        and bool(config.get("deployment", {}).get("one_model"))
        and bool(config.get("deployment", {}).get("one_forward"))
        and not bool(config.get("deployment", {}).get("ensemble", True))
    )
    add_check(checks, "one_checkpoint_deployment", deployment_ok, f"seed=42 checkpoint_sha256={deployment_hash}")
    add_check(checks, "no_ensemble_or_averaging", not bool(config.get("deployment", {}).get("ensemble", True)), "deployment.ensemble=false")

    reproduction_ok = False
    reproduction_detail = "reproduction CSV missing"
    reproduction_max_error = float("nan")
    if reproduction_path.exists() and 42 in val_frames:
        reproduction = load_frame(reproduction_path)
        saved = val_frames[42]
        saved_ids = saved["patient_id"].tolist()
        repro_ids = reproduction["patient_id"].tolist()
        saved_prob = saved[probability_column(saved)].to_numpy(dtype=float)
        repro_prob = reproduction[probability_column(reproduction)].to_numpy(dtype=float)
        reproduction_max_error = float(np.max(np.abs(saved_prob - repro_prob))) if len(saved_prob) == len(repro_prob) else float("inf")
        reproduction_auc = float(roc_auc_score(reproduction["label"], repro_prob)) if len(reproduction) else float("nan")
        saved_auc = float(roc_auc_score(saved["label"], saved_prob)) if len(saved) else float("nan")
        class_exact = np.array_equal(class_values(saved), class_values(reproduction))
        reproduction_ok = (
            len(reproduction) == len(saved)
            and repro_ids == saved_ids
            and np.array_equal(reproduction["label"].to_numpy(dtype=int), saved["label"].to_numpy(dtype=int))
            and class_exact
            and reproduction_max_error <= 1e-6
            and math.isclose(reproduction_auc, saved_auc, abs_tol=1e-6, rel_tol=0.0)
        )
        reproduction_detail = (
            f"rows={len(reproduction)}; patient_id_exact={repro_ids == saved_ids}; label_exact="
            f"{np.array_equal(reproduction['label'].to_numpy(dtype=int), saved['label'].to_numpy(dtype=int))}; "
            f"class_exact={class_exact}; auc={reproduction_auc:.10f}; max_abs_prob_error={reproduction_max_error:.12g}"
        )
    add_check(checks, "single_checkpoint_reproduction", reproduction_ok, reproduction_detail)
    add_check(checks, "no_further_training", True, "final delivery tool only reads frozen artifacts and writes delivery documents")

    final_pass = all(bool(item["passed"]) for item in checks)
    status = "FINAL_C61_DELIVERY_PASS" if final_pass else "FINAL_C61_DELIVERY_REPRODUCTION_FAIL"
    current_commit = git_value("rev-parse", "HEAD")
    validation_output = (
        official_val_report if not official_val_report.empty else pd.DataFrame(computed_val)
    )
    validation_output.to_csv(output_dir / "c61_final_validation_metrics.csv", index=False)
    pd.DataFrame(computed_test).to_csv(output_dir / "c61_final_test_reporting_only_metrics.csv", index=False)

    report_copies = {
        "c61_positive_preservation.csv": "c61_final_positive_preservation.csv",
        "c61_pairwise_inversion_summary.csv": "c61_final_pairwise_ranking.csv",
        "c61_shortcut_audit.csv": "c61_final_shortcut_audit.csv",
        "c61_training_health.csv": "c61_final_training_health.csv",
    }
    for source_name, target_name in report_copies.items():
        source = report_dir / source_name
        target = output_dir / target_name
        if source.exists():
            target.write_bytes(source.read_bytes())
        else:
            pd.DataFrame().to_csv(target, index=False)

    write_text(
        output_dir / "FINAL_MODEL_SELECTION.md",
        f"""# Final Model Selection

Status: `{status}`

Official model: `DEMA-HT C61-CBPI` (Continuous Biochemical Patient-Instance Fusion).

Goal: `GOAL_REACHED_DEMA_HT_AUC_090_PLUS`.

Formal Validation AUC: `0.8981439565 / 0.9013128112 / 0.9262109552`, mean/std `0.9085559077 +/- 0.0153715952` for seeds `[0, 42, 3407]`.

Deployment uses the median-Validation seed `42` checkpoint:
`{deployment}`

The seed 3407 AUC of 0.9262 is not the deployment-selection rule. Seed 42 is selected because it is the median Validation result. Deployment is one checkpoint, one model, one forward, with no ensemble, averaging, threshold tuning, or Test-based selection.

Test is reporting-only: mean/std AUC `{test_mean:.10f} +/- {test_std:.10f}`.

Freeze labels: `GOAL_REACHED_DEMA_HT_AUC_090_PLUS`, `FREEZE_DEMA_C61_CBPI_FINAL`, `STOP_FURTHER_OPTIMIZATION`.
""",
    )
    write_text(
        output_dir / "C61_MODEL_CARD.md",
        f"""# C61 Model Card

## Intended use

Patient-level research prediction of next-year HT risk from pre-cutoff longitudinal evidence.

## Inputs

Frozen ultrasound image evidence, report text, continuous biochemical measurements, and patient historical visits before the target year.

## Excluded inputs

Patient ID, visit count, image count, report length, padding count, source folder, saved predictions, Test artifacts, and missing-count shortcut fields are excluded from the classifier. Missingness only masks valid evidence.

## Limitations

- The current data source and evaluation are limited to the present cohort; external validation is incomplete.
- Reporting-only Test AUC (`{test_mean:.10f}`) is lower than Validation AUC (`{val_mean:.10f}`).
- This is a research model, not a clinical diagnostic replacement.
- Biochemical variables are used without hand-written abnormality reference ranges.
- Raw visit/image counts retain audit-only association and are not model inputs.
""",
    )
    write_text(
        output_dir / "C61_REPRODUCIBILITY.md",
        f"""# C61 Reproducibility

- Canonical model implementation commit: `{MODEL_IMPLEMENTATION_COMMIT}`.
- Final delivery verification commit: `{current_commit}`.
- Config: `{config_path}`; SHA-256 `{config_hash}`.
- Manifest: `{manifest_path}`; SHA-256 `{manifest_hash}`.
- Runtime: `/home/linruixin/chen/conda/envs/ma` on canonical server checkout `/home/linruixin/chen/project/DMEA-HT`.
- Formal seeds: `[0, 42, 3407]`, independently trained and selected by Validation AUC.
- Validation decision was frozen before Test. Test is reporting-only and cannot select a model, seed, threshold, or checkpoint.
- Deployment uses one checkpoint and one forward; no model or prediction combination is used.

The exact read-only reproduction commands are recorded in `c61_exact_commands.sh`. The single-checkpoint output was compared against the saved seed42 Validation prediction CSV: `{reproduction_detail}`.
""",
    )
    write_text(
        output_dir / "C61_SINGLE_CHECKPOINT_DEPLOYMENT.md",
        f"""# C61 Single-Checkpoint Deployment

- Model: `DEMA-HT C61-CBPI`.
- Seed: `42` (median Validation AUC selection).
- Checkpoint: `{deployment}`.
- Checkpoint SHA-256: `{deployment_hash}`.
- Runtime: `/home/linruixin/chen/conda/envs/ma`.
- Contract: one checkpoint, one model, one forward.
- Seeds 0 and 3407 remain scientific reproduction artifacts only.
- Test metrics are reporting-only and did not influence deployment.

Single-checkpoint reproduction status: `{status}`; max saved-vs-rerun probability error `{reproduction_max_error:.12g}`.
""",
    )
    write_text(
        output_dir / "c61_environment.txt",
        runtime_text(config_path, manifest_path, deployment, current_commit),
    )
    write_text(
        output_dir / "c61_exact_commands.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/linruixin/chen/project/DMEA-HT
PYTHON=/home/linruixin/chen/conda/envs/ma/bin/python
$PYTHON scripts/predict_c61.py \\
  --config configs/dema_ht_c61_cbpi_multiseed.yaml \\
  --checkpoint runs/dema_ht_c61_cbpi_multiseed/checkpoints/seed_42_best.pt \\
  --split val \\
  --output analysis_reports/final_c61_delivery/seed42_reproduction_val.csv
$PYTHON scripts/collect_final_c61_delivery.py \\
  --config configs/dema_ht_c61_cbpi_multiseed.yaml \\
  --output-dir analysis_reports/final_c61_delivery \\
  --reproduction-csv analysis_reports/final_c61_delivery/seed42_reproduction_val.csv
""",
    )
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output_dir / "c61_delivery_checklist.csv", index=False)
    write_text(
        output_dir / "c61_final_delivery_report.md",
        f"""# C61 Final Delivery Report

## Decision

`{status}`

Model: `DEMA-HT C61-CBPI`
Goal: `GOAL_REACHED_DEMA_HT_AUC_090_PLUS`
Implementation commit: `{MODEL_IMPLEMENTATION_COMMIT}`
Verification commit: `{current_commit}`

## Frozen metrics

- Validation AUC seed 0/42/3407: `0.8981439565 / 0.9013128112 / 0.9262109552`.
- Validation mean/std: `{val_mean:.10f} +/- {val_std:.10f}`.
- Reporting-only Test mean/std: `{test_mean:.10f} +/- {test_std:.10f}`.
- Positive preservation: `{positive_detail}`.
- Pairwise ranking: `{ranking_detail}`.
- Shortcut audit: `{shortcut_detail}`.

## Deployment

Seed `42`, checkpoint `{deployment}`, SHA-256 `{deployment_hash}`. One checkpoint, one model, one forward. No ensemble, averaging, threshold tuning, or Test selection.

## Single-checkpoint validation

`{reproduction_detail}`

## Checks

Passed `{int(checks_frame['passed'].sum())}/{len(checks_frame)}` checks. See `c61_delivery_checklist.csv` and `c61_artifact_sha256.csv` for the machine-readable audit.

Further model optimization is stopped: `STOP_FURTHER_OPTIMIZATION`.
""",
    )
    artifact_inventory(required, output_dir).to_csv(output_dir / "c61_artifact_sha256.csv", index=False)

    print(
        json.dumps(
            {
                "status": status,
                "checks_passed": int(checks_frame["passed"].sum()),
                "checks_total": int(len(checks_frame)),
                "output_dir": str(output_dir),
                "validation_mean_auc": val_mean,
                "validation_std_auc": val_std,
                "test_mean_auc_reporting_only": test_mean,
                "test_std_auc_reporting_only": test_std,
                "reproduction_max_abs_probability_error": reproduction_max_error,
                "deployment_seed": 42,
                "deployment_checkpoint_sha256": deployment_hash,
                "verification_git_commit": current_commit,
            },
            sort_keys=True,
        )
    )
    if not final_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
