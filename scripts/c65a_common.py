#!/usr/bin/env python3
"""Shared read-only utilities for Phase C65-A variance attribution."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dmea_ht.config import load_config
from dmea_ht.visit_data import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
FOLD_COUNT = 5
FOLD_SEED = 20260716
DEVELOPMENT_COUNT = 696


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_c65a_config(path: str | Path) -> Dict[str, Any]:
    config = load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c65a":
        raise RuntimeError("C65-A configuration phase mismatch")
    if tuple(int(seed) for seed in config.get("formal_seeds", [])) != SEEDS:
        raise RuntimeError("C65 formal seeds must remain [0, 42, 3407]")
    folds = config.get("folds", {})
    if int(folds.get("count", -1)) != FOLD_COUNT or int(folds.get("seed", -1)) != FOLD_SEED:
        raise RuntimeError("C65 must reuse the exact C64 five-fold assignments")
    if int(folds.get("development_patient_count", -1)) != DEVELOPMENT_COUNT:
        raise RuntimeError("C65 development pool must contain 696 patients")
    if bool(config.get("analysis", {}).get("test_loaded", True)):
        raise RuntimeError("C65-A is read-only and Test must remain locked")
    return config


def c64_report_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["c64_report_dir"])


def c64_cv_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["c64_cv_output_dir"])


def report_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["report_dir"])


def c61_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return load_config(resolve_path(config["project"]["c61_config"]))


def development_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = read_jsonl(config["project"]["manifest"])
    development = [dict(row) for row in rows if str(row.get("split", "")).lower() in {"train", "val"}]
    if len(development) != DEVELOPMENT_COUNT:
        raise RuntimeError(f"C65 development pool must contain 696 patients, got {len(development)}")
    patient_ids = [str(row["patient_id"]) for row in development]
    if len(set(patient_ids)) != len(patient_ids):
        raise RuntimeError("C65 development patient IDs are not unique")
    labels = {int(row["label"]) for row in development}
    if labels != {0, 1}:
        raise RuntimeError(f"C65 development labels are invalid: {labels}")
    return development


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) != len(right_array) or len(left_array) < 2:
        return float("nan")
    if np.all(left_array == left_array[0]) or np.all(right_array == right_array[0]):
        return float("nan")
    return float(pd.Series(left_array).corr(pd.Series(right_array), method="spearman"))


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    squared = np.sum(np.square(values), axis=1, dtype=np.float64)
    distances = squared[:, None] + squared[None, :] - 2.0 * (values @ values.T)
    return np.sqrt(np.maximum(distances, 0.0))


def upper_triangle(values: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(values.shape[0], k=1)
    return values[indices]


def zscore_columns(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.where(std > 1e-8, std, 1.0)


def knn_indices(values: np.ndarray, k: int) -> list[set[int]]:
    standardized = zscore_columns(values)
    distances = pairwise_euclidean(standardized)
    result: list[set[int]] = []
    for index in range(len(values)):
        order = np.argsort(distances[index], kind="stable")
        neighbors = [int(item) for item in order if int(item) != index][:k]
        result.append(set(neighbors))
    return result


def mean_jaccard(left: Iterable[set[int]], right: Iterable[set[int]]) -> float:
    scores = []
    for left_set, right_set in zip(left, right):
        union = left_set | right_set
        scores.append(len(left_set & right_set) / max(len(union), 1))
    return float(np.mean(scores)) if scores else float("nan")
