#!/usr/bin/env python3
"""Shared helpers for the validation-only C20 identifiability audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


EXPECTED_SEEDS = (0, 42, 3407)
SEED_PAIRS = ((0, 42), (0, 3407), (42, 3407))


def finite_matrix(value: np.ndarray) -> np.ndarray:
    """Return a two-dimensional float matrix without silently dropping rows."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def load_representation_npz(path: str | Path) -> Dict[int, Dict[str, Any]]:
    """Load one C20 split NPZ and expose seed, labels, IDs, layers, and shortcuts."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        result: Dict[int, Dict[str, Any]] = {}
        pattern = re.compile(r"^seed_(\d+)__(.+)$")
        for key in payload.files:
            match = pattern.match(key)
            if match is None:
                continue
            seed = int(match.group(1))
            name = match.group(2)
            result.setdefault(seed, {"layers": {}, "shortcuts": {}})
            if name == "patient_id":
                result[seed]["patient_id"] = payload[key].astype(str)
            elif name == "labels":
                result[seed]["labels"] = payload[key].astype(np.int64)
            elif name.startswith("shortcut__"):
                result[seed]["shortcuts"][name[len("shortcut__") :]] = payload[key].copy()
            elif name.startswith("layer__"):
                result[seed]["layers"][name[len("layer__") :]] = finite_matrix(payload[key])
        for seed in EXPECTED_SEEDS:
            if seed not in result:
                raise RuntimeError(f"missing seed {seed} in {path}")
            for required in ("patient_id", "labels"):
                if required not in result[seed]:
                    raise RuntimeError(f"missing {required} for seed {seed} in {path}")
        return result


def common_index(left_ids: Sequence[str], right_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    right = {str(value): index for index, value in enumerate(right_ids)}
    left_indices: List[int] = []
    right_indices: List[int] = []
    for index, value in enumerate(left_ids):
        key = str(value)
        if key in right:
            left_indices.append(index)
            right_indices.append(right[key])
    if not left_indices:
        raise RuntimeError("no common patient IDs between representation exports")
    return np.asarray(left_indices, dtype=np.int64), np.asarray(right_indices, dtype=np.int64)


def align_pair(
    left: Mapping[str, Any], right: Mapping[str, Any], layer: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_index, right_index = common_index(left["patient_id"], right["patient_id"])
    left_values = finite_matrix(left["layers"][layer])[left_index]
    right_values = finite_matrix(right["layers"][layer])[right_index]
    labels = np.asarray(left["labels"])[left_index].astype(np.int64)
    patient_ids = np.asarray(left["patient_id"])[left_index].astype(str)
    right_labels = np.asarray(right["labels"])[right_index].astype(np.int64)
    if not np.array_equal(labels, right_labels):
        raise RuntimeError(f"label mismatch while aligning layer {layer}")
    return left_values, right_values, labels, patient_ids


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group, count in enumerate(counts):
            if count > 1:
                ranks[inverse == group] = float(np.mean(ranks[inverse == group]))
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(rankdata(np.asarray(left).reshape(-1)), rankdata(np.asarray(right).reshape(-1)))


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (positive.size * negative.size))


def pairwise_distances(values: np.ndarray) -> np.ndarray:
    values = finite_matrix(values)
    gram = values @ values.T
    squared = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(squared)


def upper_triangle(values: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(values.shape[0], k=1)
    return np.asarray(values[indices], dtype=np.float64)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA using sample Gram matrices, invariant to rotations and scale."""
    left = finite_matrix(left)
    right = finite_matrix(right)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    left_gram -= left_gram.mean(axis=0, keepdims=True)
    left_gram -= left_gram.mean(axis=1, keepdims=True)
    right_gram -= right_gram.mean(axis=0, keepdims=True)
    right_gram -= right_gram.mean(axis=1, keepdims=True)
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(np.sqrt(np.sum(left_gram * left_gram) * np.sum(right_gram * right_gram)))
    return numerator / denominator if denominator > 1e-12 else float("nan")


def normalized_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = finite_matrix(left)
    right = finite_matrix(right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    values = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    return values


def knn_indices(values: np.ndarray, k: int = 10) -> np.ndarray:
    distances = pairwise_distances(values)
    distances[np.arange(distances.shape[0]), np.arange(distances.shape[0])] = np.inf
    effective_k = min(int(k), max(distances.shape[0] - 1, 0))
    if effective_k <= 0:
        return np.empty((distances.shape[0], 0), dtype=np.int64)
    return np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]


def jaccard_by_patient(left: np.ndarray, right: np.ndarray, k: int = 10) -> np.ndarray:
    left_neighbors = knn_indices(left, k=k)
    right_neighbors = knn_indices(right, k=k)
    values: List[float] = []
    for left_row, right_row in zip(left_neighbors, right_neighbors):
        left_set = set(int(item) for item in left_row)
        right_set = set(int(item) for item in right_row)
        union = left_set | right_set
        values.append(float(len(left_set & right_set) / len(union)) if union else float("nan"))
    return np.asarray(values, dtype=np.float64)


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if array.size and np.isfinite(array).any() else float("nan")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
