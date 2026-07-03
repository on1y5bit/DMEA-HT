from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class BinaryMetrics:
    auc: float
    auprc: float
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    balanced_accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, probs))


def _safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, probs))


def compute_binary_metrics(labels: Iterable[int], probs: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(list(labels), dtype=int)
    probs_np = np.asarray(list(probs), dtype=float)
    preds = (probs_np >= threshold).astype(int)

    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    sensitivity = recall
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(labels_np), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return {
        "AUC": _safe_auc(labels_np, probs_np),
        "AUPRC": _safe_auprc(labels_np, probs_np),
        "ACC": float(accuracy),
        "F1": float(f1),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "Recall": float(recall),
        "Balanced_ACC": float(balanced_accuracy),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def summarize_metrics(rows: List[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary

