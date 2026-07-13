#!/usr/bin/env python3
"""Run fixed train-fit/validation-eval probes for C20 layers."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_c20_common import EXPECTED_SEEDS, auc_score, finite_matrix, load_representation_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c20_dema")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-label-seed", type=int, default=20260714)
    return parser.parse_args()


def path_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, max_iter: int, seed: int) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return {"val_auc": float("nan"), "train_auc": float("nan"), "score_gap": float("nan"), "coef_norm": float("nan"), "converged": False, "error": "single-class split"}
    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=max_iter, random_state=seed)),
        ]
    )
    try:
        model.fit(x_train, y_train)
        train_score = model.decision_function(x_train)
        val_score = model.decision_function(x_val)
        classifier = model.named_steps["logistic"]
        iterations = int(np.max(classifier.n_iter_))
        return {
            "val_auc": auc_score(y_val, val_score),
            "train_auc": auc_score(y_train, train_score),
            "score_gap": float(np.mean(val_score[y_val == 1]) - np.mean(val_score[y_val == 0])),
            "coef_norm": float(np.linalg.norm(classifier.coef_)),
            "converged": iterations < max_iter,
            "iterations": iterations,
            "error": "",
        }
    except Exception as exc:
        return {"val_auc": float("nan"), "train_auc": float("nan"), "score_gap": float("nan"), "coef_norm": float("nan"), "converged": False, "iterations": -1, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    output_dir = path_from_root(args.output_dir)
    train = load_representation_npz(output_dir / "c20_internal_representations_train.npz")
    val = load_representation_npz(output_dir / "c20_internal_representations_val.npz")
    layers = sorted(set(train[0]["layers"]) & set(val[0]["layers"]))
    for seed in EXPECTED_SEEDS:
        layers = sorted(set(layers) & set(train[seed]["layers"]) & set(val[seed]["layers"]))
    if not layers:
        raise RuntimeError("no common representation layers for C20 probes")

    rng = np.random.default_rng(args.random_label_seed)
    random_train_labels = {seed: rng.permutation(train[seed]["labels"]) for seed in EXPECTED_SEEDS}
    rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for layer in layers:
        layer_values: List[float] = []
        random_values: List[float] = []
        seed_passes = 0
        for seed in EXPECTED_SEEDS:
            x_train = finite_matrix(train[seed]["layers"][layer])
            x_val = finite_matrix(val[seed]["layers"][layer])
            y_train = np.asarray(train[seed]["labels"], dtype=np.int64)
            y_val = np.asarray(val[seed]["labels"], dtype=np.int64)
            result = fit_probe(x_train, y_train, x_val, y_val, args.max_iter, seed)
            if math.isfinite(float(result["val_auc"])):
                layer_values.append(float(result["val_auc"]))
                seed_passes += int(float(result["val_auc"]) >= 0.83)
            rows.append({"layer": layer, "seed": seed, "split": "val", "probe_type": "true_labels", **result, "shortcut_fields_used": False})
            random_result = fit_probe(x_train, random_train_labels[seed], x_val, y_val, args.max_iter, args.random_label_seed)
            if math.isfinite(float(random_result["val_auc"])):
                random_values.append(float(random_result["val_auc"]))
            random_rows.append({"layer": layer, "seed": seed, "split": "val", "random_label_seed": args.random_label_seed, **random_result, "shortcut_fields_used": False})
        mean_auc = float(np.mean(layer_values)) if layer_values else float("nan")
        random_mean = float(np.mean(random_values)) if random_values else float("nan")
        random_stable = bool(sum(value >= 0.83 for value in random_values) >= 2)
        summary_rows.append(
            {
                "layer": layer,
                "mean_validation_probe_auc": mean_auc,
                "min_validation_probe_auc": float(np.min(layer_values)) if layer_values else float("nan"),
                "max_validation_probe_auc": float(np.max(layer_values)) if layer_values else float("nan"),
                "seeds_ge_0_83": seed_passes,
                "random_label_mean_auc": random_mean,
                "random_label_max_auc": float(np.max(random_values)) if random_values else float("nan"),
                "random_label_stable_signal": random_stable,
                "probe_leakage_or_overfit_concern": random_stable,
                "shortcut_fields_used": False,
                "fixed_C": 1.0,
                "fixed_max_iter": args.max_iter,
                "standardization": "train_fit_only",
            }
        )
    write_rows(output_dir / "c20_layer_probe_auc_by_seed.csv", rows)
    write_rows(output_dir / "c20_layer_probe_summary.csv", summary_rows)
    write_rows(output_dir / "c20_random_label_sanity_check.csv", random_rows)
    print(f"C20 layer probes complete: {len(layers)} layers")


if __name__ == "__main__":
    main()
