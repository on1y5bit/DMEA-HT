#!/usr/bin/env python3
"""Run validation-only one-edge-at-a-time C21-A inference ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from phase_c21a_common import (
    ALL_EDGES,
    build_loader,
    c17_forward_from_encoded,
    load_config,
    load_model,
    load_rows,
    move_batch,
    resolve_path,
    write_rows,
)


SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c17_formal_multiseed.yaml")
    parser.add_argument("--run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (positive.size * negative.size))


def pairwise_changes(labels: np.ndarray, baseline: np.ndarray, ablated: np.ndarray) -> Dict[str, int]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    baseline_scores = baseline[positive, None] - baseline[negative][None, :]
    ablated_scores = ablated[positive, None] - ablated[negative][None, :]
    baseline_bad = baseline_scores < 0.0
    ablated_bad = ablated_scores < 0.0
    return {
        "baseline_inverted_pairs": int(baseline_bad.sum()),
        "ablated_inverted_pairs": int(ablated_bad.sum()),
        "repaired_pairs": int((baseline_bad & ~ablated_bad).sum()),
        "introduced_pairs": int((~baseline_bad & ablated_bad).sum()),
        "strict_pair_count": int(baseline_bad.size),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
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


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(resolve_path(args.config))
    rows = load_rows(config, manifest=args.manifest, data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        model = load_model(config, resolve_path(args.run_dir), seed, device)
        batch_outputs: Dict[str, Dict[str, List[float]]] = {edge: {"baseline": [], "ablated": []} for edge in ALL_EDGES}
        labels_all: List[int] = []
        ids_all: List[str] = []
        with torch.no_grad():
            loader = build_loader(config, rows, "val", args.batch_size, args.num_workers)
            for batch in loader:
                moved = move_batch(batch, device)
                encoded = model.base_model.encode_modalities(moved)
                base_outputs = model.base_model.forward_from_encoded(moved, encoded)
                baseline_result = c17_forward_from_encoded(model, moved, encoded, base_outputs)
                baseline_logit = baseline_result["outputs"]["logit"].detach().cpu().numpy().reshape(-1)
                baseline_prob = baseline_result["outputs"]["prob"].detach().cpu().numpy().reshape(-1)
                labels = moved["label"].detach().cpu().numpy().reshape(-1).astype(np.int64)
                ids = [str(value) for value in moved["patient_id"]]
                labels_all.extend(labels.tolist())
                ids_all.extend(ids)
                for edge in ALL_EDGES:
                    intervention = c17_forward_from_encoded(
                        model,
                        moved,
                        encoded,
                        base_outputs,
                        zero_edges=(edge,),
                    )
                    ablated_logit = intervention["outputs"]["logit"].detach().cpu().numpy().reshape(-1)
                    ablated_prob = intervention["outputs"]["prob"].detach().cpu().numpy().reshape(-1)
                    batch_outputs[edge]["baseline"].extend(baseline_prob.tolist())
                    batch_outputs[edge]["ablated"].extend(ablated_prob.tolist())
                    for patient_id, label, base_l, abl_l, base_p, abl_p in zip(
                        ids,
                        labels.tolist(),
                        baseline_logit.tolist(),
                        ablated_logit.tolist(),
                        baseline_prob.tolist(),
                        ablated_prob.tolist(),
                    ):
                        patient_rows.append(
                            {
                                "seed": seed,
                                "split": "val",
                                "patient_id": patient_id,
                                "label": label,
                                "edge": edge,
                                "baseline_logit": base_l,
                                "ablated_logit": abl_l,
                                "delta_logit_ablated_minus_baseline": abl_l - base_l,
                                "baseline_prob": base_p,
                                "ablated_prob": abl_p,
                                "delta_prob_ablated_minus_baseline": abl_p - base_p,
                            }
                        )
        labels_array = np.asarray(labels_all, dtype=np.int64)
        for edge in ALL_EDGES:
            baseline = np.asarray(batch_outputs[edge]["baseline"], dtype=np.float64)
            ablated = np.asarray(batch_outputs[edge]["ablated"], dtype=np.float64)
            changes = pairwise_changes(labels_array, baseline, ablated)
            summary_rows.append(
                {
                    "seed": seed,
                    "split": "val",
                    "edge": edge,
                    "n": len(labels_array),
                    "baseline_auc": auc_score(labels_array, baseline),
                    "ablated_auc": auc_score(labels_array, ablated),
                    "auc_delta_ablated_minus_baseline": auc_score(labels_array, ablated) - auc_score(labels_array, baseline),
                    "mean_delta_prob": float(np.mean(ablated - baseline)),
                    "mean_abs_delta_prob": float(np.mean(np.abs(ablated - baseline))),
                    "positive_mean_delta_prob": float(np.mean((ablated - baseline)[labels_array == 1])),
                    "negative_mean_delta_prob": float(np.mean((ablated - baseline)[labels_array == 0])),
                    **changes,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "c21a_edge_ablation_patient.csv", patient_rows)
    write_csv(output_dir / "c21a_edge_ablation_summary.csv", summary_rows)
    summary = {
        "seeds": list(SEEDS),
        "split": "val",
        "edges": list(ALL_EDGES),
        "patient_row_count": len(patient_rows),
        "summary_row_count": len(summary_rows),
        "test_data_read": False,
        "training_performed": False,
    }
    (output_dir / "c21a_edge_ablation_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
