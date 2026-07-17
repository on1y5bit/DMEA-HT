#!/usr/bin/env python3
"""Compare C61 frozen source representations on the common development pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c61_cbpi import C61CBPIModel  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch  # noqa: E402
from scripts import c65a_common as common  # noqa: E402


STREAM_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65a.yaml")
    return parser.parse_args()


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError(f"Invalid C61 checkpoint payload: {path}")
    return payload


def source_representation(source: Mapping[str, torch.Tensor]) -> torch.Tensor:
    states = source["states"].float()
    visit_mask = source["visit_mask"].bool()
    valid = source["valid"].bool() & visit_mask.unsqueeze(-1)
    weights = valid.to(states.dtype)
    sums = (states * weights.unsqueeze(-1)).sum(dim=1)
    counts = weights.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
    pooled = sums / counts
    return pooled.reshape(pooled.shape[0], STREAM_COUNT * pooled.shape[-1])


def forward_seed(config: Mapping[str, Any], c61_config: Mapping[str, Any], seed: int, rows: list[Dict[str, Any]], output: Path) -> Dict[str, Any]:
    checkpoint = Path(str(config["project"]["c61_checkpoint"]).replace("{seed}", str(seed)))
    if not checkpoint.is_absolute():
        checkpoint = common.resolve_path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"C61 checkpoint missing: {checkpoint}")
    payload = checkpoint_payload(checkpoint)
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"C61 checkpoint seed mismatch: {checkpoint}")
    model_cfg = c61_config["model"]
    dataset_rows = [dict(row, split="val") for row in rows]
    dataset = VisitPatientDataset(
        rows=dataset_rows,
        data_root=c61_config["project"]["data_root"],
        split="val",
        image_size=int(model_cfg["image_size"]),
        text_max_length=int(model_cfg["text_max_length"]),
        text_vocab_size=int(model_cfg["text_vocab_size"]),
        bio_dim=int(model_cfg["bio_dim"]),
        max_images_per_visit=int(model_cfg["max_images_per_visit"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(c61_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(c61_config["training"].get("num_workers", 0)),
        collate_fn=collate_visit_batch,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.set_seed(seed)
    model = C61CBPIModel(dict(c61_config), seed=seed).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    patient_ids = []
    labels = []
    representations = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            source = model._source_states(batch)
            representation = source_representation(source).cpu().numpy().astype(np.float32)
            representations.append(representation)
            patient_ids.extend(str(item) for item in batch["patient_id"])
            labels.extend(int(item) for item in batch["label"].detach().cpu().numpy())
    frame = pd.DataFrame({"patient_id": patient_ids, "label": labels})
    values = np.concatenate(representations, axis=0)
    order = np.argsort(frame["patient_id"].astype(str).to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    values = values[order]
    if len(frame) != common.DEVELOPMENT_COUNT or frame["patient_id"].duplicated().any():
        raise RuntimeError(f"C65 frozen representation coverage failed for seed {seed}")
    np.savez_compressed(output / f"c65a_frozen_source_representation_seed_{seed}.npz", patient_id=frame["patient_id"].to_numpy(dtype=str), label=frame["label"].to_numpy(dtype=np.int64), representation=values)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "patient_count": int(len(frame)),
        "representation_dimension": int(values.shape[1]),
        "norm_mean": float(np.linalg.norm(values, axis=1).mean()),
        "norm_std": float(np.linalg.norm(values, axis=1).std(ddof=1)),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "test_loaded": False,
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    cross = left_centered.T @ right_centered
    left_gram = left_centered.T @ left_centered
    right_gram = right_centered.T @ right_centered
    denominator = float(np.linalg.norm(left_gram, ord="fro") * np.linalg.norm(right_gram, ord="fro"))
    return float(np.square(np.linalg.norm(cross, ord="fro")) / denominator) if denominator > 0.0 else float("nan")


def representation_pair(left: np.ndarray, right: np.ndarray, k: int) -> Dict[str, Any]:
    left_distance = common.pairwise_euclidean(left)
    right_distance = common.pairwise_euclidean(right)
    distance_spearman = common.safe_spearman(common.upper_triangle(left_distance), common.upper_triangle(right_distance))
    left_knn = common.knn_indices(left, k)
    right_knn = common.knn_indices(right, k)
    return {
        "linear_cka": linear_cka(left, right),
        "patient_distance_spearman": distance_spearman,
        "knn_jaccard_mean": common.mean_jaccard(left_knn, right_knn),
        "distance_definition": "raw Euclidean distance; Spearman across patient pairs",
        "knn_definition": "z-scored feature Euclidean distance; k=10 excluding self",
    }


def main() -> None:
    args = parse_args()
    config = common.load_c65a_config(args.config)
    rows = common.development_rows(config)
    c61_config = common.c61_config(config)
    output = common.report_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for seed in common.SEEDS:
        metadata.append(forward_seed(config, c61_config, seed, rows, output))
    arrays = {}
    for seed in common.SEEDS:
        payload = np.load(output / f"c65a_frozen_source_representation_seed_{seed}.npz")
        arrays[seed] = {key: payload[key] for key in payload.files}
    reference_ids = arrays[common.SEEDS[0]]["patient_id"].astype(str)
    reference_labels = arrays[common.SEEDS[0]]["label"].astype(int)
    for seed in common.SEEDS[1:]:
        if not np.array_equal(reference_ids, arrays[seed]["patient_id"].astype(str)) or not np.array_equal(reference_labels, arrays[seed]["label"].astype(int)):
            raise RuntimeError("C65 frozen representation patient or label alignment failed")
    pair_rows = []
    for index, left_seed in enumerate(common.SEEDS):
        for right_seed in common.SEEDS[index + 1 :]:
            pair_rows.append({"seed_a": left_seed, "seed_b": right_seed, "patient_count": common.DEVELOPMENT_COUNT, **representation_pair(arrays[left_seed]["representation"].astype(np.float64), arrays[right_seed]["representation"].astype(np.float64), int(config["analysis"]["knn_k"]))})
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "c65a_frozen_representation_stability.csv", index=False)
    summary = {
        "status": "C65A_FROZEN_REPRESENTATION_ANALYSIS_COMPLETE",
        "representation": "frozen_c61_source_patient_stream_mean_before_c64_head",
        "patient_count": common.DEVELOPMENT_COUNT,
        "metadata_by_seed": metadata,
        "mean_linear_cka": float(pair_frame["linear_cka"].mean()),
        "mean_patient_distance_spearman": float(pair_frame["patient_distance_spearman"].mean()),
        "mean_knn_jaccard": float(pair_frame["knn_jaccard_mean"].mean()),
        "test_loaded": False,
    }
    common.write_json(output / "c65a_frozen_backbone_summary.json", summary)
    print(json.dumps({"status": summary["status"], "mean_linear_cka": summary["mean_linear_cka"], "mean_patient_distance_spearman": summary["mean_patient_distance_spearman"], "mean_knn_jaccard": summary["mean_knn_jaccard"]}, sort_keys=True))


if __name__ == "__main__":
    main()
