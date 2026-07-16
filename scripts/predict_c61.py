#!/usr/bin/env python3
"""Run one read-only C61 seed-42 Validation forward for final reproduction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c61_cbpi import C61CBPIModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c61_cbpi_multiseed.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"Invalid C61 checkpoint payload: {path}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError("Final single-checkpoint reproduction accepts only seed 42")
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c61":
        raise RuntimeError("C61 prediction requires the final C61 config")

    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    payload = load_checkpoint(checkpoint)
    set_seed(42)

    rows = read_jsonl(config["project"]["manifest"])
    model_cfg = config["model"]
    dataset = VisitPatientDataset(
        rows=rows,
        data_root=config["project"]["data_root"],
        split=args.split,
        image_size=int(model_cfg["image_size"]),
        text_max_length=int(model_cfg["text_max_length"]),
        text_vocab_size=int(model_cfg["text_vocab_size"]),
        bio_dim=int(model_cfg["bio_dim"]),
        max_images_per_visit=int(model_cfg["max_images_per_visit"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_visit_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C61CBPIModel(config, seed=42).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            probabilities = outputs["prob"].detach().cpu().numpy().astype(float)
            logits = outputs["logit"].detach().cpu().numpy().astype(float)
            labels = batch["label"].detach().cpu().numpy().astype(int)
            for index, patient_id in enumerate(batch["patient_id"]):
                probability = float(probabilities[index])
                predictions.append(
                    {
                        "patient_id": str(patient_id),
                        "label": int(labels[index]),
                        "logit": float(logits[index]),
                        "prob": probability,
                        "predicted_class": int(probability >= 0.5),
                    }
                )

    frame = pd.DataFrame(predictions).sort_values("patient_id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        json.dumps(
            {
                "status": "FINAL_C61_SINGLE_CHECKPOINT_FORWARD_COMPLETE",
                "split": args.split,
                "seed": 42,
                "checkpoint": str(checkpoint),
                "output": str(output),
                "rows": int(len(frame)),
                "device": str(device),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
