#!/usr/bin/env python3
"""Train the final authorized C25 pairwise-ranking residual on the server."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c25_pairwise_ranking_residual import (  # noqa: E402
    C25PairwiseRankingResidualModel,
    c25_loss_terms,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loaders(config: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, DataLoader]:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = PatientHTDataset(
            rows=rows,
            data_root=project["data_root"],
            split=split,
            max_images=int(model_cfg.get("max_images_per_patient", 4)),
            image_size=int(model_cfg.get("image_size", 224)),
            text_max_length=int(model_cfg.get("text_max_length", 256)),
            text_vocab_size=int(model_cfg.get("text_vocab_size", 50000)),
            bio_dim=int(model_cfg.get("bio_dim", 32)),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training.get("batch_size", 8)),
            shuffle=split == "train",
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate_patient_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def binary_metrics(labels: Iterable[int], probs: Iterable[float]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probs), dtype=float)
    pred = p >= 0.5
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_ACC": float(0.5 * (sensitivity + specificity)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def pairwise_inversions(labels: np.ndarray, probs: np.ndarray) -> int:
    positive = probs[labels == 1]
    negative = probs[labels == 0]
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def run_epoch(
    model: C25PairwiseRankingResidualModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    predictions: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, float]] = []
    mixed_class_batches = 0
    single_class_batches = 0
    pair_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            if is_train:
                terms = c25_loss_terms(outputs, batch, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                optimizer.step()
                tensor_terms = {key: value for key, value in terms.items() if torch.is_tensor(value)}
                loss_rows.append({key: float(value.detach().cpu()) for key, value in tensor_terms.items()})
                batch_pairs = int(terms["pair_count"])
                pair_count += batch_pairs
                if batch_pairs > 0:
                    mixed_class_batches += 1
                else:
                    single_class_batches += 1
        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items() if torch.is_tensor(value)}
        labels = batch["label"].detach().cpu().numpy().astype(int)
        for index, patient_id in enumerate(batch["patient_id"]):
            abs_logit = abs(float(arrays["frozen_c17_logit"][index]))
            group = "low" if abs_logit < 0.75 else ("medium" if abs_logit < 2.0 else "high")
            frozen_class = int(float(arrays["frozen_c17_logit"][index]) >= 0.0)
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "frozen_c17_logit": float(arrays["frozen_c17_logit"][index]),
                "frozen_c17_prob": float(arrays["frozen_c17_prob"][index]),
                "frozen_c17_predicted_class": frozen_class,
                "frozen_c17_correct_audit_only": bool(frozen_class == int(labels[index])),
                "abs_frozen_c17_logit": abs_logit,
                "confidence_gate": float(arrays["confidence_gate"][index]),
                "confidence_group": group,
                "mechanism_representation_norm": float(arrays["mechanism_representation_norm"][index]),
                "raw_delta_c25": float(arrays["raw_delta_c25"][index]),
                "delta_c25": float(arrays["delta_c25"][index]),
                "logit": float(arrays["logit"][index]),
                "prob": float(arrays["prob"][index]),
                "final_predicted_class": int(float(arrays["logit"][index]) >= 0.0),
            }
            row.update(batch["shortcuts"][index])
            predictions.append(row)

    frame = pd.DataFrame(predictions)
    y = frame["label"].to_numpy(dtype=int)
    p = frame["prob"].to_numpy(dtype=float)
    delta = frame["delta_c25"].to_numpy(dtype=float)
    gate = frame["confidence_gate"].to_numpy(dtype=float)
    positive, negative = y == 1, y == 0
    frozen_correct = frame["frozen_c17_correct_audit_only"].astype(bool).to_numpy()
    correct_positive = frozen_correct & positive
    correct_negative = frozen_correct & negative
    incorrect_positive = (~frozen_correct) & positive
    incorrect_negative = (~frozen_correct) & negative
    metrics = binary_metrics(y, p)
    for key in loss_rows[0] if loss_rows else ():
        metrics[f"{key}_loss"] = float(np.mean([row[key] for row in loss_rows]))
    metrics.update(
        {
            "positive_prob_mean": _mean(p[positive]),
            "negative_prob_mean": _mean(p[negative]),
            "probability_gap": _mean(p[positive]) - _mean(p[negative]),
            "confidence_gate_mean": _mean(gate),
            "confidence_gate_positive_mean": _mean(gate[positive]),
            "confidence_gate_negative_mean": _mean(gate[negative]),
            "mixed_class_batch_count": int(mixed_class_batches),
            "single_class_batch_count": int(single_class_batches),
            "mixed_class_batch_fraction": float(mixed_class_batches / max(mixed_class_batches + single_class_batches, 1)) if is_train else 0.0,
            "pair_count": int(pair_count),
            "delta_mean": _mean(delta),
            "delta_median": float(np.median(delta)) if delta.size else 0.0,
            "delta_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
            "positive_delta_mean": _mean(delta[positive]),
            "negative_delta_mean": _mean(delta[negative]),
            "correct_positive_delta_mean": _mean(delta[correct_positive]),
            "correct_negative_delta_mean": _mean(delta[correct_negative]),
            "incorrect_positive_delta_mean": _mean(delta[incorrect_positive]),
            "incorrect_negative_delta_mean": _mean(delta[incorrect_negative]),
            "fraction_correct_positive_wrong_direction": _mean((delta[correct_positive] < 0.0).astype(float)),
            "fraction_correct_negative_wrong_direction": _mean((delta[correct_negative] > 0.0).astype(float)),
            "positive_delta_fraction": _mean((delta > 0.0).astype(float)),
            "negative_delta_fraction": _mean((delta < 0.0).astype(float)),
            "mean_abs_delta_low": _mean(np.abs(delta[frame["confidence_group"].eq("low").to_numpy()])),
            "mean_abs_delta_medium": _mean(np.abs(delta[frame["confidence_group"].eq("medium").to_numpy()])),
            "mean_abs_delta_high": _mean(np.abs(delta[frame["confidence_group"].eq("high").to_numpy()])),
            "fraction_near_negative_bound": _mean((delta <= -0.99 * 0.15 * gate).astype(float)),
            "fraction_near_positive_bound": _mean((delta >= 0.99 * 0.15 * gate).astype(float)),
            "pairwise_inversion_count": pairwise_inversions(y, p),
            "n_rows": int(len(frame)),
        }
    )
    return {"metrics": metrics, "predictions": predictions}


def train_seed(
    config: Dict[str, Any], rows: List[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, [dict(row) for row in rows])
    model = C25PairwiseRankingResidualModel(config, seed).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("residual_head.") for name, _ in trainable):
        raise RuntimeError(f"C25 trainable scope violation: {[name for name, _ in trainable]}")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["training"].get("lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device, config["loss"])
        val_result = run_epoch(model, loaders["val"], None, device, config["loss"])
        row: Dict[str, Any] = {"seed": seed, "epoch": epoch, "selected_by_val_auc": False}
        row.update({f"train_{key}": value for key, value in train_result["metrics"].items()})
        row.update({f"val_{key}": value for key, value in val_result["metrics"].items()})
        epoch_rows.append(row)
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["training"].get("patience", 8)):
            break
    if best_state is None:
        raise RuntimeError(f"C25 seed {seed} produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device, config["loss"])
    test_result = run_epoch(model, loaders["test"], None, device, config["loss"]) if config["training"].get("evaluate_test", True) else None
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch},
        checkpoint_dir / f"seed_{seed}_best.pt",
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "test": test_result,
        "trainable_parameter_names": [name for name, _ in trainable],
    }


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c25" or not config.get("c25"):
        raise RuntimeError("C25 config/phase contract is missing")
    for key, value in (("data_root", args.data_root), ("manifest", args.manifest), ("output_dir", args.output_dir)):
        if value:
            config["project"][key] = value
    rows = read_manifest(config["project"]["manifest"])
    if not all(str(row.get("split", "")).strip() for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split
    out_dir = resolve_path(config["project"]["output_dir"])
    for child in ("reports", "predictions", "checkpoints"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = timestamp()
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    status: Dict[str, Any] = {
        "phase": "C25", "status": "RUNNING", "started_at": started, "completed_seeds": [],
        "seeds": seeds, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    status_path = out_dir / "reports" / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metrics_rows, epoch_rows = [], []
    trainable_by_seed: Dict[str, List[str]] = {}
    for seed in seeds:
        result = train_seed(config, rows, seed, out_dir, device)
        epoch_rows.extend(result["epoch_history"])
        trainable_by_seed[str(seed)] = result["trainable_parameter_names"]
        for split in ("val", "test"):
            split_result = result.get(split)
            if split_result is None:
                continue
            frame = pd.DataFrame(split_result["predictions"])
            frame.insert(0, "split", split)
            frame.insert(0, "seed", seed)
            frame.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
            metrics_rows.append({"seed": seed, "split": split, "best_epoch": result["best_epoch"], **split_result["metrics"]})
        status["completed_seeds"].append(seed)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    summary_rows = []
    for split, split_frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "delta_mean", "delta_std"):
            values = split_frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)
    status.update({"status": "COMPLETE", "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "config": config, "started_at": started, "finished_at": status["finished_at"],
        "device": str(device), "gpu": status["gpu"], "seeds": seeds,
        "trainable_parameter_names_by_seed": trainable_by_seed,
        "selection_metric": "validation_AUC_only", "test_role": "reporting_only_after_validation_selection",
    }
    (out_dir / "reports" / "run_config.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output_dir": str(out_dir), "seeds": seeds}))


if __name__ == "__main__":
    main()
