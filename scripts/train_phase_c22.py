#!/usr/bin/env python3
"""Train the authorized C22 stable-evidence-pooling residual on the server."""

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

from dmea_ht.c22_stable_pooling import C22StableEvidencePoolingModel, c22_loss_terms  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest  # noqa: E402


SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "raw_n_visits",
    "raw_n_images",
    "source_folder",
)


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
    project = config["project"]
    model_cfg = config["model"]
    training = config["training"]
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

    labels_np = np.asarray(list(labels), dtype=int)
    probs_np = np.asarray(list(probs), dtype=float)
    preds = (probs_np >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "AUC": float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else 0.0,
        "ACC": float((tp + tn) / max(len(labels_np), 1)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "Recall": float(sensitivity),
        "Balanced_ACC": float(0.5 * (sensitivity + specificity)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def _float_array(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def run_epoch(
    model: C22StableEvidencePoolingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    labels: List[int] = []
    probs: List[float] = []
    predictions: List[Dict[str, Any]] = []
    losses: List[float] = []
    cls_losses: List[float] = []
    residual_losses: List[float] = []
    positive_losses: List[float] = []
    deltas: List[float] = []
    positive_deltas: List[float] = []
    negative_deltas: List[float] = []
    evidence_counts: List[float] = []
    evidence_norms: List[float] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            terms = c22_loss_terms(outputs, batch, loss_cfg)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                optimizer.step()

        batch_labels = batch["label"].detach().cpu().numpy().astype(int)
        batch_probs = outputs["prob"].detach().cpu().numpy().astype(float)
        batch_delta = outputs["delta_c22"].detach().cpu().numpy().astype(float)
        batch_count = outputs["valid_evidence_count"].detach().cpu().numpy().astype(float)
        batch_norm = outputs["stable_evidence_norm"].detach().cpu().numpy().astype(float)
        labels.extend(int(value) for value in batch_labels)
        probs.extend(float(value) for value in batch_probs)
        deltas.extend(float(value) for value in batch_delta)
        positive_deltas.extend(float(value) for value in batch_delta[batch_labels == 1])
        negative_deltas.extend(float(value) for value in batch_delta[batch_labels == 0])
        evidence_counts.extend(float(value) for value in batch_count)
        evidence_norms.extend(float(value) for value in batch_norm)
        losses.append(float(terms["total"].detach().cpu()))
        cls_losses.append(float(terms["classification"].detach().cpu()))
        residual_losses.append(float(terms["residual"].detach().cpu()))
        positive_losses.append(float(terms["positive_preserve"].detach().cpu()))

        base_logit = outputs["base_logit"].detach().cpu().numpy()
        base_prob = outputs["base_prob"].detach().cpu().numpy()
        final_logit = outputs["logit"].detach().cpu().numpy()
        for index, patient_id in enumerate(batch["patient_id"]):
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(batch_labels[index]),
                "logit": float(final_logit[index]),
                "prob": float(batch_probs[index]),
                "base_logit": float(base_logit[index]),
                "base_prob": float(base_prob[index]),
                "delta_c22": float(batch_delta[index]),
                "valid_evidence_count": float(batch_count[index]),
                "stable_evidence_norm": float(batch_norm[index]),
            }
            row.update(batch["shortcuts"][index])
            predictions.append(row)

    metrics = binary_metrics(labels, probs)
    delta_np = _float_array(deltas)
    positive_np = _float_array(positive_deltas)
    negative_np = _float_array(negative_deltas)
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "classification_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "residual_loss": float(np.mean(residual_losses)) if residual_losses else 0.0,
            "positive_preservation_loss": float(np.mean(positive_losses)) if positive_losses else 0.0,
            "mean_delta_c22": float(delta_np.mean()) if delta_np.size else 0.0,
            "std_delta_c22": float(delta_np.std(ddof=1)) if delta_np.size > 1 else 0.0,
            "min_delta_c22": float(delta_np.min()) if delta_np.size else 0.0,
            "max_delta_c22": float(delta_np.max()) if delta_np.size else 0.0,
            "mean_positive_delta_c22": float(positive_np.mean()) if positive_np.size else 0.0,
            "mean_negative_delta_c22": float(negative_np.mean()) if negative_np.size else 0.0,
            "fraction_positive_delta_below_minus_0_10": float((positive_np < -0.10).mean()) if positive_np.size else 0.0,
            "fraction_delta_at_lower_bound": float((delta_np <= -0.50 + 1e-5).mean()) if delta_np.size else 0.0,
            "fraction_delta_at_upper_bound": float((delta_np >= 0.50 - 1e-5).mean()) if delta_np.size else 0.0,
            "mean_valid_evidence_count": float(np.mean(evidence_counts)) if evidence_counts else 0.0,
            "mean_stable_evidence_norm": float(np.mean(evidence_norms)) if evidence_norms else 0.0,
            "n_rows": int(len(labels)),
        }
    )
    return {"metrics": metrics, "predictions": predictions}


def train_seed(
    config: Dict[str, Any], rows: List[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, [dict(row) for row in rows])
    model = C22StableEvidencePoolingModel(config, seed).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("residual_head.") for name, _ in trainable):
        raise RuntimeError(f"C22 trainable scope violation: {[name for name, _ in trainable]}")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["training"].get("lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(config["training"].get("epochs", 30))
    patience = int(config["training"].get("patience", 8))
    best_auc = -float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device, config.get("loss", {}))
        val_result = run_epoch(model, loaders["val"], None, device, config.get("loss", {}))
        val_metrics = val_result["metrics"]
        epoch_row: Dict[str, Any] = {
            "seed": int(seed),
            "epoch": int(epoch),
            "split": "train_val",
            "train_total_loss": train_result["metrics"]["loss"],
            "train_classification_loss": train_result["metrics"]["classification_loss"],
            "train_residual_loss": train_result["metrics"]["residual_loss"],
            "train_positive_preservation_loss": train_result["metrics"]["positive_preservation_loss"],
            "val_AUC": val_metrics["AUC"],
            "val_ACC": val_metrics["ACC"],
            "val_Sensitivity": val_metrics["Sensitivity"],
            "val_Specificity": val_metrics["Specificity"],
            "val_Balanced_ACC": val_metrics["Balanced_ACC"],
            "val_loss": val_metrics["loss"],
            "val_mean_delta_c22": val_metrics["mean_delta_c22"],
            "val_std_delta_c22": val_metrics["std_delta_c22"],
            "val_mean_positive_delta_c22": val_metrics["mean_positive_delta_c22"],
            "val_mean_negative_delta_c22": val_metrics["mean_negative_delta_c22"],
            "val_fraction_positive_delta_below_minus_0_10": val_metrics[
                "fraction_positive_delta_below_minus_0_10"
            ],
            "selected_by_val_auc": False,
        }
        epoch_rows.append(epoch_row)
        val_auc = float(val_metrics["AUC"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"C22 seed {seed} produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == int(best_epoch)
    val_result = run_epoch(model, loaders["val"], None, device, config.get("loss", {}))
    test_result = (
        run_epoch(model, loaders["test"], None, device, config.get("loss", {}))
        if bool(config["training"].get("evaluate_test", True))
        else None
    )
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "seed": int(seed), "best_epoch": int(best_epoch)},
        checkpoint_dir / f"seed_{seed}_best.pt",
    )
    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epoch_history": epoch_rows,
        "val": val_result,
        "test": test_result,
        "trainable_parameter_names": [name for name, _ in trainable],
    }


def write_predictions(result: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    metrics_rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        split_result = result.get(split)
        if split_result is None:
            continue
        frame = pd.DataFrame(split_result["predictions"])
        frame.insert(0, "split", split)
        frame.insert(0, "seed", int(result["seed"]))
        frame.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{result['seed']}.csv", index=False)
        row: Dict[str, Any] = {"seed": int(result["seed"]), "split": split, "best_epoch": int(result["best_epoch"])}
        row.update(split_result["metrics"])
        metrics_rows.append(row)
    return metrics_rows


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c22" or not config.get("c22"):
        raise RuntimeError("C22 config/phase contract is missing")
    if args.data_root:
        config["project"]["data_root"] = args.data_root
    if args.manifest:
        config["project"]["manifest"] = args.manifest
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    rows = read_manifest(config["project"]["manifest"])
    if not all(str(row.get("split", "")).strip() for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split

    out_dir = resolve_path(config["project"]["output_dir"])
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = timestamp()
    status = {
        "phase": "C22",
        "status": "RUNNING",
        "started_at": start,
        "completed_seeds": [],
        "seeds": [int(seed) for seed in config["training"].get("seeds", [0, 42, 3407])],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "reports" / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    metrics_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    trainable_names_by_seed: Dict[str, List[str]] = {}
    for seed_value in config["training"].get("seeds", [0, 42, 3407]):
        seed = int(seed_value)
        result = train_seed(config, rows, seed, out_dir, device)
        metrics_rows.extend(write_predictions(result, out_dir))
        epoch_rows.extend(result["epoch_history"])
        trainable_names_by_seed[str(seed)] = result["trainable_parameter_names"]
        status["completed_seeds"].append(seed)
        (out_dir / "reports" / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        split_frame = metrics_frame[metrics_frame["split"] == split]
        if split_frame.empty:
            continue
        summary: Dict[str, Any] = {"split": split}
        for key in ("AUC", "ACC", "Sensitivity", "Specificity", "Balanced_ACC", "mean_delta_c22", "std_delta_c22"):
            values = pd.to_numeric(split_frame[key], errors="coerce").dropna().to_numpy(dtype=float)
            summary[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
            summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)
    metrics_frame[["seed", "split", "TN", "FP", "FN", "TP"]].to_csv(
        out_dir / "reports" / "confusion_matrix_by_seed.csv", index=False
    )

    status.update({"status": "COMPLETE", "finished_at": timestamp()})
    (out_dir / "reports" / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "phase": "C22",
        "started_at": start,
        "finished_at": status["finished_at"],
        "device": str(device),
        "gpu": status["gpu"],
        "seeds": status["seeds"],
        "trainable_parameter_names_by_seed": trainable_names_by_seed,
        "selection_metric": "validation_AUC_only",
        "test_role": "reporting_only_after_validation_selection",
    }
    (out_dir / "reports" / "run_config.json").write_text(
        json.dumps({"config": config, "runtime": runtime}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "output_dir": str(out_dir), "seeds": status["seeds"]}))


if __name__ == "__main__":
    main()
