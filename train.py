from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmea_ht.config import load_config
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest
from dmea_ht.evidence_losses import confidence_weighted_bce_with_logits
from dmea_ht.metrics import compute_binary_metrics, summarize_metrics
from dmea_ht.models import DMEAHTModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def make_loaders(config: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, DataLoader]:
    project = config["project"]
    model = config["model"]
    training = config["training"]
    if not all("split" in row and row["split"] for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = PatientHTDataset(
            rows=rows,
            data_root=project["data_root"],
            split=split,
            max_images=int(model.get("max_images_per_patient", 4)),
            image_size=int(model.get("image_size", 224)),
            text_max_length=int(model.get("text_max_length", 256)),
            text_vocab_size=int(model.get("text_vocab_size", 50000)),
            bio_dim=int(model.get("bio_dim", 32)),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training.get("batch_size", 8)),
            shuffle=(split == "train"),
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate_patient_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def evidence_loss_terms(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], loss_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    terms: Dict[str, torch.Tensor] = {}
    text_weight = float(loss_cfg.get("text_morphology_weight", 0.0))
    image_weight = float(loss_cfg.get("image_morphology_weight", 0.0))
    if text_weight > 0 and "text_morphology_logit" in outputs:
        terms["text_morphology_loss"] = confidence_weighted_bce_with_logits(
            outputs["text_morphology_logit"],
            batch["txt_morphology_label"],
            batch["txt_morphology_confidence"],
        )
    if image_weight > 0 and "image_morphology_logit" in outputs:
        terms["image_morphology_loss"] = confidence_weighted_bce_with_logits(
            outputs["image_morphology_logit"],
            batch["image_morphology_weak_label"],
            batch["image_morphology_weak_confidence"],
        )
    return terms


def add_evidence_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    labels: List[int],
    probs: List[float],
    confidences: List[float],
    losses: List[float],
) -> None:
    metrics[f"{prefix}_loss"] = float(np.mean(losses)) if losses else 0.0
    metrics[f"valid_{prefix}_count"] = int(len(labels))
    metrics[f"mean_{prefix}_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    if labels:
        evidence_metrics = compute_binary_metrics(labels, probs)
        metrics[f"{prefix}_auc"] = evidence_metrics["AUC"]
        metrics[f"{prefix}_acc"] = evidence_metrics["ACC"]
    else:
        metrics[f"{prefix}_auc"] = 0.0
        metrics[f"{prefix}_acc"] = 0.0


def run_epoch(
    model: DMEAHTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    labels: List[int] = []
    probs: List[float] = []
    losses: List[float] = []
    text_morphology_labels: List[int] = []
    text_morphology_probs: List[float] = []
    text_morphology_confidences: List[float] = []
    text_morphology_losses: List[float] = []
    image_morphology_labels: List[int] = []
    image_morphology_probs: List[float] = []
    image_morphology_confidences: List[float] = []
    image_morphology_losses: List[float] = []
    predictions: List[Dict[str, Any]] = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    cls_weight = float(loss_cfg.get("cls_weight", 1.0))
    text_weight = float(loss_cfg.get("text_morphology_weight", 0.0))
    image_weight = float(loss_cfg.get("image_morphology_weight", 0.0))

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            raw_loss = criterion(outputs["logit"], batch["label"])
            loss = cls_weight * (raw_loss * batch["sample_weight"]).mean()
            evidence_terms = evidence_loss_terms(outputs, batch, loss_cfg)
            if "text_morphology_loss" in evidence_terms:
                loss = loss + text_weight * evidence_terms["text_morphology_loss"]
                text_morphology_losses.append(float(evidence_terms["text_morphology_loss"].detach().cpu()))
            if "image_morphology_loss" in evidence_terms:
                loss = loss + image_weight * evidence_terms["image_morphology_loss"]
                image_morphology_losses.append(float(evidence_terms["image_morphology_loss"].detach().cpu()))
            if "role_alignment_loss" in outputs:
                loss = loss + outputs["role_alignment_loss"] * 0.0
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        batch_probs = outputs["prob"].detach().cpu().numpy().tolist()
        batch_labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        labels.extend(batch_labels)
        probs.extend(batch_probs)

        if "text_morphology_logit" in outputs:
            valid = (batch["txt_morphology_label"] != -1) & (batch["txt_morphology_confidence"] > 0)
            if bool(valid.any().item()):
                text_morphology_labels.extend(batch["txt_morphology_label"][valid].detach().cpu().numpy().astype(int).tolist())
                text_morphology_probs.extend(torch.sigmoid(outputs["text_morphology_logit"][valid]).detach().cpu().numpy().tolist())
                text_morphology_confidences.extend(batch["txt_morphology_confidence"][valid].detach().cpu().numpy().tolist())
        if "image_morphology_logit" in outputs:
            valid = (batch["image_morphology_weak_label"] != -1) & (batch["image_morphology_weak_confidence"] > 0)
            if bool(valid.any().item()):
                image_morphology_labels.extend(batch["image_morphology_weak_label"][valid].detach().cpu().numpy().astype(int).tolist())
                image_morphology_probs.extend(torch.sigmoid(outputs["image_morphology_logit"][valid]).detach().cpu().numpy().tolist())
                image_morphology_confidences.extend(batch["image_morphology_weak_confidence"][valid].detach().cpu().numpy().tolist())

        for i, patient_id in enumerate(batch["patient_id"]):
            row = {
                "patient_id": patient_id,
                "label": batch_labels[i],
                "prob": float(batch_probs[i]),
                "pred_prob": float(batch_probs[i]),
                "logit": float(outputs["logit"].detach().cpu()[i]),
                "txt_morphology_label": int(batch["txt_morphology_label"].detach().cpu()[i]),
                "txt_morphology_confidence": float(batch["txt_morphology_confidence"].detach().cpu()[i]),
                "matched_morphology_terms": "|".join(str(term) for term in batch["matched_morphology_terms"][i]),
            }
            for key in ("e_img", "e_text", "e_bio", "e_synergy", "e_negative", "d_img_txt", "d_img_bio", "d_txt_bio"):
                if key in outputs:
                    row[key] = float(outputs[key].detach().cpu()[i])
            for key in ("text_morphology_logit", "image_morphology_logit"):
                if key in outputs:
                    row[key] = float(outputs[key].detach().cpu()[i])
            if "text_morphology_prob" in outputs:
                row["text_morphology_prob"] = float(outputs["text_morphology_prob"].detach().cpu()[i])
            if "text_morphology_anchor" in outputs:
                anchor = outputs["text_morphology_anchor"].detach().cpu()[i]
                row["text_morphology_anchor_norm"] = float(anchor.norm())
                row["text_morphology_anchor_mean"] = float(anchor.mean())
            row.update(batch["shortcuts"][i])
            predictions.append(row)

    metrics = compute_binary_metrics(labels, probs)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    add_evidence_metrics(
        metrics,
        "text_morphology",
        text_morphology_labels,
        text_morphology_probs,
        text_morphology_confidences,
        text_morphology_losses,
    )
    add_evidence_metrics(
        metrics,
        "image_morphology",
        image_morphology_labels,
        image_morphology_probs,
        image_morphology_confidences,
        image_morphology_losses,
    )
    return {"metrics": metrics, "predictions": predictions}


def train_seed(config: Dict[str, Any], rows: List[Dict[str, Any]], seed: int, out_dir: Path) -> Dict[str, Any]:
    set_seed(seed)
    loaders = make_loaders(config, [dict(row) for row in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMEAHTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(config["training"].get("epochs", 30))
    patience = int(config["training"].get("patience", 8))
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        run_epoch(model, loaders["train"], optimizer, device, config.get("loss", {}))
        val_result = run_epoch(model, loaders["val"], None, device, config.get("loss", {}))
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_result = run_epoch(model, loaders["val"], None, device, config.get("loss", {}))
    test_result = run_epoch(model, loaders["test"], None, device, config.get("loss", {}))
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch}, checkpoints / f"seed_{seed}_best.pt")
    return {"seed": seed, "best_epoch": best_epoch, "val": val_result, "test": test_result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["project"]["data_root"] = args.data_root
    if args.manifest:
        config["project"]["manifest"] = args.manifest
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    rows = read_manifest(config["project"]["manifest"])
    out_dir = Path(config["project"]["output_dir"])
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    for seed in config["training"].get("seeds", [0, 42, 3407]):
        result = train_seed(config, rows, int(seed), out_dir)
        for split in ("val", "test"):
            pred_df = pd.DataFrame(result[split]["predictions"])
            pred_df.insert(3, "split", split)
            pred_df.insert(4, "seed", int(seed))
            pred_df.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
            metric_row = {"seed": int(seed), "split": split, "best_epoch": result["best_epoch"]}
            metric_row.update(result[split]["metrics"])
            metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    summary_rows = []
    for split in ("val", "test"):
        split_rows = [row for row in metrics_rows if row["split"] == split]
        summary = {"split": split}
        summary.update(
            summarize_metrics(
                split_rows,
                [
                    "AUC",
                    "AUPRC",
                    "ACC",
                    "F1",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "Recall",
                    "Balanced_ACC",
                    "text_morphology_auc",
                    "text_morphology_acc",
                    "valid_text_morphology_count",
                    "mean_text_morphology_confidence",
                    "image_morphology_auc",
                    "image_morphology_acc",
                    "valid_image_morphology_count",
                    "mean_image_morphology_confidence",
                ],
            )
        )
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)
    cm_cols = ["seed", "split", "TN", "FP", "FN", "TP"]
    metrics_df[cm_cols].to_csv(out_dir / "reports" / "confusion_matrix_by_seed.csv", index=False)
    (out_dir / "reports" / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
