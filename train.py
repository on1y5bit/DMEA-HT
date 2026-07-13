from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmea_ht.config import load_config
from dmea_ht.alignment import MODALITIES
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest
from dmea_ht.evidence_losses import confidence_weighted_bce_with_logits
from dmea_ht.metrics import compute_binary_metrics, summarize_metrics
from dmea_ht.models import DMEAHTModel


DSSA_LOSS_WEIGHTS = {
    "proto_classification_loss": "proto",
    "shared_consistency_loss": "shared",
    "shared_specific_orth_loss": "orth",
    "specific_variance_loss": "var",
    "prototype_separation_loss": "sep",
    "pairwise_ranking_loss": "rank",
}

DSSA_PREDICTION_KEYS = (
    "prototype_similarity_non_ht",
    "prototype_similarity_ht",
    "disease_margin",
    "patient_shared_norm",
    "specific_residual_norm",
    "specific_residual_shared_ratio",
    "soft_disease_anchor_norm",
    "shared_attention_img",
    "shared_attention_txt",
    "shared_attention_bio",
    "specific_gate_img",
    "specific_gate_txt",
    "specific_gate_bio",
    "shared_img_norm",
    "shared_txt_norm",
    "shared_bio_norm",
    "specific_img_norm",
    "specific_txt_norm",
    "specific_bio_norm",
    "shared_specific_cosine_img",
    "shared_specific_cosine_txt",
    "shared_specific_cosine_bio",
    "shared_cosine_img_txt",
    "shared_cosine_img_bio",
    "shared_cosine_txt_bio",
    "shared_pair_available_img_txt",
    "shared_pair_available_img_bio",
    "shared_pair_available_txt_bio",
    "modality_available_img",
    "modality_available_txt",
    "modality_available_bio",
)


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


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive_logits = logits[labels > 0.5]
    negative_logits = logits[labels <= 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return logits.sum() * 0.0
    margins = positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
    return F.softplus(-margins).mean()


def dssa_loss_terms(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    include_ranking: bool,
) -> Dict[str, torch.Tensor]:
    if "dssa_prototype_logits" not in outputs:
        return {}
    logits = outputs["dssa_prototype_logits"]
    available = outputs["dssa_available_mask"].bool()
    targets = labels.long().unsqueeze(1).expand(-1, logits.shape[1])
    if bool(available.any().item()):
        prototype_loss = F.cross_entropy(logits[available], targets[available])
    else:
        prototype_loss = logits.sum() * 0.0
    zero = outputs["logit"].sum() * 0.0
    return {
        "proto_classification_loss": prototype_loss,
        "shared_consistency_loss": outputs["dssa_shared_consistency_loss"],
        "shared_specific_orth_loss": outputs["dssa_shared_specific_orth_loss"],
        "specific_variance_loss": outputs["dssa_specific_variance_loss"],
        "prototype_separation_loss": outputs["dssa_prototype_separation_loss"],
        "pairwise_ranking_loss": pairwise_ranking_loss(outputs["logit"], labels) if include_ranking else zero,
    }


def effective_dssa_weight(loss_cfg: Dict[str, Any], short_name: str) -> float:
    return float(loss_cfg.get(f"effective_lambda_{short_name}", loss_cfg.get(f"lambda_{short_name}", 0.0)))


def add_representation_health_metrics(
    metrics: Dict[str, Any],
    shared_batches: List[torch.Tensor],
    specific_batches: List[torch.Tensor],
    availability_batches: List[torch.Tensor],
) -> None:
    if not shared_batches or not specific_batches or not availability_batches:
        return
    shared = torch.cat(shared_batches, dim=0)
    specific = torch.cat(specific_batches, dim=0)
    available = torch.cat(availability_batches, dim=0).bool()
    for index, modality in enumerate(MODALITIES):
        for prefix, representations in (("shared", shared), ("specific", specific)):
            selected = representations[available[:, index], index]
            metrics[f"{prefix}_{modality}_sample_count"] = int(selected.shape[0])
            if selected.shape[0] < 2:
                metrics[f"{prefix}_{modality}_feature_std_mean"] = 0.0
                metrics[f"{prefix}_{modality}_offdiag_cosine_mean"] = 0.0
                continue
            metrics[f"{prefix}_{modality}_feature_std_mean"] = float(
                selected.std(dim=0, unbiased=False).mean()
            )
            normalized = F.normalize(selected, dim=-1, eps=1e-8)
            cosine_matrix = normalized @ normalized.transpose(0, 1)
            offdiag_sum = cosine_matrix.sum() - cosine_matrix.diag().sum()
            denominator = selected.shape[0] * (selected.shape[0] - 1)
            metrics[f"{prefix}_{modality}_offdiag_cosine_mean"] = float(offdiag_sum / denominator)


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
    logits: List[float] = []
    losses: List[float] = []
    cls_losses: List[float] = []
    text_morphology_labels: List[int] = []
    text_morphology_probs: List[float] = []
    text_morphology_confidences: List[float] = []
    text_morphology_losses: List[float] = []
    image_morphology_labels: List[int] = []
    image_morphology_probs: List[float] = []
    image_morphology_confidences: List[float] = []
    image_morphology_losses: List[float] = []
    dssa_loss_values: Dict[str, List[float]] = defaultdict(list)
    dssa_diagnostics: Dict[str, List[float]] = defaultdict(list)
    dssa_shared_batches: List[torch.Tensor] = []
    dssa_specific_batches: List[torch.Tensor] = []
    dssa_availability_batches: List[torch.Tensor] = []
    dssa_enabled = False
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
            cls_loss = cls_weight * (raw_loss * batch["sample_weight"]).mean()
            loss = cls_loss
            evidence_terms = evidence_loss_terms(outputs, batch, loss_cfg)
            if "text_morphology_loss" in evidence_terms:
                loss = loss + text_weight * evidence_terms["text_morphology_loss"]
                text_morphology_losses.append(float(evidence_terms["text_morphology_loss"].detach().cpu()))
            if "image_morphology_loss" in evidence_terms:
                loss = loss + image_weight * evidence_terms["image_morphology_loss"]
                image_morphology_losses.append(float(evidence_terms["image_morphology_loss"].detach().cpu()))
            if "role_alignment_loss" in outputs:
                loss = loss + outputs["role_alignment_loss"] * 0.0
            dssa_terms = dssa_loss_terms(outputs, batch["label"], include_ranking=is_train)
            for term_name, term in dssa_terms.items():
                short_name = DSSA_LOSS_WEIGHTS[term_name]
                loss = loss + effective_dssa_weight(loss_cfg, short_name) * term
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        cls_losses.append(float(cls_loss.detach().cpu()))
        batch_probs = outputs["prob"].detach().cpu().numpy().tolist()
        batch_logits = outputs["logit"].detach().cpu().numpy().tolist()
        batch_labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        labels.extend(batch_labels)
        probs.extend(batch_probs)
        logits.extend(batch_logits)

        if dssa_terms:
            dssa_enabled = True
            for term_name, term in dssa_terms.items():
                dssa_loss_values[term_name].append(float(term.detach().cpu()))
            for key in (
                "prototype_cosine",
                "prototype_distance",
                "prototype_similarity_non_ht",
                "prototype_similarity_ht",
                "disease_margin",
                "patient_shared_norm",
                "specific_residual_norm",
                "specific_residual_shared_ratio",
                "soft_disease_anchor_norm",
                "shared_attention_img",
                "shared_attention_txt",
                "shared_attention_bio",
                "specific_gate_img",
                "specific_gate_txt",
                "specific_gate_bio",
                "shared_specific_cosine_img",
                "shared_specific_cosine_txt",
                "shared_specific_cosine_bio",
                "shared_cosine_img_txt",
                "shared_cosine_img_bio",
                "shared_cosine_txt_bio",
                "shared_pair_available_img_txt",
                "shared_pair_available_img_bio",
                "shared_pair_available_txt_bio",
                "modality_available_img",
                "modality_available_txt",
                "modality_available_bio",
            ):
                if key not in outputs:
                    continue
                values = outputs[key].detach().cpu().reshape(-1).numpy().tolist()
                dssa_diagnostics[key].extend(float(value) for value in values)
            dssa_shared_batches.append(outputs["dssa_shared_representations"].detach().cpu())
            dssa_specific_batches.append(outputs["dssa_specific_representations"].detach().cpu())
            dssa_availability_batches.append(outputs["dssa_available_mask"].detach().cpu())

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
            for key in DSSA_PREDICTION_KEYS:
                if key in outputs:
                    row[key] = float(outputs[key].detach().cpu()[i])
            row.update(batch["shortcuts"][i])
            predictions.append(row)

    metrics = compute_binary_metrics(labels, probs)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["cls_loss"] = float(np.mean(cls_losses)) if cls_losses else 0.0
    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    pos_probs = probs_np[labels_np == 1]
    neg_probs = probs_np[labels_np == 0]
    metrics["positive_prob_mean"] = float(pos_probs.mean()) if pos_probs.size else 0.0
    metrics["negative_prob_mean"] = float(neg_probs.mean()) if neg_probs.size else 0.0
    metrics["pos_neg_gap"] = metrics["positive_prob_mean"] - metrics["negative_prob_mean"]
    metrics["pred_prob_mean"] = float(probs_np.mean()) if probs_np.size else 0.0
    metrics["pred_prob_std"] = float(probs_np.std(ddof=1)) if probs_np.size > 1 else 0.0
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
    if dssa_enabled:
        metrics["classification_loss"] = metrics["cls_loss"]
        for term_name, values in dssa_loss_values.items():
            metrics[term_name] = float(np.mean(values)) if values else 0.0
        for short_name in DSSA_LOSS_WEIGHTS.values():
            metrics[f"effective_lambda_{short_name}"] = effective_dssa_weight(loss_cfg, short_name)

        logits_np = np.asarray(logits, dtype=float)
        positive_logits = logits_np[labels_np == 1]
        negative_logits = logits_np[labels_np == 0]
        if positive_logits.size and negative_logits.size:
            pair_margins = positive_logits[:, None] - negative_logits[None, :]
            metrics["pairwise_pair_count"] = int(pair_margins.size)
            metrics["pairwise_inversion_count"] = int((pair_margins <= 0.0).sum())
        else:
            metrics["pairwise_pair_count"] = 0
            metrics["pairwise_inversion_count"] = 0

        for key in ("prototype_cosine", "prototype_distance"):
            values = dssa_diagnostics.get(key, [])
            metrics[key] = float(np.mean(values)) if values else 0.0
        disease_margins = np.asarray(dssa_diagnostics.get("disease_margin", []), dtype=float)
        if disease_margins.size == labels_np.size:
            metrics["mean_disease_margin_positive"] = float(disease_margins[labels_np == 1].mean()) if bool((labels_np == 1).any()) else 0.0
            metrics["mean_disease_margin_negative"] = float(disease_margins[labels_np == 0].mean()) if bool((labels_np == 0).any()) else 0.0
            metrics["prototype_assignment_accuracy"] = float(((disease_margins >= 0).astype(int) == labels_np).mean())
        else:
            metrics["mean_disease_margin_positive"] = 0.0
            metrics["mean_disease_margin_negative"] = 0.0
            metrics["prototype_assignment_accuracy"] = 0.0

        for modality in MODALITIES:
            available = np.asarray(dssa_diagnostics.get(f"modality_available_{modality}", []), dtype=float) > 0.5
            for source, output_name in (
                (f"shared_attention_{modality}", f"mean_shared_attention_{modality}"),
                (f"specific_gate_{modality}", f"mean_specific_gate_{modality}"),
                (f"shared_specific_cosine_{modality}", f"mean_shared_specific_cosine_{modality}"),
            ):
                values = np.asarray(dssa_diagnostics.get(source, []), dtype=float)
                metrics[output_name] = float(values[available].mean()) if values.size == available.size and bool(available.any()) else 0.0
        for left, right in (("img", "txt"), ("img", "bio"), ("txt", "bio")):
            values = np.asarray(dssa_diagnostics.get(f"shared_cosine_{left}_{right}", []), dtype=float)
            available = np.asarray(
                dssa_diagnostics.get(f"shared_pair_available_{left}_{right}", []), dtype=float
            ) > 0.5
            metrics[f"mean_shared_cosine_{left}_{right}"] = (
                float(values[available].mean())
                if values.size == available.size and bool(available.any())
                else 0.0
            )
        for source, output_name in (
            ("patient_shared_norm", "mean_shared_norm"),
            ("specific_residual_norm", "mean_specific_residual_norm"),
            ("specific_residual_shared_ratio", "specific_residual_shared_ratio"),
            ("soft_disease_anchor_norm", "mean_soft_disease_anchor_norm"),
        ):
            values = dssa_diagnostics.get(source, [])
            metrics[output_name] = float(np.mean(values)) if values else 0.0
        add_representation_health_metrics(
            metrics,
            dssa_shared_batches,
            dssa_specific_batches,
            dssa_availability_batches,
        )
        attention_means = [metrics[f"mean_shared_attention_{modality}"] for modality in MODALITIES]
        gate_means = [metrics[f"mean_specific_gate_{modality}"] for modality in MODALITIES]
        metrics["prototype_collapse_flag"] = int(metrics["prototype_cosine"] >= 0.95)
        metrics["global_attention_collapse_flag"] = int(max(attention_means) >= 0.95)
        metrics["global_gate_saturation_flag"] = int(any(value <= 0.01 or value >= 0.99 for value in gate_means))
        metrics["specific_dominates_shared_flag"] = int(metrics["specific_residual_shared_ratio"] >= 1.0)
    return {"metrics": metrics, "predictions": predictions}


def loss_config_for_epoch(loss_cfg: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    active_cfg = dict(loss_cfg)
    start_epoch = int(float(loss_cfg.get("text_morphology_start_epoch", 0)))
    if epoch < start_epoch:
        active_cfg["text_morphology_weight"] = 0.0
        active_cfg["text_morphology_active"] = False
    else:
        active_cfg["text_morphology_active"] = float(loss_cfg.get("text_morphology_weight", 0.0)) > 0
    warmup_epochs = int(loss_cfg.get("dssa_warmup_epochs", 3))
    ramp_epochs = max(int(loss_cfg.get("dssa_ramp_epochs", 5)), 1)
    if epoch <= warmup_epochs:
        dssa_scale = 0.0
    else:
        dssa_scale = min(max((epoch - warmup_epochs) / ramp_epochs, 0.0), 1.0)
    active_cfg["dssa_weight_scale"] = dssa_scale
    for short_name in DSSA_LOSS_WEIGHTS.values():
        active_cfg[f"effective_lambda_{short_name}"] = float(loss_cfg.get(f"lambda_{short_name}", 0.0)) * dssa_scale
    return active_cfg


def epoch_log_row(seed: int, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "seed": int(seed),
        "epoch": int(epoch),
        "split": "train_val",
        "train_total_loss": train_metrics.get("loss", 0.0),
        "train_cls_loss": train_metrics.get("cls_loss", 0.0),
        "train_text_morphology_loss": train_metrics.get("text_morphology_loss", 0.0),
        "val_total_loss": val_metrics.get("loss", 0.0),
        "val_cls_loss": val_metrics.get("cls_loss", 0.0),
        "val_text_morphology_loss": val_metrics.get("text_morphology_loss", 0.0),
        "val_auc": val_metrics.get("AUC", 0.0),
        "val_auprc": val_metrics.get("AUPRC", 0.0),
        "val_acc": val_metrics.get("ACC", 0.0),
        "val_f1": val_metrics.get("F1", 0.0),
        "val_sensitivity": val_metrics.get("Sensitivity", 0.0),
        "val_specificity": val_metrics.get("Specificity", 0.0),
        "val_balanced_accuracy": val_metrics.get("Balanced_ACC", 0.0),
        "val_positive_prob_mean": val_metrics.get("positive_prob_mean", 0.0),
        "val_negative_prob_mean": val_metrics.get("negative_prob_mean", 0.0),
        "val_pos_neg_gap": val_metrics.get("pos_neg_gap", 0.0),
        "val_pred_prob_mean": val_metrics.get("pred_prob_mean", 0.0),
        "val_pred_prob_std": val_metrics.get("pred_prob_std", 0.0),
        "selected_by_val_auc": False,
    }
    if "proto_classification_loss" in train_metrics:
        row.update(
            {
                "train_classification_loss": train_metrics.get("classification_loss", 0.0),
                "train_proto_classification_loss": train_metrics.get("proto_classification_loss", 0.0),
                "train_shared_consistency_loss": train_metrics.get("shared_consistency_loss", 0.0),
                "train_shared_specific_orth_loss": train_metrics.get("shared_specific_orth_loss", 0.0),
                "train_specific_variance_loss": train_metrics.get("specific_variance_loss", 0.0),
                "train_prototype_separation_loss": train_metrics.get("prototype_separation_loss", 0.0),
                "train_pairwise_ranking_loss": train_metrics.get("pairwise_ranking_loss", 0.0),
                "effective_lambda_proto": train_metrics.get("effective_lambda_proto", 0.0),
                "effective_lambda_shared": train_metrics.get("effective_lambda_shared", 0.0),
                "effective_lambda_orth": train_metrics.get("effective_lambda_orth", 0.0),
                "effective_lambda_var": train_metrics.get("effective_lambda_var", 0.0),
                "effective_lambda_sep": train_metrics.get("effective_lambda_sep", 0.0),
                "effective_lambda_rank": train_metrics.get("effective_lambda_rank", 0.0),
                "prototype_cosine": val_metrics.get("prototype_cosine", 0.0),
                "prototype_distance": val_metrics.get("prototype_distance", 0.0),
                "mean_disease_margin_positive": val_metrics.get("mean_disease_margin_positive", 0.0),
                "mean_disease_margin_negative": val_metrics.get("mean_disease_margin_negative", 0.0),
                "mean_shared_attention_img": val_metrics.get("mean_shared_attention_img", 0.0),
                "mean_shared_attention_txt": val_metrics.get("mean_shared_attention_txt", 0.0),
                "mean_shared_attention_bio": val_metrics.get("mean_shared_attention_bio", 0.0),
                "mean_specific_gate_img": val_metrics.get("mean_specific_gate_img", 0.0),
                "mean_specific_gate_txt": val_metrics.get("mean_specific_gate_txt", 0.0),
                "mean_specific_gate_bio": val_metrics.get("mean_specific_gate_bio", 0.0),
                "mean_shared_norm": val_metrics.get("mean_shared_norm", 0.0),
                "mean_specific_residual_norm": val_metrics.get("mean_specific_residual_norm", 0.0),
                "specific_residual_shared_ratio": val_metrics.get("specific_residual_shared_ratio", 0.0),
                "val_pairwise_inversion_count": val_metrics.get("pairwise_inversion_count", 0),
            }
        )
    return row


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
    epoch_history: List[Dict[str, Any]] = []
    base_loss_cfg = config.get("loss", {})

    for epoch in range(1, epochs + 1):
        epoch_loss_cfg = loss_config_for_epoch(base_loss_cfg, epoch)
        train_result = run_epoch(model, loaders["train"], optimizer, device, epoch_loss_cfg)
        val_result = run_epoch(model, loaders["val"], None, device, epoch_loss_cfg)
        epoch_history.append(epoch_log_row(seed, epoch, train_result["metrics"], val_result["metrics"]))
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

    for row in epoch_history:
        row["selected_by_val_auc"] = int(row["epoch"]) == int(best_epoch)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_loss_cfg = dict(config.get("loss", {}))
    for short_name in DSSA_LOSS_WEIGHTS.values():
        final_loss_cfg[f"effective_lambda_{short_name}"] = float(final_loss_cfg.get(f"lambda_{short_name}", 0.0))
    val_result = run_epoch(model, loaders["val"], None, device, final_loss_cfg)
    test_result = run_epoch(model, loaders["test"], None, device, final_loss_cfg)
    train_result = run_epoch(model, loaders["train"], None, device, final_loss_cfg) if model.dssa is not None else None
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch}, checkpoints / f"seed_{seed}_best.pt")
    result = {"seed": seed, "best_epoch": best_epoch, "epoch_history": epoch_history, "val": val_result, "test": test_result}
    if train_result is not None:
        result["train"] = train_result
    return result


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
    epoch_rows: List[Dict[str, Any]] = []
    for seed in config["training"].get("seeds", [0, 42, 3407]):
        result = train_seed(config, rows, int(seed), out_dir)
        epoch_rows.extend(result.get("epoch_history", []))
        output_splits = ("train", "val", "test") if "train" in result else ("val", "test")
        for split in output_splits:
            pred_df = pd.DataFrame(result[split]["predictions"])
            pred_df.insert(3, "split", split)
            pred_df.insert(4, "seed", int(seed))
            pred_df.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
            metric_row = {"seed": int(seed), "split": split, "best_epoch": result["best_epoch"]}
            metric_row.update(result[split]["metrics"])
            metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    if epoch_rows:
        pd.DataFrame(epoch_rows).to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
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
                    "pos_neg_gap",
                    "pairwise_inversion_count",
                    "prototype_cosine",
                    "prototype_distance",
                    "prototype_assignment_accuracy",
                    "mean_disease_margin_positive",
                    "mean_disease_margin_negative",
                    "mean_shared_attention_img",
                    "mean_shared_attention_txt",
                    "mean_shared_attention_bio",
                    "mean_specific_gate_img",
                    "mean_specific_gate_txt",
                    "mean_specific_gate_bio",
                    "specific_residual_shared_ratio",
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
