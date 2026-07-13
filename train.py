from __future__ import annotations

import argparse
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
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest
from dmea_ht.evidence_losses import confidence_weighted_bce_with_logits
from dmea_ht.metrics import compute_binary_metrics, summarize_metrics
from dmea_ht.mea_losses import mea_loss_weights_for_epoch, pairwise_ranking_loss, state_margin_loss
from dmea_ht.c17_residual import C17ResidualModel
from dmea_ht.models import C18DirectionalResidualModel, DMEAHTModel


MEA_PATIENT_DIAGNOSTIC_KEYS = (
    "patient_support_strength",
    "patient_opposition_strength",
    "patient_uncertainty_strength",
    "patient_conflict_score",
    "image_support_score",
    "image_opposition_score",
    "image_uncertainty_score",
    "text_support_score",
    "text_opposition_score",
    "text_uncertainty_score",
    "text_temporal_conflict_score",
    "text_temporal_available",
    "text_latest_support_score",
    "text_latest_opposition_score",
    "text_latest_available",
    "text_history_support_score",
    "text_history_opposition_score",
    "text_history_available",
    "bio_support_score",
    "bio_opposition_score",
    "bio_uncertainty_score",
    "bio_evidence_reliability",
    "bio_valid_fraction",
    "image_evidence_weight",
    "text_evidence_weight",
    "bio_evidence_weight",
    "morphology_alignment_cosine",
    "morphology_alignment_available",
    "support_opposition_cosine",
    "mechanism_state_norm",
    "mechanism_attention_max",
    "evidence_role_entropy",
    "evidence_role_prob_sum_error",
    "evidence_reliability_mean",
    "image_evidence_attention_entropy",
    "image_evidence_slot_norm_mean",
    "text_support_attention_mass",
    "text_opposition_attention_mass",
    "text_uncertainty_attention_mass",
    "text_role_norm_mean",
    "bio_evidence_norm_mean",
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


def is_c17_config(config: Dict[str, Any]) -> bool:
    return bool(config.get("c17")) or str(config.get("phase", "")).lower() == "c17"


def is_c18_config(config: Dict[str, Any]) -> bool:
    return bool(config.get("c18")) or str(config.get("phase", "")).lower() == "c18"


def hard_pairwise_ranking_loss(
    base_logit: torch.Tensor,
    final_logit: torch.Tensor,
    labels: torch.Tensor,
    margin_threshold: float,
) -> tuple[torch.Tensor, int]:
    """Rank only training-batch positive-negative pairs with a hard base margin."""
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any().item()) or not bool(negative.any().item()):
        return final_logit.sum() * 0.0, 0
    base_positive = base_logit[positive].unsqueeze(1)
    base_negative = base_logit[negative].unsqueeze(0)
    hard = (base_positive - base_negative) < float(margin_threshold)
    hard_count = int(hard.sum().detach().cpu())
    if hard_count == 0:
        return final_logit.sum() * 0.0, 0
    final_positive = final_logit[positive].unsqueeze(1)
    final_negative = final_logit[negative].unsqueeze(0)
    loss = F.softplus(-(final_positive - final_negative)[hard]).mean()
    return loss, hard_count


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
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    is_c17 = bool(loss_cfg.get("c17", False))
    is_c18 = bool(loss_cfg.get("c18", False))
    is_directional = is_c17 or is_c18
    model.train(is_train)
    labels: List[int] = []
    probs: List[float] = []
    losses: List[float] = []
    cls_losses: List[float] = []
    mea_loss_values: Dict[str, List[float]] = {
        "state_margin_loss": [],
        "mechanism_alignment_loss": [],
        "role_separation_loss": [],
        "pairwise_ranking_loss": [],
    }
    mea_diagnostic_values: Dict[str, List[float]] = {key: [] for key in MEA_PATIENT_DIAGNOSTIC_KEYS}
    text_morphology_labels: List[int] = []
    text_morphology_probs: List[float] = []
    text_morphology_confidences: List[float] = []
    text_morphology_losses: List[float] = []
    image_morphology_labels: List[int] = []
    image_morphology_probs: List[float] = []
    image_morphology_confidences: List[float] = []
    image_morphology_losses: List[float] = []
    residual_regularization_losses: List[float] = []
    positive_preservation_losses: List[float] = []
    hard_ranking_losses: List[float] = []
    hard_pair_counts: List[int] = []
    delta_values: List[float] = []
    positive_delta_values: List[float] = []
    negative_delta_values: List[float] = []
    support_delta_values: List[float] = []
    opposition_delta_values: List[float] = []
    support_gate_values: List[float] = []
    opposition_gate_values: List[float] = []
    conflict_suppression_values: List[float] = []
    predictions: List[Dict[str, Any]] = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    cls_weight = float(loss_cfg.get("cls_weight", 1.0))
    text_weight = float(loss_cfg.get("text_morphology_weight", 0.0))
    image_weight = float(loss_cfg.get("image_morphology_weight", 0.0))
    lambda_state = float(loss_cfg.get("effective_lambda_state", 0.0))
    lambda_mech = float(loss_cfg.get("effective_lambda_mech", 0.0))
    lambda_role = float(loss_cfg.get("effective_lambda_role", 0.0))
    lambda_rank = float(loss_cfg.get("effective_lambda_rank", 0.0))
    lambda_residual = float(
        loss_cfg.get(
            "lambda_directional_residual" if is_c18 else "lambda_residual",
            0.001 if is_c18 else 0.0,
        )
    )
    lambda_positive_preserve = float(loss_cfg.get("lambda_positive_preserve", 0.0))
    allowed_negative_delta = float(loss_cfg.get("allowed_negative_delta", 0.05))
    lambda_hard_rank = float(loss_cfg.get("lambda_hard_rank", 0.0))
    hard_margin_threshold = float(loss_cfg.get("hard_margin_threshold", 0.50))

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            raw_loss = criterion(outputs["logit"], batch["label"])
            cls_loss = cls_weight * (raw_loss * batch["sample_weight"]).mean()
            loss = cls_loss
            if is_c18:
                directional_delta = outputs["directional_delta"]
                residual_loss = (
                    outputs["effective_support_delta"].square()
                    + outputs["effective_opposition_delta"].square()
                ).mean()
                positive_mask = batch["label"] > 0.5
                if bool(positive_mask.any().item()):
                    positive_loss = F.relu(-directional_delta[positive_mask] - allowed_negative_delta).mean()
                else:
                    positive_loss = directional_delta.sum() * 0.0
                if bool(loss_cfg.get("hard_rank", False)) and is_train:
                    hard_rank_loss, hard_pair_count = hard_pairwise_ranking_loss(
                        outputs["base_logit"],
                        outputs["logit"],
                        batch["label"],
                        hard_margin_threshold,
                    )
                else:
                    hard_rank_loss = outputs["logit"].sum() * 0.0
                    hard_pair_count = 0
                loss = (
                    loss
                    + lambda_residual * residual_loss
                    + lambda_positive_preserve * positive_loss
                    + lambda_hard_rank * hard_rank_loss
                )
                residual_regularization_losses.append(float(residual_loss.detach().cpu()))
                positive_preservation_losses.append(float(positive_loss.detach().cpu()))
                hard_ranking_losses.append(float(hard_rank_loss.detach().cpu()))
                hard_pair_counts.append(hard_pair_count)
            elif is_c17:
                delta = outputs["delta_logit"]
                residual_loss = delta.square().mean()
                positive_mask = batch["label"] > 0.5
                if bool(positive_mask.any().item()):
                    positive_loss = F.relu(-delta[positive_mask] - allowed_negative_delta).mean()
                else:
                    positive_loss = delta.sum() * 0.0
                loss = loss + lambda_residual * residual_loss + lambda_positive_preserve * positive_loss
                residual_regularization_losses.append(float(residual_loss.detach().cpu()))
                positive_preservation_losses.append(float(positive_loss.detach().cpu()))
            else:
                evidence_terms = evidence_loss_terms(outputs, batch, loss_cfg)
                if "text_morphology_loss" in evidence_terms:
                    loss = loss + text_weight * evidence_terms["text_morphology_loss"]
                    text_morphology_losses.append(float(evidence_terms["text_morphology_loss"].detach().cpu()))
                if "image_morphology_loss" in evidence_terms:
                    loss = loss + image_weight * evidence_terms["image_morphology_loss"]
                    image_morphology_losses.append(float(evidence_terms["image_morphology_loss"].detach().cpu()))
                if "role_alignment_loss" in outputs:
                    loss = loss + outputs["role_alignment_loss"] * 0.0
                if "state_margin" in outputs:
                    state_loss = state_margin_loss(outputs["state_margin"], batch["label"])
                    mechanism_loss = outputs["mea_mechanism_alignment_loss"]
                    role_loss = outputs["mea_role_separation_loss"]
                    rank_loss = pairwise_ranking_loss(outputs["logit"], batch["label"]) if is_train else outputs["logit"].sum() * 0.0
                    loss = (
                        loss
                        + lambda_state * state_loss
                        + lambda_mech * mechanism_loss
                        + lambda_role * role_loss
                        + lambda_rank * rank_loss
                    )
                    for key, value in (
                        ("state_margin_loss", state_loss),
                        ("mechanism_alignment_loss", mechanism_loss),
                        ("role_separation_loss", role_loss),
                        ("pairwise_ranking_loss", rank_loss),
                    ):
                        mea_loss_values[key].append(float(value.detach().cpu()))
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        cls_losses.append(float(cls_loss.detach().cpu()))
        batch_probs = outputs["prob"].detach().cpu().numpy().tolist()
        batch_labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        labels.extend(batch_labels)
        probs.extend(batch_probs)

        if is_directional:
            delta_key = "directional_delta" if is_c18 else "delta_logit"
            batch_delta = outputs[delta_key].detach().float().cpu().numpy()
            delta_values.extend(float(value) for value in batch_delta)
            batch_labels_np = np.asarray(batch_labels, dtype=int)
            positive_delta_values.extend(float(value) for value in batch_delta[batch_labels_np == 1])
            negative_delta_values.extend(float(value) for value in batch_delta[batch_labels_np == 0])
            if is_c18:
                support_delta_values.extend(float(value) for value in outputs["support_delta"].detach().float().cpu().numpy())
                opposition_delta_values.extend(float(value) for value in outputs["opposition_delta"].detach().float().cpu().numpy())
                support_gate_values.extend(float(value) for value in outputs["support_gate"].detach().float().cpu().numpy())
                opposition_gate_values.extend(float(value) for value in outputs["opposition_gate"].detach().float().cpu().numpy())
                conflict_suppression_values.extend(float(value) for value in outputs["conflict_suppression"].detach().float().cpu().numpy())

        for key in MEA_PATIENT_DIAGNOSTIC_KEYS:
            if key in outputs:
                values = outputs[key].detach().float().cpu().reshape(-1).numpy().tolist()
                mea_diagnostic_values[key].extend(float(value) for value in values)

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
            if is_c17:
                row.update(
                    {
                        "base_logit": float(outputs["base_logit"].detach().cpu()[i]),
                        "base_prob": float(outputs["base_prob"].detach().cpu()[i]),
                        "delta_logit": float(outputs["delta_logit"].detach().cpu()[i]),
                        "final_logit": float(outputs["logit"].detach().cpu()[i]),
                        "final_prob": float(outputs["prob"].detach().cpu()[i]),
                    }
                )
            elif is_c18:
                row.update(
                    {
                        "base_logit": float(outputs["base_logit"].detach().cpu()[i]),
                        "base_prob": float(outputs["base_prob"].detach().cpu()[i]),
                        "support_delta": float(outputs["support_delta"].detach().cpu()[i]),
                        "opposition_delta": float(outputs["opposition_delta"].detach().cpu()[i]),
                        "support_gate": float(outputs["support_gate"].detach().cpu()[i]),
                        "opposition_gate": float(outputs["opposition_gate"].detach().cpu()[i]),
                        "conflict_suppression": float(outputs["conflict_suppression"].detach().cpu()[i]),
                        "effective_support_delta": float(outputs["effective_support_delta"].detach().cpu()[i]),
                        "effective_opposition_delta": float(outputs["effective_opposition_delta"].detach().cpu()[i]),
                        "directional_delta": float(outputs["directional_delta"].detach().cpu()[i]),
                        "final_logit": float(outputs["logit"].detach().cpu()[i]),
                        "final_prob": float(outputs["prob"].detach().cpu()[i]),
                    }
                )
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
            for key in MEA_PATIENT_DIAGNOSTIC_KEYS:
                if key in outputs:
                    value = outputs[key].detach().cpu()[i]
                    if value.numel() == 1:
                        row[key] = float(value)
            row.update(batch["shortcuts"][i])
            predictions.append(row)

    metrics = compute_binary_metrics(labels, probs)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["cls_loss"] = float(np.mean(cls_losses)) if cls_losses else 0.0
    for key, values in mea_loss_values.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
    for key, values in mea_diagnostic_values.items():
        metrics[f"mean_{key}"] = float(np.mean(values)) if values else 0.0
    metrics["effective_lambda_state"] = lambda_state
    metrics["effective_lambda_mech"] = lambda_mech
    metrics["effective_lambda_role"] = lambda_role
    metrics["effective_lambda_rank"] = lambda_rank
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
    if is_c17:
        delta_np = np.asarray(delta_values, dtype=float)
        positive_delta_np = np.asarray(positive_delta_values, dtype=float)
        negative_delta_np = np.asarray(negative_delta_values, dtype=float)
        delta_max = float(loss_cfg.get("delta_max", 0.50))
        metrics.update(
            {
                "residual_regularization_loss": float(np.mean(residual_regularization_losses)) if residual_regularization_losses else 0.0,
                "positive_preservation_loss": float(np.mean(positive_preservation_losses)) if positive_preservation_losses else 0.0,
                "mean_delta_logit": float(delta_np.mean()) if delta_np.size else 0.0,
                "mean_positive_delta_logit": float(positive_delta_np.mean()) if positive_delta_np.size else 0.0,
                "mean_negative_delta_logit": float(negative_delta_np.mean()) if negative_delta_np.size else 0.0,
                "std_delta_logit": float(delta_np.std(ddof=1)) if delta_np.size > 1 else 0.0,
                "fraction_delta_at_lower_bound": float((delta_np <= -delta_max + 1e-5).mean()) if delta_np.size else 0.0,
                "fraction_delta_at_upper_bound": float((delta_np >= delta_max - 1e-5).mean()) if delta_np.size else 0.0,
                "fraction_positive_delta_below_minus_0_10": float((positive_delta_np < -0.10).mean()) if positive_delta_np.size else 0.0,
                "_c17_route": True,
            }
        )
        metrics.pop("AUPRC", None)
    elif is_c18:
        delta_np = np.asarray(delta_values, dtype=float)
        positive_delta_np = np.asarray(positive_delta_values, dtype=float)
        negative_delta_np = np.asarray(negative_delta_values, dtype=float)
        support_np = np.asarray(support_delta_values, dtype=float)
        opposition_np = np.asarray(opposition_delta_values, dtype=float)
        support_gate_np = np.asarray(support_gate_values, dtype=float)
        opposition_gate_np = np.asarray(opposition_gate_values, dtype=float)
        conflict_suppression_np = np.asarray(conflict_suppression_values, dtype=float)
        metrics.update(
            {
                "directional_residual_loss": float(np.mean(residual_regularization_losses)) if residual_regularization_losses else 0.0,
                "positive_preservation_loss": float(np.mean(positive_preservation_losses)) if positive_preservation_losses else 0.0,
                "hard_rank_loss": float(np.mean(hard_ranking_losses)) if hard_ranking_losses else 0.0,
                "hard_pair_count": int(np.sum(hard_pair_counts)) if hard_pair_counts else 0,
                "mean_support_delta": float(support_np.mean()) if support_np.size else 0.0,
                "mean_opposition_delta": float(opposition_np.mean()) if opposition_np.size else 0.0,
                "mean_directional_delta": float(delta_np.mean()) if delta_np.size else 0.0,
                "std_directional_delta": float(delta_np.std(ddof=1)) if delta_np.size > 1 else 0.0,
                "mean_positive_directional_delta": float(positive_delta_np.mean()) if positive_delta_np.size else 0.0,
                "mean_negative_directional_delta": float(negative_delta_np.mean()) if negative_delta_np.size else 0.0,
                "mean_support_gate": float(support_gate_np.mean()) if support_gate_np.size else 0.0,
                "mean_opposition_gate": float(opposition_gate_np.mean()) if opposition_gate_np.size else 0.0,
                "mean_conflict_suppression": float(conflict_suppression_np.mean()) if conflict_suppression_np.size else 0.0,
                "fraction_support_at_upper_bound": float((support_np >= 0.50 - 1e-5).mean()) if support_np.size else 0.0,
                "fraction_opposition_at_upper_bound": float((opposition_np >= 0.50 - 1e-5).mean()) if opposition_np.size else 0.0,
                "fraction_positive_delta_below_minus_0_10": float((positive_delta_np < -0.10).mean()) if positive_delta_np.size else 0.0,
                "_c18_route": True,
            }
        )
        metrics.pop("AUPRC", None)
    return {"metrics": metrics, "predictions": predictions}


def loss_config_for_epoch(loss_cfg: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    active_cfg = dict(loss_cfg)
    start_epoch = int(float(loss_cfg.get("text_morphology_start_epoch", 0)))
    if epoch < start_epoch:
        active_cfg["text_morphology_weight"] = 0.0
        active_cfg["text_morphology_active"] = False
    else:
        active_cfg["text_morphology_active"] = float(loss_cfg.get("text_morphology_weight", 0.0)) > 0
    if not bool(loss_cfg.get("c17", False)) and not bool(loss_cfg.get("c18", False)):
        active_cfg.update(mea_loss_weights_for_epoch(loss_cfg, epoch))
    return active_cfg


def epoch_log_row(seed: int, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if bool(train_metrics.get("_c18_route", False) or val_metrics.get("_c18_route", False)):
        return {
            "seed": int(seed),
            "epoch": int(epoch),
            "split": "train_val",
            "train_total_loss": train_metrics.get("loss", 0.0),
            "train_classification_loss": train_metrics.get("cls_loss", 0.0),
            "train_directional_residual_loss": train_metrics.get("directional_residual_loss", 0.0),
            "train_positive_preservation_loss": train_metrics.get("positive_preservation_loss", 0.0),
            "train_hard_rank_loss": train_metrics.get("hard_rank_loss", 0.0),
            "train_hard_pair_count": train_metrics.get("hard_pair_count", 0),
            "val_auc": val_metrics.get("AUC", 0.0),
            "val_sensitivity": val_metrics.get("Sensitivity", 0.0),
            "val_specificity": val_metrics.get("Specificity", 0.0),
            "val_balanced_accuracy": val_metrics.get("Balanced_ACC", 0.0),
            "val_positive_probability_mean": val_metrics.get("positive_prob_mean", 0.0),
            "val_negative_probability_mean": val_metrics.get("negative_prob_mean", 0.0),
            "val_positive_negative_gap": val_metrics.get("pos_neg_gap", 0.0),
            "mean_support_delta": val_metrics.get("mean_support_delta", 0.0),
            "mean_opposition_delta": val_metrics.get("mean_opposition_delta", 0.0),
            "mean_directional_delta": val_metrics.get("mean_directional_delta", 0.0),
            "std_directional_delta": val_metrics.get("std_directional_delta", 0.0),
            "mean_positive_directional_delta": val_metrics.get("mean_positive_directional_delta", 0.0),
            "mean_negative_directional_delta": val_metrics.get("mean_negative_directional_delta", 0.0),
            "mean_support_gate": val_metrics.get("mean_support_gate", 0.0),
            "mean_opposition_gate": val_metrics.get("mean_opposition_gate", 0.0),
            "mean_conflict_suppression": val_metrics.get("mean_conflict_suppression", 0.0),
            "fraction_support_at_upper_bound": val_metrics.get("fraction_support_at_upper_bound", 0.0),
            "fraction_opposition_at_upper_bound": val_metrics.get("fraction_opposition_at_upper_bound", 0.0),
            "fraction_positive_delta_below_minus_0_10": val_metrics.get("fraction_positive_delta_below_minus_0_10", 0.0),
            "selected_by_val_auc": False,
        }
    if bool(train_metrics.get("_c17_route", False) or val_metrics.get("_c17_route", False)):
        return {
            "seed": int(seed),
            "epoch": int(epoch),
            "split": "train_val",
            "train_total_loss": train_metrics.get("loss", 0.0),
            "train_classification_loss": train_metrics.get("cls_loss", 0.0),
            "train_residual_regularization_loss": train_metrics.get("residual_regularization_loss", 0.0),
            "train_positive_preservation_loss": train_metrics.get("positive_preservation_loss", 0.0),
            "val_auc": val_metrics.get("AUC", 0.0),
            "val_sensitivity": val_metrics.get("Sensitivity", 0.0),
            "val_specificity": val_metrics.get("Specificity", 0.0),
            "val_balanced_accuracy": val_metrics.get("Balanced_ACC", 0.0),
            "val_positive_probability_mean": val_metrics.get("positive_prob_mean", 0.0),
            "val_negative_probability_mean": val_metrics.get("negative_prob_mean", 0.0),
            "val_positive_negative_gap": val_metrics.get("pos_neg_gap", 0.0),
            "mean_delta_logit": val_metrics.get("mean_delta_logit", 0.0),
            "mean_positive_delta_logit": val_metrics.get("mean_positive_delta_logit", 0.0),
            "mean_negative_delta_logit": val_metrics.get("mean_negative_delta_logit", 0.0),
            "std_delta_logit": val_metrics.get("std_delta_logit", 0.0),
            "fraction_delta_at_lower_bound": val_metrics.get("fraction_delta_at_lower_bound", 0.0),
            "fraction_delta_at_upper_bound": val_metrics.get("fraction_delta_at_upper_bound", 0.0),
            "fraction_positive_delta_below_minus_0_10": val_metrics.get("fraction_positive_delta_below_minus_0_10", 0.0),
            "support_strength": val_metrics.get("mean_patient_support_strength", 0.0),
            "opposition_strength": val_metrics.get("mean_patient_opposition_strength", 0.0),
            "uncertainty_strength": val_metrics.get("mean_patient_uncertainty_strength", 0.0),
            "conflict_score": val_metrics.get("mean_patient_conflict_score", 0.0),
            "selected_by_val_auc": False,
        }
    return {
        "seed": int(seed),
        "epoch": int(epoch),
        "split": "train_val",
        "train_total_loss": train_metrics.get("loss", 0.0),
        "train_cls_loss": train_metrics.get("cls_loss", 0.0),
        "train_classification_loss": train_metrics.get("cls_loss", 0.0),
        "train_state_margin_loss": train_metrics.get("state_margin_loss", 0.0),
        "train_mechanism_alignment_loss": train_metrics.get("mechanism_alignment_loss", 0.0),
        "train_role_separation_loss": train_metrics.get("role_separation_loss", 0.0),
        "train_pairwise_ranking_loss": train_metrics.get("pairwise_ranking_loss", 0.0),
        "effective_lambda_state": train_metrics.get("effective_lambda_state", 0.0),
        "effective_lambda_mech": train_metrics.get("effective_lambda_mech", 0.0),
        "effective_lambda_role": train_metrics.get("effective_lambda_role", 0.0),
        "effective_lambda_rank": train_metrics.get("effective_lambda_rank", 0.0),
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
        "mean_patient_support_strength": val_metrics.get("mean_patient_support_strength", 0.0),
        "mean_patient_opposition_strength": val_metrics.get("mean_patient_opposition_strength", 0.0),
        "mean_patient_uncertainty_strength": val_metrics.get("mean_patient_uncertainty_strength", 0.0),
        "mean_patient_conflict_score": val_metrics.get("mean_patient_conflict_score", 0.0),
        "mean_image_evidence_weight": val_metrics.get("mean_image_evidence_weight", 0.0),
        "mean_text_evidence_weight": val_metrics.get("mean_text_evidence_weight", 0.0),
        "mean_bio_evidence_weight": val_metrics.get("mean_bio_evidence_weight", 0.0),
        "mean_morphology_alignment_cosine": val_metrics.get("mean_morphology_alignment_cosine", 0.0),
        "mean_support_opposition_cosine": val_metrics.get("mean_support_opposition_cosine", 0.0),
        "mechanism_state_norm": val_metrics.get("mean_mechanism_state_norm", 0.0),
        "selected_by_val_auc": False,
    }


def build_model(config: Dict[str, Any], seed: int) -> torch.nn.Module:
    if is_c18_config(config):
        return C18DirectionalResidualModel(config, seed)
    if is_c17_config(config):
        return C17ResidualModel(config, seed)
    return DMEAHTModel(config)


def train_seed(config: Dict[str, Any], rows: List[Dict[str, Any]], seed: int, out_dir: Path) -> Dict[str, Any]:
    set_seed(seed)
    loaders = make_loaders(config, [dict(row) for row in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, seed).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for the selected route.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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

    selected_loss_cfg = loss_config_for_epoch(base_loss_cfg, best_epoch)
    val_result = run_epoch(model, loaders["val"], None, device, selected_loss_cfg)
    test_result = (
        run_epoch(model, loaders["test"], None, device, selected_loss_cfg)
        if bool(config["training"].get("evaluate_test", True))
        else None
    )
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch}, checkpoints / f"seed_{seed}_best.pt")
    result = {"seed": seed, "best_epoch": best_epoch, "epoch_history": epoch_history, "val": val_result}
    if test_result is not None:
        result["test"] = test_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    c17_route = is_c17_config(config)
    c18_route = is_c18_config(config)
    auc_only_route = c17_route or c18_route
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
        for split in ("val", "test"):
            if split not in result:
                continue
            pred_df = pd.DataFrame(result[split]["predictions"])
            pred_df.insert(3, "split", split)
            pred_df.insert(4, "seed", int(seed))
            pred_df.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
            metric_row = {"seed": int(seed), "split": split, "best_epoch": result["best_epoch"]}
            metric_row.update(
                {
                    key: value
                    for key, value in result[split]["metrics"].items()
                    if not str(key).startswith("_") and (not auc_only_route or key != "AUPRC")
                }
            )
            metrics_rows.append(metric_row)

    metrics_df = pd.DataFrame(metrics_rows)
    if auc_only_route and "AUPRC" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["AUPRC"])
    metrics_df.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    if epoch_rows:
        epoch_df = pd.DataFrame(epoch_rows)
        if auc_only_route and "AUPRC" in epoch_df.columns:
            epoch_df = epoch_df.drop(columns=["AUPRC"])
        epoch_df.to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    summary_rows = []
    for split in ("val", "test"):
        split_rows = [row for row in metrics_rows if row["split"] == split]
        if not split_rows:
            continue
        summary = {"split": split}
        summary_keys = [
            "AUC",
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
        ]
        if not auc_only_route:
            summary_keys.insert(1, "AUPRC")
        summary.update(summarize_metrics(split_rows, summary_keys))
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)
    cm_cols = ["seed", "split", "TN", "FP", "FN", "TP"]
    metrics_df[cm_cols].to_csv(out_dir / "reports" / "confusion_matrix_by_seed.csv", index=False)
    (out_dir / "reports" / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
