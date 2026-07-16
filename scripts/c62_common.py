#!/usr/bin/env python3
"""Shared C62 end-to-end model, optimizer, audit, and prediction utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
GROUPS = (
    "modality_encoders",
    "evidence_source_modules",
    "c61_task_specific_path",
)
GROUP_PREFIXES = {
    "modality_encoders": (
        "sources.image_encoder.",
        "sources.text_encoder.",
        "sources.bio_encoder.",
    ),
    "evidence_source_modules": (
        "sources.image_projector.",
        "sources.text_projector.",
        "sources.bio_projector.",
    ),
    "c61_task_specific_path": (
        "multimodal_encoder.",
        "continuous_bio_encoder.",
        "joint_instance_encoder.",
        "patient_readout.",
        "classifier.",
    ),
}

from dmea_ht.c61_cbpi import C61CBPIModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import train_phase_c40 as core  # noqa: E402


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


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def build_model(
    config: Dict[str, Any], seed: int, device: torch.device, checkpoint: Path | None = None
) -> C61CBPIModel:
    model = C61CBPIModel(config, seed).to(device)
    if checkpoint is not None:
        payload = checkpoint_payload(checkpoint)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C61 initialization checkpoint seed mismatch: {checkpoint}")
        model.load_state_dict(payload["model"], strict=True)
    return model


def group_for_parameter(name: str) -> str:
    matches = [group for group, prefixes in GROUP_PREFIXES.items() if name.startswith(prefixes)]
    if len(matches) != 1:
        raise RuntimeError(f"C62 parameter is not assigned to exactly one group: {name} -> {matches}")
    return matches[0]


def parameter_inventory(model: torch.nn.Module) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        group = group_for_parameter(name)
        if name in seen:
            raise RuntimeError(f"Duplicate C62 parameter name: {name}")
        seen.add(name)
        rows.append(
            {
                "parameter_name": name,
                "group": group,
                "parameter_count": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "shape": json.dumps(list(parameter.shape)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != sum(1 for _ in model.named_parameters()):
        raise RuntimeError("C62 parameter inventory is incomplete")
    return frame.sort_values(["group", "parameter_name"]).reset_index(drop=True)


def optimizer_parameter_groups(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> tuple[torch.optim.Optimizer, pd.DataFrame]:
    inventory = parameter_inventory(model)
    if not inventory["requires_grad"].astype(bool).all():
        frozen = inventory.loc[~inventory["requires_grad"].astype(bool), "parameter_name"].tolist()
        raise RuntimeError(f"C62 predictive parameters are frozen: {frozen[:8]}")
    factors = config["learning_rate_groups"]
    base_lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    groups: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        names = inventory.loc[inventory["group"] == group, "parameter_name"].tolist()
        if not names:
            raise RuntimeError(f"C62 optimizer group is empty: {group}")
        factor = float(factors[group])
        lr = base_lr * factor
        if lr <= 0.0:
            raise RuntimeError(f"C62 learning rate must be positive: {group}={lr}")
        parameters = [dict(model.named_parameters())[name] for name in names]
        groups.append({"params": parameters, "lr": lr, "weight_decay": weight_decay})
        audit_rows.append(
            {
                "group": group,
                "learning_rate_factor": factor,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "parameter_tensor_count": len(names),
                "parameter_count": int(inventory.loc[inventory["group"] == group, "parameter_count"].sum()),
                "all_requires_grad": True,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    return optimizer, pd.DataFrame(audit_rows)


def group_gradient_norms(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, List[float]] = {group: [] for group in GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = group_for_parameter(name)
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"C62 non-finite gradient: {name}")
        values[group].append(float(gradient.pow(2).sum().cpu()))
    result: Dict[str, Dict[str, Any]] = {}
    for group in GROUPS:
        squared = float(sum(values[group]))
        result[group] = {
            "norm": float(np.sqrt(squared)),
            "finite": True,
            "nonzero_tensor_count": int(sum(value > 0.0 for value in values[group])),
        }
    return result


def prediction_row(
    arrays: Mapping[str, np.ndarray],
    batch: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    history_weights = arrays["history_weights"][index]
    history_weights = np.asarray(history_weights, dtype=float).clip(1e-8, 1.0)
    attention = np.asarray(arrays["attention"][index], dtype=float).clip(1e-8, 1.0)
    row: Dict[str, Any] = {
        "patient_id": str(batch["patient_id"][index]),
        "label": int(batch["label"].detach().cpu().numpy()[index]),
        "visit_count_audit_only": int(batch["visit_mask"].detach().cpu().numpy()[index].sum()),
        "evidence_valid_count": int(arrays["evidence_valid"][index].sum()),
        "bio_valid_count": int(
            bool(arrays["latest_bio_valid"][index]) or bool(arrays["history_bio_valid"][index])
        ),
        "patient_state_norm": float(np.linalg.norm(arrays["patient_state"][index])),
        "patient_state_component_std": float(np.std(arrays["patient_state"][index])),
        "attended_evidence_norm": float(np.linalg.norm(arrays["attended_evidence"][index])),
        "bio_state_norm": float(np.linalg.norm(arrays["bio_state"][index])),
        "final_logit": float(arrays["logit"][index]),
        "final_prob": float(arrays["prob"][index]),
        "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
        "history_kernel_entropy": float(-(history_weights * np.log(history_weights)).sum()),
        "evidence_attention_entropy": float(-(attention * np.log(attention)).sum()),
    }
    for field in core.AUDIT_SHORTCUT_FIELDS:
        row[field] = batch["shortcuts"][index].get(field, np.nan)
    return row


def run_epoch(
    model: C61CBPIModel,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_values: List[float] = []
    predictions: List[Dict[str, Any]] = []
    patient_states: List[np.ndarray] = []
    evidence_states: List[np.ndarray] = []
    bio_states: List[np.ndarray] = []
    group_gradients: Dict[str, List[float]] = {group: [] for group in GROUPS}
    group_nonzero: Dict[str, int] = {group: 0 for group in GROUPS}
    for raw_batch in loader:
        batch = core.move_batch(raw_batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C62 non-finite BCE loss")
        if is_train:
            loss.backward()
            gradient_info = group_gradient_norms(model)
            for group in GROUPS:
                group_gradients[group].append(float(gradient_info[group]["norm"]))
                group_nonzero[group] += int(gradient_info[group]["nonzero_tensor_count"] > 0)
            optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if torch.is_tensor(value)
        }
        patient_states.append(arrays["patient_state"])
        evidence_states.append(arrays["attended_evidence"])
        bio_states.append(arrays["bio_state"])
        for index in range(len(batch["patient_id"])):
            predictions.append(prediction_row(arrays, batch, index))
    frame = pd.DataFrame(predictions)
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["final_prob"].to_numpy(dtype=float)
    metrics = core.binary_metrics(labels, probabilities)
    metrics.update(
        {
            "bce_loss": float(np.mean(loss_values)),
            "positive_probability_mean": float(probabilities[labels == 1].mean()),
            "negative_probability_mean": float(probabilities[labels == 0].mean()),
            "positive_negative_gap": float(probabilities[labels == 1].mean() - probabilities[labels == 0].mean()),
            "prediction_std": float(probabilities.std(ddof=1)),
            "mean_patient_state_norm": float(frame["patient_state_norm"].mean()),
            "std_patient_state_norm": float(frame["patient_state_norm"].std(ddof=1)),
            "patient_state_component_std": float(np.concatenate(patient_states, axis=0).std()),
            "mean_history_kernel_entropy": float(frame["history_kernel_entropy"].mean()),
            "mean_evidence_attention_entropy": float(frame["evidence_attention_entropy"].mean()),
            "pairwise_inversion_count": core.pairwise_inversions(labels, probabilities),
            "n_rows": int(len(frame)),
        }
    )
    gradient_summary = {
        group: {
            "mean_norm": float(np.mean(values)) if values else 0.0,
            "max_norm": float(np.max(values)) if values else 0.0,
            "active_batch_count": int(group_nonzero[group]),
        }
        for group, values in group_gradients.items()
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "patient_states": np.concatenate(patient_states, axis=0),
        "evidence_states": np.concatenate(evidence_states, axis=0),
        "bio_states": np.concatenate(bio_states, axis=0),
        "patient_diagnostics": frame,
        "gradient_summary": gradient_summary,
    }


def save_split(result: Mapping[str, Any], out_dir: Path, split: str) -> Dict[str, Any]:
    seed = int(result["seed"])
    split_result = result[split]
    frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id").reset_index(drop=True)
    prediction_dir = out_dir / "predictions"
    representation_dir = out_dir / "representations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    representation_dir.mkdir(parents=True, exist_ok=True)
    frame.insert(0, "split", split)
    frame.insert(0, "seed", seed)
    frame.to_csv(prediction_dir / f"{split}_predictions_seed_{seed}.csv", index=False)
    order = np.asarray([str(row["patient_id"]) for row in split_result["predictions"]])
    order = np.argsort(order)
    np.savez_compressed(
        representation_dir / f"{split}_patient_state_seed_{seed}.npz",
        patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
        label=frame["label"].to_numpy(dtype=np.int64),
        patient_state=split_result["patient_states"][order].astype(np.float32),
        evidence_states=split_result["evidence_states"][order].astype(np.float32),
        bio_states=split_result["bio_states"][order].astype(np.float32),
    )
    return {"seed": seed, "split": split, "best_epoch": int(result["best_epoch"]), **split_result["metrics"]}


def parameter_update_audit(
    model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor], seed: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu().float()
        initial = initial_state[name].float()
        delta = current - initial
        initial_norm = float(torch.linalg.vector_norm(initial))
        delta_norm = float(torch.linalg.vector_norm(delta))
        rows.append(
            {
                "seed": seed,
                "kind": "parameter",
                "group": group_for_parameter(name),
                "parameter_name": name,
                "parameter_count": int(current.numel()),
                "initial_l2": initial_norm,
                "delta_l2": delta_norm,
                "relative_parameter_change": delta_norm / max(initial_norm, 1e-8),
                "updated": bool(delta_norm > 0.0),
                "finite": bool(torch.isfinite(current).all() and torch.isfinite(delta).all()),
            }
        )
    frame = pd.DataFrame(rows)
    summaries: List[Dict[str, Any]] = []
    for group, group_frame in frame.groupby("group", sort=True):
        summaries.append(
            {
                "seed": seed,
                "kind": "group_summary",
                "group": group,
                "parameter_name": "",
                "parameter_count": int(group_frame["parameter_count"].sum()),
                "initial_l2": float(np.sqrt((group_frame["initial_l2"] ** 2).sum())),
                "delta_l2": float(np.sqrt((group_frame["delta_l2"] ** 2).sum())),
                "relative_parameter_change": float(group_frame["delta_l2"].sum() / max(group_frame["initial_l2"].sum(), 1e-8)),
                "updated": bool(group_frame["updated"].any()),
                "updated_parameter_tensor_count": int(group_frame["updated"].sum()),
                "finite": bool(group_frame["finite"].all()),
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def load_c62_config(path: str | Path) -> Dict[str, Any]:
    config = load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c62" or not bool(config.get("end_to_end")):
        raise RuntimeError("C62 configuration must enable end_to_end")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C62 formal seeds must remain [0, 42, 3407]")
    return config


def manifest_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return read_jsonl(config["project"]["manifest"])
