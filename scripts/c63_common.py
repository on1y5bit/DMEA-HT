#!/usr/bin/env python3
"""Shared C63 from-base initialization, training, and audit utilities."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
OPTIMIZER_GROUPS = (
    "image_text_encoders",
    "bio_encoder_and_evidence_projectors",
    "c61_task_specific_path",
)
OPTIMIZER_PREFIXES = {
    "image_text_encoders": (
        "sources.image_encoder.",
        "sources.text_encoder.",
    ),
    "bio_encoder_and_evidence_projectors": (
        "sources.bio_encoder.",
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
MODULE_GROUPS = (
    "image_encoder",
    "text_encoder",
    "bio_encoder",
    "image_projector",
    "text_projector",
    "bio_projector",
    "multimodal_instance_encoder",
    "continuous_bio_encoder",
    "joint_instance_encoder",
    "patient_readout",
    "classifier",
)
MODULE_PREFIXES = {
    "image_encoder": ("sources.image_encoder.",),
    "text_encoder": ("sources.text_encoder.",),
    "bio_encoder": ("sources.bio_encoder.",),
    "image_projector": ("sources.image_projector.",),
    "text_projector": ("sources.text_projector.",),
    "bio_projector": ("sources.bio_projector.",),
    "multimodal_instance_encoder": ("multimodal_encoder.",),
    "continuous_bio_encoder": ("continuous_bio_encoder.",),
    "joint_instance_encoder": ("joint_instance_encoder.",),
    "patient_readout": ("patient_readout.",),
    "classifier": ("classifier.",),
}

from dmea_ht.c61_cbpi import C61CBPIModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402
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


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loaders(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], seed: int, splits: Sequence[str]
) -> Dict[str, DataLoader]:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = VisitPatientDataset(
            rows=rows,
            data_root=project["data_root"],
            split=split,
            image_size=int(model_cfg["image_size"]),
            text_max_length=int(model_cfg["text_max_length"]),
            text_vocab_size=int(model_cfg["text_vocab_size"]),
            bio_dim=int(model_cfg["bio_dim"]),
            max_images_per_visit=int(model_cfg["max_images_per_visit"]),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate_visit_batch,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            worker_init_fn=seed_worker,
        )
    return loaders


def build_from_base_model(
    config: Dict[str, Any], seed: int, device: torch.device
) -> C61CBPIModel:
    if not bool(config.get("from_base")) or config.get("initialization", {}).get("mode") != "from_base":
        raise RuntimeError("C63 model construction requires explicit from_base initialization")
    model = C61CBPIModel(config, seed).to(device)
    if getattr(model.sources, "initialization_type", "") != "random_task_specific":
        raise RuntimeError("C63 source modules were not randomly initialized from base")
    return model


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid C63 checkpoint payload: {path}")
    return payload


def build_from_c63_checkpoint(
    config: Dict[str, Any], seed: int, device: torch.device, path: Path
) -> C61CBPIModel:
    model = build_from_base_model(config, seed, device)
    payload = checkpoint_payload(path)
    if int(payload.get("seed", -1)) != seed or str(payload.get("phase", "")) != "C63-FS-CBPI":
        raise RuntimeError(f"C63 reporting checkpoint metadata mismatch: {path}")
    model.load_state_dict(payload["model"], strict=True)
    return model


def optimizer_group_for_parameter(name: str) -> str:
    matches = [group for group, prefixes in OPTIMIZER_PREFIXES.items() if name.startswith(prefixes)]
    if len(matches) != 1:
        raise RuntimeError(f"C63 optimizer group assignment is not unique: {name} -> {matches}")
    return matches[0]


def module_group_for_parameter(name: str) -> str:
    matches = [group for group, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
    if len(matches) != 1:
        raise RuntimeError(f"C63 module group assignment is not unique: {name} -> {matches}")
    return matches[0]


def parameter_inventory(model: torch.nn.Module) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "parameter_name": name,
                "module_group": module_group_for_parameter(name),
                "optimizer_group": optimizer_group_for_parameter(name),
                "parameter_count": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "shape": json.dumps(list(parameter.shape)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != sum(1 for _ in model.named_parameters()):
        raise RuntimeError("C63 parameter inventory is incomplete")
    return frame.sort_values(["optimizer_group", "module_group", "parameter_name"]).reset_index(drop=True)


def optimizer_parameter_groups(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> tuple[torch.optim.Optimizer, pd.DataFrame]:
    inventory = parameter_inventory(model)
    if not inventory["requires_grad"].astype(bool).all():
        frozen = inventory.loc[~inventory["requires_grad"].astype(bool), "parameter_name"].tolist()
        raise RuntimeError(f"C63 predictive parameters are frozen: {frozen[:8]}")
    factors = config["learning_rate_groups"]
    base_lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    named_parameters = dict(model.named_parameters())
    groups: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for group in OPTIMIZER_GROUPS:
        names = inventory.loc[inventory["optimizer_group"] == group, "parameter_name"].tolist()
        factor = float(factors[group])
        learning_rate = base_lr * factor
        if not names or learning_rate <= 0.0:
            raise RuntimeError(f"C63 invalid optimizer group: {group}")
        groups.append(
            {"params": [named_parameters[name] for name in names], "lr": learning_rate, "weight_decay": weight_decay}
        )
        group_frame = inventory[inventory["optimizer_group"] == group]
        audit_rows.append(
            {
                "group": group,
                "learning_rate_factor": factor,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_tensor_count": len(names),
                "parameter_count": int(group_frame["parameter_count"].sum()),
                "all_requires_grad": True,
            }
        )
    return torch.optim.AdamW(groups), pd.DataFrame(audit_rows)


def parameter_hashes(model: torch.nn.Module, seed: int) -> tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    overall = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        value = digest.hexdigest()
        overall.update(name.encode("utf-8"))
        overall.update(value.encode("ascii"))
        rows.append(
            {
                "seed": seed,
                "parameter_name": name,
                "module_group": module_group_for_parameter(name),
                "optimizer_group": optimizer_group_for_parameter(name),
                "parameter_sha256": value,
            }
        )
    return pd.DataFrame(rows), overall.hexdigest()


def initialization_inventory(
    model: torch.nn.Module, seed: int, optimizer_audit: pd.DataFrame
) -> pd.DataFrame:
    lr_by_group = optimizer_audit.set_index("group")["learning_rate"].to_dict()
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        optimizer_group = optimizer_group_for_parameter(name)
        rows.append(
            {
                "seed": seed,
                "module_group": module_group_for_parameter(name),
                "parameter_name": name,
                "initialization_type": "random_task_specific",
                "source_path_or_rule": "PyTorch default module initialization after deterministic formal seed",
                "source_sha256": "NONE",
                "public_pretrained_backbone": False,
                "task_trained_checkpoint_used": False,
                "requires_grad": bool(parameter.requires_grad),
                "optimizer_group": optimizer_group,
                "learning_rate": float(lr_by_group[optimizer_group]),
                "parameter_count": int(parameter.numel()),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "module_group", "parameter_name"]).reset_index(drop=True)


def module_gradient_norms(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, List[float]] = {group: [] for group in MODULE_GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"C63 non-finite gradient: {name}")
        values[module_group_for_parameter(name)].append(float(gradient.pow(2).sum().cpu()))
    result: Dict[str, Dict[str, Any]] = {}
    for group in MODULE_GROUPS:
        squared = float(sum(values[group]))
        result[group] = {
            "norm": float(np.sqrt(squared)),
            "finite": True,
            "nonzero_tensor_count": int(sum(value > 0.0 for value in values[group])),
        }
    return result


def prediction_row(
    arrays: Mapping[str, np.ndarray], batch: Mapping[str, Any], index: int
) -> Dict[str, Any]:
    history_weights = np.asarray(arrays["history_weights"][index], dtype=float).clip(1e-8, 1.0)
    attention = np.asarray(arrays["attention"][index], dtype=float).clip(1e-8, 1.0)
    row: Dict[str, Any] = {
        "patient_id": str(batch["patient_id"][index]),
        "label": int(batch["label"].detach().cpu().numpy()[index]),
        "visit_count_audit_only": int(batch["visit_mask"].detach().cpu().numpy()[index].sum()),
        "evidence_valid_count": int(arrays["evidence_valid"][index].sum()),
        "bio_valid_count": int(
            np.logical_or(
                np.asarray(arrays["latest_bio_valid"][index], dtype=bool),
                np.asarray(arrays["history_bio_valid"][index], dtype=bool),
            ).sum()
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
    loader: DataLoader,
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
    gradients: Dict[str, List[float]] = {group: [] for group in MODULE_GROUPS}
    active_batches: Dict[str, int] = {group: 0 for group in MODULE_GROUPS}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C63 non-finite BCE loss")
        if is_train:
            loss.backward()
            gradient_info = module_gradient_norms(model)
            for group in MODULE_GROUPS:
                gradients[group].append(float(gradient_info[group]["norm"]))
                active_batches[group] += int(gradient_info[group]["nonzero_tensor_count"] > 0)
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
            "active_batch_count": int(active_batches[group]),
        }
        for group, values in gradients.items()
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
    order = np.argsort(np.asarray([str(row["patient_id"]) for row in split_result["predictions"]]))
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
                "module_group": module_group_for_parameter(name),
                "optimizer_group": optimizer_group_for_parameter(name),
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
    for group, group_frame in frame.groupby("module_group", sort=True):
        summaries.append(
            {
                "seed": seed,
                "kind": "module_summary",
                "module_group": group,
                "optimizer_group": str(group_frame["optimizer_group"].iloc[0]),
                "parameter_name": "",
                "parameter_count": int(group_frame["parameter_count"].sum()),
                "initial_l2": float(np.sqrt((group_frame["initial_l2"] ** 2).sum())),
                "delta_l2": float(np.sqrt((group_frame["delta_l2"] ** 2).sum())),
                "relative_parameter_change": float(
                    group_frame["delta_l2"].sum() / max(group_frame["initial_l2"].sum(), 1e-8)
                ),
                "updated": bool(group_frame["updated"].any()),
                "updated_parameter_tensor_count": int(group_frame["updated"].sum()),
                "finite": bool(group_frame["finite"].all()),
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def load_c63_config(path: str | Path) -> Dict[str, Any]:
    config = load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c63":
        raise RuntimeError("C63 configuration phase mismatch")
    if not bool(config.get("from_base")) or not bool(config.get("end_to_end")):
        raise RuntimeError("C63 requires from_base and end_to_end")
    if config.get("initialization", {}).get("mode") != "from_base":
        raise RuntimeError("C63 initialization mode must be from_base")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C63 formal seeds must remain [0, 42, 3407]")
    return config


def manifest_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return read_jsonl(config["project"]["manifest"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
