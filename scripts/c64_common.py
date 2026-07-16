#!/usr/bin/env python3
"""Shared C64 staged-tuning, fold, and final-training utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmea_ht.c61_cbpi import C61CBPIModel
from dmea_ht.config import load_config
from dmea_ht.visit_data import read_jsonl
from scripts import c63_common as c63
from scripts import train_phase_c40 as core


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
FOLD_COUNT = 5
FOLD_SEED = 20260716
CANDIDATES = (
    "A_CBPI_HEAD_ONLY",
    "B_PROJECTOR_CBPI",
    "C_FULL_LOW_LR",
)
MODULE_GROUPS = c63.MODULE_GROUPS
C64_OPTIMIZER_GROUPS = (
    "image_encoder",
    "text_encoder",
    "bio_source_encoder",
    "evidence_projectors",
    "c61_task_path",
)
C64_OPTIMIZER_PREFIXES = {
    "image_encoder": ("sources.image_encoder.",),
    "text_encoder": ("sources.text_encoder.",),
    "bio_source_encoder": ("sources.bio_encoder.",),
    "evidence_projectors": (
        "sources.image_projector.",
        "sources.text_projector.",
        "sources.bio_projector.",
    ),
    "c61_task_path": (
        "multimodal_encoder.",
        "continuous_bio_encoder.",
        "joint_instance_encoder.",
        "patient_readout.",
        "classifier.",
    ),
}
CANDIDATE_TRAINABLE_PREFIXES = {
    "A_CBPI_HEAD_ONLY": C64_OPTIMIZER_PREFIXES["c61_task_path"],
    "B_PROJECTOR_CBPI": (
        *C64_OPTIMIZER_PREFIXES["evidence_projectors"],
        *C64_OPTIMIZER_PREFIXES["c61_task_path"],
    ),
    "C_FULL_LOW_LR": tuple(
        prefix for group in C64_OPTIMIZER_GROUPS for prefix in C64_OPTIMIZER_PREFIXES[group]
    ),
}
CANDIDATE_END_TO_END = {
    "A_CBPI_HEAD_ONLY": False,
    "B_PROJECTOR_CBPI": True,
    "C_FULL_LOW_LR": True,
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_c64_config(path: str | Path) -> Dict[str, Any]:
    config = load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c64":
        raise RuntimeError("C64 configuration phase mismatch")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C64 formal seeds must remain [0, 42, 3407]")
    if int(config["training"]["patience"]) != 15:
        raise RuntimeError("C64 Early Stopping patience must be 15")
    if int(config["training"]["max_epochs"]) != 60:
        raise RuntimeError("C64 maximum epochs must be 60")
    return config


def manifest_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return read_jsonl(config["project"]["manifest"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def candidate_runtime_config(config: Mapping[str, Any], candidate: str) -> Dict[str, Any]:
    if candidate not in CANDIDATES:
        raise RuntimeError(f"Unsupported C64 candidate: {candidate}")
    runtime = copy.deepcopy(dict(config))
    runtime["end_to_end"] = bool(CANDIDATE_END_TO_END[candidate])
    runtime["c64_candidate"] = candidate
    return runtime


def c64_optimizer_group(name: str) -> str:
    matches = [
        group for group, prefixes in C64_OPTIMIZER_PREFIXES.items() if name.startswith(prefixes)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"C64 optimizer group assignment is not unique: {name} -> {matches}")
    return matches[0]


def expected_trainable_groups(candidate: str) -> Tuple[str, ...]:
    if candidate == "A_CBPI_HEAD_ONLY":
        return ("c61_task_path",)
    if candidate == "B_PROJECTOR_CBPI":
        return ("evidence_projectors", "c61_task_path")
    if candidate == "C_FULL_LOW_LR":
        return C64_OPTIMIZER_GROUPS
    raise RuntimeError(f"Unsupported C64 candidate: {candidate}")


def candidate_trainable(name: str, candidate: str) -> bool:
    return name.startswith(CANDIDATE_TRAINABLE_PREFIXES[candidate])


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError(f"Invalid C61 checkpoint payload: {path}")
    return payload


def c61_checkpoint_path(config: Mapping[str, Any], seed: int) -> Path:
    template = str(config["initialization"]["c61_checkpoint"])
    return resolve_path(template.replace("{seed}", str(seed)))


def build_c61_warm_start(
    config: Mapping[str, Any], candidate: str, seed: int, device: torch.device
) -> Tuple[C61CBPIModel, Mapping[str, Any], Path]:
    runtime = candidate_runtime_config(config, candidate)
    path = c61_checkpoint_path(config, seed)
    if not path.exists():
        raise FileNotFoundError(f"C64 C61 warm-start checkpoint missing: {path}")
    payload = checkpoint_payload(path)
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"C61 checkpoint seed mismatch: {path}")
    model = C61CBPIModel(runtime, seed).to(device)
    model.load_state_dict(payload["model"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(candidate_trainable(name, candidate))
    expected = expected_trainable_groups(candidate)
    actual = {c64_optimizer_group(name) for name, p in model.named_parameters() if p.requires_grad}
    if actual != set(expected):
        raise RuntimeError(f"C64 trainable scope mismatch for {candidate}: {actual} != {set(expected)}")
    return model, payload, path


def set_train_mode(model: C61CBPIModel, candidate: str, training: bool) -> None:
    model.train(training)
    if candidate == "A_CBPI_HEAD_ONLY":
        model.sources.eval()
    elif candidate == "B_PROJECTOR_CBPI":
        for module_name in ("image_encoder", "text_encoder", "bio_encoder"):
            getattr(model.sources, module_name).eval()
        for module_name in ("image_projector", "text_projector", "bio_projector"):
            getattr(model.sources, module_name).train(training)


def parameter_inventory(model: torch.nn.Module, candidate: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "candidate": candidate,
                "parameter_name": name,
                "module_group": c63.module_group_for_parameter(name),
                "optimizer_group": c64_optimizer_group(name),
                "parameter_count": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "shape": json.dumps(list(parameter.shape)),
            }
        )
    return pd.DataFrame(rows).sort_values(["optimizer_group", "module_group", "parameter_name"]).reset_index(drop=True)


def optimizer_parameter_groups(
    model: torch.nn.Module, config: Mapping[str, Any], candidate: str
) -> Tuple[torch.optim.Optimizer, pd.DataFrame]:
    inventory = parameter_inventory(model, candidate)
    factors = config["learning_rate_factors"]
    base_lr = float(config["training"]["base_lr"])
    weight_decay = float(config["training"]["weight_decay"])
    named = dict(model.named_parameters())
    groups: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    expected = set(expected_trainable_groups(candidate))
    for group in C64_OPTIMIZER_GROUPS:
        frame = inventory[inventory["optimizer_group"] == group]
        names = frame.loc[frame["requires_grad"].astype(bool), "parameter_name"].tolist()
        active = group in expected
        if active and not names:
            raise RuntimeError(f"C64 active optimizer group has no parameters: {candidate}/{group}")
        factor_key = {
            "image_encoder": "image_text_encoders",
            "text_encoder": "image_text_encoders",
            "bio_source_encoder": "bio_source_encoder",
            "evidence_projectors": "evidence_projectors",
            "c61_task_path": "c61_task_path",
        }[group]
        factor = float(factors[factor_key]) if active else 0.0
        learning_rate = base_lr * factor
        if active and learning_rate <= 0.0:
            raise RuntimeError(f"C64 active optimizer group has nonpositive lr: {candidate}/{group}")
        if active:
            groups.append(
                {
                    "params": [named[name] for name in names],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        audit.append(
            {
                "candidate": candidate,
                "group": group,
                "factor_key": factor_key,
                "learning_rate_factor": factor,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_tensor_count": len(names),
                "parameter_count": int(frame["parameter_count"].sum()),
                "included_in_optimizer": active,
                "all_active_requires_grad": bool(active and frame["requires_grad"].astype(bool).all()),
            }
        )
    if not groups:
        raise RuntimeError(f"C64 candidate has no optimizer groups: {candidate}")
    return torch.optim.AdamW(groups), pd.DataFrame(audit)


def parameter_hashes(model: torch.nn.Module, seed: int, candidate: str) -> Tuple[pd.DataFrame, str]:
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
                "candidate": candidate,
                "seed": seed,
                "parameter_name": name,
                "module_group": c63.module_group_for_parameter(name),
                "optimizer_group": c64_optimizer_group(name),
                "parameter_sha256": value,
            }
        )
    return pd.DataFrame(rows), overall.hexdigest()


def initialization_inventory(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    candidate: str,
    seed: int,
    optimizer_audit: pd.DataFrame,
    checkpoint: Path,
) -> pd.DataFrame:
    lr_by_group = optimizer_audit.set_index("group")["learning_rate"].to_dict()
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        group = c64_optimizer_group(name)
        rows.append(
            {
                "candidate": candidate,
                "seed": seed,
                "module_group": c63.module_group_for_parameter(name),
                "optimizer_group": group,
                "parameter_name": name,
                "initialization_type": "c61_validation_checkpoint_warm_start",
                "source_path": str(checkpoint),
                "source_sha256": source_hash,
                "task_trained_checkpoint_used": True,
                "requires_grad": bool(parameter.requires_grad),
                "learning_rate": float(lr_by_group[group]),
                "parameter_count": int(parameter.numel()),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate", "seed", "module_group", "parameter_name"]).reset_index(drop=True)


def module_gradient_summary(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    return c63.module_gradient_norms(model)


def run_epoch(
    model: C61CBPIModel,
    candidate: str,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    set_train_mode(model, candidate, is_train)
    loss_values: List[float] = []
    predictions: List[Dict[str, Any]] = []
    patient_states: List[np.ndarray] = []
    evidence_states: List[np.ndarray] = []
    bio_states: List[np.ndarray] = []
    gradients: Dict[str, List[float]] = {group: [] for group in MODULE_GROUPS}
    active_batches: Dict[str, int] = {group: 0 for group in MODULE_GROUPS}
    for raw_batch in loader:
        batch = c63.move_batch(raw_batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C64 non-finite BCE loss")
        if is_train:
            loss.backward()
            gradient_info = module_gradient_summary(model)
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
            predictions.append(c63.prediction_row(arrays, batch, index))
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


def save_split(result: Mapping[str, Any], out_dir: Path, split: str, candidate: str) -> Dict[str, Any]:
    seed = int(result["seed"])
    split_result = result[split]
    frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id").reset_index(drop=True)
    frame.insert(0, "candidate", candidate)
    frame.insert(0, "split", split)
    frame.insert(0, "seed", seed)
    prediction_dir = out_dir / "predictions"
    representation_dir = out_dir / "representations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    representation_dir.mkdir(parents=True, exist_ok=True)
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
    return {"candidate": candidate, "seed": seed, "split": split, "best_epoch": int(result.get("best_epoch", 0)), **split_result["metrics"]}


def parameter_update_audit(
    model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor], seed: int, candidate: str
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
                "candidate": candidate,
                "seed": seed,
                "kind": "parameter",
                "module_group": c63.module_group_for_parameter(name),
                "optimizer_group": c64_optimizer_group(name),
                "parameter_name": name,
                "parameter_count": int(current.numel()),
                "requires_grad": bool(parameter.requires_grad),
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
                "candidate": candidate,
                "seed": seed,
                "kind": "module_summary",
                "module_group": group,
                "optimizer_group": str(group_frame["optimizer_group"].iloc[0]),
                "parameter_name": "",
                "parameter_count": int(group_frame["parameter_count"].sum()),
                "requires_grad": bool(group_frame["requires_grad"].any()),
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


def write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def train_validation_seed(
    config: Mapping[str, Any],
    candidate: str,
    seed: int,
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    max_epochs: int | None = None,
    patience: int | None = None,
    run_label: str = "stage_a",
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    train_rows = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    val_rows = [row for row in rows if str(row.get("split", "")).lower() == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError("C64 validation training requires nonempty train and val rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, payload, checkpoint = build_c61_warm_start(config, candidate, seed, device)
    inventory = parameter_inventory(model, candidate)
    optimizer, optimizer_audit = optimizer_parameter_groups(model, config, candidate)
    init_hash_frame, init_hash = parameter_hashes(model, seed, candidate)
    initialization = initialization_inventory(model, config, candidate, seed, optimizer_audit, checkpoint)
    inventory.to_csv(out_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(out_dir / "optimizer_parameter_groups.csv", index=False)
    init_hash_frame.to_csv(out_dir / "initial_parameter_hash.csv", index=False)
    initialization.to_csv(out_dir / "initialization_inventory.csv", index=False)
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    loaders = c63.build_loaders(config, list(train_rows) + list(val_rows), seed, ("train", "val"))
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    max_epochs = int(max_epochs if max_epochs is not None else config["training"]["max_epochs"])
    patience = int(patience if patience is not None else config["training"]["patience"])
    for epoch in range(1, max_epochs + 1):
        train_result = run_epoch(model, candidate, loaders["train"], optimizer, device)
        val_result = run_epoch(model, candidate, loaders["val"], None, device)
        val_auc = float(val_result["metrics"]["AUC"])
        epoch_row: Dict[str, Any] = {
            "candidate": candidate,
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": train_result["metrics"]["bce_loss"],
            "val_auc": val_auc,
            "val_sensitivity": val_result["metrics"]["Sensitivity"],
            "val_specificity": val_result["metrics"]["Specificity"],
            "val_balanced_accuracy": val_result["metrics"]["Balanced_ACC"],
            "val_pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
            "selected_by_val_auc": False,
        }
        for group in MODULE_GROUPS:
            summary = train_result["gradient_summary"][group]
            epoch_row[f"{group}_grad_norm"] = summary["mean_norm"]
            epoch_row[f"{group}_active_batch_count"] = summary["active_batch_count"]
        epoch_rows.append(epoch_row)
        for group in MODULE_GROUPS:
            summary = train_result["gradient_summary"][group]
            gradient_rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "epoch": epoch,
                    "module_group": group,
                    "gradient_norm": summary["mean_norm"],
                    "active_batch_count": summary["active_batch_count"],
                    "expected_trainable": any(
                        c63.module_group_for_parameter(name) == group and parameter.requires_grad
                        for name, parameter in model.named_parameters()
                    ),
                }
            )
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"C64 {candidate} seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    best_val = run_epoch(model, candidate, loaders["val"], None, device)
    if best_val["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C64 {candidate} seed {seed} produced constant predictions")
    updates = parameter_update_audit(model, initial_state, seed, candidate)
    checkpoint_out = out_dir / f"{candidate}_seed_{seed}_best.pt"
    torch.save(
        {
            "phase": "C64-STCV",
            "stage": run_label,
            "candidate": candidate,
            "model": model.state_dict(),
            "config": dict(config),
            "seed": seed,
            "best_epoch": best_epoch,
            "initialization_type": "c61_validation_checkpoint_warm_start",
            "c61_checkpoint": str(checkpoint),
            "c61_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection_metric": "validation_auc_only",
            "max_epochs": max_epochs,
            "patience": patience,
            "trainable_groups": list(expected_trainable_groups(candidate)),
            "initial_parameter_hash": init_hash,
        },
        checkpoint_out,
    )
    result = {"candidate": candidate, "seed": seed, "best_epoch": best_epoch, "val": best_val}
    metric = save_split({"seed": seed, "best_epoch": best_epoch, "val": best_val}, out_dir, "val", candidate)
    pd.DataFrame([metric]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(out_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(out_dir / "parameter_update_audit.csv", index=False)
    best_val["patient_diagnostics"].sort_values("patient_id").to_csv(out_dir / "patient_diagnostics_val.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "candidate": candidate,
                "seed": seed,
                "best_epoch": best_epoch,
                "max_epochs": max_epochs,
                "patience": patience,
                "selection_metric": "validation_AUC_only",
                "initialization_type": "c61_validation_checkpoint_warm_start",
                "c61_checkpoint": str(checkpoint),
                "trainable_parameter_count": int(inventory.loc[inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "frozen_predictive_parameter_count": int(inventory.loc[~inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "trainable_groups": list(expected_trainable_groups(candidate)),
                "early_stopping": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_status(
        out_dir / "run_status.json",
        {
            "phase": "C64-STCV",
            "stage": run_label,
            "status": "COMPLETE",
            "candidate": candidate,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "test_loaded": False,
        },
    )
    return metric


def train_fixed_seed(
    config: Mapping[str, Any],
    candidate: str,
    seed: int,
    rows: Sequence[Dict[str, Any]],
    fixed_epoch: int,
    out_dir: Path,
) -> Dict[str, Any]:
    if fixed_epoch < 3 or fixed_epoch > 60:
        raise RuntimeError(f"C64 final epoch outside contract: {fixed_epoch}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    train_rows = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    if len(train_rows) != 696:
        raise RuntimeError(f"C64 final training requires 696 development patients, got {len(train_rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, payload, checkpoint = build_c61_warm_start(config, candidate, seed, device)
    inventory = parameter_inventory(model, candidate)
    optimizer, optimizer_audit = optimizer_parameter_groups(model, config, candidate)
    init_hash_frame, init_hash = parameter_hashes(model, seed, candidate)
    initialization = initialization_inventory(model, config, candidate, seed, optimizer_audit, checkpoint)
    inventory.to_csv(out_dir / "trainable_parameter_inventory.csv", index=False)
    optimizer_audit.to_csv(out_dir / "optimizer_parameter_groups.csv", index=False)
    init_hash_frame.to_csv(out_dir / "initial_parameter_hash.csv", index=False)
    initialization.to_csv(out_dir / "initialization_inventory.csv", index=False)
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    loader = c63.build_loaders(config, train_rows, seed, ("train",))["train"]
    epoch_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    for epoch in range(1, fixed_epoch + 1):
        result = run_epoch(model, candidate, loader, optimizer, device)
        row = {
            "candidate": candidate,
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": result["metrics"]["bce_loss"],
            "train_rows": result["metrics"]["n_rows"],
            "early_stopping": False,
        }
        for group in MODULE_GROUPS:
            summary = result["gradient_summary"][group]
            row[f"{group}_grad_norm"] = summary["mean_norm"]
            row[f"{group}_active_batch_count"] = summary["active_batch_count"]
            gradient_rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "epoch": epoch,
                    "module_group": group,
                    "gradient_norm": summary["mean_norm"],
                    "active_batch_count": summary["active_batch_count"],
                }
            )
        epoch_rows.append(row)
    updates = parameter_update_audit(model, initial_state, seed, candidate)
    checkpoint_out = out_dir / f"{candidate}_seed_{seed}_final.pt"
    torch.save(
        {
            "phase": "C64-STCV",
            "stage": "final",
            "candidate": candidate,
            "model": model.state_dict(),
            "config": dict(config),
            "seed": seed,
            "fixed_epoch": fixed_epoch,
            "early_stopping": False,
            "initialization_type": "c61_validation_checkpoint_warm_start",
            "c61_checkpoint": str(checkpoint),
            "c61_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection_metric": "frozen_cv_median_epoch",
            "trainable_groups": list(expected_trainable_groups(candidate)),
            "initial_parameter_hash": init_hash,
        },
        checkpoint_out,
    )
    pd.DataFrame(epoch_rows).to_csv(out_dir / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(out_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(out_dir / "parameter_update_audit.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "candidate": candidate,
                "seed": seed,
                "fixed_epoch": fixed_epoch,
                "early_stopping": False,
                "initialization_type": "c61_validation_checkpoint_warm_start",
                "c61_checkpoint": str(checkpoint),
                "trainable_parameter_count": int(inventory.loc[inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "frozen_predictive_parameter_count": int(inventory.loc[~inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "trainable_groups": list(expected_trainable_groups(candidate)),
                "test_role": "reporting_only_after_final_contract",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_status(
        out_dir / "run_status.json",
        {
            "phase": "C64-STCV",
            "stage": "final",
            "status": "COMPLETE",
            "candidate": candidate,
            "seed": seed,
            "fixed_epoch": fixed_epoch,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "test_loaded": False,
        },
    )
    return {"candidate": candidate, "seed": seed, "fixed_epoch": fixed_epoch}


def evaluate_test_seed(
    config: Mapping[str, Any], candidate: str, seed: int, rows: Sequence[Dict[str, Any]], checkpoint: Path, out_dir: Path
) -> Dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    test_rows = [row for row in rows if str(row.get("split", "")).lower() == "test"]
    if len(test_rows) != 84:
        raise RuntimeError(f"C64 Test requires 84 patients, got {len(test_rows)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    model, _, _ = build_c61_warm_start(config, candidate, seed, device)
    payload = checkpoint_payload(checkpoint)
    if str(payload.get("stage", "")) != "final" or str(payload.get("candidate", "")) != candidate:
        raise RuntimeError(f"C64 final checkpoint metadata mismatch: {checkpoint}")
    model.load_state_dict(payload["model"], strict=True)
    loader = c63.build_loaders(config, test_rows, seed, ("test",))["test"]
    result = run_epoch(model, candidate, loader, None, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = save_split({"seed": seed, "best_epoch": int(payload["fixed_epoch"]), "test": result}, out_dir, "test", candidate)
    return metric


def development_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = [copy.deepcopy(row) for row in rows if str(row.get("split", "")).lower() in {"train", "val"}]
    if len(result) != 696:
        raise RuntimeError(f"C64 development pool must contain 696 patients, got {len(result)}")
    return result


def make_fold_assignments(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    dev = development_rows(rows)
    by_patient: Dict[str, Dict[str, Any]] = {}
    for row in dev:
        patient_id = str(row["patient_id"])
        if patient_id in by_patient:
            raise RuntimeError(f"C64 development patient duplicated: {patient_id}")
        by_patient[patient_id] = row
    rng = random.Random(FOLD_SEED)
    assignments: Dict[str, int] = {}
    for label in (0, 1):
        patient_ids = [patient_id for patient_id, row in by_patient.items() if int(row["label"]) == label]
        rng.shuffle(patient_ids)
        for index, patient_id in enumerate(patient_ids):
            assignments[patient_id] = index % FOLD_COUNT
    if len(assignments) != 696:
        raise RuntimeError("C64 fold assignment did not cover the development pool")
    return assignments


def write_fold_artifacts(config: Mapping[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    assignments = make_fold_assignments(rows)
    report_dir = resolve_path(config["project"]["report_dir"])
    cv_dir = resolve_path(config["project"]["cv_output_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)
    dev = development_rows(rows)
    inventory = pd.DataFrame(
        [
            {
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                "original_split": str(row["split"]),
                "fold": int(assignments[str(row["patient_id"])]),
            }
            for row in dev
        ]
    ).sort_values("patient_id")
    inventory.to_csv(report_dir / "c64_fold_patient_inventory.csv", index=False)
    folds = {
        str(fold): sorted(patient_id for patient_id, assigned in assignments.items() if assigned == fold)
        for fold in range(FOLD_COUNT)
    }
    (cv_dir / "fold_assignments.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    integrity = {
        "fold_seed": FOLD_SEED,
        "fold_count": FOLD_COUNT,
        "development_patient_count": len(dev),
        "fold_patient_counts": {fold: len(patient_ids) for fold, patient_ids in folds.items()},
        "fold_label_counts": {
            fold: {
                "label0": int(inventory.loc[(inventory["fold"] == int(fold)) & (inventory["label"] == 0)].shape[0]),
                "label1": int(inventory.loc[(inventory["fold"] == int(fold)) & (inventory["label"] == 1)].shape[0]),
            }
            for fold in folds
        },
        "test_overlap": 0,
        "patient_overlap": 0,
        "test_loaded": False,
    }
    (report_dir / "c64_fold_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    return assignments


def load_fold_assignments(config: Mapping[str, Any]) -> Dict[str, int]:
    path = resolve_path(config["project"]["cv_output_dir"]) / "fold_assignments.json"
    if not path.exists():
        raise FileNotFoundError(f"C64 fold assignment missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(patient_id): int(fold) for fold, patient_ids in payload.items() for patient_id in patient_ids}


def fold_rows(rows: Sequence[Dict[str, Any]], assignments: Mapping[str, int], fold: int) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in development_rows(rows):
        copy_row = copy.deepcopy(row)
        patient_id = str(copy_row["patient_id"])
        copy_row["split"] = "val" if int(assignments[patient_id]) == int(fold) else "train"
        result.append(copy_row)
    return result


def all_development_as_train(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = development_rows(rows)
    for row in result:
        row["split"] = "train"
    return result
