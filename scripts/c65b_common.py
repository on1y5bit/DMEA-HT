#!/usr/bin/env python3
"""Shared C65-B common-backbone training utilities."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmea_ht.c61_cbpi import C61CBPIModel
from scripts import c63_common as c63
from scripts import c64_common as c64
from scripts import c65a_common as c65a


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "A_CBPI_HEAD_ONLY"
SEEDS = c65a.SEEDS
FOLD_COUNT = c65a.FOLD_COUNT
FOLD_SEED = c65a.FOLD_SEED

write_json = c65a.write_json
write_markdown = c65a.write_markdown


def resolve_path(value: str | Path) -> Path:
    return c65a.resolve_path(value)


def load_c65b_config(path: str | Path) -> Dict[str, Any]:
    config = c65a.load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c65b":
        raise RuntimeError("C65-B configuration phase mismatch")
    if tuple(int(seed) for seed in config.get("training", {}).get("seeds", [])) != SEEDS:
        raise RuntimeError("C65-B formal seeds must remain [0, 42, 3407]")
    if int(config["training"]["patience"]) != 15 or int(config["training"]["max_epochs"]) != 60:
        raise RuntimeError("C65-B must retain C64 max_epochs=60 and patience=15")
    if float(config["learning_rate_factors"]["c61_task_path"]) <= 0.0:
        raise RuntimeError("C65-B head learning rate must be positive")
    if any(float(config["learning_rate_factors"][name]) != 0.0 for name in ("image_text_encoders", "bio_source_encoder", "evidence_projectors")):
        raise RuntimeError("C65-B source and projector learning rates must be zero")
    decision_path = resolve_path(config["project"]["c65a_report_dir"]) / "c65a_route_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "C65B_COMMON_BACKBONE_CV_AUTHORIZED":
        raise RuntimeError(f"C65-B route is not authorized: {decision.get('status')}")
    return config


def report_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["report_dir"])


def cv_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["cv_output_dir"])


def common_checkpoint(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["initialization"]["common_c61_checkpoint"])


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError(f"Invalid C61 checkpoint payload: {path}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"C65-B common checkpoint must be C61 seed 42: {path}")
    return payload


def build_common_backbone_model(
    config: Mapping[str, Any], head_seed: int, device: torch.device
) -> Tuple[C61CBPIModel, Mapping[str, Any], Path]:
    checkpoint = common_checkpoint(config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"C65-B common checkpoint missing: {checkpoint}")
    payload = checkpoint_payload(checkpoint)
    runtime = copy.deepcopy(dict(config))
    runtime["end_to_end"] = False
    runtime["c64_candidate"] = CANDIDATE
    # The source checkpoint is fixed to seed 42; the task head is initialized by head_seed.
    c65a.set_seed(head_seed)
    model = C61CBPIModel(runtime, seed=42).to(device)
    source_state = {name: tensor for name, tensor in payload["model"].items() if str(name).startswith("sources.")}
    if not source_state:
        raise RuntimeError("C61 seed-42 checkpoint contains no sources.* parameters")
    model_state = model.state_dict()
    missing_source = [name for name in model_state if name.startswith("sources.") and name not in source_state]
    unexpected_source = [name for name in source_state if name not in model_state]
    if missing_source or unexpected_source:
        raise RuntimeError(f"C65-B common source scope mismatch: missing={missing_source[:3]}, unexpected={unexpected_source[:3]}")
    model.load_state_dict(source_state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(c64.CANDIDATE_TRAINABLE_PREFIXES[CANDIDATE]))
    actual_groups = {c64.c64_optimizer_group(name) for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual_groups != {"c61_task_path"}:
        raise RuntimeError(f"C65-B trainable scope mismatch: {actual_groups}")
    model.sources.eval()
    return model, payload, checkpoint


def initialization_inventory(model: nn.Module, seed: int, checkpoint: Path) -> pd.DataFrame:
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    for name, parameter in model.named_parameters():
        is_source = name.startswith("sources.")
        rows.append(
            {
                "candidate": CANDIDATE,
                "seed": seed,
                "module_group": c63.module_group_for_parameter(name),
                "optimizer_group": c64.c64_optimizer_group(name),
                "parameter_name": name,
                "initialization_type": "c61_seed42_frozen_backbone" if is_source else "independent_head_seed_initialization",
                "source_checkpoint": str(checkpoint),
                "source_sha256": source_hash if is_source else "",
                "backbone_seed": 42,
                "head_seed": seed,
                "requires_grad": bool(parameter.requires_grad),
                "learning_rate": 0.0 if is_source else float("nan"),
                "parameter_count": int(parameter.numel()),
            }
        )
    return pd.DataFrame(rows).sort_values(["optimizer_group", "module_group", "parameter_name"]).reset_index(drop=True)


def write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def train_validation_seed(
    config: Mapping[str, Any],
    seed: int,
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c65a.set_seed(seed)
    train_rows = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    val_rows = [row for row in rows if str(row.get("split", "")).lower() == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError("C65-B fold training requires nonempty train and val rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, payload, checkpoint = build_common_backbone_model(config, seed, device)
    inventory = c64.parameter_inventory(model, CANDIDATE)
    optimizer, optimizer_audit = c64.optimizer_parameter_groups(model, config, CANDIDATE)
    init_hash_frame, init_hash = c64.parameter_hashes(model, seed, CANDIDATE)
    initialization = initialization_inventory(model, seed, checkpoint)
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
    epoch_rows = []
    gradient_rows = []
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["patience"])
    for epoch in range(1, max_epochs + 1):
        train_result = c64.run_epoch(model, CANDIDATE, loaders["train"], optimizer, device)
        val_result = c64.run_epoch(model, CANDIDATE, loaders["val"], None, device)
        val_auc = float(val_result["metrics"]["AUC"])
        epoch_row = {
            "candidate": CANDIDATE,
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
        for group in c64.MODULE_GROUPS:
            summary = train_result["gradient_summary"][group]
            epoch_row[f"{group}_grad_norm"] = summary["mean_norm"]
            epoch_row[f"{group}_active_batch_count"] = summary["active_batch_count"]
            gradient_rows.append(
                {
                    "candidate": CANDIDATE,
                    "seed": seed,
                    "epoch": epoch,
                    "module_group": group,
                    "gradient_norm": summary["mean_norm"],
                    "active_batch_count": summary["active_batch_count"],
                    "expected_trainable": any(c63.module_group_for_parameter(name) == group and parameter.requires_grad for name, parameter in model.named_parameters()),
                }
            )
        epoch_rows.append(epoch_row)
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"C65-B seed {seed} produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    best_val = c64.run_epoch(model, CANDIDATE, loaders["val"], None, device)
    if best_val["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C65-B seed {seed} produced constant predictions")
    updates = c64.parameter_update_audit(model, initial_state, seed, CANDIDATE)
    checkpoint_out = out_dir / f"{CANDIDATE}_seed_{seed}_best.pt"
    torch.save(
        {
            "phase": "C65-VACS",
            "stage": "cv",
            "candidate": CANDIDATE,
            "model": model.state_dict(),
            "config": dict(config),
            "seed": seed,
            "best_epoch": best_epoch,
            "initialization_type": "c61_seed42_frozen_backbone_independent_head",
            "common_c61_checkpoint": str(checkpoint),
            "common_c61_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection_metric": "validation_auc_only",
            "max_epochs": max_epochs,
            "patience": patience,
            "trainable_groups": ["c61_task_path"],
            "initial_parameter_hash": init_hash,
            "test_loaded": False,
        },
        checkpoint_out,
    )
    metric = c64.save_split({"seed": seed, "best_epoch": best_epoch, "val": best_val}, out_dir, "val", CANDIDATE)
    pd.DataFrame([metric]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(out_dir / "gradient_connectivity.csv", index=False)
    updates.to_csv(out_dir / "parameter_update_audit.csv", index=False)
    best_val["patient_diagnostics"].sort_values("patient_id").to_csv(out_dir / "patient_diagnostics_val.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "seed": seed,
                "backbone_seed": 42,
                "best_epoch": best_epoch,
                "max_epochs": max_epochs,
                "patience": patience,
                "selection_metric": "validation_AUC_only",
                "initialization_type": "c61_seed42_frozen_backbone_independent_head",
                "common_c61_checkpoint": str(checkpoint),
                "trainable_parameter_count": int(inventory.loc[inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "frozen_predictive_parameter_count": int(inventory.loc[~inventory["requires_grad"].astype(bool), "parameter_count"].sum()),
                "trainable_groups": ["c61_task_path"],
                "early_stopping": True,
                "test_loaded": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_status(
        out_dir / "run_status.json",
        {
            "phase": "C65-VACS",
            "stage": "cv",
            "status": "COMPLETE",
            "candidate": CANDIDATE,
            "seed": seed,
            "backbone_seed": 42,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "test_loaded": False,
        },
    )
    return metric


def development_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return c65a.development_rows(config)


def fold_assignments(config: Mapping[str, Any]) -> Dict[str, int]:
    path = resolve_path(config["project"]["c64_cv_output_dir"]) / "fold_assignments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(patient_id): int(fold) for fold, patient_ids in payload.items() for patient_id in patient_ids}
