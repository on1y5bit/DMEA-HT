#!/usr/bin/env python3
"""Shared runtime, data, optimization, and audit helpers for C66-LFFC."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dmea_ht.c66_lffc import C66CBPIModel, C66SourceModel, C66VisitPatientDataset, collate_c66_visit_batch

from scripts import c66_common as protocol


REPO_ROOT = protocol.REPO_ROOT
GROUPS = ("image_text_encoders", "bio_source_encoder", "evidence_projectors", "source_task_path", "cbpi_task_path")
PATIENT_ID_PATTERN = re.compile(r'"patient_id"\s*:\s*"([^"\\]+)"')


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_preflight_path(config: Mapping[str, Any]) -> Path:
    return protocol.report_dir(config) / "c66_runtime_preflight.json"


def generic_provenance(config: Mapping[str, Any], verify_hashes: bool = False) -> Dict[str, Any]:
    generic = dict(config["generic_initialization"])
    image = dict(generic["image"])
    text = dict(generic["text"])
    image_path = Path(str(image["local_weight_path"]))
    text_path = Path(str(text["local_weight_path"]))
    for path in (image_path, text_path, Path(str(text["tokenizer_path"])), Path(str(text["text_encoder_path"]))):
        if not path.exists():
            raise FileNotFoundError(f"C66 required public initialization artifact is missing: {path}")
    payload: Dict[str, Any] = {
        "image": {
            "provider": image["provider"],
            "source": image["source"],
            "version": image["version"],
            "local_weight_path": str(image_path),
            "expected_sha256": str(image["sha256"]),
        },
        "text": {
            "provider": text["provider"],
            "source": text["source"],
            "snapshot_revision": text["snapshot_revision"],
            "transformers_version": text["transformers_version"],
            "tokenizer_path": str(text["tokenizer_path"]),
            "text_encoder_path": str(text["text_encoder_path"]),
            "local_weight_path": str(text_path),
            "expected_sha256": str(text["sha256"]),
        },
    }
    if verify_hashes:
        import transformers

        actual_image_sha = sha256_file(image_path)
        actual_text_sha = sha256_file(text_path)
        if actual_image_sha != payload["image"]["expected_sha256"]:
            raise RuntimeError("C66 public ResNet50 SHA256 mismatch")
        if actual_text_sha != payload["text"]["expected_sha256"]:
            raise RuntimeError("C66 public CLIP text SHA256 mismatch")
        if str(transformers.__version__) != str(text["transformers_version"]):
            raise RuntimeError("C66 transformers version does not match the frozen text provenance")
        payload["image"]["verified_sha256"] = actual_image_sha
        payload["text"]["verified_sha256"] = actual_text_sha
        payload["text"]["verified_transformers_version"] = str(transformers.__version__)
    return payload


def write_runtime_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = generic_provenance(config, verify_hashes=True)
    set_seed(0)
    model = C66SourceModel(dict(config))
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    source_count = int(sum(parameter.numel() for parameter in model.sources.parameters()))
    payload = {
        "phase": "C66-LFFC",
        "status": "C66_RUNTIME_PUBLIC_INITIALIZATION_AUTHORIZED",
        "test_loaded": False,
        "test_rows_read": 0,
        "task_checkpoint_loaded": False,
        "historical_prediction_or_representation_input": False,
        "provenance": provenance,
        "source_model_parameter_count": parameter_count,
        "source_module_parameter_count": source_count,
    }
    protocol.write_json(runtime_preflight_path(config), payload)
    del model
    return payload


def require_runtime_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = runtime_preflight_path(config)
    if not path.exists():
        raise RuntimeError(f"C66 formal training requires the runtime preflight artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C66_RUNTIME_PUBLIC_INITIALIZATION_AUTHORIZED":
        raise RuntimeError("C66 runtime public initialization is not authorized")
    expected = generic_provenance(config, verify_hashes=False)
    recorded = payload.get("provenance", {})
    for modality in ("image", "text"):
        if recorded.get(modality, {}).get("expected_sha256") != expected[modality]["expected_sha256"]:
            raise RuntimeError(f"C66 {modality} provenance changed after preflight")
    return payload


def _stream_development_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Parse only patient lines present in C64's 696-patient development inventory."""
    inventory = protocol.read_c64_development_inventory(config)
    development_ids = set(inventory["patient_id"].astype(str).tolist())
    manifest_path = Path(str(config["project"]["manifest"]))
    if not manifest_path.exists():
        raise FileNotFoundError(f"C66 manifest is missing: {manifest_path}")
    rows: list[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = PATIENT_ID_PATTERN.search(raw_line)
            if match is None:
                raise RuntimeError("C66 could not identify a manifest patient ID without parsing a row")
            patient_id = match.group(1)
            if patient_id in development_ids:
                rows.append(json.loads(raw_line))
    observed = {str(row["patient_id"]) for row in rows}
    if len(rows) != protocol.DEVELOPMENT_PATIENT_COUNT or observed != development_ids:
        raise RuntimeError("C66 development-only manifest stream did not recover exactly 696 patients")
    if any(str(row.get("split", "")).lower() not in {"train", "val"} for row in rows):
        raise RuntimeError("C66 development stream unexpectedly parsed a non-development row")
    return rows


def development_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return _stream_development_rows(config)


def test_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Open Test only from the final collector after its irreversible lock is written."""
    manifest_path = Path(str(config["project"]["manifest"]))
    rows: list[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            if str(row.get("split", "")).lower() == "test":
                rows.append(row)
    if len(rows) != int(config["data_contract"]["locked_test_patient_count"]):
        raise RuntimeError("C66 final Test loader did not recover the locked 84-patient Test split")
    return rows


def nested_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = protocol.nested_split_path(config)
    if not path.exists():
        raise FileNotFoundError(f"C66 nested split artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_rows(rows: Sequence[Dict[str, Any]], assignments: Mapping[str, Sequence[str]]) -> list[Dict[str, Any]]:
    split_by_patient: Dict[str, str] = {}
    for split, patient_ids in assignments.items():
        for patient_id in patient_ids:
            patient_id = str(patient_id)
            if patient_id in split_by_patient:
                raise RuntimeError(f"C66 patient appears in more than one split: {patient_id}")
            split_by_patient[patient_id] = str(split)
    result: list[Dict[str, Any]] = []
    for raw in rows:
        patient_id = str(raw["patient_id"])
        if patient_id not in split_by_patient:
            continue
        copy_row = copy.deepcopy(raw)
        copy_row["split"] = split_by_patient[patient_id]
        result.append(copy_row)
    if len(result) != len(split_by_patient):
        raise RuntimeError("C66 split assignment references a missing development patient")
    return result


def fold_inner_rows(config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], fold: int) -> list[Dict[str, Any]]:
    entry = nested_payload(config)["folds"][str(int(fold))]
    return split_rows(
        rows,
        {"train": entry["inner_train_patient_ids"], "val": entry["inner_val_patient_ids"]},
    )


def fold_outer_rows(config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], fold: int) -> list[Dict[str, Any]]:
    entry = nested_payload(config)["folds"][str(int(fold))]
    return split_rows(
        rows,
        {"train": entry["outer_train_patient_ids"], "val": entry["outer_val_patient_ids"]},
    )


def all_development_train_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return split_rows(rows, {"train": [str(row["patient_id"]) for row in rows]})


def build_loaders(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], seed: int, splits: Sequence[str]
) -> Dict[str, DataLoader]:
    model_cfg = dict(config["model"])
    data_cfg = dict(config["data_loader"])
    text_cfg = dict(config["generic_initialization"])["text"]
    loaders: Dict[str, DataLoader] = {}
    for offset, split in enumerate(splits):
        dataset = C66VisitPatientDataset(
            rows,
            data_root=config["project"]["data_root"],
            split=split,
            image_size=int(model_cfg["image_size"]),
            text_max_length=int(model_cfg["text_max_length"]),
            text_vocab_size=int(model_cfg["text_vocab_size"]),
            bio_dim=int(model_cfg["bio_dim"]),
            max_images_per_visit=int(model_cfg["max_images_per_visit"]),
            clip_tokenizer_path=text_cfg["tokenizer_path"],
            clip_max_length=int(model_cfg["clip_max_length"]),
        )
        if len(dataset) == 0:
            raise RuntimeError(f"C66 {split} loader is empty")
        generator = torch.Generator()
        generator.manual_seed(int(seed) + offset * 100003)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            generator=generator,
            collate_fn=collate_c66_visit_batch,
        )
    return loaders


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("C66 formal training requires CUDA")
    return torch.device("cuda")


def source_model(config: Mapping[str, Any], seed: int, target_device: torch.device) -> C66SourceModel:
    set_seed(seed)
    return C66SourceModel(dict(config)).to(target_device)


def route_model(config: Mapping[str, Any], seed: int, target_device: torch.device, route: str) -> C66CBPIModel:
    runtime = copy.deepcopy(dict(config))
    runtime["from_base"] = True
    runtime["end_to_end"] = route == "E"
    runtime["initialization"] = {"mode": "from_base"}
    set_seed(seed)
    model = C66CBPIModel(runtime, seed).to(target_device)
    model.configure_route(route)
    return model


def source_group(name: str, stage: str) -> str:
    if name.startswith(("sources.image_encoder.", "sources.text_encoder.")):
        return "image_text_encoders"
    if name.startswith("sources.bio_encoder."):
        return "bio_source_encoder"
    if name.startswith(("sources.image_projector.", "sources.text_projector.", "sources.bio_projector.")):
        return "evidence_projectors"
    if name.startswith("source_evidence_stack."):
        return "source_task_path" if stage == "source" else "evidence_projectors"
    if name.startswith(("source_patient_readout.", "source_classifier.")):
        return "source_task_path"
    if name.startswith(("multimodal_encoder.", "continuous_bio_encoder.", "joint_instance_encoder.", "patient_readout.", "classifier.")):
        return "cbpi_task_path"
    raise RuntimeError(f"C66 cannot assign an optimizer group to parameter: {name}")


def expected_groups(stage: str, route: str | None = None) -> set[str]:
    if stage == "source":
        return {"image_text_encoders", "bio_source_encoder", "evidence_projectors", "source_task_path"}
    if stage == "route":
        if route == "F":
            return {"cbpi_task_path"}
        if route == "E":
            return {"image_text_encoders", "bio_source_encoder", "evidence_projectors", "cbpi_task_path"}
    raise RuntimeError(f"Invalid C66 training scope: stage={stage}, route={route}")


def optimizer_and_inventory(
    model: torch.nn.Module, config: Mapping[str, Any], stage: str, route: str | None = None
) -> tuple[torch.optim.Optimizer, pd.DataFrame, pd.DataFrame]:
    section = dict(config["source_learning"] if stage == "source" else config["route_training"])
    if stage == "source":
        factors = dict(section["learning_rate_factors"])
        factors = {
            "image_text_encoders": float(factors["image_text_encoders"]),
            "bio_source_encoder": float(factors["bio_source_encoder"]),
            "evidence_projectors": float(factors["evidence_projectors"]),
            "source_task_path": float(factors["source_task_path"]),
        }
    else:
        route_factors = dict(section["route_f" if route == "F" else "route_e"]["learning_rate_factors"])
        factors = {
            "image_text_encoders": float(route_factors["image_text_encoders"]),
            "bio_source_encoder": float(route_factors["bio_source_encoder"]),
            "evidence_projectors": float(route_factors["evidence_projectors"]),
            "cbpi_task_path": float(route_factors["cbpi_task_path"]),
        }

    inventory_rows = []
    by_group: Dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        group = source_group(name, stage)
        inventory_rows.append(
            {
                "parameter_name": name,
                "optimizer_group": group,
                "parameter_count": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "shape": json.dumps(list(parameter.shape)),
            }
        )
        if parameter.requires_grad:
            by_group.setdefault(group, []).append((name, parameter))
    inventory = pd.DataFrame(inventory_rows).sort_values(["optimizer_group", "parameter_name"]).reset_index(drop=True)
    actual_groups = set(by_group)
    expected = expected_groups(stage, route)
    if actual_groups != expected:
        raise RuntimeError(f"C66 trainable scope mismatch: expected={sorted(expected)}, actual={sorted(actual_groups)}")
    groups = []
    audit_rows = []
    base_lr = float(section["base_lr"])
    weight_decay = float(section["weight_decay"])
    for group in sorted(expected):
        factor = float(factors[group])
        if factor <= 0.0 or not by_group[group]:
            raise RuntimeError(f"C66 optimizer group is inactive: {group}")
        lr = base_lr * factor
        groups.append({"params": [parameter for _, parameter in by_group[group]], "lr": lr, "weight_decay": weight_decay})
        audit_rows.append(
            {
                "optimizer_group": group,
                "learning_rate_factor": factor,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "parameter_count": int(sum(parameter.numel() for _, parameter in by_group[group])),
                "all_requires_grad": True,
            }
        )
    return torch.optim.AdamW(groups), inventory, pd.DataFrame(audit_rows)


def initialization_inventory(model: torch.nn.Module, stage: str, source_checkpoint: str = "") -> pd.DataFrame:
    rows = []
    for name, parameter in model.named_parameters():
        is_public = name.startswith(("sources.image_encoder.backbone.", "sources.text_encoder.backbone."))
        is_fold_local = bool(source_checkpoint) and name.startswith(("sources.", "source_evidence_stack."))
        rows.append(
            {
                "parameter_name": name,
                "stage": stage,
                "parameter_count": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "initialization_type": (
                    "fold_local_current_outer_checkpoint" if is_fold_local else "public_generic_pretrained" if is_public else "seed_random_task_specific"
                ),
                "source_checkpoint": source_checkpoint if is_fold_local else "",
                "task_checkpoint_used": False,
                "historical_prediction_input": False,
                "historical_representation_input": False,
            }
        )
    return pd.DataFrame(rows).sort_values("parameter_name").reset_index(drop=True)


def initial_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def parameter_update_audit(model: torch.nn.Module, before: Mapping[str, torch.Tensor], stage: str) -> pd.DataFrame:
    rows = []
    for name, parameter in model.named_parameters():
        start = before[name].float()
        current = parameter.detach().cpu().float()
        delta = current - start
        group = source_group(name, stage)
        rows.append(
            {
                "kind": "parameter",
                "parameter_name": name,
                "optimizer_group": group,
                "initial_l2": float(torch.linalg.vector_norm(start)),
                "delta_l2": float(torch.linalg.vector_norm(delta)),
                "updated": bool(float(torch.linalg.vector_norm(delta)) > 0.0) if parameter.requires_grad else True,
                "finite": bool(torch.isfinite(current).all() and torch.isfinite(delta).all()),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    frame = pd.DataFrame(rows)
    summaries = []
    for group, group_frame in frame.groupby("optimizer_group"):
        active = group_frame[group_frame["requires_grad"].astype(bool)]
        summaries.append(
            {
                "kind": "module_summary",
                "parameter_name": "",
                "optimizer_group": group,
                "initial_l2": float(np.sqrt((active["initial_l2"] ** 2).sum())) if len(active) else 0.0,
                "delta_l2": float(np.sqrt((active["delta_l2"] ** 2).sum())) if len(active) else 0.0,
                "updated": bool(active["updated"].all()) if len(active) else True,
                "finite": bool(group_frame["finite"].all()),
                "requires_grad": bool(len(active)),
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True)


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def training_health_details(
    gradient: pd.DataFrame,
    updates: pd.DataFrame,
    stage: str,
    selected_epoch: int,
    route: str | None = None,
) -> pd.DataFrame:
    """Verify gradient connectivity and aggregate optimizer-group updates.

    Some imported evidence modules contain role branches that are unavailable for
    a particular batch. They remain visible in the per-parameter audit, but the
    health gate is intentionally defined at the declared optimizer-group level.
    """
    expected = sorted(expected_groups(stage, route))
    selected = gradient[gradient["epoch"].astype(int) == int(selected_epoch)]
    summary = updates[updates["kind"].astype(str) == "module_summary"]
    rows = []
    for group in expected:
        gradient_rows = selected[selected["optimizer_group"].astype(str) == group]
        summary_rows = summary[summary["optimizer_group"].astype(str) == group]
        gradient_values = pd.to_numeric(gradient_rows.get("max_norm"), errors="coerce")
        update_values = pd.to_numeric(summary_rows.get("delta_l2"), errors="coerce")
        gradient_nonzero = bool(
            len(gradient_rows) > 0 and np.isfinite(gradient_values).all() and float(gradient_values.max()) > 0.0
        )
        aggregate_update_nonzero = bool(
            len(summary_rows) == 1 and np.isfinite(update_values).all() and float(update_values.iloc[0]) > 0.0
        )
        aggregate_finite = bool(
            len(summary_rows) == 1 and _bool_value(summary_rows.iloc[0]["finite"])
        )
        aggregate_trainable = bool(
            len(summary_rows) == 1 and _bool_value(summary_rows.iloc[0]["requires_grad"])
        )
        rows.append(
            {
                "optimizer_group": group,
                "selected_epoch": int(selected_epoch),
                "selected_epoch_gradient_nonzero": gradient_nonzero,
                "aggregate_update_nonzero": aggregate_update_nonzero,
                "aggregate_finite": aggregate_finite,
                "aggregate_trainable": aggregate_trainable,
                "training_health_pass": bool(
                    gradient_nonzero and aggregate_update_nonzero and aggregate_finite and aggregate_trainable
                ),
            }
        )
    return pd.DataFrame(rows)


def training_health_pass(
    gradient: pd.DataFrame,
    updates: pd.DataFrame,
    stage: str,
    selected_epoch: int,
    route: str | None = None,
) -> bool:
    details = training_health_details(gradient, updates, stage, selected_epoch, route)
    return bool(len(details) > 0 and details["training_health_pass"].astype(bool).all())


def move_batch(batch: Mapping[str, Any], target_device: torch.device) -> Dict[str, Any]:
    return {key: value.to(target_device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def gradient_summary(model: torch.nn.Module, stage: str) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, list[float]] = {group: [] for group in GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = source_group(name, stage)
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"C66 non-finite gradient: {name}")
        values[group].append(float(torch.linalg.vector_norm(gradient).detach().cpu()))
    return {
        group: {
            "mean_norm": float(np.mean(group_values)) if group_values else 0.0,
            "max_norm": float(np.max(group_values)) if group_values else 0.0,
            "nonzero_tensor_count": int(sum(value > 0.0 for value in group_values)),
        }
        for group, group_values in values.items()
    }


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise RuntimeError("C66 evaluation requires both classes")
    if not np.isfinite(probabilities).all() or float(np.std(probabilities, ddof=1)) <= 0.0:
        raise RuntimeError("C66 evaluation produced non-finite or constant predictions")
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((negative & ~predicted).sum())
    fp = int((negative & predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_ACC": 0.5 * (sensitivity + specificity),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "positive_sensitivity_damage": 1.0 - sensitivity,
        "positive_negative_gap": float(probabilities[positive].mean() - probabilities[negative].mean()),
        "prediction_std": float(np.std(probabilities, ddof=1)),
        "pairwise_inversion_count": int((probabilities[positive, None] < probabilities[negative][None, :]).sum()),
        "n_rows": int(len(labels)),
    }


def _prediction_row(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any], index: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "patient_id": str(batch["patient_id"][index]),
        "label": int(batch["label"][index].detach().cpu()),
        "final_prob": float(outputs["prob"][index].detach().cpu()),
        "final_logit": float(outputs["logit"][index].detach().cpu()),
        "predicted_class": int(float(outputs["prob"][index].detach().cpu()) >= 0.5),
        "visit_count_audit_only": int(batch["visit_mask"][index].detach().cpu().sum()),
        "patient_state_norm": float(torch.linalg.vector_norm(outputs["patient_state"][index]).detach().cpu()),
    }
    for key, value in batch["shortcuts"][index].items():
        row[str(key)] = value
    return row


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    target_device: torch.device,
    stage: str,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    rows: list[Dict[str, Any]] = []
    active: Dict[str, int] = {group: 0 for group in GROUPS}
    gradient_rows: list[Dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, target_device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C66 BCEWithLogitsLoss is non-finite")
        if is_train:
            loss.backward()
            gradients = gradient_summary(model, stage)
            for group, summary in gradients.items():
                active[group] += int(summary["nonzero_tensor_count"] > 0)
                gradient_rows.append({"optimizer_group": group, **summary})
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for index in range(len(batch["patient_id"])):
            rows.append(_prediction_row(outputs, batch, index))
    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    metrics = _binary_metrics(frame["label"].to_numpy(dtype=int), frame["final_prob"].to_numpy(dtype=float))
    metrics["bce_loss"] = float(np.mean(losses))
    metrics["gradient_active_batches"] = {group: active[group] for group in GROUPS}
    return {"metrics": metrics, "predictions": frame, "gradient_rows": gradient_rows}


def early_stop_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_device: torch.device,
    stage: str,
    max_epochs: int,
    patience: int,
) -> Dict[str, Any]:
    best_auc = -float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[Dict[str, Any]] = []
    gradient_rows: list[Dict[str, Any]] = []
    for epoch in range(1, int(max_epochs) + 1):
        train_result = run_epoch(model, train_loader, optimizer, target_device, stage)
        val_result = run_epoch(model, val_loader, None, target_device, stage)
        val_auc = float(val_result["metrics"]["AUC"])
        history.append(
            {
                "epoch": epoch,
                "train_bce_loss": train_result["metrics"]["bce_loss"],
                "val_auc": val_auc,
                "val_sensitivity": val_result["metrics"]["Sensitivity"],
                "val_specificity": val_result["metrics"]["Specificity"],
                "val_positive_sensitivity_damage": val_result["metrics"]["positive_sensitivity_damage"],
                "selected_by_val_auc": False,
            }
        )
        for row in train_result["gradient_rows"]:
            gradient_rows.append({"epoch": epoch, **row})
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("C66 did not create an inner Validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in history:
        row["selected_by_val_auc"] = int(row["epoch"]) == int(best_epoch)
    selected_val = run_epoch(model, val_loader, None, target_device, stage)
    return {
        "best_epoch": int(best_epoch),
        "best_auc": float(best_auc),
        "history": pd.DataFrame(history),
        "gradient": pd.DataFrame(gradient_rows),
        "val": selected_val,
    }


def fixed_epoch_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_device: torch.device,
    stage: str,
    fixed_epochs: int,
) -> Dict[str, Any]:
    if int(fixed_epochs) <= 0:
        raise RuntimeError("C66 fixed refit epoch must be positive")
    history: list[Dict[str, Any]] = []
    gradient_rows: list[Dict[str, Any]] = []
    for epoch in range(1, int(fixed_epochs) + 1):
        result = run_epoch(model, train_loader, optimizer, target_device, stage)
        history.append({"epoch": epoch, "train_bce_loss": result["metrics"]["bce_loss"], "fixed_epoch_refit": True})
        for row in result["gradient_rows"]:
            gradient_rows.append({"epoch": epoch, **row})
    return {"history": pd.DataFrame(history), "gradient": pd.DataFrame(gradient_rows)}


def save_run_status(path: Path, payload: Mapping[str, Any]) -> None:
    protocol.write_json(path, payload)


def save_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("patient_id").to_csv(path, index=False)


def inner_source_dir(config: Mapping[str, Any], fold: int, seed: int) -> Path:
    return protocol.nested_cv_dir(config) / "inner_source" / f"fold_{fold}" / f"seed_{seed}"


def inner_route_dir(config: Mapping[str, Any], fold: int, route: str, seed: int) -> Path:
    return protocol.nested_cv_dir(config) / "inner_routes" / f"fold_{fold}" / f"route_{route}" / f"seed_{seed}"


def fold_decision_path(config: Mapping[str, Any], fold: int) -> Path:
    return protocol.nested_cv_dir(config) / "inner_decisions" / f"fold_{fold}.json"


def outer_refit_dir(config: Mapping[str, Any], fold: int, seed: int) -> Path:
    return protocol.nested_cv_dir(config) / "outer_refit" / f"fold_{fold}" / f"seed_{seed}"


def final_seed_dir(config: Mapping[str, Any], seed: int) -> Path:
    return protocol.resolve_path(config["project"]["final_output_dir"]) / "seed_runs" / f"seed_{seed}"


def torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"C66 invalid checkpoint payload: {path}")
    return payload


def torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
