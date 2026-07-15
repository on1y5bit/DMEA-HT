#!/usr/bin/env python3
"""Train C37-E2E-VRL as three independent direct validation-selected seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, trainable_parameter_count  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import (  # noqa: E402
    binary_metrics,
    build_loaders,
    move_batch,
    pairwise_inversions,
    resolve_path,
    set_seed,
    timestamp,
)


SEEDS = (0, 42, 3407)
MODULE_CATEGORIES = (
    "image_encoder",
    "text_encoder",
    "bio_encoder",
    "image_projector",
    "text_projector",
    "bio_projector",
    "temporal_path",
    "patient_projection",
    "classifier",
)
LR_SCALES = {
    "image_encoder": "encoder_lr_scale",
    "text_encoder": "encoder_lr_scale",
    "bio_encoder": "encoder_lr_scale",
    "image_projector": "projector_lr_scale",
    "text_projector": "projector_lr_scale",
    "bio_projector": "projector_lr_scale",
    "temporal_path": "prediction_path_lr_scale",
    "patient_projection": "prediction_path_lr_scale",
    "classifier": "prediction_path_lr_scale",
}
SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
    "raw_n_visits",
    "raw_n_images",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c37_e2e_vrl_multiseed.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validation-seed",
            "validation-finalize",
            "reporting-test",
            "direct-multiseed",
        ),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def parameter_category(name: str) -> str | None:
    if name.startswith("frozen_sources.image_encoder."):
        return "image_encoder"
    if name.startswith("frozen_sources.text_encoder."):
        return "text_encoder"
    if name.startswith("frozen_sources.bio_encoder."):
        return "bio_encoder"
    if name.startswith("frozen_sources.image_projector."):
        return "image_projector"
    if name.startswith("frozen_sources.text_projector."):
        return "text_projector"
    if name.startswith("frozen_sources.bio_projector."):
        return "bio_projector"
    if name.startswith("core.empty_slot_tokens") or name.startswith("core.temporal_"):
        return "temporal_path"
    if name.startswith("core.patient_projection."):
        return "patient_projection"
    if name.startswith("core.classifier."):
        return "classifier"
    return None


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid C27 checkpoint payload: {path}")
    return payload


def init_from_c27(config: Dict[str, Any], seed: int) -> tuple[C27VTMEModel, Mapping[str, Any], str]:
    checkpoint_path = Path(
        str(config["c27"]["c27_checkpoint"]).replace("{seed}", str(seed))
    )
    payload = checkpoint_payload(checkpoint_path)
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"C27 checkpoint seed mismatch for {seed}: {checkpoint_path}")
    model = C27VTMEModel(config, seed)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload, str(checkpoint_path)


def trainable_gradient_norms(model: C27VTMEModel) -> Dict[str, float]:
    result = {category: 0.0 for category in MODULE_CATEGORIES}
    for name, parameter in model.named_parameters():
        category = parameter_category(name)
        if category is None or parameter.grad is None:
            continue
        result[category] += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return {category: float(np.sqrt(value)) for category, value in result.items()}


def parameter_drift_rows(
    model: C27VTMEModel, initial_state: Mapping[str, torch.Tensor]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name)
        if category is None:
            raise RuntimeError(f"Unknown C37 trainable parameter: {name}")
        value = parameter.detach().cpu()
        baseline = initial_state[name]
        denominator = max(float(torch.linalg.vector_norm(baseline)), 1e-8)
        relative = float(torch.linalg.vector_norm(value - baseline)) / denominator
        rows.append(
            {
                "seed": model.seed,
                "category": category,
                "parameter_name": name,
                "parameter_count": int(value.numel()),
                "relative_parameter_drift": relative,
                "finite": bool(np.isfinite(relative)),
            }
        )
    return rows


def build_optimizer(
    config: Dict[str, Any], model: C27VTMEModel
) -> torch.optim.Optimizer:
    base_lr = float(config["training"]["lr"])
    groups: List[Dict[str, Any]] = []
    for category in MODULE_CATEGORIES:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter_category(name) == category
        ]
        if not parameters:
            raise RuntimeError(f"C37 optimizer group has no parameters: {category}")
        scale = float(config["c37"][LR_SCALES[category]])
        groups.append(
            {
                "params": parameters,
                "lr": base_lr * scale,
                "category": category,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=float(config["training"]["weight_decay"]))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1)) if len(finite) > 1 else 0.0


def run_epoch(
    model: C27VTMEModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    predictions: List[Dict[str, Any]] = []
    patient_states: List[np.ndarray] = []
    mechanism_states: List[np.ndarray] = []
    temporal_latest: List[np.ndarray] = []
    conflict_states: List[np.ndarray] = []
    loss_values: List[float] = []
    gradient_values: Dict[str, List[float]] = {
        category: [] for category in MODULE_CATEGORIES
    }

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C37 non-finite BCE loss")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = trainable_gradient_norms(model)
                for category, value in norms.items():
                    gradient_values[category].append(value)
                optimizer.step()
        loss_values.append(float(loss.detach().cpu()))

        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if torch.is_tensor(value)
        }
        labels = batch["label"].detach().cpu().numpy().astype(int)
        visit_mask = batch["visit_mask"].detach().cpu().numpy().astype(bool)
        support = batch["visit_support_present"].detach().cpu().numpy().astype(bool)
        opposition = batch["visit_opposition_present"].detach().cpu().numpy().astype(bool)
        image_mask = batch["image_mask"].detach().cpu().numpy().astype(bool)
        text_valid = batch["visit_text_valid"].detach().cpu().numpy().astype(bool)
        patient_states.append(arrays["patient_state"])
        mechanism_states.append(arrays["mechanism_states"])
        temporal_latest.append(arrays["temporal_latest_weights"])
        conflict_states.append(arrays["conflicts"])

        for index, patient_id in enumerate(batch["patient_id"]):
            count = int(visit_mask[index].sum())
            latest_index = max(count - 1, 0)
            weights = arrays["temporal_weights"][index, :count]
            patient_state = arrays["patient_state"][index]
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "visit_count_audit_only": count,
                "reconstructable_visit_count_audit_only": int(
                    batch["shortcuts"][index]["reconstructable_visit_count"]
                ),
                "visit_report_coverage_audit_only": float(
                    batch["shortcuts"][index]["visit_report_coverage"]
                ),
                "latest_visit_rank": latest_index,
                "latest_visit_has_image": bool(image_mask[index, latest_index].any()),
                "latest_visit_has_text": bool(text_valid[index, latest_index]),
                "latest_visit_has_dated_bio": bool(
                    batch["visit_dated_bio_present"][index][latest_index]
                ),
                "mean_temporal_weight_latest": float(
                    arrays["temporal_latest_weights"][index].mean()
                ),
                "mean_temporal_weight_history": float(
                    1.0 - arrays["temporal_latest_weights"][index].mean()
                ),
                "mean_temporal_weight_entropy": float(
                    arrays["temporal_entropy"][index].mean()
                ),
                "mean_normalized_temporal_entropy": float(
                    arrays["temporal_normalized_entropy"][index].mean()
                ),
                "fraction_latest_weight_above_0_90": float(
                    (arrays["temporal_latest_weights"][index] > 0.90).mean()
                ),
                "fraction_uniform_temporal_weight": float(
                    np.mean((weights.max(axis=0) - weights.min(axis=0)) < 1e-3)
                    if count > 1
                    else 0.0
                ),
                "patient_state_norm": float(np.linalg.norm(patient_state)),
                "patient_state_component_std": float(np.std(patient_state)),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "same_visit_image_text_cosine": float(
                    arrays["same_visit_alignment_mean"][index]
                ),
                "cross_visit_image_text_cosine": float(
                    arrays["cross_visit_alignment_mean"][index]
                ),
                "latest_same_visit_alignment": float(
                    arrays["latest_same_visit_alignment"][index]
                ),
                "history_same_visit_alignment": float(
                    arrays["history_same_visit_alignment"][index]
                ),
                "same_visit_alignment_count": int(
                    arrays["same_visit_alignment_count"][index]
                ),
                "cross_visit_alignment_pair_count": int(
                    arrays["cross_visit_alignment_pair_count"][index]
                ),
            }
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                row[f"temporal_weight_latest_{mechanism}"] = float(
                    arrays["temporal_latest_weights"][index, mechanism_index]
                )
                row[f"conflict_{mechanism}"] = float(
                    arrays["conflicts"][index, mechanism_index]
                )
                row[f"history_available_{mechanism}"] = bool(
                    arrays["history_available"][index, mechanism_index]
                )
                row[f"H_{mechanism}_norm"] = float(
                    np.linalg.norm(arrays["mechanism_states"][index, mechanism_index])
                )
            for field in SHORTCUT_FIELDS:
                row[field] = batch["shortcuts"][index].get(field, float("nan"))
            predictions.append(row)

    frame = pd.DataFrame(predictions)
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = dict(binary_metrics(labels, probabilities))
    metrics.update(
        {
            "bce_loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "positive_probability_mean": float(probabilities[labels == 1].mean()),
            "negative_probability_mean": float(probabilities[labels == 0].mean()),
            "positive_negative_gap": float(
                probabilities[labels == 1].mean() - probabilities[labels == 0].mean()
            ),
            "mean_temporal_weight_latest": float(
                frame["mean_temporal_weight_latest"].mean()
            ),
            "mean_temporal_weight_history": float(
                frame["mean_temporal_weight_history"].mean()
            ),
            "mean_temporal_weight_entropy": float(
                frame["mean_temporal_weight_entropy"].mean()
            ),
            "mean_normalized_temporal_entropy": float(
                frame["mean_normalized_temporal_entropy"].mean()
            ),
            "fraction_latest_weight_above_0_90": float(
                frame["fraction_latest_weight_above_0_90"].mean()
            ),
            "fraction_uniform_temporal_weight": float(
                frame["fraction_uniform_temporal_weight"].mean()
            ),
            "mean_patient_state_norm": float(frame["patient_state_norm"].mean()),
            "std_patient_state_norm": float(frame["patient_state_norm"].std(ddof=1)),
            "patient_state_component_std": float(
                np.concatenate(patient_states, axis=0).std()
            ),
            "prediction_std": float(frame["final_prob"].std(ddof=1)),
            "pairwise_inversion_count": pairwise_inversions(labels, probabilities),
            "n_rows": int(len(frame)),
        }
    )
    for mechanism in MECHANISM_NAMES:
        metrics[f"mean_conflict_{mechanism}"] = float(
            frame[f"conflict_{mechanism}"].mean()
        )
    for category in MODULE_CATEGORIES:
        values = gradient_values[category]
        metrics[f"{category}_grad_norm"] = float(np.mean(values)) if values else 0.0
    return {
        "metrics": metrics,
        "predictions": predictions,
        "patient_states": np.concatenate(patient_states, axis=0),
        "mechanism_states": np.concatenate(mechanism_states, axis=0),
        "temporal_latest": np.concatenate(temporal_latest, axis=0),
        "conflicts": np.concatenate(conflict_states, axis=0),
        "patient_diagnostics": frame,
    }


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model, init_payload, source_checkpoint = init_from_c27(config, seed)
    model = model.to(device)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(parameter_category(name) is None for name, _ in trainable):
        raise RuntimeError(f"C37 trainable scope violation: {[name for name, _ in trainable]}")
    count = trainable_parameter_count(model)
    if count > int(config["c37"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C37_CAPACITY_CONTRACT_FAIL: {count}")
    initial_state = {
        name: parameter.detach().cpu().clone() for name, parameter in trainable
    }
    optimizer = build_optimizer(config, model)
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device)
        val_result = run_epoch(model, loaders["val"], None, device)
        drift_frame = pd.DataFrame(parameter_drift_rows(model, initial_state))
        drift_summary = {
            category: float(
                drift_frame.loc[
                    drift_frame["category"] == category,
                    "relative_parameter_drift",
                ].mean()
            )
            for category in MODULE_CATEGORIES
        }
        row: Dict[str, Any] = {
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": train_result["metrics"]["bce_loss"],
            "val_auc": val_result["metrics"]["AUC"],
            "val_sensitivity": val_result["metrics"]["Sensitivity"],
            "val_specificity": val_result["metrics"]["Specificity"],
            "val_balanced_accuracy": val_result["metrics"]["Balanced_ACC"],
            "val_positive_probability_mean": val_result["metrics"]["positive_probability_mean"],
            "val_negative_probability_mean": val_result["metrics"]["negative_probability_mean"],
            "val_positive_negative_gap": val_result["metrics"]["positive_negative_gap"],
            "pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
            "mean_temporal_weight_latest": val_result["metrics"]["mean_temporal_weight_latest"],
            "mean_temporal_weight_history": val_result["metrics"]["mean_temporal_weight_history"],
            "mean_temporal_weight_entropy": val_result["metrics"]["mean_temporal_weight_entropy"],
            "mean_patient_state_norm": val_result["metrics"]["mean_patient_state_norm"],
            "std_patient_state_norm": val_result["metrics"]["std_patient_state_norm"],
            "prediction_std": val_result["metrics"]["prediction_std"],
            "selected_by_val_auc": False,
        }
        for category in MODULE_CATEGORIES:
            row[f"{category}_grad_norm"] = train_result["metrics"][
                f"{category}_grad_norm"
            ]
            row[f"{category}_relative_drift"] = drift_summary[category]
        epoch_rows.append(row)
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"C37 seed {seed} produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C37 seed {seed} produced constant validation predictions")
    checkpoint_path = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "init_c27_checkpoint": source_checkpoint,
            "init_c27_best_epoch": int(init_payload.get("best_epoch", -1)),
            "selection_metric": "validation_auc_only",
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "drift": parameter_drift_rows(model, initial_state),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": count,
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "source_c27_checkpoint": source_checkpoint,
        "init_c27_best_epoch": int(init_payload.get("best_epoch", -1)),
    }


def save_split(result: Dict[str, Any], out_dir: Path, split: str) -> Dict[str, Any]:
    seed = int(result["seed"])
    split_result = result[split]
    original_ids = np.asarray([str(row["patient_id"]) for row in split_result["predictions"]])
    order = np.argsort(original_ids)
    frame = (
        pd.DataFrame(split_result["predictions"])
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    frame.insert(0, "split", split)
    frame.insert(0, "seed", seed)
    frame.to_csv(
        out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False
    )
    np.savez_compressed(
        out_dir / "representations" / f"{split}_patient_state_seed_{seed}.npz",
        patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
        label=frame["label"].to_numpy(dtype=np.int64),
        patient_state=split_result["patient_states"][order].astype(np.float32),
        mechanism_states=split_result["mechanism_states"][order].astype(np.float32),
        temporal_latest=split_result["temporal_latest"][order].astype(np.float32),
        conflicts=split_result["conflicts"][order].astype(np.float32),
    )
    return {
        "seed": seed,
        "split": split,
        "best_epoch": int(result["best_epoch"]),
        **split_result["metrics"],
    }


def write_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for split, frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "prediction_std"):
            values = frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)


def validation_seed_stage(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    if seed_dir.exists():
        raise RuntimeError(f"C37 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C37-E2E-VRL",
        "stage": "validation-seed",
        "status": "RUNNING",
        "seed": seed,
        "started_at": timestamp(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    result = train_seed(config, rows, seed, seed_dir, device)
    metric = save_split(result, seed_dir, "val")
    pd.DataFrame([metric]).to_csv(seed_dir / "reports" / "metrics.csv", index=False)
    pd.DataFrame(result["epoch_history"]).to_csv(
        seed_dir / "reports" / "metrics_by_epoch.csv", index=False
    )
    pd.DataFrame(result["drift"]).to_csv(
        seed_dir / "reports" / "parameter_drift.csv", index=False
    )
    result["val"]["patient_diagnostics"].assign(seed=seed).to_csv(
        seed_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    runtime = {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "source_c27_checkpoint": result["source_c27_checkpoint"],
        "init_c27_best_epoch": result["init_c27_best_epoch"],
        "trainable_parameter_names": result["trainable_parameter_names"],
        "trainable_parameter_count": int(result["trainable_parameter_count"]),
        "frozen_parameter_count": int(result["frozen_parameter_count"]),
        "learning_rate_scales": {
            "encoders": float(config["c37"]["encoder_lr_scale"]),
            "projectors": float(config["c37"]["projector_lr_scale"]),
            "prediction_path": float(config["c37"]["prediction_path_lr_scale"]),
        },
        "selection_metric": "validation_auc_only",
    }
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C37_VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(
    config: Dict[str, Any], out_dir: Path, device: torch.device
) -> None:
    metrics_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    drift_parts: List[pd.DataFrame] = []
    diagnostic_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    runtime_by_seed: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads(
            (seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C37 seed {seed} validation shard incomplete")
        metrics_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        drift_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_drift.csv"))
        diagnostic_parts.append(
            pd.read_csv(seed_dir / "reports" / "patient_diagnostics_val.csv")
        )
        runtime_by_seed[str(seed)] = json.loads(
            (seed_dir / "reports" / "run_config.json").read_text(encoding="utf-8")
        )
        statuses.append(status)
        for source, target in (
            (
                seed_dir / "checkpoints" / f"seed_{seed}_best.pt",
                out_dir / "checkpoints" / f"seed_{seed}_best.pt",
            ),
            (
                seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv",
                out_dir / "predictions" / f"val_predictions_seed_{seed}.csv",
            ),
            (
                seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz",
                out_dir / "representations" / f"val_patient_state_seed_{seed}.npz",
            ),
        ):
            shutil.copy2(source, target)
    metrics = pd.concat(metrics_parts, ignore_index=True).sort_values("seed")
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        out_dir / "reports" / "metrics_by_epoch.csv", index=False
    )
    pd.concat(drift_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "parameter_drift.csv", index=False
    )
    pd.concat(diagnostic_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    write_summary(metrics, out_dir)
    status = {
        "phase": "C37-E2E-VRL",
        "status": "VALIDATION_COMPLETE",
        "started_at": min(str(item["started_at"]) for item in statuses),
        "validation_finished_at": timestamp(),
        "completed_seeds": list(SEEDS),
        "parallel_seed_training": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    (out_dir / "reports" / "run_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "runtime_by_seed": runtime_by_seed,
                "selection_metric": "validation_AUC_only",
                "test_role": "reporting_only_after_validation_decision",
                "deployment_contract": "one_checkpoint_one_model_one_forward",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "C37_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c37_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C37 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not bool(decision.get("validation_decision_frozen_before_test", False))
        or bool(decision.get("test_used_for_decision", True))
        or bool(decision.get("ensemble_used", True))
    ):
        raise RuntimeError("C37 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C37 reporting-only test requires validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model, _, _ = init_from_c27(config, seed)
        model = model.to(device)
        checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        payload = checkpoint_payload(checkpoint_path)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C37 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = run_epoch(model, loader, None, device)
        metric = save_split(
            {"seed": seed, "best_epoch": int(payload["best_epoch"]), "test": result},
            out_dir,
            "test",
        )
        metrics = pd.concat([metrics, pd.DataFrame([metric])], ignore_index=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "status": "COMPLETE",
            "test_started_after_validation_decision": True,
            "finished_at": timestamp(),
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C37_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c37_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C37 direct execution requires the completed 16-check gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C37_E2E_VRL_DIRECT_MULTI_SEED_AUTHORIZED" or int(
        gate.get("passed", 0)
    ) != 16:
        raise RuntimeError("C37 direct execution requires an authorized 16/16 gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C37 formal seed outputs already exist")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--config",
                str(config_path),
                "--stage",
                "validation-seed",
                "--seed",
                str(seed),
            ]
        )
        for seed in SEEDS
    ]
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"C37 formal validation seed failure codes: {failures}")
    validation_finalize_stage(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c37_report.py"
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"],
        check=True,
    )
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "final"],
        check=True,
    )
    print(json.dumps({"status": "C37_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c37":
        raise RuntimeError("C37 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C37 formal seeds must remain [0, 42, 3407]")
    if not bool(config["loss"]["bce_only"]):
        raise RuntimeError("C37 requires BCE-only training")
    if not bool(config["c37"]["init_from_c27"]) or not bool(config["c37"]["train_e2e"]):
        raise RuntimeError("C37 requires E2E initialization from C27")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"--seed must be one of {SEEDS}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
