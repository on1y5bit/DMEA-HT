#!/usr/bin/env python3
"""Train C36-JTSA as three independent direct validation-selected seeds."""

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

from dmea_ht.c36_jtsa import (  # noqa: E402
    MECHANISM_LABELS,
    MECHANISM_NAMES,
    TRAINABLE_MODULES,
    C36JTSAModel,
    named_trainable_parameters,
    trainable_parameter_count,
)
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
    parser.add_argument("--config", default="configs/dema_ht_c36_jtsa_multiseed.yaml")
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


def _prefix_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def trainable_gradient_norms(model: C36JTSAModel) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for category, prefix in TRAINABLE_MODULES.items():
        squared = 0.0
        for name, parameter in model.named_parameters():
            if _prefix_matches(name, prefix) and parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
        result[category] = float(np.sqrt(squared))
    return result


def trainable_drift_summary(model: C36JTSAModel) -> Dict[str, float]:
    frame = pd.DataFrame(model.parameter_drift_rows())
    result: Dict[str, float] = {}
    for category in TRAINABLE_MODULES:
        values = frame.loc[
            frame["category"] == category, "relative_parameter_drift"
        ].to_numpy(dtype=float)
        result[category] = float(values.mean()) if len(values) else float("nan")
    return result


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1)) if len(finite) > 1 else 0.0


def run_epoch(
    model: C36JTSAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    prediction_rows: List[Dict[str, Any]] = []
    embedding_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    gradient_values: Dict[str, List[float]] = {
        category: [] for category in TRAINABLE_MODULES
    }
    anchor_distances: List[float] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C36 non-finite BCE loss")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = trainable_gradient_norms(model)
                for category, value in norms.items():
                    gradient_values[category].append(value)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))

        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if torch.is_tensor(value)
        }
        labels = batch["label"].detach().cpu().numpy().astype(int)
        visit_mask = batch["visit_mask"].detach().cpu().numpy().astype(bool)
        source_valid = arrays["mechanism_source_valid"].astype(bool)
        source_states = arrays["mechanism_source_states"]
        visit_states = arrays["mechanism_visit_state"]
        embeddings = arrays["mechanism_embedding"]
        patient_states = arrays["patient_disease_state"]
        anchor_distances.append(float(arrays["anchor_distance"]))

        for index, patient_id in enumerate(batch["patient_id"]):
            count = int(visit_mask[index].sum())
            patient_state = patient_states[index]
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "visit_count_audit_only": count,
                "d_non_ht": float(arrays["d_non_ht"][index]),
                "d_ht": float(arrays["d_ht"][index]),
                "state_margin": float(arrays["state_margin"][index]),
                "anchor_cosine": float(arrays["anchor_cosine"][index]),
                "patient_state_norm": float(arrays["patient_state_norm"][index]),
            }
            row.update(
                {f"patient_state_{component}": float(value) for component, value in enumerate(patient_state)}
            )
            for field in SHORTCUT_FIELDS:
                row[field] = batch["shortcuts"][index].get(field, float("nan"))
            prediction_rows.append(row)

            anchor_rows.append(
                {
                    "seed": model.seed,
                    "patient_id": str(patient_id),
                    "label": int(labels[index]),
                    "d_non_ht": float(arrays["d_non_ht"][index]),
                    "d_ht": float(arrays["d_ht"][index]),
                    "state_margin": float(arrays["state_margin"][index]),
                    "anchor_cosine": float(arrays["anchor_cosine"][index]),
                    "patient_state_norm": float(arrays["patient_state_norm"][index]),
                    **{
                        f"patient_state_{component}": float(value)
                        for component, value in enumerate(patient_state)
                    },
                }
            )
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                valid = source_valid[index, :count, mechanism_index]
                source_values = source_states[index, :count, mechanism_index]
                visit_values = visit_states[index, :count, mechanism_index]
                embedding = embeddings[index, mechanism_index]
                embedding_rows.append(
                    {
                        "seed": model.seed,
                        "patient_id": str(patient_id),
                        "label": int(labels[index]),
                        "mechanism": mechanism,
                        "mechanism_label": MECHANISM_LABELS[mechanism_index],
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "embedding_component_std": float(np.std(embedding)),
                        "embedding_component_mean": float(np.mean(embedding)),
                        **{
                            f"embedding_{component}": float(value)
                            for component, value in enumerate(embedding)
                        },
                        "source_state_norm_mean": float(np.linalg.norm(source_values, axis=-1).mean())
                        if count
                        else 0.0,
                        "source_state_norm_std": float(np.linalg.norm(source_values, axis=-1).std(ddof=1))
                        if count > 1
                        else 0.0,
                        "visit_state_norm_mean": float(np.linalg.norm(visit_values, axis=-1).mean())
                        if count
                        else 0.0,
                        "visit_state_norm_std": float(np.linalg.norm(visit_values, axis=-1).std(ddof=1))
                        if count > 1
                        else 0.0,
                        "valid_visit_count": int(valid.sum()),
                        "valid_visit_fraction": float(valid.mean()) if count else 0.0,
                        "observed": bool(valid.any()),
                    }
                )
                trajectory_rows.append(
                    {
                        "seed": model.seed,
                        "patient_id": str(patient_id),
                        "label": int(labels[index]),
                        "mechanism": mechanism,
                        "single_visit": count == 1,
                        "latest_norm": float(
                            np.linalg.norm(arrays["latest_mechanism_state"][index, mechanism_index])
                        ),
                        "history_norm": float(
                            np.linalg.norm(arrays["history_mechanism_state"][index, mechanism_index])
                        ),
                        "delta_norm": float(
                            np.linalg.norm(arrays["mechanism_state_delta"][index, mechanism_index])
                        ),
                    }
                )

    frame = pd.DataFrame(prediction_rows)
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = dict(binary_metrics(labels, probabilities))
    metrics.update(
        {
            "bce_loss": float(np.mean(losses)) if losses else 0.0,
            "positive_probability_mean": float(probabilities[labels == 1].mean()),
            "negative_probability_mean": float(probabilities[labels == 0].mean()),
            "positive_negative_gap": float(
                probabilities[labels == 1].mean() - probabilities[labels == 0].mean()
            ),
            "pairwise_inversion_count": pairwise_inversions(labels, probabilities),
            "prediction_std": float(frame["final_prob"].std(ddof=1)),
            "patient_state_norm_mean": float(frame["patient_state_norm"].mean()),
            "patient_state_norm_std": float(frame["patient_state_norm"].std(ddof=1)),
            "patient_state_component_std": float(
                frame[[f"patient_state_{i}" for i in range(32)]].to_numpy(dtype=float).std()
            ),
            "anchor_distance": float(np.mean(anchor_distances)),
            "d_non_ht_mean": float(frame["d_non_ht"].mean()),
            "d_ht_mean": float(frame["d_ht"].mean()),
            "state_margin_mean": float(frame["state_margin"].mean()),
            "state_margin_std": float(frame["state_margin"].std(ddof=1)),
            "anchor_cosine_mean": float(frame["anchor_cosine"].mean()),
            "n_rows": int(len(frame)),
        }
    )
    embedding_frame = pd.DataFrame(embedding_rows)
    for mechanism in MECHANISM_NAMES:
        selected = embedding_frame[embedding_frame["mechanism"] == mechanism]
        for field in ("embedding_norm", "embedding_component_std"):
            mean, std = _mean_std(selected[field].to_numpy(dtype=float))
            metrics[f"{mechanism}_{field}_mean"] = mean
            metrics[f"{mechanism}_{field}_std"] = std
    for category in TRAINABLE_MODULES:
        values = gradient_values[category]
        metrics[f"{category}_grad_norm"] = float(np.mean(values)) if values else 0.0
    diagnostics_frame = frame.copy()
    diagnostics_frame.insert(0, "seed", model.seed)
    return {
        "metrics": metrics,
        "predictions": prediction_rows,
        "mechanism_embedding_health": embedding_frame,
        "patient_state_anchor_audit": pd.DataFrame(anchor_rows),
        "trajectory_state_audit": pd.DataFrame(trajectory_rows),
        "patient_diagnostics": diagnostics_frame,
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
    model = C36JTSAModel(config, seed).to(device)
    trainable = list(named_trainable_parameters(model))
    if not trainable or any(
        not C36JTSAModel.is_trainable_parameter(name) for name, _ in trainable
    ):
        raise RuntimeError(f"C36 trainable scope violation: {[name for name, _ in trainable]}")
    parameter_count = trainable_parameter_count(model)
    if parameter_count > int(config["c36"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C36_CAPACITY_CONTRACT_FAIL: {parameter_count}")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device)
        val_result = run_epoch(model, loaders["val"], None, device)
        drift = trainable_drift_summary(model)
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
            "val_patient_state_norm_mean": val_result["metrics"]["patient_state_norm_mean"],
            "val_patient_state_norm_std": val_result["metrics"]["patient_state_norm_std"],
            "val_patient_state_component_std": val_result["metrics"]["patient_state_component_std"],
            "val_d_non_ht_mean": val_result["metrics"]["d_non_ht_mean"],
            "val_d_ht_mean": val_result["metrics"]["d_ht_mean"],
            "val_state_margin_mean": val_result["metrics"]["state_margin_mean"],
            "val_state_margin_std": val_result["metrics"]["state_margin_std"],
            "val_anchor_distance": val_result["metrics"]["anchor_distance"],
            "val_anchor_cosine_mean": val_result["metrics"]["anchor_cosine_mean"],
            "selected_by_val_auc": False,
        }
        for mechanism in MECHANISM_NAMES:
            for field in ("embedding_norm", "embedding_component_std"):
                row[f"val_{mechanism}_{field}_mean"] = val_result["metrics"][
                    f"{mechanism}_{field}_mean"
                ]
                row[f"val_{mechanism}_{field}_std"] = val_result["metrics"][
                    f"{mechanism}_{field}_std"
                ]
        for category in TRAINABLE_MODULES:
            row[f"{category}_grad_norm"] = train_result["metrics"][
                f"{category}_grad_norm"
            ]
            row[f"{category}_relative_drift"] = drift[category]
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
        raise RuntimeError(f"C36 seed {seed} produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C36 seed {seed} produced constant validation predictions")
    checkpoint_path = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    source_checkpoint = str(
        Path(str(config["c36"]["c17_checkpoint"]).replace("{seed}", str(seed)))
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "source_c17_checkpoint": source_checkpoint,
            "selection_metric": "validation_auc_only",
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "drift": model.parameter_drift_rows(),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": parameter_count,
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "source_c17_checkpoint": source_checkpoint,
    }


def save_split(result: Dict[str, Any], out_dir: Path, split: str) -> Dict[str, Any]:
    seed = int(result["seed"])
    split_result = result[split]
    frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id")
    frame.insert(0, "split", split)
    frame.insert(0, "seed", seed)
    frame.to_csv(
        out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False
    )
    return {"seed": seed, "split": split, "best_epoch": int(result["best_epoch"]), **split_result["metrics"]}


def write_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for split, frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC"):
            values = frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)


def validation_seed_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    if seed_dir.exists():
        raise RuntimeError(f"C36 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C36-JTSA",
        "stage": "validation-seed",
        "status": "RUNNING",
        "seed": seed,
        "started_at": timestamp(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    result = train_seed(config, rows, int(seed), seed_dir, device)
    metric = save_split(result, seed_dir, "val")
    pd.DataFrame([metric]).to_csv(seed_dir / "reports" / "metrics.csv", index=False)
    pd.DataFrame(result["epoch_history"]).to_csv(seed_dir / "reports" / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(result["drift"]).to_csv(seed_dir / "reports" / "parameter_drift.csv", index=False)
    result["val"]["mechanism_embedding_health"].to_csv(
        seed_dir / "reports" / "mechanism_embedding_health.csv", index=False
    )
    result["val"]["patient_state_anchor_audit"].to_csv(
        seed_dir / "reports" / "patient_state_anchor_audit.csv", index=False
    )
    result["val"]["trajectory_state_audit"].to_csv(
        seed_dir / "reports" / "trajectory_state_audit.csv", index=False
    )
    result["val"]["patient_diagnostics"].to_csv(
        seed_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    runtime = {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "source_c17_checkpoint": result["source_c17_checkpoint"],
        "trainable_parameter_names": result["trainable_parameter_names"],
        "trainable_parameter_count": int(result["trainable_parameter_count"]),
        "frozen_parameter_count": int(result["frozen_parameter_count"]),
        "selection_metric": "validation_auc_only",
    }
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C36_VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metrics_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    drift_parts: List[pd.DataFrame] = []
    embedding_parts: List[pd.DataFrame] = []
    anchor_parts: List[pd.DataFrame] = []
    trajectory_parts: List[pd.DataFrame] = []
    patient_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    runtime_by_seed: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads((seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C36 seed {seed} validation shard incomplete")
        metrics_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        drift_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_drift.csv"))
        embedding_parts.append(pd.read_csv(seed_dir / "reports" / "mechanism_embedding_health.csv"))
        anchor_parts.append(pd.read_csv(seed_dir / "reports" / "patient_state_anchor_audit.csv"))
        trajectory_parts.append(pd.read_csv(seed_dir / "reports" / "trajectory_state_audit.csv"))
        patient_parts.append(pd.read_csv(seed_dir / "reports" / "patient_diagnostics_val.csv"))
        runtime_by_seed[str(seed)] = json.loads(
            (seed_dir / "reports" / "run_config.json").read_text(encoding="utf-8")
        )
        statuses.append(status)
        for source, target in (
            (seed_dir / "checkpoints" / f"seed_{seed}_best.pt", out_dir / "checkpoints" / f"seed_{seed}_best.pt"),
            (seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv", out_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
        ):
            shutil.copy2(source, target)
    metrics = pd.concat(metrics_parts, ignore_index=True).sort_values("seed")
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        out_dir / "reports" / "metrics_by_epoch.csv", index=False
    )
    pd.concat(drift_parts, ignore_index=True).to_csv(out_dir / "reports" / "parameter_drift.csv", index=False)
    pd.concat(embedding_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "mechanism_embedding_health_val.csv", index=False
    )
    pd.concat(anchor_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "patient_state_anchor_audit_val.csv", index=False
    )
    pd.concat(trajectory_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "trajectory_state_audit_val.csv", index=False
    )
    pd.concat(patient_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    write_summary(metrics, out_dir)
    status = {
        "phase": "C36-JTSA",
        "status": "VALIDATION_COMPLETE",
        "started_at": min(str(item["started_at"]) for item in statuses),
        "validation_finished_at": timestamp(),
        "completed_seeds": list(SEEDS),
        "parallel_seed_training": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "reports" / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps({"status": "C36_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c36_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C36 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not bool(decision.get("validation_decision_frozen_before_test", False))
        or bool(decision.get("test_used_for_decision", True))
        or bool(decision.get("ensemble_used", True))
    ):
        raise RuntimeError("C36 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C36 reporting-only test requires validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C36JTSAModel(config, seed).to(device)
        checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C36 checkpoint seed mismatch for {seed}")
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
    print(json.dumps({"status": "C36_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c36_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C36 direct execution requires the completed 20-check gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C36_JTSA_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != 20:
        raise RuntimeError("C36 direct execution requires an authorized 20/20 gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C36 formal seed outputs already exist")
    for child in ("reports", "predictions", "checkpoints"):
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
        raise RuntimeError(f"C36 formal validation seed failure codes: {failures}")
    validation_finalize_stage(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c36_report.py"
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"],
        check=True,
    )
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "final"],
        check=True,
    )
    print(json.dumps({"status": "C36_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c36":
        raise RuntimeError("C36 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C36 formal seeds must remain [0, 42, 3407]")
    if not bool(config["loss"]["bce_only"]):
        raise RuntimeError("C36 requires BCE-only training")
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
