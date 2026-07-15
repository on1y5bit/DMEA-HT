#!/usr/bin/env python3
"""Train C38-MPES as direct, independent formal validation shards."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c38_mpes import (  # noqa: E402
    C38MPESModel,
    HEAD_PREFIXES,
    MECHANISM_NAMES,
    trainable_parameter_count,
    trainable_parameter_names,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import SHORTCUT_FIELDS  # noqa: E402
from dmea_ht.visit_data import (  # noqa: E402
    VisitPatientDataset,
    collate_visit_batch,
    read_jsonl,
)


SEEDS = (0, 42, 3407)
AUDIT_SHORTCUT_FIELDS = tuple(SHORTCUT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c38_mpes_multiseed.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation-seed", "validation-finalize", "reporting-test", "direct-multiseed"),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loaders(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], splits: Sequence[str]
) -> Dict[str, DataLoader]:
    project, model_cfg, training = config["project"], config["model"], config["training"]
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
        )
    return loaders


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    predicted = p >= 0.5
    positive = y == 1
    negative = y == 0
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((negative & ~predicted).sum())
    fp = int((negative & predicted).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_ACC": float(0.5 * (sensitivity + specificity)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def pairwise_inversions(labels: np.ndarray, probabilities: np.ndarray) -> int:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def gradient_norm(model: C38MPESModel) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith("sources.") or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return float(np.sqrt(total))


def parameter_drift_rows(
    model: C38MPESModel, initial_state: Mapping[str, torch.Tensor], seed: int
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        baseline = initial_state[name]
        value = parameter.detach().cpu()
        denominator = max(float(torch.linalg.vector_norm(baseline)), 1e-8)
        relative = float(torch.linalg.vector_norm(value - baseline)) / denominator
        rows.append(
            {
                "seed": seed,
                "parameter_name": name,
                "parameter_count": int(value.numel()),
                "relative_parameter_drift": relative,
                "finite": bool(np.isfinite(relative)),
            }
        )
    return rows


def run_epoch(
    model: C38MPESModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_values: List[float] = []
    predictions: List[Dict[str, Any]] = []
    patient_states: List[np.ndarray] = []
    mechanism_states: List[np.ndarray] = []
    gradient_values: List[float] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C38 non-finite BCE loss")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_values.append(gradient_norm(model))
                optimizer.step()
        loss_values.append(float(loss.detach().cpu()))

        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if torch.is_tensor(value)
        }
        labels = batch["label"].detach().cpu().numpy().astype(int)
        visit_mask = batch["visit_mask"].detach().cpu().numpy().astype(bool)
        patient_states.append(arrays["patient_state"])
        mechanism_states.append(arrays["mechanism_states"])
        for index, patient_id in enumerate(batch["patient_id"]):
            count = int(visit_mask[index].sum())
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "visit_count_audit_only": count,
                "latest_valid_mechanism_count": int(arrays["latest_valid"][index].sum()),
                "history_valid_mechanism_count": int(arrays["history_valid"][index].sum()),
                "patient_state_norm": float(np.linalg.norm(arrays["patient_state"][index])),
                "patient_state_component_std": float(np.std(arrays["patient_state"][index])),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "latest_set_entropy": float(arrays["latest_entropy"][index].mean()),
                "history_set_entropy": float(arrays["history_entropy"][index].mean()),
                "patient_mechanism_attention_entropy": float(
                    (-(arrays["patient_attention"][index].clip(1e-8, 1.0)
                      * np.log(arrays["patient_attention"][index].clip(1e-8, 1.0))).sum())
                ),
            }
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                row[f"mechanism_state_norm_{mechanism}"] = float(
                    np.linalg.norm(arrays["mechanism_states"][index, mechanism_index])
                )
                row[f"latest_valid_{mechanism}"] = bool(
                    arrays["latest_valid"][index, mechanism_index]
                )
                row[f"history_valid_{mechanism}"] = bool(
                    arrays["history_valid"][index, mechanism_index]
                )
            for field in AUDIT_SHORTCUT_FIELDS:
                row[field] = batch["shortcuts"][index].get(field, np.nan)
            predictions.append(row)

    frame = pd.DataFrame(predictions)
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = binary_metrics(labels, probabilities)
    metrics.update(
        {
            "bce_loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "positive_probability_mean": float(probabilities[labels == 1].mean()),
            "negative_probability_mean": float(probabilities[labels == 0].mean()),
            "positive_negative_gap": float(
                probabilities[labels == 1].mean() - probabilities[labels == 0].mean()
            ),
            "prediction_std": float(probabilities.std(ddof=1)),
            "mean_patient_state_norm": float(frame["patient_state_norm"].mean()),
            "std_patient_state_norm": float(frame["patient_state_norm"].std(ddof=1)),
            "patient_state_component_std": float(np.concatenate(patient_states, axis=0).std()),
            "mean_latest_set_entropy": float(frame["latest_set_entropy"].mean()),
            "mean_history_set_entropy": float(frame["history_set_entropy"].mean()),
            "mean_patient_mechanism_attention_entropy": float(
                frame["patient_mechanism_attention_entropy"].mean()
            ),
            "pairwise_inversion_count": pairwise_inversions(labels, probabilities),
            "n_rows": int(len(frame)),
            "head_grad_norm": float(np.mean(gradient_values)) if gradient_values else 0.0,
        }
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "patient_states": np.concatenate(patient_states, axis=0),
        "mechanism_states": np.concatenate(mechanism_states, axis=0),
        "patient_diagnostics": frame,
    }


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid C38 checkpoint payload: {path}")
    return payload


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model = C38MPESModel(config, seed).to(device)
    names = trainable_parameter_names(model)
    if not names or any(not name.startswith(HEAD_PREFIXES) for name in names):
        raise RuntimeError(f"C38 trainable scope violation: {names}")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("sources.")):
        raise RuntimeError("C38 source evidence must remain frozen")
    count = trainable_parameter_count(model)
    if count > int(config["c38"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C38 capacity contract failed: {count}")
    initial_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device)
        val_result = run_epoch(model, loaders["val"], None, device)
        drift = pd.DataFrame(parameter_drift_rows(model, initial_state, seed))
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_bce_loss": train_result["metrics"]["bce_loss"],
                "val_auc": val_result["metrics"]["AUC"],
                "val_sensitivity": val_result["metrics"]["Sensitivity"],
                "val_specificity": val_result["metrics"]["Specificity"],
                "val_balanced_accuracy": val_result["metrics"]["Balanced_ACC"],
                "val_prediction_std": val_result["metrics"]["prediction_std"],
                "val_pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
                "head_grad_norm": train_result["metrics"]["head_grad_norm"],
                "mean_relative_drift": float(drift["relative_parameter_drift"].mean()),
                "selected_by_val_auc": False,
            }
        )
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"C38 seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C38 seed {seed} produced constant validation predictions")
    checkpoint_path = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "source_c17_checkpoint": str(
                Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))
            ),
            "selection_metric": "validation_auc_only",
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "drift": parameter_drift_rows(model, initial_state, seed),
        "trainable_parameter_names": names,
        "trainable_parameter_count": count,
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
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
    frame.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
    np.savez_compressed(
        out_dir / "representations" / f"{split}_patient_state_seed_{seed}.npz",
        patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
        label=frame["label"].to_numpy(dtype=np.int64),
        patient_state=split_result["patient_states"][order].astype(np.float32),
        mechanism_states=split_result["mechanism_states"][order].astype(np.float32),
    )
    return {"seed": seed, "split": split, "best_epoch": int(result["best_epoch"]), **split_result["metrics"]}


def write_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for split, frame in metrics.groupby("split"):
        rows.append(
            {
                "split": split,
                "AUC_mean": float(frame["AUC"].mean()),
                "AUC_std": float(frame["AUC"].std(ddof=1)),
                "Sensitivity_mean": float(frame["Sensitivity"].mean()),
                "Specificity_mean": float(frame["Specificity"].mean()),
                "Balanced_ACC_mean": float(frame["Balanced_ACC"].mean()),
                "prediction_std_mean": float(frame["prediction_std"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)


def validation_seed_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    if seed_dir.exists():
        raise RuntimeError(f"C38 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C38-MPES",
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
    pd.DataFrame(result["epoch_history"]).to_csv(seed_dir / "reports" / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(result["drift"]).to_csv(seed_dir / "reports" / "parameter_drift.csv", index=False)
    result["val"]["patient_diagnostics"].assign(seed=seed).to_csv(
        seed_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": int(result["best_epoch"]),
                "trainable_parameter_names": result["trainable_parameter_names"],
                "trainable_parameter_count": int(result["trainable_parameter_count"]),
                "frozen_parameter_count": int(result["frozen_parameter_count"]),
                "selection_metric": "validation_AUC_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status.update({"status": "COMPLETE", "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C38_VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metric_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    drift_parts: List[pd.DataFrame] = []
    diagnostic_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads((seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C38 seed {seed} shard incomplete")
        statuses.append(status)
        metric_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        drift_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_drift.csv"))
        diagnostic_parts.append(pd.read_csv(seed_dir / "reports" / "patient_diagnostics_val.csv"))
        for source, target in (
            (seed_dir / "checkpoints" / f"seed_{seed}_best.pt", out_dir / "checkpoints" / f"seed_{seed}_best.pt"),
            (seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv", out_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
            (seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz", out_dir / "representations" / f"val_patient_state_seed_{seed}.npz"),
        ):
            shutil.copy2(source, target)
    metrics = pd.concat(metric_parts, ignore_index=True).sort_values("seed")
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        out_dir / "reports" / "metrics_by_epoch.csv", index=False
    )
    pd.concat(drift_parts, ignore_index=True).to_csv(out_dir / "reports" / "parameter_drift.csv", index=False)
    pd.concat(diagnostic_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    write_summary(metrics, out_dir)
    (out_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "seeds": list(SEEDS),
                "parallel_seed_training": True,
                "selection_metric": "validation_AUC_only",
                "test_role": "reporting_only_after_validation_decision",
                "deployment_contract": "one_checkpoint_one_model_one_forward",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "reports" / "run_status.json").write_text(
        json.dumps(
            {
                "phase": "C38-MPES",
                "status": "VALIDATION_COMPLETE",
                "started_at": min(item["started_at"] for item in statuses),
                "finished_at": timestamp(),
                "completed_seeds": list(SEEDS),
                "parallel_seed_training": True,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "C38_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c38_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C38 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not decision.get("validation_decision_frozen_before_test", False)
        or decision.get("test_used_for_decision", True)
        or decision.get("ensemble_used", True)
    ):
        raise RuntimeError("C38 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C38 reporting-only test requires validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C38MPESModel(config, seed).to(device)
        payload = checkpoint_payload(out_dir / "checkpoints" / f"seed_{seed}_best.pt")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C38 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = run_epoch(model, loader, None, device)
        metrics = pd.concat(
            [metrics, pd.DataFrame([save_split({"seed": seed, "best_epoch": payload["best_epoch"], "test": result}, out_dir, "test")])],
            ignore_index=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C38_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path, config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c38_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C38 direct execution requires the completed gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C38_MPES_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C38 direct execution requires an authorized gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C38 formal seed outputs already exist")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), "--config", str(config_path), "--stage", "validation-seed", "--seed", str(seed)]
        )
        for seed in SEEDS
    ]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"C38 validation shard failed: {codes}")
    subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--stage", "validation-finalize"],
        check=True,
    )
    collector = REPO_ROOT / "scripts" / "collect_phase_c38_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True)
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True)
    print(json.dumps({"status": "C38_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c38":
        raise RuntimeError("C38 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C38 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C38 seed: {args.seed}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
