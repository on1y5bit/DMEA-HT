#!/usr/bin/env python3
"""Train the formal C26-SM stable mechanism mixer on three independent seeds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c26sm_stable_mechanism_mixer import (  # noqa: E402
    MECHANISM_NAMES,
    RELATION_NAMES,
    C26SMStableMechanismModel,
    c26sm_loss_terms,
    propagation_capacity,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--stage",
        choices=("validation", "validation-seed", "validation-finalize", "reporting-test"),
        default="validation",
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loaders(config: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, DataLoader]:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = PatientHTDataset(
            rows=rows,
            data_root=project["data_root"],
            split=split,
            max_images=int(model_cfg["max_images_per_patient"]),
            image_size=int(model_cfg["image_size"]),
            text_max_length=int(model_cfg["text_max_length"]),
            text_vocab_size=int(model_cfg["text_vocab_size"]),
            bio_dim=int(model_cfg["bio_dim"]),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate_patient_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def binary_metrics(labels: Iterable[int], probs: Iterable[float]) -> Dict[str, float | int]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probs), dtype=float)
    pred = p >= 0.5
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_ACC": float(0.5 * (sensitivity + specificity)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def pairwise_inversions(labels: np.ndarray, probs: np.ndarray) -> int:
    positive, negative = probs[labels == 1], probs[labels == 0]
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def run_epoch(
    model: C26SMStableMechanismModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    predictions: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, float]] = []
    representations: List[np.ndarray] = []
    relation_gate_rows: List[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            if is_train:
                terms = c26sm_loss_terms(outputs, batch, loss_cfg)
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                optimizer.step()
                loss_rows.append({key: float(value.detach().cpu()) for key, value in terms.items()})
        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items() if torch.is_tensor(value)}
        labels = batch["label"].detach().cpu().numpy().astype(int)
        representations.append(arrays["mechanism_state"])
        relation_gate_rows.append(arrays["relation_gates"])
        for index, patient_id in enumerate(batch["patient_id"]):
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "base_logit": float(arrays["base_logit"][index]),
                "base_prob": float(arrays["base_prob"][index]),
                "raw_delta": float(arrays["raw_delta"][index]),
                "delta_logit": float(arrays["delta_logit"][index]),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "mechanism_core_norm": float(arrays["mechanism_core_norm"][index]),
                "context_norm": float(arrays["context_norm"][index]),
                "mechanism_final_norm": float(arrays["mechanism_final_norm"][index]),
            }
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                row[f"{mechanism}_norm"] = float(arrays["mechanism_node_norms"][index, mechanism_index])
                row[f"{mechanism}_weight"] = float(arrays["mechanism_node_weights"][index, mechanism_index])
                row[f"valid_source_count_{mechanism}"] = float(arrays["valid_source_counts"][index, mechanism_index])
                row[f"empty_slot_{mechanism}"] = bool(arrays["empty_slot_mask"][index, mechanism_index])
            row.update(batch["shortcuts"][index])
            predictions.append(row)

    frame = pd.DataFrame(predictions)
    y = frame["label"].to_numpy(dtype=int)
    p = frame["final_prob"].to_numpy(dtype=float)
    delta = frame["delta_logit"].to_numpy(dtype=float)
    positive, negative = y == 1, y == 0
    weights = frame[[f"{name}_weight" for name in MECHANISM_NAMES]].to_numpy(dtype=float)
    entropy = -(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    relation_gates = np.stack(relation_gate_rows).mean(axis=0)
    metrics: Dict[str, Any] = dict(binary_metrics(y, p))
    for key in loss_rows[0] if loss_rows else ():
        metrics[f"{key}_loss"] = float(np.mean([row[key] for row in loss_rows]))
    metrics.update({
        "positive_probability_mean": _mean(p[positive]),
        "negative_probability_mean": _mean(p[negative]),
        "positive_negative_gap": _mean(p[positive]) - _mean(p[negative]),
        "mean_delta_c26sm": _mean(delta),
        "std_delta_c26sm": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
        "mean_positive_delta_c26sm": _mean(delta[positive]),
        "mean_negative_delta_c26sm": _mean(delta[negative]),
        "fraction_positive_delta_below_minus_0_10": _mean((delta[positive] < -0.10).astype(float)),
        "fraction_delta_near_positive_bound": _mean((delta >= 0.495).astype(float)),
        "fraction_delta_near_negative_bound": _mean((delta <= -0.495).astype(float)),
        "node_weight_entropy": _mean(entropy),
        "fraction_node_weight_max_above_0_90": _mean((weights.max(axis=1) > 0.90).astype(float)),
        "mechanism_core_norm": float(frame["mechanism_core_norm"].mean()),
        "mechanism_final_norm": float(frame["mechanism_final_norm"].mean()),
        "pairwise_inversion_count": pairwise_inversions(y, p),
        "n_rows": int(len(frame)),
    })
    for index, name in enumerate(RELATION_NAMES):
        metrics[f"relation_gate_{name}"] = float(relation_gates[index])
    for name in MECHANISM_NAMES:
        metrics[f"mean_node_weight_{name}"] = float(frame[f"{name}_weight"].mean())
    return {
        "metrics": metrics,
        "predictions": predictions,
        "representations": np.concatenate(representations, axis=0),
    }


def train_seed(
    config: Dict[str, Any], rows: List[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, [dict(row) for row in rows])
    model = C26SMStableMechanismModel(config, seed).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    allowed_prefixes = ("mixer.", "residual_mlp.")
    if not trainable or any(not name.startswith(allowed_prefixes) for name, _ in trainable):
        raise RuntimeError(f"C26-SM trainable scope violation: {[name for name, _ in trainable]}")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device, config["loss"])
        val_result = run_epoch(model, loaders["val"], None, device, config["loss"])
        row: Dict[str, Any] = {"seed": seed, "epoch": epoch, "selected_by_val_auc": False}
        row.update({f"train_{key}": value for key, value in train_result["metrics"].items() if key.endswith("_loss")})
        row.update({f"val_{key}": value for key, value in val_result["metrics"].items()})
        epoch_rows.append(row)
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"C26-SM seed {seed} produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device, config["loss"])
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch},
        checkpoint_dir / f"seed_{seed}_best.pt",
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "trainable_parameter_names": [name for name, _ in trainable],
        "capacity": propagation_capacity(model),
    }


def save_validation_seed(result: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    seed = int(result["seed"])
    split_result = result["val"]
    frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id").reset_index(drop=True)
    frame.insert(0, "split", "val")
    frame.insert(0, "seed", seed)
    frame.to_csv(out_dir / "predictions" / f"val_predictions_seed_{seed}.csv", index=False)
    order = np.argsort(np.asarray([str(row["patient_id"]) for row in split_result["predictions"]]))
    np.savez_compressed(
        out_dir / "representations" / f"val_mechanism_state_seed_{seed}.npz",
        patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
        label=frame["label"].to_numpy(dtype=np.int64),
        mechanism_state=split_result["representations"][order].astype(np.float32),
    )
    metrics_row = {
        "seed": seed,
        "split": "val",
        "best_epoch": int(result["best_epoch"]),
        **split_result["metrics"],
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "reports" / f"metrics_seed_{seed}.csv", index=False)
    pd.DataFrame(result["epoch_history"]).to_csv(
        out_dir / "reports" / f"metrics_by_epoch_seed_{seed}.csv", index=False
    )
    return metrics_row


def write_metrics_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    summary_rows = []
    for split, split_frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "mean_delta_c26sm", "std_delta_c26sm"):
            values = split_frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c26sm":
        raise RuntimeError("C26-SM phase contract is missing")
    for key, value in (("data_root", args.data_root), ("manifest", args.manifest), ("output_dir", args.output_dir)):
        if value:
            config["project"][key] = value
    rows = read_manifest(config["project"]["manifest"])
    if not all(str(row.get("split", "")).strip() for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split
    out_dir = resolve_path(config["project"]["output_dir"])
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    if args.stage == "validation-seed":
        if args.seed not in seeds:
            raise RuntimeError(f"--seed must be one of the formal C26-SM seeds: {seeds}")
        seed = int(args.seed)
        started = timestamp()
        shard_status_path = out_dir / "reports" / f"run_status_seed_{seed}.json"
        shard_status = {
            "phase": "C26-SM",
            "stage": "validation-seed",
            "status": "RUNNING",
            "seed": seed,
            "started_at": started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "deployment_contract": "one_checkpoint_one_model_one_forward",
        }
        shard_status_path.write_text(json.dumps(shard_status, indent=2) + "\n", encoding="utf-8")
        result = train_seed(config, rows, seed, out_dir, device)
        save_validation_seed(result, out_dir)
        shard_runtime = {
            "seed": seed,
            "trainable_parameter_names": result["trainable_parameter_names"],
            "capacity": result["capacity"],
            "best_epoch": int(result["best_epoch"]),
        }
        (out_dir / "reports" / f"run_config_seed_{seed}.json").write_text(
            json.dumps(shard_runtime, indent=2) + "\n", encoding="utf-8"
        )
        shard_status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
        shard_status_path.write_text(json.dumps(shard_status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "VALIDATION_SEED_COMPLETE", "seed": seed}))
        return

    if args.stage == "validation-finalize":
        if args.seed is not None:
            raise RuntimeError("validation-finalize does not accept --seed")
        metrics_parts = []
        epoch_parts = []
        shard_statuses: List[Dict[str, Any]] = []
        trainable_by_seed: Dict[str, List[str]] = {}
        capacity_by_seed: Dict[str, Dict[str, Any]] = {}
        for seed in seeds:
            shard_status_path = out_dir / "reports" / f"run_status_seed_{seed}.json"
            shard_metrics_path = out_dir / "reports" / f"metrics_seed_{seed}.csv"
            shard_epoch_path = out_dir / "reports" / f"metrics_by_epoch_seed_{seed}.csv"
            shard_runtime_path = out_dir / "reports" / f"run_config_seed_{seed}.json"
            required = (shard_status_path, shard_metrics_path, shard_epoch_path, shard_runtime_path)
            if not all(path.exists() for path in required):
                raise RuntimeError(f"C26-SM validation shard is incomplete for seed {seed}")
            shard_status = json.loads(shard_status_path.read_text(encoding="utf-8"))
            if shard_status.get("status") != "COMPLETE" or int(shard_status.get("seed", -1)) != seed:
                raise RuntimeError(f"C26-SM validation shard did not complete for seed {seed}")
            shard_metrics = pd.read_csv(shard_metrics_path)
            if len(shard_metrics) != 1 or int(shard_metrics.iloc[0]["seed"]) != seed or shard_metrics.iloc[0]["split"] != "val":
                raise RuntimeError(f"C26-SM validation metrics shard contract failed for seed {seed}")
            shard_runtime = json.loads(shard_runtime_path.read_text(encoding="utf-8"))
            metrics_parts.append(shard_metrics)
            epoch_parts.append(pd.read_csv(shard_epoch_path))
            shard_statuses.append(shard_status)
            trainable_by_seed[str(seed)] = shard_runtime["trainable_parameter_names"]
            capacity_by_seed[str(seed)] = shard_runtime["capacity"]
        metrics = pd.concat(metrics_parts, ignore_index=True).sort_values("seed").reset_index(drop=True)
        metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
        pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
            out_dir / "reports" / "metrics_by_epoch.csv", index=False
        )
        write_metrics_summary(metrics, out_dir)
        finished = timestamp()
        status = {
            "phase": "C26-SM",
            "status": "VALIDATION_COMPLETE",
            "started_at": min(str(item["started_at"]) for item in shard_statuses),
            "validation_finished_at": finished,
            "completed_seeds": seeds,
            "seeds": seeds,
            "parallel_seed_training": True,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "deployment_contract": "one_checkpoint_one_model_one_forward",
        }
        (out_dir / "reports" / "run_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        runtime = {
            "config": config,
            "started_at": status["started_at"],
            "validation_finished_at": finished,
            "device": str(device),
            "gpu": status["gpu"],
            "seeds": seeds,
            "parallel_seed_training": True,
            "trainable_parameter_names_by_seed": trainable_by_seed,
            "capacity_by_seed": capacity_by_seed,
            "selection_metric": "validation_AUC_only",
            "test_role": "reporting_only_after_validation_selection",
            "deployment_contract": "one_checkpoint_one_model_one_forward",
        }
        (out_dir / "reports" / "run_config.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "VALIDATION_COMPLETE", "output_dir": str(out_dir), "seeds": seeds}))
        return

    if args.stage == "reporting-test":
        if args.seed is not None:
            raise RuntimeError("reporting-test does not accept --seed")
        decision_path = resolve_path(config["project"]["report_dir"]) / "c26sm_final_decision.json"
        if not decision_path.exists():
            raise RuntimeError("C26-SM validation decision must be frozen before reporting-only test")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if bool(decision.get("test_used_for_decision", True)) or bool(decision.get("ensemble_used", True)):
            raise RuntimeError("C26-SM validation decision isolation contract failed")
        metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
        metrics = pd.read_csv(metrics_path)
        if set(metrics["split"]) != {"val"}:
            raise RuntimeError("Reporting-only test requires validation-only metrics as input")
        for seed in seeds:
            loaders = build_loaders(config, [dict(row) for row in rows])
            model = C26SMStableMechanismModel(config, seed).to(device)
            checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
            try:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint_path, map_location="cpu")
            if int(payload.get("seed", -1)) != seed:
                raise RuntimeError(f"C26-SM checkpoint seed mismatch for {seed}")
            model.load_state_dict(payload["model"], strict=True)
            result = run_epoch(model, loaders["test"], None, device, config["loss"])
            frame = pd.DataFrame(result["predictions"]).sort_values("patient_id").reset_index(drop=True)
            frame.insert(0, "split", "test")
            frame.insert(0, "seed", seed)
            frame.to_csv(out_dir / "predictions" / f"test_predictions_seed_{seed}.csv", index=False)
            original_ids = np.asarray([str(row["patient_id"]) for row in result["predictions"]])
            order = np.argsort(original_ids)
            np.savez_compressed(
                out_dir / "representations" / f"test_mechanism_state_seed_{seed}.npz",
                patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
                label=frame["label"].to_numpy(dtype=np.int64),
                mechanism_state=result["representations"][order].astype(np.float32),
            )
            metrics = pd.concat([
                metrics,
                pd.DataFrame([{"seed": seed, "split": "test", "best_epoch": int(payload["best_epoch"]), **result["metrics"]}]),
            ], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)
        write_metrics_summary(metrics, out_dir)
        status_path = out_dir / "reports" / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "finished_at": timestamp()})
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "REPORTING_TEST_COMPLETE", "seeds": seeds}))
        return

    if args.seed is not None:
        raise RuntimeError("validation does not accept --seed; use validation-seed")
    started = timestamp()
    status: Dict[str, Any] = {
        "phase": "C26-SM", "status": "RUNNING", "started_at": started,
        "completed_seeds": [], "seeds": seeds, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    status_path = out_dir / "reports" / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metrics_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    trainable_by_seed: Dict[str, List[str]] = {}
    capacity_by_seed: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        result = train_seed(config, rows, seed, out_dir, device)
        epoch_rows.extend(result["epoch_history"])
        trainable_by_seed[str(seed)] = result["trainable_parameter_names"]
        capacity_by_seed[str(seed)] = result["capacity"]
        for split in ("val",):
            split_result = result.get(split)
            if split_result is None:
                continue
            frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id").reset_index(drop=True)
            frame.insert(0, "split", split)
            frame.insert(0, "seed", seed)
            frame.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
            order = np.argsort(np.asarray([str(row["patient_id"]) for row in split_result["predictions"]]))
            np.savez_compressed(
                out_dir / "representations" / f"{split}_mechanism_state_seed_{seed}.npz",
                patient_id=np.asarray(frame["patient_id"].astype(str).tolist(), dtype=np.str_),
                label=frame["label"].to_numpy(dtype=np.int64),
                mechanism_state=split_result["representations"][order].astype(np.float32),
            )
            metrics_rows.append({"seed": seed, "split": split, "best_epoch": result["best_epoch"], **split_result["metrics"]})
        status["completed_seeds"].append(seed)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    summary_rows = []
    for split, split_frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "mean_delta_c26sm", "std_delta_c26sm"):
            values = split_frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)
    status.update({"status": "VALIDATION_COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "config": config, "started_at": started, "validation_finished_at": status["validation_finished_at"],
        "device": str(device), "gpu": status["gpu"], "seeds": seeds,
        "trainable_parameter_names_by_seed": trainable_by_seed,
        "capacity_by_seed": capacity_by_seed,
        "selection_metric": "validation_AUC_only",
        "test_role": "reporting_only_after_validation_selection",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    (out_dir / "reports" / "run_config.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATION_COMPLETE", "output_dir": str(out_dir), "seeds": seeds}))


if __name__ == "__main__":
    main()
