#!/usr/bin/env python3
"""Train C27-VTME as three independent single-model validation shards."""

from __future__ import annotations

import argparse
import json
import random
import shutil
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

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, trainable_parameter_count  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402


SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("validation-seed", "validation-finalize", "reporting-test"), required=True)
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
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def pairwise_inversions(labels: np.ndarray, probs: np.ndarray) -> int:
    positive, negative = probs[labels == 1], probs[labels == 0]
    return int((positive[:, None] < negative[None, :]).sum()) if positive.size and negative.size else 0


def temporal_group(
    support: torch.Tensor,
    opposition: torch.Tensor,
    visit_count: int,
) -> str:
    if visit_count <= 1:
        return "single_visit"
    latest_support = bool(support[visit_count - 1])
    latest_opposition = bool(opposition[visit_count - 1])
    history_support = bool(support[: visit_count - 1].any())
    history_opposition = bool(opposition[: visit_count - 1].any())
    if latest_support and history_opposition:
        return "latest_positive_like_history_negative_like"
    if latest_opposition and history_support:
        return "latest_negative_like_history_positive_like"
    if (latest_support or latest_opposition) and (history_support or history_opposition):
        return "latest_history_conflict"
    return "latest_history_agreement"


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

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_values.append(float(loss.detach().cpu()))

        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items() if torch.is_tensor(value)}
        labels = batch["label"].detach().cpu().numpy().astype(int)
        visit_mask = batch["visit_mask"].detach().cpu()
        support = batch["visit_support_present"].detach().cpu()
        opposition = batch["visit_opposition_present"].detach().cpu()
        image_mask = batch["image_mask"].detach().cpu()
        text_valid = batch["visit_text_valid"].detach().cpu()
        patient_states.append(arrays["patient_state"])
        mechanism_states.append(arrays["mechanism_states"])
        temporal_latest.append(arrays["temporal_latest_weights"])
        conflict_states.append(arrays["conflicts"])

        for index, patient_id in enumerate(batch["patient_id"]):
            count = int(visit_mask[index].sum())
            latest_index = count - 1
            weights = arrays["temporal_weights"][index, :count]
            uniform_fraction = float(np.mean((weights.max(axis=0) - weights.min(axis=0)) < 1e-3)) if count > 1 else 0.0
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "visit_count_audit_only": count,
                "reconstructable_visit_count_audit_only": int(batch["shortcuts"][index]["reconstructable_visit_count"]),
                "visit_report_coverage_audit_only": float(batch["shortcuts"][index]["visit_report_coverage"]),
                "latest_visit_rank": latest_index,
                "latest_visit_has_image": bool(image_mask[index, latest_index].any()),
                "latest_visit_has_text": bool(text_valid[index, latest_index]),
                "latest_visit_has_dated_bio": bool(batch["visit_dated_bio_present"][index][latest_index]),
                "mean_temporal_weight_latest": float(arrays["temporal_latest_weights"][index].mean()),
                "mean_temporal_weight_history": float(1.0 - arrays["temporal_latest_weights"][index].mean()),
                "mean_temporal_weight_entropy": float(arrays["temporal_entropy"][index].mean()),
                "mean_normalized_temporal_entropy": float(arrays["temporal_normalized_entropy"][index].mean()),
                "fraction_latest_weight_above_0_90": float((arrays["temporal_latest_weights"][index] > 0.90).mean()),
                "fraction_uniform_temporal_weight": uniform_fraction,
                "patient_state_norm": float(np.linalg.norm(arrays["patient_state"][index])),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "temporal_group": temporal_group(support[index], opposition[index], count),
                "same_visit_image_text_cosine": float(arrays["same_visit_alignment_mean"][index]),
                "cross_visit_image_text_cosine": float(arrays["cross_visit_alignment_mean"][index]),
                "latest_same_visit_alignment": float(arrays["latest_same_visit_alignment"][index]),
                "history_same_visit_alignment": float(arrays["history_same_visit_alignment"][index]),
                "same_visit_alignment_count": int(arrays["same_visit_alignment_count"][index]),
                "cross_visit_alignment_pair_count": int(arrays["cross_visit_alignment_pair_count"][index]),
            }
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                row[f"temporal_weight_latest_{mechanism}"] = float(
                    arrays["temporal_latest_weights"][index, mechanism_index]
                )
                row[f"conflict_{mechanism}"] = float(arrays["conflicts"][index, mechanism_index])
                row[f"history_available_{mechanism}"] = bool(
                    arrays["history_available"][index, mechanism_index]
                )
                row[f"H_{mechanism}_norm"] = float(
                    np.linalg.norm(arrays["mechanism_states"][index, mechanism_index])
                )
            row.update(batch["shortcuts"][index])
            predictions.append(row)

    frame = pd.DataFrame(predictions)
    labels = frame["label"].to_numpy(dtype=int)
    probs = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = dict(binary_metrics(labels, probs))
    metrics.update(
        {
            "bce_loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "positive_probability_mean": float(probs[labels == 1].mean()),
            "negative_probability_mean": float(probs[labels == 0].mean()),
            "positive_negative_gap": float(probs[labels == 1].mean() - probs[labels == 0].mean()),
            "mean_temporal_weight_latest": float(frame["mean_temporal_weight_latest"].mean()),
            "mean_temporal_weight_history": float(frame["mean_temporal_weight_history"].mean()),
            "mean_temporal_weight_entropy": float(frame["mean_temporal_weight_entropy"].mean()),
            "mean_normalized_temporal_entropy": float(frame["mean_normalized_temporal_entropy"].mean()),
            "fraction_latest_weight_above_0_90": float(frame["fraction_latest_weight_above_0_90"].mean()),
            "fraction_uniform_temporal_weight": float(frame["fraction_uniform_temporal_weight"].mean()),
            "mean_patient_state_norm": float(frame["patient_state_norm"].mean()),
            "std_patient_state_norm": float(frame["patient_state_norm"].std(ddof=1)),
            "prediction_std": float(frame["final_prob"].std(ddof=1)),
            "pairwise_inversion_count": pairwise_inversions(labels, probs),
            "n_rows": int(len(frame)),
        }
    )
    for mechanism in MECHANISM_NAMES:
        metrics[f"mean_conflict_{mechanism}"] = float(frame[f"conflict_{mechanism}"].mean())
    return {
        "metrics": metrics,
        "predictions": predictions,
        "patient_states": np.concatenate(patient_states, axis=0),
        "mechanism_states": np.concatenate(mechanism_states, axis=0),
        "temporal_latest": np.concatenate(temporal_latest, axis=0),
        "conflicts": np.concatenate(conflict_states, axis=0),
    }


def train_seed(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model = C27VTMEModel(config, seed).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("core.") for name, _ in trainable):
        raise RuntimeError(f"C27 trainable scope violation: {[name for name, _ in trainable]}")
    count = trainable_parameter_count(model)
    if count > int(config["c27"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C27_CAPACITY_CONTRACT_FAIL: {count}")
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
            "mean_temporal_weight_latest": val_result["metrics"]["mean_temporal_weight_latest"],
            "mean_temporal_weight_history": val_result["metrics"]["mean_temporal_weight_history"],
            "mean_temporal_weight_entropy": val_result["metrics"]["mean_temporal_weight_entropy"],
            "fraction_latest_weight_above_0_90": val_result["metrics"]["fraction_latest_weight_above_0_90"],
            "fraction_uniform_temporal_weight": val_result["metrics"]["fraction_uniform_temporal_weight"],
            "mean_patient_state_norm": val_result["metrics"]["mean_patient_state_norm"],
            "std_patient_state_norm": val_result["metrics"]["std_patient_state_norm"],
            "prediction_std": val_result["metrics"]["prediction_std"],
            "pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
            "selected_by_val_auc": False,
        }
        for mechanism in MECHANISM_NAMES:
            row[f"mean_conflict_{mechanism}"] = val_result["metrics"][f"mean_conflict_{mechanism}"]
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
        raise RuntimeError(f"C27 seed {seed} produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {"model": model.state_dict(), "config": config, "seed": seed, "best_epoch": best_epoch},
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "trainable_parameter_names": [name for name, _ in trainable],
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
    frame = pd.DataFrame(split_result["predictions"]).sort_values("patient_id").reset_index(drop=True)
    frame.insert(0, "split", split)
    frame.insert(0, "seed", seed)
    frame.to_csv(out_dir / "predictions" / f"{split}_predictions_seed_{seed}.csv", index=False)
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
    rows = []
    for split, frame in metrics.groupby("split"):
        row: Dict[str, Any] = {"split": split}
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "prediction_std"):
            values = frame[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "reports" / "metrics_summary.csv", index=False)


def validation_seed_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C27-VTME",
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
    runtime = {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "trainable_parameter_names": result["trainable_parameter_names"],
        "trainable_parameter_count": int(result["trainable_parameter_count"]),
        "frozen_parameter_count": int(result["frozen_parameter_count"]),
    }
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metrics_parts = []
    epoch_parts = []
    statuses = []
    trainable_by_seed: Dict[str, List[str]] = {}
    counts_by_seed: Dict[str, Dict[str, int]] = {}
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status_path = seed_dir / "reports" / "run_status.json"
        metric_path = seed_dir / "reports" / "metrics.csv"
        epoch_path = seed_dir / "reports" / "metrics_by_epoch.csv"
        config_path = seed_dir / "reports" / "run_config.json"
        if not all(path.exists() for path in (status_path, metric_path, epoch_path, config_path)):
            raise RuntimeError(f"C27 validation shard incomplete for seed {seed}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE" or int(status.get("seed", -1)) != seed:
            raise RuntimeError(f"C27 validation shard did not complete for seed {seed}")
        metric = pd.read_csv(metric_path)
        if len(metric) != 1 or int(metric.iloc[0]["seed"]) != seed or metric.iloc[0]["split"] != "val":
            raise RuntimeError(f"C27 validation metric shard invalid for seed {seed}")
        runtime = json.loads(config_path.read_text(encoding="utf-8"))
        metrics_parts.append(metric)
        epoch_parts.append(pd.read_csv(epoch_path))
        statuses.append(status)
        trainable_by_seed[str(seed)] = runtime["trainable_parameter_names"]
        counts_by_seed[str(seed)] = {
            "trainable": int(runtime["trainable_parameter_count"]),
            "frozen": int(runtime["frozen_parameter_count"]),
        }
        for source, target in (
            (seed_dir / "checkpoints" / f"seed_{seed}_best.pt", out_dir / "checkpoints" / f"seed_{seed}_best.pt"),
            (seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv", out_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
            (seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz", out_dir / "representations" / f"val_patient_state_seed_{seed}.npz"),
        ):
            shutil.copy2(source, target)
    metrics = pd.concat(metrics_parts, ignore_index=True).sort_values("seed").reset_index(drop=True)
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(
        out_dir / "reports" / "metrics_by_epoch.csv", index=False
    )
    write_summary(metrics, out_dir)
    finished = timestamp()
    status = {
        "phase": "C27-VTME",
        "status": "VALIDATION_COMPLETE",
        "started_at": min(str(item["started_at"]) for item in statuses),
        "validation_finished_at": finished,
        "completed_seeds": list(SEEDS),
        "parallel_seed_training": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    (out_dir / "reports" / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "config": config,
        "started_at": status["started_at"],
        "validation_finished_at": finished,
        "seeds": list(SEEDS),
        "trainable_parameter_names_by_seed": trainable_by_seed,
        "parameter_counts_by_seed": counts_by_seed,
        "selection_metric": "validation_AUC_only",
        "test_role": "reporting_only_after_validation_decision",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    (out_dir / "reports" / "run_config.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c27_final_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C27 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if bool(decision.get("test_used_for_decision", True)) or bool(decision.get("ensemble_used", True)):
        raise RuntimeError("C27 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C27 reporting-only test requires validation-only metrics")
    for seed in SEEDS:
        loader = build_loaders(config, rows, ("test",))["test"]
        model = C27VTMEModel(config, seed).to(device)
        checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C27 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = run_epoch(model, loader, None, device)
        wrapped = {"seed": seed, "best_epoch": int(payload["best_epoch"]), "test": result}
        metric = save_split(wrapped, out_dir, "test")
        metrics = pd.concat([metrics, pd.DataFrame([metric])], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c27":
        raise RuntimeError("C27 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C27 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"--seed must be one of {SEEDS}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        if args.seed is not None:
            raise RuntimeError("validation-finalize does not accept --seed")
        validation_finalize_stage(config, out_dir, device)
    else:
        if args.seed is not None:
            raise RuntimeError("reporting-test does not accept --seed")
        reporting_test_stage(config, rows, out_dir, device)


if __name__ == "__main__":
    main()
