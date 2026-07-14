#!/usr/bin/env python3
"""Train the frozen-C27 C30-VTCA route as independent validation shards."""

from __future__ import annotations

import argparse
import json
import shutil
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

from dmea_ht.c27_vtme import MECHANISM_NAMES  # noqa: E402
from dmea_ht.c30_vtca import C30VTCAModel, trainable_parameter_count  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import (  # noqa: E402
    binary_metrics,
    build_loaders,
    move_batch,
    pairwise_inversions,
    resolve_path,
    set_seed,
    temporal_group,
    timestamp,
)


SEEDS = (0, 42, 3407)
ADAPTER_DIAGNOSTICS = (
    "adapter_delta_abs_mean",
    "adapter_delta_abs_std",
    "adapter_delta_abs_max",
    "adapter_near_bound_fraction",
    "padding_delta_abs_max",
    "latest_visit_adapter_delta_abs",
    "history_visit_adapter_delta_abs",
    "text_token_norm_before_mean",
    "text_token_norm_after_mean",
    "text_token_cosine_before_after",
    "text_evidence_state_cosine_before_after",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation-seed", "validation-finalize", "reporting-test"),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _mask_present(batch: Mapping[str, Any], key: str, index: int, count: int) -> bool:
    return bool(batch[key][index, :count].bool().any().detach().cpu())


def _text_groups(batch: Mapping[str, Any], index: int, count: int) -> Dict[str, bool]:
    support = batch["visit_support_present"][index, :count].bool()
    opposition = batch["visit_opposition_present"][index, :count].bool()
    latest = count - 1
    history_support = bool(support[:latest].any().detach().cpu()) if latest else False
    history_opposition = bool(opposition[:latest].any().detach().cpu()) if latest else False
    latest_support = bool(support[latest].detach().cpu())
    latest_opposition = bool(opposition[latest].detach().cpu())
    return {
        "group_morphology_visible": _mask_present(batch, "text_nonspecific_mask", index, count),
        "group_diffuse_ht_like_visible": _mask_present(batch, "text_support_mask", index, count)
        or _mask_present(batch, "text_diagnostic_hint_mask", index, count),
        "group_opposition_normal_visible": _mask_present(batch, "text_opposition_mask", index, count),
        "group_uncertainty_visible": _mask_present(batch, "text_uncertainty_mask", index, count),
        "group_latest_history_mixed": bool(
            count > 1
            and (latest_support or latest_opposition)
            and (history_support or history_opposition)
        ),
        "group_latest_positive_history_negative": latest_support and history_opposition,
        "group_latest_negative_history_positive": latest_opposition and history_support,
    }


def run_epoch(
    model: C30VTCAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: List[Dict[str, Any]] = []
    patient_states: List[np.ndarray] = []
    mechanism_states: List[np.ndarray] = []
    temporal_latest: List[np.ndarray] = []
    conflict_states: List[np.ndarray] = []
    losses: List[float] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C30 non-finite BCE loss")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))

        arrays = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if torch.is_tensor(value)
        }
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
            uniform_fraction = (
                float(np.mean((weights.max(axis=0) - weights.min(axis=0)) < 1e-3))
                if count > 1
                else 0.0
            )
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
                "mean_temporal_weight_entropy": float(arrays["temporal_entropy"][index].mean()),
                "mean_normalized_temporal_entropy": float(
                    arrays["temporal_normalized_entropy"][index].mean()
                ),
                "fraction_latest_weight_above_0_90": float(
                    (arrays["temporal_latest_weights"][index] > 0.90).mean()
                ),
                "fraction_uniform_temporal_weight": uniform_fraction,
                "patient_state_norm": float(np.linalg.norm(arrays["patient_state"][index])),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "temporal_group": temporal_group(support[index], opposition[index], count),
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
                "same_visit_alignment_count": int(arrays["same_visit_alignment_count"][index]),
                "cross_visit_alignment_pair_count": int(
                    arrays["cross_visit_alignment_pair_count"][index]
                ),
            }
            for key in ADAPTER_DIAGNOSTICS:
                row[key] = float(arrays[key][index])
            row["mean_adapter_delta_abs"] = row["adapter_delta_abs_mean"]
            row["max_adapter_delta_abs"] = row["adapter_delta_abs_max"]
            row.update(_text_groups(batch, index, count))
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
            row.update(batch["shortcuts"][index])
            rows.append(row)

    frame = pd.DataFrame(rows)
    labels = frame["label"].to_numpy(dtype=int)
    probs = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = dict(binary_metrics(labels, probs))
    positive = frame[frame["label"].astype(int) == 1]
    negative = frame[frame["label"].astype(int) == 0]
    metrics.update(
        {
            "bce_loss": float(np.mean(losses)),
            "positive_probability_mean": float(positive["final_prob"].mean()),
            "negative_probability_mean": float(negative["final_prob"].mean()),
            "positive_negative_gap": float(
                positive["final_prob"].mean() - negative["final_prob"].mean()
            ),
            "adapter_delta_abs_mean": float(frame["adapter_delta_abs_mean"].mean()),
            "adapter_delta_abs_std": float(frame["adapter_delta_abs_mean"].std(ddof=1)),
            "adapter_delta_abs_max": float(frame["adapter_delta_abs_max"].max()),
            "adapter_near_bound_fraction": float(frame["adapter_near_bound_fraction"].mean()),
            "adapter_delta_positive_mean": float(positive["adapter_delta_abs_mean"].mean()),
            "adapter_delta_negative_mean": float(negative["adapter_delta_abs_mean"].mean()),
            "text_token_norm_before_mean": float(frame["text_token_norm_before_mean"].mean()),
            "text_token_norm_after_mean": float(frame["text_token_norm_after_mean"].mean()),
            "text_token_cosine_before_after": float(
                frame["text_token_cosine_before_after"].mean()
            ),
            "text_evidence_state_cosine_before_after": float(
                frame["text_evidence_state_cosine_before_after"].mean()
            ),
            "padding_delta_abs_max": float(frame["padding_delta_abs_max"].max()),
            "latest_visit_adapter_delta_abs": float(
                frame["latest_visit_adapter_delta_abs"].mean()
            ),
            "history_visit_adapter_delta_abs": float(
                frame["history_visit_adapter_delta_abs"].mean()
            ),
            "prediction_std": float(frame["final_prob"].std(ddof=1)),
            "pairwise_inversion_count": pairwise_inversions(labels, probs),
            "n_rows": int(len(frame)),
        }
    )
    if not np.isfinite(frame["final_prob"].to_numpy(dtype=float)).all():
        raise RuntimeError("C30 non-finite predictions")
    return {
        "metrics": metrics,
        "predictions": rows,
        "patient_states": np.concatenate(patient_states, axis=0),
        "mechanism_states": np.concatenate(mechanism_states, axis=0),
        "temporal_latest": np.concatenate(temporal_latest, axis=0),
        "conflicts": np.concatenate(conflict_states, axis=0),
    }


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model = C30VTCAModel(config, seed).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("adapter.") for name, _ in trainable):
        raise RuntimeError(f"C30 trainable scope violation: {[name for name, _ in trainable]}")
    count = trainable_parameter_count(model)
    if count > int(config["c30"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C30_CAPACITY_CONTRACT_FAIL: {count}")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_adapter_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = run_epoch(model, loaders["train"], optimizer, device)
        val_result = run_epoch(model, loaders["val"], None, device)
        val_metrics = val_result["metrics"]
        row: Dict[str, Any] = {
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": train_result["metrics"]["bce_loss"],
            "val_auc": val_metrics["AUC"],
            "val_sensitivity": val_metrics["Sensitivity"],
            "val_specificity": val_metrics["Specificity"],
            "val_balanced_accuracy": val_metrics["Balanced_ACC"],
            "val_positive_probability_mean": val_metrics["positive_probability_mean"],
            "val_negative_probability_mean": val_metrics["negative_probability_mean"],
            "val_positive_negative_gap": val_metrics["positive_negative_gap"],
            "adapter_delta_abs_mean": val_metrics["adapter_delta_abs_mean"],
            "adapter_delta_abs_std": val_metrics["adapter_delta_abs_std"],
            "adapter_delta_abs_max": val_metrics["adapter_delta_abs_max"],
            "adapter_near_bound_fraction": val_metrics["adapter_near_bound_fraction"],
            "adapter_delta_positive_mean": val_metrics["adapter_delta_positive_mean"],
            "adapter_delta_negative_mean": val_metrics["adapter_delta_negative_mean"],
            "text_token_norm_before_mean": val_metrics["text_token_norm_before_mean"],
            "text_token_norm_after_mean": val_metrics["text_token_norm_after_mean"],
            "text_token_cosine_before_after": val_metrics["text_token_cosine_before_after"],
            "padding_delta_abs_max": val_metrics["padding_delta_abs_max"],
            "pairwise_inversion_count": val_metrics["pairwise_inversion_count"],
            "selected_by_val_auc": False,
        }
        epoch_rows.append(row)
        val_auc = float(val_metrics["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in model.adapter.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_adapter_state is None:
        raise RuntimeError(f"C30 seed {seed} produced no validation-selected checkpoint")
    model.adapter.load_state_dict(best_adapter_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C30 seed {seed} produced constant validation predictions")
    checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "source_c27_checkpoint": model.c27_checkpoint,
            "selection_metric": "validation_auc_only",
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": count,
        "frozen_parameter_names": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "source_c27_checkpoint": model.c27_checkpoint,
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
    if (seed_dir / "reports" / "run_status.json").exists():
        raise RuntimeError(f"C30 validation shard already exists for seed {seed}")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C30-VTCA",
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
        "source_c27_checkpoint": result["source_c27_checkpoint"],
        "trainable_parameter_names": result["trainable_parameter_names"],
        "trainable_parameter_count": int(result["trainable_parameter_count"]),
        "frozen_parameter_names": result["frozen_parameter_names"],
        "frozen_parameter_count": int(result["frozen_parameter_count"]),
    }
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metrics_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    runtime_by_seed: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        required = (
            seed_dir / "reports" / "run_status.json",
            seed_dir / "reports" / "metrics.csv",
            seed_dir / "reports" / "metrics_by_epoch.csv",
            seed_dir / "reports" / "run_config.json",
            seed_dir / "checkpoints" / f"seed_{seed}_best.pt",
            seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv",
            seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz",
        )
        if not all(path.exists() for path in required):
            raise RuntimeError(f"C30 validation shard incomplete for seed {seed}")
        status = json.loads(required[0].read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE" or int(status.get("seed", -1)) != seed:
            raise RuntimeError(f"C30 validation shard status invalid for seed {seed}")
        metric = pd.read_csv(required[1])
        if len(metric) != 1 or int(metric.iloc[0]["seed"]) != seed or metric.iloc[0]["split"] != "val":
            raise RuntimeError(f"C30 validation metric invalid for seed {seed}")
        metrics_parts.append(metric)
        epoch_parts.append(pd.read_csv(required[2]))
        statuses.append(status)
        runtime_by_seed[str(seed)] = json.loads(required[3].read_text(encoding="utf-8"))
        for source, target in (
            (required[4], out_dir / "checkpoints" / required[4].name),
            (required[5], out_dir / "predictions" / required[5].name),
            (required[6], out_dir / "representations" / required[6].name),
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
        "phase": "C30-VTCA",
        "status": "VALIDATION_COMPLETE",
        "started_at": min(str(item["started_at"]) for item in statuses),
        "validation_finished_at": finished,
        "completed_seeds": list(SEEDS),
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
        "seeds": list(SEEDS),
        "seed_runtime": runtime_by_seed,
        "selection_metric": "validation_AUC_only",
        "test_role": "reporting_only_after_validation_decision",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    (out_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c30_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C30 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not bool(decision.get("validation_decision_frozen_before_test", False))
        or bool(decision.get("test_used_for_decision", True))
        or bool(decision.get("ensemble_used", True))
    ):
        raise RuntimeError("C30 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C30 reporting-only test requires validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C30VTCAModel(config, seed).to(device)
        checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C30 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = run_epoch(model, loader, None, device)
        metric = save_split(
            {"seed": seed, "best_epoch": int(payload["best_epoch"]), "test": result},
            out_dir,
            "test",
        )
        metrics = pd.concat([metrics, pd.DataFrame([metric])], ignore_index=True)
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
    print(json.dumps({"status": "REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c30":
        raise RuntimeError("C30 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C30 formal seeds must remain [0, 42, 3407]")
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
