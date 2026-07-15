#!/usr/bin/env python3
"""Train C33-JERA as three independent direct validation-selected seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c33_jera import (  # noqa: E402
    C33JERAModel,
    TRAINABLE_MODULES,
    named_trainable_parameters,
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
    parser.add_argument(
        "--config", default="configs/dema_ht_c33_jera_multiseed.yaml"
    )
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


def trainable_gradient_norms(model: C33JERAModel) -> Dict[str, float]:
    named = dict(model.named_parameters())
    result: Dict[str, float] = {}
    for category, prefix in TRAINABLE_MODULES.items():
        squared = 0.0
        for name, parameter in named.items():
            if name.startswith(f"{prefix}.") and parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
        result[category] = float(np.sqrt(squared))
    return result


def trainable_drift_summary(model: C33JERAModel) -> Dict[str, float]:
    rows = pd.DataFrame(model.parameter_drift_rows())
    result: Dict[str, float] = {}
    for category in TRAINABLE_MODULES:
        values = rows.loc[
            rows["category"] == category, "relative_parameter_drift"
        ].to_numpy(dtype=float)
        result[category] = float(values.mean()) if len(values) else float("nan")
    return result


def run_epoch(
    model: C33JERAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    gradient_values: Dict[str, List[float]] = {
        category: [] for category in TRAINABLE_MODULES
    }

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C33 non-finite BCE loss")
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
        visit_mask = batch["visit_mask"].detach().cpu()
        for index, patient_id in enumerate(batch["patient_id"]):
            row: Dict[str, Any] = {
                "patient_id": str(patient_id),
                "label": int(labels[index]),
                "final_logit": float(arrays["logit"][index]),
                "final_prob": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "visit_count": int(visit_mask[index].sum()),
                "mean_temporal_weight_latest": float(
                    arrays["temporal_latest_weights"][index].mean()
                ),
                "mean_conflict": float(arrays["conflicts"][index].mean()),
                "patient_state_norm": float(
                    np.linalg.norm(arrays["patient_state"][index])
                ),
            }
            for field in SHORTCUT_FIELDS:
                row[field] = batch["shortcuts"][index].get(field, float("nan"))
            rows.append(row)

    frame = pd.DataFrame(rows)
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["final_prob"].to_numpy(dtype=float)
    metrics: Dict[str, Any] = dict(binary_metrics(labels, probabilities))
    metrics.update(
        {
            "bce_loss": float(np.mean(losses)),
            "positive_probability_mean": float(probabilities[labels == 1].mean()),
            "negative_probability_mean": float(probabilities[labels == 0].mean()),
            "positive_negative_gap": float(
                probabilities[labels == 1].mean()
                - probabilities[labels == 0].mean()
            ),
            "pairwise_inversion_count": pairwise_inversions(labels, probabilities),
            "prediction_std": float(frame["final_prob"].std(ddof=1)),
            "n_rows": int(len(frame)),
        }
    )
    for category in TRAINABLE_MODULES:
        values = gradient_values[category]
        metrics[f"{category}_grad_norm"] = (
            float(np.mean(values)) if values else 0.0
        )
    return {"metrics": metrics, "predictions": rows}


def patient_state_diagnostics(
    model: C33JERAModel,
    reference: C33JERAModel,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    reference.eval()
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            adapted = model(batch)["patient_state"]
            original = reference(batch)["patient_state"]
            for index, patient_id in enumerate(batch["patient_id"]):
                left = original[index]
                right = adapted[index]
                cosine = F.cosine_similarity(
                    left.unsqueeze(0), right.unsqueeze(0), dim=-1
                ).squeeze(0)
                delta = torch.linalg.vector_norm(right - left)
                rows.append(
                    {
                        "seed": model.seed,
                        "patient_id": str(patient_id),
                        "label": int(batch["label"][index].detach().cpu()),
                        "original_patient_state_norm": float(
                            torch.linalg.vector_norm(left).detach().cpu()
                        ),
                        "adapted_patient_state_norm": float(
                            torch.linalg.vector_norm(right).detach().cpu()
                        ),
                        "original_vs_adapted_cosine": float(cosine.detach().cpu()),
                        "original_vs_adapted_l2_delta": float(delta.detach().cpu()),
                    }
                )
    return pd.DataFrame(rows)


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model = C33JERAModel(config, seed).to(device)
    trainable = list(named_trainable_parameters(model))
    if not trainable or any(
        not C33JERAModel.is_trainable_parameter(name) for name, _ in trainable
    ):
        raise RuntimeError(
            f"C33 trainable scope violation: {[name for name, _ in trainable]}"
        )
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
            "val_positive_probability_mean": val_result["metrics"][
                "positive_probability_mean"
            ],
            "val_negative_probability_mean": val_result["metrics"][
                "negative_probability_mean"
            ],
            "val_positive_negative_gap": val_result["metrics"][
                "positive_negative_gap"
            ],
            "pairwise_inversion_count": val_result["metrics"][
                "pairwise_inversion_count"
            ],
            "selected_by_val_auc": False,
        }
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
        raise RuntimeError(f"C33 seed {seed} produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C33 seed {seed} produced constant validation predictions")
    reference = C33JERAModel(config, seed).to(device)
    diagnostics = patient_state_diagnostics(model, reference, loaders["val"], device)
    del reference
    checkpoint_path = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
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
        "drift": model.parameter_drift_rows(),
        "diagnostics": diagnostics,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "source_c27_checkpoint": model.c27_checkpoint,
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
        for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC"):
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
        raise RuntimeError(f"C33 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C33-JERA",
        "stage": "validation-seed",
        "status": "RUNNING",
        "seed": seed,
        "started_at": timestamp(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
    result["diagnostics"].to_csv(
        seed_dir / "reports" / "patient_state_change.csv", index=False
    )
    runtime = {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "source_c27_checkpoint": result["source_c27_checkpoint"],
        "trainable_parameter_names": result["trainable_parameter_names"],
        "trainable_parameter_count": int(result["trainable_parameter_count"]),
        "frozen_parameter_count": int(result["frozen_parameter_count"]),
    }
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "COMPLETE", "validation_finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C33_VALIDATION_SEED_COMPLETE", "seed": seed}))


def validation_finalize_stage(
    config: Dict[str, Any], out_dir: Path, device: torch.device
) -> None:
    metrics_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    drift_parts: List[pd.DataFrame] = []
    state_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    runtime_by_seed: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads(
            (seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C33 seed {seed} validation shard incomplete")
        metrics_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        drift_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_drift.csv"))
        state_parts.append(pd.read_csv(seed_dir / "reports" / "patient_state_change.csv"))
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
    pd.concat(state_parts, ignore_index=True).to_csv(
        out_dir / "reports" / "patient_state_change_val.csv", index=False
    )
    write_summary(metrics, out_dir)
    status = {
        "phase": "C33-JERA",
        "status": "VALIDATION_COMPLETE",
        "started_at": min(str(item["started_at"]) for item in statuses),
        "validation_finished_at": timestamp(),
        "completed_seeds": list(SEEDS),
        "parallel_seed_training": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
    print(json.dumps({"status": "C33_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c33_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C33 validation decision must be frozen before reporting-only test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not bool(decision.get("validation_decision_frozen_before_test", False))
        or bool(decision.get("test_used_for_decision", True))
        or bool(decision.get("ensemble_used", True))
    ):
        raise RuntimeError("C33 validation/test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C33 reporting-only test requires validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C33JERAModel(config, seed).to(device)
        checkpoint_path = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C33 checkpoint seed mismatch for {seed}")
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
    print(json.dumps({"status": "C33_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c33_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C33_JERA_DIRECT_MULTI_SEED_AUTHORIZED" or int(
        gate.get("passed", 0)
    ) != 15:
        raise RuntimeError("C33 direct execution requires an authorized 15/15 gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C33 formal seed outputs already exist")
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
        raise RuntimeError(f"C33 formal validation seed failure codes: {failures}")
    validation_finalize_stage(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c33_report.py"
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"],
        check=True,
    )
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run(
        [sys.executable, str(collector), "--config", str(config_path), "--stage", "final"],
        check=True,
    )
    print(json.dumps({"status": "C33_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c33":
        raise RuntimeError("C33 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C33 formal seeds must remain [0, 42, 3407]")
    expected_lr = float(config["c33"]["c27_base_lr"]) * float(
        config["c33"]["lr_scale"]
    )
    if abs(float(config["training"]["lr"]) - expected_lr) > 1e-12:
        raise RuntimeError("C33 joint adaptation learning rate contract failed")
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
