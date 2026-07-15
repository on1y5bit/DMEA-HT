#!/usr/bin/env python3
"""Train C42-E2E-PET as direct, independent formal validation shards."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c42_e2e_pet import C42E2EPETModel, trainable_parameter_count, trainable_parameter_names  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import train_phase_c41 as base  # noqa: E402


SEEDS = base.SEEDS
build_loaders = base.build_loaders
move_batch = base.move_batch
run_epoch = base.run_epoch
save_split = base.save_split
write_summary = base.write_summary
parameter_drift_rows = base.parameter_drift_rows
set_seed = base.set_seed
timestamp = base.timestamp
resolve_path = base.resolve_path
checkpoint_payload = base.checkpoint_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c42_e2e_pet_multiseed.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation-seed", "validation-finalize", "reporting-test", "direct-multiseed"),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def train_seed(config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, seed_dir: Path, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    loaders = build_loaders(config, rows, ("train", "val"))
    model = C42E2EPETModel(config, seed).to(device)
    names = trainable_parameter_names(model)
    if not names or any(not parameter.requires_grad for _, parameter in model.named_parameters()):
        raise RuntimeError("C42 requires end-to-end trainable source and patient graph parameters")
    count = trainable_parameter_count(model)
    if count > int(config["c42"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C42 capacity contract failed: {count}")
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"C42 seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C42 seed {seed} produced constant validation predictions")
    checkpoint_path = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "source_c17_checkpoint": str(Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))),
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
        "frozen_parameter_count": 0,
    }


def validation_seed_stage(config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    if seed_dir.exists():
        raise RuntimeError(f"C42 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C42-E2E-PET",
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
    result["val"]["patient_diagnostics"].assign(seed=seed).to_csv(seed_dir / "reports" / "patient_diagnostics_val.csv", index=False)
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": int(result["best_epoch"]),
                "trainable_parameter_names": result["trainable_parameter_names"],
                "trainable_parameter_count": int(result["trainable_parameter_count"]),
                "frozen_parameter_count": 0,
                "selection_metric": "validation_AUC_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status.update({"status": "COMPLETE", "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C42_VALIDATION_SEED_COMPLETE", "seed": seed}))


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
            raise RuntimeError(f"C42 seed {seed} shard incomplete")
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
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    pd.concat(drift_parts, ignore_index=True).sort_values(["seed", "parameter_name"]).to_csv(out_dir / "reports" / "parameter_drift.csv", index=False)
    pd.concat(diagnostic_parts, ignore_index=True).to_csv(out_dir / "reports" / "patient_diagnostics_val.csv", index=False)
    write_summary(metrics, out_dir)
    (out_dir / "reports" / "run_config.json").write_text(json.dumps({"config": config, "seeds": list(SEEDS), "parallel_seed_training": True, "selection_metric": "validation_AUC_only", "test_role": "reporting_only_after_validation_decision", "deployment_contract": "one_checkpoint_one_model_one_forward"}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "reports" / "run_status.json").write_text(json.dumps({"phase": "C42-E2E-PET", "status": "VALIDATION_COMPLETE", "started_at": min(item["started_at"] for item in statuses), "finished_at": timestamp(), "completed_seeds": list(SEEDS), "parallel_seed_training": True, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C42_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c42_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C42 Validation decision must be frozen before reporting-only Test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("validation_decision_frozen_before_test", False) or decision.get("test_used_for_decision", True) or decision.get("ensemble_used", True):
        raise RuntimeError("C42 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C42 reporting-only Test requires Validation-only metrics")
    loader = build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C42E2EPETModel(config, seed).to(device)
        payload = checkpoint_payload(out_dir / "checkpoints" / f"seed_{seed}_best.pt")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C42 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = run_epoch(model, loader, None, device)
        test_metric = save_split({"seed": seed, "best_epoch": payload["best_epoch"], "test": result}, out_dir, "test")
        metrics = pd.concat([metrics, pd.DataFrame([test_metric])], ignore_index=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "finished_at": timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C42_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(config_path: Path, config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c42_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C42 direct execution requires the completed gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C42_E2E_PET_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C42 direct execution requires an authorized gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C42 formal seed outputs already exist")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes = [subprocess.Popen([sys.executable, str(script), "--config", str(config_path), "--stage", "validation-seed", "--seed", str(seed)]) for seed in SEEDS]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"C42 validation shard failed: {codes}")
    subprocess.run([sys.executable, str(script), "--config", str(config_path), "--stage", "validation-finalize"], check=True)
    collector = REPO_ROOT / "scripts" / "collect_phase_c42_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True)
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True)
    print(json.dumps({"status": "C42_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c42":
        raise RuntimeError("C42 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C42 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C42 seed: {args.seed}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
