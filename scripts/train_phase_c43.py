#!/usr/bin/env python3
"""Train C43-MCFE as direct, independent formal validation shards."""

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

from dmea_ht.c43_mcfe import (  # noqa: E402
    C43MCFEModel,
    HEAD_PREFIXES,
    trainable_parameter_count,
    trainable_parameter_names,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import train_phase_c40 as base  # noqa: E402


SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c43_mcfe_multiseed.yaml")
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


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    base.set_seed(seed)
    loaders = base.build_loaders(config, rows, ("train", "val"))
    model = C43MCFEModel(config, seed).to(device)
    names = trainable_parameter_names(model)
    if not names or any(not name.startswith(HEAD_PREFIXES) for name in names):
        raise RuntimeError(f"C43 trainable scope violation: {names}")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("sources.")):
        raise RuntimeError("C43 C17 sources must remain frozen")
    count = trainable_parameter_count(model)
    if count > int(config["c43"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C43 capacity contract failed: {count}")
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
        train_result = base.run_epoch(model, loaders["train"], optimizer, device)
        val_result = base.run_epoch(model, loaders["val"], None, device)
        drift = pd.DataFrame(base.parameter_drift_rows(model, initial_state, seed))
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
        raise RuntimeError(f"C43 seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = base.run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C43 seed {seed} produced constant validation predictions")
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
        "drift": base.parameter_drift_rows(model, initial_state, seed),
        "trainable_parameter_names": names,
        "trainable_parameter_count": count,
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
    }


def validation_seed_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> None:
    seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
    if seed_dir.exists():
        raise RuntimeError(f"C43 seed output already exists: {seed_dir}")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "reports" / "run_status.json"
    status = {
        "phase": "C43-MCFE",
        "stage": "validation-seed",
        "status": "RUNNING",
        "seed": seed,
        "started_at": base.timestamp(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deployment_contract": "one_checkpoint_one_model_one_forward",
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    result = train_seed(config, rows, seed, seed_dir, device)
    metric = base.save_split(result, seed_dir, "val")
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
    status.update({"status": "COMPLETE", "finished_at": base.timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C43_VALIDATION_SEED_COMPLETE", "seed": seed}))


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
            raise RuntimeError(f"C43 seed {seed} shard incomplete")
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
    base.write_summary(metrics, out_dir)
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
                "phase": "C43-MCFE",
                "status": "VALIDATION_COMPLETE",
                "started_at": min(item["started_at"] for item in statuses),
                "finished_at": base.timestamp(),
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
    print(json.dumps({"status": "C43_VALIDATION_COMPLETE", "seeds": list(SEEDS)}))


def reporting_test_stage(config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c43_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C43 Validation decision must be frozen before reporting-only Test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("validation_decision_frozen_before_test", False) or decision.get("test_used_for_decision", True) or decision.get("ensemble_used", True):
        raise RuntimeError("C43 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C43 reporting-only Test requires Validation-only metrics")
    loader = base.build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C43MCFEModel(config, seed).to(device)
        payload = base.checkpoint_payload(out_dir / "checkpoints" / f"seed_{seed}_best.pt")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C43 checkpoint seed mismatch for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = base.run_epoch(model, loader, None, device)
        metrics = pd.concat(
            [metrics, pd.DataFrame([base.save_split({"seed": seed, "best_epoch": payload["best_epoch"], "test": result}, out_dir, "test")])],
            ignore_index=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    base.write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "finished_at": base.timestamp()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C43_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(config_path: Path, config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c43_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C43 direct execution requires the completed gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C43_MCFE_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C43 direct execution requires an authorized gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C43 formal seed outputs already exist")
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes = [
        subprocess.Popen([sys.executable, str(script), "--config", str(config_path), "--stage", "validation-seed", "--seed", str(seed)])
        for seed in SEEDS
    ]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"C43 validation shard failed: {codes}")
    subprocess.run([sys.executable, str(script), "--config", str(config_path), "--stage", "validation-finalize"], check=True)
    collector = REPO_ROOT / "scripts" / "collect_phase_c43_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True)
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True)
    print(json.dumps({"status": "C43_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c43":
        raise RuntimeError("C43 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C43 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C43 seed: {args.seed}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
