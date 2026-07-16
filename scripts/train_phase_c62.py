#!/usr/bin/env python3
"""Train C62-E2E-CBPI as direct full end-to-end validation shards."""

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

from scripts import c62_common as common  # noqa: E402


SEEDS = common.SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c62_e2e_cbpi_multiseed.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation-seed", "validation-finalize", "reporting-test", "direct-multiseed"),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def current_group_drift(model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    numerators = {group: 0.0 for group in common.GROUPS}
    denominators = {group: 0.0 for group in common.GROUPS}
    for name, parameter in model.named_parameters():
        group = common.group_for_parameter(name)
        initial = initial_state[name].float()
        current = parameter.detach().cpu().float()
        numerators[group] += float(torch.linalg.vector_norm(current - initial))
        denominators[group] += float(torch.linalg.vector_norm(initial))
    return {
        group: numerators[group] / max(denominators[group], 1e-8)
        for group in common.GROUPS
    }


def checkpoint_path(config: Mapping[str, Any], seed: int) -> Path:
    return common.resolve_path(
        str(config["c62"]["initialization_checkpoint"]).replace("{seed}", str(seed))
    )


def write_seed_status(seed_dir: Path, payload: Mapping[str, Any]) -> None:
    report_dir = seed_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_status.json").write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> None:
    common.set_seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    for child in ("reports", "predictions", "checkpoints", "representations"):
        (seed_dir / child).mkdir(parents=True, exist_ok=True)
    init_path = checkpoint_path(config, seed)
    model = common.build_model(config, seed, device, init_path)
    initial_state = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
    }
    inventory = common.parameter_inventory(model)
    inventory.to_csv(seed_dir / "reports" / "trainable_parameter_inventory.csv", index=False)
    optimizer, optimizer_audit = common.optimizer_parameter_groups(model, config)
    optimizer_audit.insert(0, "seed", seed)
    optimizer_audit.to_csv(seed_dir / "reports" / "optimizer_parameter_groups.csv", index=False)
    loaders = common.core.build_loaders(config, rows, ("train", "val"))
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows: List[Dict[str, Any]] = []
    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    for epoch in range(1, max_epochs + 1):
        train_result = common.run_epoch(model, loaders["train"], optimizer, device)
        val_result = common.run_epoch(model, loaders["val"], None, device)
        drift = current_group_drift(model, initial_state)
        epoch_row: Dict[str, Any] = {
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": train_result["metrics"]["bce_loss"],
            "val_auc": val_result["metrics"]["AUC"],
            "val_sensitivity": val_result["metrics"]["Sensitivity"],
            "val_specificity": val_result["metrics"]["Specificity"],
            "val_balanced_accuracy": val_result["metrics"]["Balanced_ACC"],
            "val_prediction_std": val_result["metrics"]["prediction_std"],
            "val_pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
            "mean_relative_parameter_change": float(np.mean(list(drift.values()))),
            "selected_by_val_auc": False,
        }
        for group in common.GROUPS:
            summary = train_result["gradient_summary"][group]
            epoch_row[f"{group}_grad_norm"] = summary["mean_norm"]
            epoch_row[f"{group}_active_batch_count"] = summary["active_batch_count"]
            epoch_row[f"{group}_relative_parameter_change"] = drift[group]
        epoch_rows.append(epoch_row)
        val_auc = float(val_result["metrics"]["AUC"])
        print(json.dumps({"phase": "C62", "seed": seed, "epoch": epoch, "val_auc": val_auc}), flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"C62 seed {seed} produced no Validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    model.eval()
    best_val = common.run_epoch(model, loaders["val"], None, device)
    if best_val["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C62 seed {seed} produced constant Validation predictions")
    update_audit = common.parameter_update_audit(model, initial_state, seed)
    update_audit.to_csv(seed_dir / "reports" / "parameter_update_audit.csv", index=False)
    checkpoint = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "phase": "C62-E2E-CBPI",
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "initial_c61_checkpoint": str(init_path),
            "selection_metric": "validation_auc_only",
            "optimizer_parameter_groups": optimizer_audit.to_dict(orient="records"),
        },
        checkpoint,
    )
    result = {"seed": seed, "best_epoch": best_epoch, "val": best_val}
    val_metric = common.save_split(result, seed_dir, "val")
    pd.DataFrame([val_metric]).to_csv(seed_dir / "reports" / "metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(seed_dir / "reports" / "metrics_by_epoch.csv", index=False)
    best_val["patient_diagnostics"].sort_values("patient_id").to_csv(
        seed_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    run_config = {
        "phase": "C62-E2E-CBPI",
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "validation_auc_only",
        "initial_c61_checkpoint": str(init_path),
        "trainable_parameter_count": int(inventory["parameter_count"].sum()),
        "frozen_predictive_parameter_count": int((~inventory["requires_grad"].astype(bool)).sum()),
        "optimizer_parameter_groups": optimizer_audit.to_dict(orient="records"),
        "full_end_to_end": True,
        "test_role": "reporting_only_after_validation_decision",
    }
    (seed_dir / "reports" / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    write_seed_status(
        seed_dir,
        {
            "phase": "C62-E2E-CBPI",
            "status": "COMPLETE",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "full_end_to_end": True,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    )
    print(json.dumps({"status": "C62_VALIDATION_SEED_COMPLETE", "seed": seed, "best_epoch": best_epoch}), flush=True)
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validation_finalize(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metric_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    update_parts: List[pd.DataFrame] = []
    inventory_parts: List[pd.DataFrame] = []
    optimizer_parts: List[pd.DataFrame] = []
    statuses: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads((seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C62 seed {seed} shard incomplete")
        statuses.append(status)
        metric_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        update_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_update_audit.csv"))
        inventory = pd.read_csv(seed_dir / "reports" / "trainable_parameter_inventory.csv")
        inventory["seed"] = seed
        inventory_parts.append(inventory)
        optimizer_parts.append(pd.read_csv(seed_dir / "reports" / "optimizer_parameter_groups.csv"))
        for source, target in (
            (seed_dir / "checkpoints" / f"seed_{seed}_best.pt", out_dir / "checkpoints" / f"seed_{seed}_best.pt"),
            (seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv", out_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
            (seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz", out_dir / "representations" / f"val_patient_state_seed_{seed}.npz"),
        ):
            shutil.copy2(source, target)
    metrics = pd.concat(metric_parts, ignore_index=True).sort_values("seed")
    metrics.to_csv(out_dir / "reports" / "metrics_by_seed.csv", index=False)
    epochs = pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"])
    epochs.to_csv(out_dir / "reports" / "metrics_by_epoch.csv", index=False)
    updates = pd.concat(update_parts, ignore_index=True).sort_values(["seed", "kind", "group", "parameter_name"])
    updates.to_csv(out_dir / "reports" / "parameter_update_audit.csv", index=False)
    pd.concat(inventory_parts, ignore_index=True).to_csv(out_dir / "reports" / "trainable_parameter_inventory.csv", index=False)
    pd.concat(optimizer_parts, ignore_index=True).to_csv(out_dir / "reports" / "optimizer_parameter_groups.csv", index=False)
    diagnostics = []
    for seed in SEEDS:
        frame = pd.read_csv(out_dir / "seed_runs" / f"seed_{seed}" / "reports" / "patient_diagnostics_val.csv")
        frame.insert(0, "seed", seed)
        diagnostics.append(frame)
    pd.concat(diagnostics, ignore_index=True).sort_values(["seed", "patient_id"]).to_csv(
        out_dir / "reports" / "patient_diagnostics_val.csv", index=False
    )
    common.core.write_summary(metrics, out_dir)
    report_dir = common.resolve_path(config["project"]["report_dir"])
    for name in ("c62_initial_c61_reproduction.csv", "c62_trainable_parameter_inventory.csv", "c62_optimizer_parameter_groups.csv", "c62_gradient_connectivity_audit.csv", "c62_gate.json"):
        source = report_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / "reports" / name)
    (out_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "C62-E2E-CBPI",
                "config": config,
                "seeds": list(SEEDS),
                "parallel_seed_training": True,
                "full_end_to_end": True,
                "selection_metric": "validation_auc_only",
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
                "phase": "C62-E2E-CBPI",
                "status": "VALIDATION_COMPLETE",
                "completed_seeds": list(SEEDS),
                "parallel_seed_training": True,
                "full_end_to_end": True,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "C62_VALIDATION_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def reporting_test(config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    report_dir = common.resolve_path(config["project"]["report_dir"])
    decision_path = report_dir / "c62_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C62 Test requires a frozen Validation decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("validation_decision_frozen_before_test") or decision.get("test_used_for_decision", True):
        raise RuntimeError("C62 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"].astype(str)) != {"val"}:
        raise RuntimeError("C62 reporting-only Test requires Validation-only metrics")
    loader = common.core.build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        checkpoint = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        model = common.build_model(config, seed, device, checkpoint)
        result = common.run_epoch(model, loader, None, device)
        test_metric = common.save_split({"seed": seed, "best_epoch": int(common.checkpoint_payload(checkpoint)["best_epoch"]), "test": result}, out_dir, "test")
        metrics = pd.concat([metrics, pd.DataFrame([test_metric])], ignore_index=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    common.core.write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True, "full_end_to_end": True})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C62_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def direct_multiseed(config_path: Path, config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device) -> None:
    report_dir = common.resolve_path(config["project"]["report_dir"])
    gate_path = report_dir / "c62_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C62 direct execution requires the completed Gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C62_E2E_CBPI_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C62 direct execution requires 20/20 authorized Gate")
    if out_dir.exists():
        raise RuntimeError(f"C62 output directory already exists: {out_dir}")
    for child in ("reports", "predictions", "checkpoints", "representations", "seed_runs"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), "--config", str(config_path), "--stage", "validation-seed", "--seed", str(seed)],
            cwd=REPO_ROOT,
        )
        for seed in SEEDS
    ]
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"C62 validation shards failed: {codes}")
    validation_finalize(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c62_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True, cwd=REPO_ROOT)
    reporting_test(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True, cwd=REPO_ROOT)
    print(json.dumps({"status": "C62_E2E_CBPI_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c62_config(config_path)
    rows = common.manifest_rows(config)
    out_dir = common.resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C62 seed: {args.seed}")
        train_seed(config, rows, int(args.seed), out_dir / "seed_runs" / f"seed_{args.seed}", device)
    elif args.stage == "validation-finalize":
        validation_finalize(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test(config, rows, out_dir, device)
    else:
        direct_multiseed(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
