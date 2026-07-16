#!/usr/bin/env python3
"""Train C63-FS-CBPI from a clean task initialization."""

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

from scripts import c63_common as common  # noqa: E402


SEEDS = common.SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c63_from_base_e2e_cbpi_multiseed.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation-seed", "validation-finalize", "reporting-test", "direct-multiseed"),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def current_module_drift(
    model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor]
) -> Dict[str, float]:
    numerators = {group: 0.0 for group in common.MODULE_GROUPS}
    denominators = {group: 0.0 for group in common.MODULE_GROUPS}
    for name, parameter in model.named_parameters():
        group = common.module_group_for_parameter(name)
        initial = initial_state[name].float()
        current = parameter.detach().cpu().float()
        numerators[group] += float(torch.linalg.vector_norm(current - initial))
        denominators[group] += float(torch.linalg.vector_norm(initial))
    return {group: numerators[group] / max(denominators[group], 1e-8) for group in common.MODULE_GROUPS}


def write_status(seed_dir: Path, payload: Mapping[str, Any]) -> None:
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

    model = common.build_from_base_model(config, seed, device)
    initial_state = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer, optimizer_audit = common.optimizer_parameter_groups(model, config)
    inventory = common.parameter_inventory(model)
    initialization = common.initialization_inventory(model, seed, optimizer_audit)
    hashes, overall_hash = common.parameter_hashes(model, seed)
    inventory.insert(0, "seed", seed)
    optimizer_audit.insert(0, "seed", seed)
    inventory.to_csv(seed_dir / "reports" / "trainable_parameter_inventory.csv", index=False)
    initialization.to_csv(seed_dir / "reports" / "initialization_inventory.csv", index=False)
    optimizer_audit.to_csv(seed_dir / "reports" / "optimizer_parameter_groups.csv", index=False)
    hashes["overall_parameter_hash"] = overall_hash
    hashes.to_csv(seed_dir / "reports" / "initial_parameter_hash.csv", index=False)

    loaders = common.build_loaders(config, rows, seed, ("train", "val"))
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
        drift = current_module_drift(model, initial_state)
        epoch_row: Dict[str, Any] = {
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
            "val_pairwise_inversion_count": val_result["metrics"]["pairwise_inversion_count"],
            "mean_relative_parameter_change": float(np.mean(list(drift.values()))),
            "selected_by_val_auc": False,
        }
        for group in common.MODULE_GROUPS:
            summary = train_result["gradient_summary"][group]
            epoch_row[f"{group}_grad_norm"] = summary["mean_norm"]
            epoch_row[f"{group}_active_batch_count"] = summary["active_batch_count"]
            epoch_row[f"{group}_relative_parameter_change"] = drift[group]
        epoch_rows.append(epoch_row)
        val_auc = float(val_result["metrics"]["AUC"])
        print(json.dumps({"phase": "C63-FS-CBPI", "seed": seed, "epoch": epoch, "val_auc": val_auc}), flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"C63 seed {seed} produced no Validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    model.eval()
    best_val = common.run_epoch(model, loaders["val"], None, device)
    if best_val["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C63 seed {seed} produced constant Validation predictions")
    update_audit = common.parameter_update_audit(model, initial_state, seed)
    update_audit.to_csv(seed_dir / "reports" / "parameter_update_audit.csv", index=False)
    checkpoint = seed_dir / "checkpoints" / f"seed_{seed}_best.pt"
    torch.save(
        {
            "phase": "C63-FS-CBPI",
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "initialization_mode": "from_base",
            "task_checkpoint_loaded_for_initialization": False,
            "selection_metric": "validation_auc_only",
            "optimizer_parameter_groups": optimizer_audit.to_dict(orient="records"),
            "initial_parameter_hash": overall_hash,
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
    (seed_dir / "reports" / "run_config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "selection_metric": "validation_AUC_only",
                "initialization_mode": "from_base",
                "task_checkpoint_loaded_for_initialization": False,
                "trainable_parameter_count": int(inventory["parameter_count"].sum()),
                "frozen_predictive_parameter_count": int((~inventory["requires_grad"].astype(bool)).sum()),
                "initial_parameter_hash": overall_hash,
                "optimizer_parameter_groups": optimizer_audit.to_dict(orient="records"),
                "full_end_to_end": True,
                "test_role": "reporting_only_after_validation_decision",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_status(
        seed_dir,
        {
            "phase": "C63-FS-CBPI",
            "status": "COMPLETE",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "initialization_mode": "from_base",
            "task_checkpoint_loaded_for_initialization": False,
            "full_end_to_end": True,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    )
    print(json.dumps({"status": "C63_VALIDATION_SEED_COMPLETE", "seed": seed, "best_epoch": best_epoch}), flush=True)
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validation_finalize(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    metric_parts: List[pd.DataFrame] = []
    epoch_parts: List[pd.DataFrame] = []
    update_parts: List[pd.DataFrame] = []
    inventory_parts: List[pd.DataFrame] = []
    initialization_parts: List[pd.DataFrame] = []
    hash_parts: List[pd.DataFrame] = []
    optimizer_parts: List[pd.DataFrame] = []
    for seed in SEEDS:
        seed_dir = out_dir / "seed_runs" / f"seed_{seed}"
        status = json.loads((seed_dir / "reports" / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"C63 seed {seed} shard incomplete")
        metric_parts.append(pd.read_csv(seed_dir / "reports" / "metrics.csv"))
        epoch_parts.append(pd.read_csv(seed_dir / "reports" / "metrics_by_epoch.csv"))
        update_parts.append(pd.read_csv(seed_dir / "reports" / "parameter_update_audit.csv"))
        inventory_parts.append(pd.read_csv(seed_dir / "reports" / "trainable_parameter_inventory.csv"))
        initialization_parts.append(pd.read_csv(seed_dir / "reports" / "initialization_inventory.csv"))
        hash_parts.append(pd.read_csv(seed_dir / "reports" / "initial_parameter_hash.csv"))
        optimizer_parts.append(pd.read_csv(seed_dir / "reports" / "optimizer_parameter_groups.csv"))
        for source, target in (
            (seed_dir / "checkpoints" / f"seed_{seed}_best.pt", out_dir / "checkpoints" / f"seed_{seed}_best.pt"),
            (seed_dir / "predictions" / f"val_predictions_seed_{seed}.csv", out_dir / "predictions" / f"val_predictions_seed_{seed}.csv"),
            (seed_dir / "representations" / f"val_patient_state_seed_{seed}.npz", out_dir / "representations" / f"val_patient_state_seed_{seed}.npz"),
        ):
            shutil.copy2(source, target)
    out_reports = out_dir / "reports"
    out_reports.mkdir(parents=True, exist_ok=True)
    pd.concat(metric_parts, ignore_index=True).sort_values("seed").to_csv(out_reports / "metrics_by_seed.csv", index=False)
    pd.concat(epoch_parts, ignore_index=True).sort_values(["seed", "epoch"]).to_csv(out_reports / "metrics_by_epoch.csv", index=False)
    pd.concat(update_parts, ignore_index=True).sort_values(["seed", "kind", "module_group", "parameter_name"]).to_csv(out_reports / "parameter_update_audit.csv", index=False)
    pd.concat(inventory_parts, ignore_index=True).to_csv(out_reports / "trainable_parameter_inventory.csv", index=False)
    pd.concat(initialization_parts, ignore_index=True).to_csv(out_reports / "initialization_inventory.csv", index=False)
    pd.concat(hash_parts, ignore_index=True).to_csv(out_reports / "initial_parameter_hash_by_seed.csv", index=False)
    pd.concat(optimizer_parts, ignore_index=True).to_csv(out_reports / "optimizer_parameter_groups.csv", index=False)
    diagnostics: List[pd.DataFrame] = []
    for seed in SEEDS:
        frame = pd.read_csv(out_dir / "seed_runs" / f"seed_{seed}" / "reports" / "patient_diagnostics_val.csv")
        frame.insert(0, "seed", seed)
        diagnostics.append(frame)
    pd.concat(diagnostics, ignore_index=True).sort_values(["seed", "patient_id"]).to_csv(
        out_reports / "patient_diagnostics_val.csv", index=False
    )
    report_dir = common.resolve_path(config["project"]["report_dir"])
    for name in (
        "c63_initialization_inventory.csv",
        "c63_initial_parameter_hash_by_seed.csv",
        "c63_task_checkpoint_exclusion_audit.csv",
        "c63_trainable_parameter_inventory.csv",
        "c63_optimizer_parameter_groups.csv",
        "c63_gradient_connectivity_audit.csv",
        "c63_gate.json",
    ):
        source = report_dir / name
        if source.exists():
            shutil.copy2(source, out_reports / name)
    common.core.write_summary(pd.concat(metric_parts, ignore_index=True), out_dir)
    (out_reports / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "C63-FS-CBPI",
                "config": config,
                "seeds": list(SEEDS),
                "parallel_seed_training": True,
                "initialization_mode": "from_base",
                "task_checkpoint_loaded_for_initialization": False,
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
    (out_reports / "run_status.json").write_text(
        json.dumps(
            {
                "phase": "C63-FS-CBPI",
                "status": "VALIDATION_COMPLETE",
                "completed_seeds": list(SEEDS),
                "parallel_seed_training": True,
                "initialization_mode": "from_base",
                "task_checkpoint_loaded_for_initialization": False,
                "full_end_to_end": True,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "C63_VALIDATION_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def reporting_test(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    report_dir = common.resolve_path(config["project"]["report_dir"])
    decision_path = report_dir / "c63_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C63 Test requires a frozen Validation decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("validation_decision_frozen_before_test") or decision.get("test_used_for_decision", True):
        raise RuntimeError("C63 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"].astype(str)) != {"val"}:
        raise RuntimeError("C63 reporting-only Test requires Validation-only metrics")
    for seed in SEEDS:
        loader = common.build_loaders(config, rows, seed, ("test",))["test"]
        checkpoint = out_dir / "checkpoints" / f"seed_{seed}_best.pt"
        model = common.build_from_c63_checkpoint(config, seed, device, checkpoint)
        result = common.run_epoch(model, loader, None, device)
        test_metric = common.save_split(
            {
                "seed": seed,
                "best_epoch": int(common.checkpoint_payload(checkpoint)["best_epoch"]),
                "test": result,
            },
            out_dir,
            "test",
        )
        metrics = pd.concat([metrics, pd.DataFrame([test_metric])], ignore_index=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    common.core.write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"status": "COMPLETE", "test_started_after_validation_decision": True})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C63_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def direct_multiseed(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    report_dir = common.resolve_path(config["project"]["report_dir"])
    gate_path = report_dir / "c63_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C63 direct execution requires the completed Gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "C63_FROM_BASE_E2E_DIRECT_MULTI_SEED_AUTHORIZED"
        or int(gate.get("passed", 0)) != 24
        or int(gate.get("total", 0)) != 24
    ):
        raise RuntimeError("C63 direct execution requires the authorized 24/24 Gate")
    if out_dir.exists():
        raise RuntimeError(f"C63 output directory already exists: {out_dir}")
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
        raise RuntimeError(f"C63 validation shards failed: {codes}")
    validation_finalize(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c63_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True, cwd=REPO_ROOT)
    reporting_test(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True, cwd=REPO_ROOT)
    print(json.dumps({"status": "C63_FROM_BASE_E2E_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}), flush=True)


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c63_config(config_path)
    rows = common.manifest_rows(config)
    out_dir = common.resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C63 seed: {args.seed}")
        train_seed(config, rows, int(args.seed), out_dir / "seed_runs" / f"seed_{args.seed}", device)
    elif args.stage == "validation-finalize":
        validation_finalize(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test(config, rows, out_dir, device)
    else:
        direct_multiseed(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
