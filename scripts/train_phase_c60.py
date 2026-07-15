#!/usr/bin/env python3
"""Train C60-EMASE as direct, independent formal validation shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c60_emase import C60EMASEModel, HEAD_PREFIXES, trainable_parameter_count, trainable_parameter_names  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import train_phase_c40 as core  # noqa: E402
from scripts import train_phase_c54 as shared  # noqa: E402


SEEDS = (0, 42, 3407)
shared.C54LRRAModel = C60EMASEModel
shared.HEAD_PREFIXES = HEAD_PREFIXES


def clone_state(value: Mapping[str, torch.Tensor] | torch.nn.Module) -> Dict[str, torch.Tensor]:
    state = value.state_dict() if isinstance(value, torch.nn.Module) else value
    return {key: tensor.detach().cpu().clone() for key, tensor in state.items()}


def update_ema(ema_state: Dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    for key, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        if current.is_floating_point():
            ema_state[key].mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            ema_state[key] = current.clone()


class EMAOptimizer:
    """Optimizer facade that updates one EMA state after every optimizer step."""

    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module, ema_state: Dict[str, torch.Tensor], decay: float) -> None:
        self.optimizer = optimizer
        self.model = model
        self.ema_state = ema_state
        self.decay = float(decay)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        result = self.optimizer.step(*args, **kwargs)
        update_ema(self.ema_state, self.model, self.decay)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c60_emase_multiseed.yaml")
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


def runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    translated = dict(config)
    translated["c54"] = dict(config["c60"])
    return translated


def train_seed(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    seed: int,
    seed_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    core.set_seed(seed)
    loaders = core.build_loaders(config, rows, ("train", "val"))
    model = C60EMASEModel(config, seed).to(device)
    names = trainable_parameter_names(model)
    if not names or any(not name.startswith(HEAD_PREFIXES) for name in names):
        raise RuntimeError(f"C60 trainable scope violation: {names}")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("sources.")):
        raise RuntimeError("C60 C17 sources must remain frozen")
    count = trainable_parameter_count(model)
    if count > int(config["c60"]["trainable_parameter_limit"]):
        raise RuntimeError(f"C60 capacity contract failed: {count}")
    decay = float(config["c60"]["ema_decay"])
    if not 0.0 < decay < 1.0:
        raise RuntimeError(f"C60 EMA decay must be in (0, 1): {decay}")
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
    ema_state = clone_state(model)
    ema_optimizer = EMAOptimizer(optimizer, model, ema_state, decay)
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state: Dict[str, torch.Tensor] | None = None
    epoch_rows = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_result = core.run_epoch(model, loaders["train"], ema_optimizer, device)
        online_state = clone_state(model)
        model.load_state_dict(ema_state, strict=True)
        val_result = core.run_epoch(model, loaders["val"], None, device)
        model.load_state_dict(online_state, strict=True)
        drift = pd.DataFrame(core.parameter_drift_rows(model, initial_state, seed))
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
                "ema_validation": True,
            }
        )
        val_auc = float(val_result["metrics"]["AUC"])
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = clone_state(ema_state)
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"C60 seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    for row in epoch_rows:
        row["selected_by_val_auc"] = int(row["epoch"]) == best_epoch
    val_result = core.run_epoch(model, loaders["val"], None, device)
    if val_result["metrics"]["prediction_std"] <= 0.0:
        raise RuntimeError(f"C60 seed {seed} produced constant validation predictions")
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "source_c17_checkpoint": str(Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))),
            "selection_metric": "validation_auc_only_ema",
            "ema_decay": decay,
            "ema_checkpoint": True,
        },
        seed_dir / "checkpoints" / f"seed_{seed}_best.pt",
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_history": epoch_rows,
        "val": val_result,
        "drift": core.parameter_drift_rows(model, initial_state, seed),
        "trainable_parameter_names": names,
        "trainable_parameter_count": count,
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
    }


shared.train_seed = train_seed


def mark_phase(path: Path, phase: str = "C60-EMASE") -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase"] = phase
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validation_seed_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], seed: int, out_dir: Path, device: torch.device
) -> None:
    shared.validation_seed_stage(runtime_config(config), rows, seed, out_dir, device)
    mark_phase(out_dir / "seed_runs" / f"seed_{seed}" / "reports" / "run_status.json")


def validation_finalize_stage(config: Dict[str, Any], out_dir: Path, device: torch.device) -> None:
    shared.validation_finalize_stage(runtime_config(config), out_dir, device)
    mark_phase(out_dir / "reports" / "run_status.json")


def reporting_test_stage(
    config: Dict[str, Any], rows: Sequence[Dict[str, Any]], out_dir: Path, device: torch.device
) -> None:
    decision_path = resolve_path(config["project"]["report_dir"]) / "c60_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C60 Validation decision must be frozen before reporting-only Test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not decision.get("validation_decision_frozen_before_test", False)
        or decision.get("test_used_for_decision", True)
        or decision.get("ensemble_used", True)
    ):
        raise RuntimeError("C60 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C60 reporting-only Test requires Validation-only metrics")
    loader = core.build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C60EMASEModel(config, seed).to(device)
        payload = core.checkpoint_payload(out_dir / "checkpoints" / f"seed_{seed}_best.pt")
        if int(payload.get("seed", -1)) != seed or not bool(payload.get("ema_checkpoint", False)):
            raise RuntimeError(f"C60 checkpoint EMA/seed contract failed for {seed}")
        model.load_state_dict(payload["model"], strict=True)
        result = core.run_epoch(model, loader, None, device)
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    [
                        core.save_split(
                            {"seed": seed, "best_epoch": payload["best_epoch"], "test": result},
                            out_dir,
                            "test",
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    core.write_summary(metrics, out_dir)
    status_path = out_dir / "reports" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "phase": "C60-EMASE",
            "status": "COMPLETE",
            "test_started_after_validation_decision": True,
            "finished_at": core.timestamp(),
            "ema_checkpoint": True,
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C60_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c60_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C60 direct execution requires the completed gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C60_EMASE_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C60 direct execution requires an authorized gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C60 formal seed outputs already exist")
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
        raise RuntimeError(f"C60 validation shard failed: {codes}")
    validation_finalize_stage(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c60_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True)
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True)
    print(json.dumps({"status": "C60_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c60":
        raise RuntimeError("C60 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C60 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C60 seed: {args.seed}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
