#!/usr/bin/env python3
"""Train C59-PMESE as direct, independent formal validation shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c59_pmese import C59PMESEModel, HEAD_PREFIXES  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import train_phase_c40 as core  # noqa: E402
from scripts import train_phase_c54 as shared  # noqa: E402


SEEDS = (0, 42, 3407)
shared.C54LRRAModel = C59PMESEModel
shared.HEAD_PREFIXES = HEAD_PREFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c59_pmese_multiseed.yaml")
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
    translated["c54"] = dict(config["c59"])
    return translated


def mark_phase(path: Path, phase: str = "C59-PMESE") -> None:
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
    decision_path = resolve_path(config["project"]["report_dir"]) / "c59_validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("C59 Validation decision must be frozen before reporting-only Test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        not decision.get("validation_decision_frozen_before_test", False)
        or decision.get("test_used_for_decision", True)
        or decision.get("ensemble_used", True)
    ):
        raise RuntimeError("C59 Validation/Test isolation contract failed")
    metrics_path = out_dir / "reports" / "metrics_by_seed.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"]) != {"val"}:
        raise RuntimeError("C59 reporting-only Test requires Validation-only metrics")
    loader = core.build_loaders(config, rows, ("test",))["test"]
    for seed in SEEDS:
        model = C59PMESEModel(config, seed).to(device)
        payload = core.checkpoint_payload(out_dir / "checkpoints" / f"seed_{seed}_best.pt")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"C59 checkpoint seed mismatch for {seed}")
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
            "phase": "C59-PMESE",
            "status": "COMPLETE",
            "test_started_after_validation_decision": True,
            "finished_at": core.timestamp(),
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "C59_REPORTING_TEST_COMPLETE", "seeds": list(SEEDS)}))


def direct_multiseed_stage(
    config_path: Path,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: torch.device,
) -> None:
    gate_path = resolve_path(config["project"]["report_dir"]) / "c59_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C59 direct execution requires the completed gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C59_PMESE_DIRECT_MULTI_SEED_AUTHORIZED" or int(gate.get("passed", 0)) != int(gate.get("total", 0)):
        raise RuntimeError("C59 direct execution requires an authorized gate")
    if (out_dir / "seed_runs").exists():
        raise RuntimeError("C59 formal seed outputs already exist")
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
        raise RuntimeError(f"C59 validation shard failed: {codes}")
    validation_finalize_stage(config, out_dir, device)
    collector = REPO_ROOT / "scripts" / "collect_phase_c59_report.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "validation"], check=True)
    reporting_test_stage(config, rows, out_dir, device)
    subprocess.run([sys.executable, str(collector), "--config", str(config_path), "--stage", "final"], check=True)
    print(json.dumps({"status": "C59_DIRECT_MULTI_SEED_COMPLETE", "seeds": list(SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if str(config.get("phase", "")).lower() != "c59":
        raise RuntimeError("C59 phase contract is missing")
    if [int(seed) for seed in config["training"]["seeds"]] != list(SEEDS):
        raise RuntimeError("C59 formal seeds must remain [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    out_dir = resolve_path(config["project"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "validation-seed":
        if args.seed not in SEEDS:
            raise RuntimeError(f"Unsupported C59 seed: {args.seed}")
        validation_seed_stage(config, rows, int(args.seed), out_dir, device)
    elif args.stage == "validation-finalize":
        validation_finalize_stage(config, out_dir, device)
    elif args.stage == "reporting-test":
        reporting_test_stage(config, rows, out_dir, device)
    else:
        direct_multiseed_stage(config_path, config, rows, out_dir, device)


if __name__ == "__main__":
    main()
