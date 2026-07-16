#!/usr/bin/env python3
"""Run the three fixed-epoch C64 final development-training shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402
from scripts import c64_reporting as reporting  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c64_final.yaml")
    parser.add_argument("--stage", required=True, choices=("seed", "direct-multiseed"))
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def contract(config: Dict[str, Any]) -> Dict[str, Any]:
    path = common.resolve_path(config["project"]["report_dir"]) / "c64_final_training_contract.json"
    if not path.exists():
        raise RuntimeError(f"C64 final training contract is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C64_FINAL_TRAINING_CONTRACT_FROZEN" or payload.get("test_loaded", True):
        raise RuntimeError(f"C64 final training contract is not frozen: {payload.get('status')}")
    return payload


def train_seed(config: Dict[str, Any], seed: int) -> None:
    if seed not in common.SEEDS:
        raise RuntimeError(f"Unsupported C64 seed: {seed}")
    frozen = contract(config)
    candidate = str(frozen["selected_candidate"])
    fixed_epoch = int(frozen["selected_epochs_by_seed"][str(seed)])
    rows = common.manifest_rows(config)
    development = common.all_development_as_train(rows)
    out_dir = common.resolve_path(config["project"]["final_output_dir"]) / "seed_runs" / f"seed_{seed}"
    if out_dir.exists():
        raise RuntimeError(f"C64 final seed output already exists: {out_dir}")
    reporting.write_json(
        out_dir / "run_status.json",
        {
            "phase": "C64-STCV",
            "stage": "final",
            "status": "RUNNING",
            "candidate": candidate,
            "seed": seed,
            "fixed_epoch": fixed_epoch,
            "early_stopping": False,
            "test_loaded": False,
        },
    )
    try:
        common.train_fixed_seed(config, candidate, seed, development, fixed_epoch, out_dir)
    except Exception as exc:
        reporting.write_json(
            out_dir / "run_status.json",
            {
                "phase": "C64-STCV",
                "stage": "final",
                "status": "FAILED",
                "candidate": candidate,
                "seed": seed,
                "fixed_epoch": fixed_epoch,
                "test_loaded": False,
                "error": repr(exc),
            },
        )
        raise
    print(json.dumps({"status": "C64_FINAL_SEED_COMPLETE", "candidate": candidate, "seed": seed, "fixed_epoch": fixed_epoch}), flush=True)


def direct_multiseed(config_path: Path, config: Dict[str, Any]) -> None:
    frozen = contract(config)
    output_root = common.resolve_path(config["project"]["final_output_dir"])
    for seed in common.SEEDS:
        shard = output_root / "seed_runs" / f"seed_{seed}"
        if shard.exists():
            raise RuntimeError(f"C64 final formal output already exists: {shard}")
    script = Path(__file__).resolve()
    processes: list[tuple[int, subprocess.Popen[Any]]] = []
    for seed in common.SEEDS:
        process = subprocess.Popen(
            [sys.executable, str(script), "--config", str(config_path), "--stage", "seed", "--seed", str(seed)]
        )
        processes.append((seed, process))
    codes = [(seed, process.wait()) for seed, process in processes]
    failed = [item for item in codes if item[1] != 0]
    if failed:
        raise RuntimeError(f"C64 final shards failed: {failed}")
    reporting.write_json(
        output_root / "final_training_status.json",
        {
            "phase": "C64-STCV",
            "stage": "final",
            "status": "FINAL_DEVELOPMENT_TRAINING_COMPLETE",
            "candidate": frozen["selected_candidate"],
            "seeds": list(common.SEEDS),
            "fixed_epochs": frozen["selected_epochs_by_seed"],
            "test_loaded": False,
        },
    )
    print(json.dumps({"status": "C64_FINAL_DEVELOPMENT_TRAINING_COMPLETE", "seeds": list(common.SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c64_config(config_path)
    if args.stage == "seed":
        if args.seed is None:
            raise RuntimeError("seed stage requires --seed")
        train_seed(config, int(args.seed))
    else:
        direct_multiseed(config_path, config)


if __name__ == "__main__":
    main()
