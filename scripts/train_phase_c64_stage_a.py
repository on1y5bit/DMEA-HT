#!/usr/bin/env python3
"""Run the nine C64 Stage-A fixed-split Validation shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402
from scripts import c64_reporting as reporting  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c64_stage_a_head_only.yaml")
    parser.add_argument("--projector-config", default="configs/dema_ht_c64_stage_a_projector_cbpi.yaml")
    parser.add_argument("--full-config", default="configs/dema_ht_c64_stage_a_full_finetune.yaml")
    parser.add_argument("--stage", required=True, choices=("candidate-seed", "direct-multiseed"))
    parser.add_argument("--candidate", choices=common.CANDIDATES)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def gate_path(config: Dict[str, Any]) -> Path:
    return common.resolve_path(config["project"]["report_dir"]) / "c64_gate.json"


def require_gate(config: Dict[str, Any]) -> None:
    path = gate_path(config)
    if not path.exists():
        raise RuntimeError(f"C64 gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C64_STAGED_TUNING_CV_AUTHORIZED" or int(payload.get("passed", 0)) != int(payload.get("total", 0)):
        raise RuntimeError(f"C64 gate is not authorized: {payload.get('status')}")


def candidate_seed_stage(config_path: Path, config: Dict[str, Any], candidate: str, seed: int) -> None:
    if str(config.get("candidate")) != candidate:
        raise RuntimeError(f"C64 candidate/config mismatch: {candidate} vs {config.get('candidate')}")
    if seed not in common.SEEDS:
        raise RuntimeError(f"Unsupported C64 seed: {seed}")
    require_gate(config)
    rows = common.manifest_rows(config)
    out_dir = common.resolve_path(config["project"]["stage_a_output_dir"]) / candidate / "seed_runs" / f"seed_{seed}"
    if out_dir.exists():
        raise RuntimeError(f"C64 Stage-A shard output already exists: {out_dir}")
    reporting.write_json(
        out_dir / "run_status.json",
        {
            "phase": "C64-STCV",
            "stage": "stage_a",
            "status": "RUNNING",
            "candidate": candidate,
            "seed": seed,
            "test_loaded": False,
        },
    )
    try:
        common.train_validation_seed(config, candidate, seed, rows, out_dir, run_label="stage_a")
    except Exception as exc:
        reporting.write_json(
            out_dir / "run_status.json",
            {
                "phase": "C64-STCV",
                "stage": "stage_a",
                "status": "FAILED",
                "candidate": candidate,
                "seed": seed,
                "test_loaded": False,
                "error": repr(exc),
            },
        )
        raise
    print(json.dumps({"status": "C64_STAGE_A_SEED_COMPLETE", "candidate": candidate, "seed": seed}), flush=True)


def direct_multiseed_stage(args: argparse.Namespace, configs: Sequence[Path]) -> None:
    loaded = [common.load_c64_config(path) for path in configs]
    for config in loaded:
        require_gate(config)
    output_root = common.resolve_path(loaded[0]["project"]["stage_a_output_dir"])
    for config in loaded:
        candidate = str(config["candidate"])
        for seed in common.SEEDS:
            shard = output_root / candidate / "seed_runs" / f"seed_{seed}"
            if shard.exists():
                raise RuntimeError(f"C64 Stage-A formal output already exists: {shard}")
    script = Path(__file__).resolve()
    processes: list[tuple[str, int, subprocess.Popen[Any]]] = []
    for config_path, config in zip(configs, loaded):
        candidate = str(config["candidate"])
        for seed in common.SEEDS:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--stage",
                    "candidate-seed",
                    "--candidate",
                    candidate,
                    "--seed",
                    str(seed),
                ]
            )
            processes.append((candidate, seed, process))
    codes = [(candidate, seed, process.wait()) for candidate, seed, process in processes]
    failed = [item for item in codes if item[2] != 0]
    if failed:
        raise RuntimeError(f"C64 Stage-A shards failed: {failed}")
    collector = REPO_ROOT / "scripts" / "collect_phase_c64_stage_a.py"
    subprocess.run(
        [
            sys.executable,
            str(collector),
            "--head-config",
            str(configs[0]),
            "--projector-config",
            str(configs[1]),
            "--full-config",
            str(configs[2]),
        ],
        check=True,
    )
    reporting.write_json(
        output_root / "stage_a_status.json",
        {"phase": "C64-STCV", "stage": "stage_a", "status": "COMPLETE", "seeds": list(common.SEEDS), "test_loaded": False},
    )
    print(json.dumps({"status": "C64_STAGE_A_COMPLETE", "candidates": list(common.CANDIDATES), "seeds": list(common.SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c64_config(config_path)
    if args.stage == "candidate-seed":
        if args.candidate is None or args.seed is None:
            raise RuntimeError("candidate-seed requires --candidate and --seed")
        candidate_seed_stage(config_path, config, args.candidate, int(args.seed))
        return
    direct_multiseed_stage(
        args,
        [
            common.resolve_path(args.config),
            common.resolve_path(args.projector_config),
            common.resolve_path(args.full_config),
        ],
    )


if __name__ == "__main__":
    main()
