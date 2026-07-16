#!/usr/bin/env python3
"""Run the fifteen patient-level C64 development CV shards."""

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
    parser.add_argument("--config", default="configs/dema_ht_c64_cv.yaml")
    parser.add_argument("--stage", required=True, choices=("fold-seed", "direct-multiseed"))
    parser.add_argument("--candidate")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def route_decision(config: Dict[str, Any]) -> Dict[str, Any]:
    path = common.resolve_path(config["project"]["report_dir"]) / "c64_stage_a_route_decision.json"
    if not path.exists():
        raise RuntimeError(f"C64 Stage-A route decision is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "C64_STAGE_A_ROUTE_SELECTED" or not payload.get("selected_candidate"):
        raise RuntimeError(f"C64 Stage-A route was not selected: {payload.get('status')}")
    return payload


def fold_seed_stage(config: Dict[str, Any], candidate: str, fold: int, seed: int) -> None:
    if candidate not in common.CANDIDATES:
        raise RuntimeError(f"Unsupported C64 candidate: {candidate}")
    if fold not in range(common.FOLD_COUNT) or seed not in common.SEEDS:
        raise RuntimeError(f"Unsupported C64 fold/seed: {fold}/{seed}")
    selected = route_decision(config)
    if selected["selected_candidate"] != candidate:
        raise RuntimeError(f"C64 CV candidate does not match Stage-A route: {candidate}")
    rows = common.manifest_rows(config)
    assignments = common.load_fold_assignments(config)
    fold_data = common.fold_rows(rows, assignments, fold)
    out_dir = common.resolve_path(config["project"]["cv_output_dir"]) / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
    if out_dir.exists():
        raise RuntimeError(f"C64 CV shard output already exists: {out_dir}")
    reporting.write_json(
        out_dir / "run_status.json",
        {
            "phase": "C64-STCV",
            "stage": "cv",
            "status": "RUNNING",
            "candidate": candidate,
            "fold": fold,
            "seed": seed,
            "test_loaded": False,
        },
    )
    try:
        common.train_validation_seed(config, candidate, seed, fold_data, out_dir, run_label=f"cv_fold_{fold}")
    except Exception as exc:
        reporting.write_json(
            out_dir / "run_status.json",
            {
                "phase": "C64-STCV",
                "stage": "cv",
                "status": "FAILED",
                "candidate": candidate,
                "fold": fold,
                "seed": seed,
                "test_loaded": False,
                "error": repr(exc),
            },
        )
        raise
    print(json.dumps({"status": "C64_CV_FOLD_SEED_COMPLETE", "candidate": candidate, "fold": fold, "seed": seed}), flush=True)


def direct_multiseed_stage(config_path: Path, config: Dict[str, Any]) -> None:
    selected = route_decision(config)
    candidate = str(selected["selected_candidate"])
    if not (common.resolve_path(config["project"]["cv_output_dir"]) / "fold_assignments.json").exists():
        raise RuntimeError("C64 fold assignments must be built before CV training")
    output_root = common.resolve_path(config["project"]["cv_output_dir"])
    for fold in range(common.FOLD_COUNT):
        for seed in common.SEEDS:
            shard = output_root / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
            if shard.exists():
                raise RuntimeError(f"C64 CV formal output already exists: {shard}")
    script = Path(__file__).resolve()
    processes: list[tuple[int, int, subprocess.Popen[Any]]] = []
    for fold in range(common.FOLD_COUNT):
        for seed in common.SEEDS:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--stage",
                    "fold-seed",
                    "--candidate",
                    candidate,
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                ]
            )
            processes.append((fold, seed, process))
    codes = [(fold, seed, process.wait()) for fold, seed, process in processes]
    failed = [item for item in codes if item[2] != 0]
    if failed:
        raise RuntimeError(f"C64 CV shards failed: {failed}")
    collector = REPO_ROOT / "scripts" / "collect_phase_c64_cv.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path)], check=True)
    reporting.write_json(
        output_root / "cv_status.json",
        {
            "phase": "C64-STCV",
            "stage": "cv",
            "status": "COMPLETE",
            "candidate": candidate,
            "folds": list(range(common.FOLD_COUNT)),
            "seeds": list(common.SEEDS),
            "test_loaded": False,
        },
    )
    print(json.dumps({"status": "C64_CV_COMPLETE", "candidate": candidate, "folds": common.FOLD_COUNT, "seeds": list(common.SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c64_config(config_path)
    if args.stage == "fold-seed":
        if args.candidate is None or args.fold is None or args.seed is None:
            raise RuntimeError("fold-seed requires --candidate, --fold, and --seed")
        fold_seed_stage(config, args.candidate, int(args.fold), int(args.seed))
    else:
        direct_multiseed_stage(config_path, config)


if __name__ == "__main__":
    main()
