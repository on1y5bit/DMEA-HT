#!/usr/bin/env python3
"""Run direct C65-B patient-level five-fold by three-head-Seed CV."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as c64  # noqa: E402
from scripts import c65b_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65b.yaml")
    parser.add_argument("--stage", choices=("cv", "fold-seed"), default="cv")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def fold_seed_stage(config: dict, fold: int, seed: int) -> None:
    if fold not in range(common.FOLD_COUNT) or seed not in common.SEEDS:
        raise RuntimeError(f"Unsupported C65-B fold/seed: {fold}/{seed}")
    rows = common.development_rows(config)
    assignments = common.fold_assignments(config)
    fold_rows = c64.fold_rows(rows, assignments, fold)
    out_dir = common.cv_dir(config) / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
    if out_dir.exists():
        raise RuntimeError(f"C65-B CV shard output already exists: {out_dir}")
    common.write_status(
        out_dir / "run_status.json",
        {"phase": "C65-VACS", "stage": "cv", "status": "RUNNING", "candidate": common.CANDIDATE, "fold": fold, "seed": seed, "backbone_seed": 42, "test_loaded": False},
    )
    try:
        common.train_validation_seed(config, seed, fold_rows, out_dir)
    except Exception as exc:
        common.write_status(
            out_dir / "run_status.json",
            {"phase": "C65-VACS", "stage": "cv", "status": "FAILED", "candidate": common.CANDIDATE, "fold": fold, "seed": seed, "backbone_seed": 42, "test_loaded": False, "error": repr(exc)},
        )
        raise
    print(json.dumps({"status": "C65B_CV_FOLD_SEED_COMPLETE", "fold": fold, "seed": seed}), flush=True)


def direct_cv(config_path: Path, config: dict) -> None:
    gate_path = common.report_dir(config) / "c65b_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C65B_COMMON_BACKBONE_CV_AUTHORIZED":
        raise RuntimeError(f"C65-B gate is not passed: {gate.get('status')}")
    output_root = common.cv_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    script = Path(__file__).resolve()
    for fold in range(common.FOLD_COUNT):
        for seed in common.SEEDS:
            shard = output_root / f"fold_{fold}" / "seed_runs" / f"seed_{seed}"
            if shard.exists():
                raise RuntimeError(f"C65-B formal output already exists: {shard}")
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(script), "--config", str(config_path), "--stage", "fold-seed", "--fold", str(fold), "--seed", str(seed)]
                )
            )
    codes = [process.wait() for process in processes]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"C65-B CV shards failed: {codes}")
    collector = REPO_ROOT / "scripts" / "collect_phase_c65_cv.py"
    subprocess.run([sys.executable, str(collector), "--config", str(config_path)], check=True)
    common.write_status(output_root / "cv_status.json", {"phase": "C65-VACS", "stage": "cv", "status": "COMPLETE", "folds": list(range(common.FOLD_COUNT)), "seeds": list(common.SEEDS), "backbone_seed": 42, "test_loaded": False})
    print(json.dumps({"status": "C65B_CV_COMPLETE", "folds": common.FOLD_COUNT, "seeds": list(common.SEEDS)}))


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c65b_config(config_path)
    if args.stage == "fold-seed":
        if args.fold is None or args.seed is None:
            raise RuntimeError("fold-seed requires --fold and --seed")
        fold_seed_stage(config, int(args.fold), int(args.seed))
    else:
        direct_cv(config_path, config)


if __name__ == "__main__":
    main()
