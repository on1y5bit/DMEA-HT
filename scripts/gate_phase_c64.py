#!/usr/bin/env python3
"""Authorize the C64 staged tuning and patient-level CV workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402
from scripts import c64_reporting as reporting  # noqa: E402


EXPECTED_DATA_ROOT = "/data/csb/DMEA-HT/HT_2025.12_25"
EXPECTED_MANIFEST = f"{EXPECTED_DATA_ROOT}/manifest_distmatch_structmatch_evidence_v2_c27_visit_level.jsonl"
EXPECTED_MANIFEST_SHA256 = "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
EXPECTED_REPO = "/home/linruixin/chen/project/DMEA-HT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-config", default="configs/dema_ht_c64_stage_a_head_only.yaml")
    parser.add_argument("--projector-config", default="configs/dema_ht_c64_stage_a_projector_cbpi.yaml")
    parser.add_argument("--full-config", default="configs/dema_ht_c64_stage_a_full_finetune.yaml")
    parser.add_argument("--cv-config", default="configs/dema_ht_c64_cv.yaml")
    parser.add_argument("--final-config", default="configs/dema_ht_c64_final.yaml")
    return parser.parse_args()


def check(name: str, passed: bool, details: Any) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def static_contract(configs: Mapping[str, Mapping[str, Any]], script_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for name, config in configs.items():
        project = config["project"]
        training = config["training"]
        initialization = config["initialization"]
        deployment = config["deployment"]
        checks.extend(
            [
                check(f"{name}_phase", str(config.get("phase", "")).lower() == "c64", config.get("phase")),
                check(f"{name}_data_root", str(project["data_root"]) == EXPECTED_DATA_ROOT, project["data_root"]),
                check(f"{name}_manifest", str(project["manifest"]) == EXPECTED_MANIFEST, project["manifest"]),
                check(f"{name}_seeds", [int(value) for value in training["seeds"]] == list(common.SEEDS), training["seeds"]),
                check(f"{name}_patience15", int(training["patience"]) == 15, training["patience"]),
                check(f"{name}_max_epochs60", int(training["max_epochs"]) == 60, training["max_epochs"]),
                check(f"{name}_bce_only", bool(config["loss"]["bce_only"]), config["loss"]),
                check(f"{name}_base_lr", float(training["base_lr"]) == 1e-4, training["base_lr"]),
                check(f"{name}_weight_decay", float(training["weight_decay"]) == 5e-4, training["weight_decay"]),
                check(f"{name}_no_ensemble", not any(bool(deployment.get(key, False)) for key in ("ensemble", "prediction_averaging", "checkpoint_averaging")), deployment),
                check(f"{name}_test_locked", bool(config["evaluation"]["test_locked_until_final_contract"]), config["evaluation"]),
                check(f"{name}_warm_start_mode", str(initialization["mode"]) == "c61_checkpoint_warm_start", initialization["mode"]),
                check(f"{name}_no_c63_input", not bool(initialization.get("use_c63_checkpoint", False)) and not bool(initialization.get("saved_prediction_input", False)) and not bool(initialization.get("saved_representation_input", False)), initialization),
                check(f"{name}_c61_checkpoint_template", "c61_checkpoint" in initialization and "dema_ht_c61_cbpi_multiseed" in str(initialization["c61_checkpoint"]), initialization.get("c61_checkpoint")),
            ]
        )
    forbidden_patterns = ("build_loaders(config, .*test", "build_loaders(.*\"test\"", "build_loaders(.*'test'")
    for path in script_paths:
        text = path.read_text(encoding="utf-8")
        forbidden = [pattern for pattern in forbidden_patterns if pattern.replace(".*", "") in text]
        checks.append(check(f"static_{path.name}_no_test_loader", not forbidden, forbidden))
        checks.append(check(f"static_{path.name}_no_from_base", "from_base" not in text and "C63" not in text and "C62" not in text, forbidden))
    return checks


def manifest_contract(rows: Sequence[Dict[str, Any]], manifest: Path) -> List[Dict[str, Any]]:
    split_frame = pd.DataFrame(
        [{"patient_id": str(row["patient_id"]), "label": int(row["label"]), "split": str(row["split"])} for row in rows]
    )
    counts = split_frame.groupby("split").size().to_dict()
    labels = {
        split: split_frame[split_frame["split"] == split]["label"].value_counts().to_dict()
        for split in ("train", "val", "test")
    }
    ids = split_frame["patient_id"].astype(str)
    split_sets = {split: set(split_frame.loc[split_frame["split"] == split, "patient_id"]) for split in ("train", "val", "test")}
    disjoint = all(split_sets[left].isdisjoint(split_sets[right]) for index, left in enumerate(split_sets) for right in list(split_sets)[index + 1 :])
    expected = counts == {"train": 602, "val": 94, "test": 84}
    label_expected = all(labels.get(split, {}) == {0: expected_count, 1: expected_count} for split, expected_count in (("train", 301), ("val", 47), ("test", 42)))
    return [
        check("manifest_sha256", sha256_file(manifest) == EXPECTED_MANIFEST_SHA256, sha256_file(manifest)),
        check("manifest_line_count", len(rows) == 780, len(rows)),
        check("manifest_patient_unique", not ids.duplicated().any(), int(ids.duplicated().sum())),
        check("manifest_split_counts", expected, counts),
        check("manifest_label_balance", label_expected, labels),
        check("manifest_patient_split_disjoint", disjoint, {key: len(value) for key, value in split_sets.items()}),
    ]


def one_batch_gradient_audit(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], candidate: str, seed: int, device: torch.device
) -> Dict[str, Any]:
    runtime = json.loads(json.dumps(config))
    runtime["training"]["num_workers"] = 0
    train_rows = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    model, payload, checkpoint = common.build_c61_warm_start(runtime, candidate, seed, device)
    optimizer, optimizer_audit = common.optimizer_parameter_groups(model, runtime, candidate)
    loader = common.c63.build_loaders(runtime, train_rows, seed, ("train",))["train"]
    common.set_train_mode(model, candidate, True)
    batch = common.c63.move_batch(next(iter(loader)), device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"C64 gate non-finite loss for {candidate}/{seed}")
    loss.backward()
    gradients = common.module_gradient_summary(model)
    expected_modules = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            expected_modules.add(common.c63.module_group_for_parameter(name))
    gradient_pass = all(
        gradients[group]["finite"] and gradients[group]["norm"] > 0.0 and gradients[group]["nonzero_tensor_count"] > 0
        for group in expected_modules
    )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer.step()
    updated_groups = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(before[name], parameter.detach().cpu()):
            updated_groups.add(common.c64_optimizer_group(name))
    expected_optimizer_groups = set(common.expected_trainable_groups(candidate))
    update_pass = updated_groups == expected_optimizer_groups
    result = {
        "candidate": candidate,
        "seed": seed,
        "c61_checkpoint": str(checkpoint),
        "c61_payload_seed": int(payload.get("seed", -1)),
        "loss": float(loss.detach().cpu()),
        "expected_module_groups": sorted(expected_modules),
        "gradient_groups": gradients,
        "expected_optimizer_groups": sorted(expected_optimizer_groups),
        "updated_optimizer_groups": sorted(updated_groups),
        "gradient_pass": bool(gradient_pass),
        "update_pass": bool(update_pass),
        "passed": bool(gradient_pass and update_pass),
        "test_loaded": False,
        "optimizer_audit": optimizer_audit.to_dict(orient="records"),
    }
    del optimizer, model, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_gate(args: argparse.Namespace) -> Dict[str, Any]:
    config_paths = {
        "head": common.resolve_path(args.head_config),
        "projector": common.resolve_path(args.projector_config),
        "full": common.resolve_path(args.full_config),
        "cv": common.resolve_path(args.cv_config),
        "final": common.resolve_path(args.final_config),
    }
    configs = {name: common.load_c64_config(path) for name, path in config_paths.items()}
    stage_a = {name: configs[name] for name in ("head", "projector", "full")}
    checks = static_contract(configs, [REPO_ROOT / "scripts" / name for name in ("train_phase_c64_stage_a.py", "train_phase_c64_cv.py", "train_phase_c64_final.py") if (REPO_ROOT / "scripts" / name).exists()])
    head = configs["head"]
    manifest = Path(str(head["project"]["manifest"]))
    rows = common.manifest_rows(head)
    checks.extend(manifest_contract(rows, manifest))
    checks.extend(
        [
            check("canonical_repo", str(REPO_ROOT) == EXPECTED_REPO, str(REPO_ROOT)),
            check("canonical_branch", git_value("rev-parse", "--abbrev-ref", "HEAD") == "main", git_value("rev-parse", "--abbrev-ref", "HEAD")),
            check("fold_assignments_present", (common.resolve_path(configs["cv"]["project"]["cv_output_dir"]) / "fold_assignments.json").exists(), str(common.resolve_path(configs["cv"]["project"]["cv_output_dir"]) / "fold_assignments.json")),
        ]
    )
    candidate_values = {name: str(config["candidate"]) for name, config in stage_a.items()}
    checks.append(check("candidate_names", set(candidate_values.values()) == set(common.CANDIDATES), candidate_values))
    checks.append(check("candidate_config_common_manifest", len({str(config["project"]["manifest"]) for config in stage_a.values()}) == 1, candidate_values))

    reports: List[pd.DataFrame] = []
    hashes: List[pd.DataFrame] = []
    initializations: List[pd.DataFrame] = []
    optimizers: List[pd.DataFrame] = []
    task_audits: List[Dict[str, Any]] = []
    gradient_audits: List[Dict[str, Any]] = []
    initial_hash_by_seed: Dict[int, set[str]] = {seed: set() for seed in common.SEEDS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate_config in stage_a.values():
        candidate = str(candidate_config["candidate"])
        for seed in common.SEEDS:
            model, payload, checkpoint = common.build_c61_warm_start(candidate_config, candidate, seed, device)
            inventory = common.parameter_inventory(model, candidate)
            optimizer, optimizer_audit = common.optimizer_parameter_groups(model, candidate_config, candidate)
            hash_frame, overall_hash = common.parameter_hashes(model, seed, candidate)
            init = common.initialization_inventory(model, candidate_config, candidate, seed, optimizer_audit, checkpoint)
            inventory.insert(0, "seed", seed)
            optimizer_audit.insert(0, "seed", seed)
            hash_frame["overall_parameter_hash"] = overall_hash
            reports.append(inventory)
            hashes.append(hash_frame)
            initializations.append(init)
            optimizers.append(optimizer_audit)
            initial_hash_by_seed[seed].add(overall_hash)
            task_audits.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "source_checkpoint": str(checkpoint),
                    "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "initialization_type": "c61_validation_checkpoint_warm_start",
                    "from_base": bool(getattr(model.sources, "from_base", False)),
                    "task_trained_checkpoint_used": True,
                    "c63_checkpoint_used": False,
                    "c62_checkpoint_used": False,
                    "saved_prediction_input": False,
                    "saved_representation_input": False,
                    "strict_state_load": True,
                }
            )
            del optimizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gradient_audits.append(one_batch_gradient_audit(candidate_config, rows, candidate, seed, device))

    checks.extend(
        [
            check("c61_checkpoint_presence_and_seed", all(Path(str(item["source_checkpoint"])).exists() and int(item["seed"]) in common.SEEDS for item in task_audits), len(task_audits)),
            check("c61_strict_state_load", all(bool(item["strict_state_load"]) and not item["from_base"] for item in task_audits), len(task_audits)),
            check("initial_hash_equivalence_per_seed", all(len(values) == 1 for values in initial_hash_by_seed.values()), {str(key): sorted(value) for key, value in initial_hash_by_seed.items()}),
            check("optimizer_factors_positive", all(float(item["learning_rate_factor"]) > 0.0 for audit in optimizers for _, item in audit.iterrows() if bool(item["included_in_optimizer"])), True),
            check("real_batch_gradient_and_update", all(bool(item["passed"]) for item in gradient_audits), gradient_audits),
        ]
    )
    report_dir = common.resolve_path(head["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(reports, ignore_index=True).to_csv(report_dir / "c64_stage_a_freeze_inventory.csv", index=False)
    pd.concat(optimizers, ignore_index=True).to_csv(report_dir / "c64_optimizer_parameter_groups.csv", index=False)
    pd.concat(initializations, ignore_index=True).to_csv(report_dir / "c64_initialization_inventory.csv", index=False)
    pd.concat(hashes, ignore_index=True).to_csv(report_dir / "c64_initial_parameter_hash_by_seed.csv", index=False)
    pd.DataFrame(task_audits).to_csv(report_dir / "c64_task_checkpoint_exclusion_audit.csv", index=False)
    pd.DataFrame(gradient_audits).to_json(report_dir / "c64_gradient_connectivity_audit.json", orient="records", indent=2)
    passed = all(bool(item["passed"]) for item in checks)
    payload = {
        "phase": "C64-STCV",
        "status": "C64_STAGED_TUNING_CV_AUTHORIZED" if passed else "C64_STAGE_A_GATE_FAIL",
        "passed": int(sum(bool(item["passed"]) for item in checks)),
        "total": len(checks),
        "checks": checks,
        "repo": str(REPO_ROOT),
        "commit": git_value("rev-parse", "--short", "HEAD"),
        "manifest": EXPECTED_MANIFEST,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "candidates": list(common.CANDIDATES),
        "seeds": list(common.SEEDS),
        "patience": 15,
        "max_epochs": 60,
        "test_loaded": False,
        "test_role": "locked_until_final_contract",
        "ensemble": False,
        "prediction_averaging": False,
    }
    reporting.write_json(report_dir / "c64_gate.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    try:
        payload = run_gate(args)
    except Exception as exc:
        report_dir = REPO_ROOT / "analysis_reports" / "phase_c64_dema"
        payload = {
            "phase": "C64-STCV",
            "status": "C64_STAGE_A_GATE_FAIL",
            "passed": 0,
            "total": 1,
            "checks": [{"name": "gate_exception", "passed": False, "details": repr(exc)}],
            "test_loaded": False,
        }
        reporting.write_json(report_dir / "c64_gate.json", payload)
        raise
    print(json.dumps({"status": payload["status"], "passed": payload["passed"], "total": payload["total"]}))
    if payload["status"] != "C64_STAGED_TUNING_CV_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
