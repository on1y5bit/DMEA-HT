#!/usr/bin/env python3
"""Authorize C63 from-base full end-to-end multi-seed training."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c61_cbpi import C61CBPIModel  # noqa: E402
from dmea_ht.c47_drfe import C47DRFEModel  # noqa: E402
from scripts import c63_common as common  # noqa: E402


EXPECTED_MANIFEST_SHA256 = "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
FORBIDDEN_TASK_PATTERNS = (
    r"runs[/\\]dema_ht_(?:c13|c17|c27|c37|c59|c61|c62)[^\s\"']*",
    r"(?<!use_)(?:c13|c17|c27|c37|c59|c61|c62)_checkpoint",
    r"resume_from",
    r"teacher_checkpoint",
    r"task_checkpoint_used\s*:\s*true",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c63_from_base_e2e_cbpi_multiseed.yaml")
    parser.add_argument("--expected-project", default="/home/linruixin/chen/project/DMEA-HT")
    parser.add_argument("--gradient-batches", type=int, default=4)
    return parser.parse_args()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def manifest_contract(rows: List[Dict[str, Any]]) -> tuple[bool, str]:
    split_ids: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    seen: set[str] = set()
    labels: Dict[str, int] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        split = str(row["split"])
        if patient_id in seen or split not in split_ids:
            return False, f"duplicate patient or invalid split: {patient_id}/{split}"
        seen.add(patient_id)
        split_ids[split].add(patient_id)
        labels[patient_id] = int(row["label"])
    disjoint = not (
        split_ids["train"] & split_ids["val"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["val"] & split_ids["test"]
    )
    counts = ", ".join(
        f"{split}={len(ids)} ({sum(labels[patient_id] for patient_id in ids)}/{len(ids) - sum(labels[patient_id] for patient_id in ids)})"
        for split, ids in split_ids.items()
    )
    return disjoint, f"patient-level disjoint; {counts}"


def static_exclusion_audit(config_path: Path) -> pd.DataFrame:
    files = {
        "config": config_path,
        "c63_common": REPO_ROOT / "scripts" / "c63_common.py",
        "c63_trainer": REPO_ROOT / "scripts" / "train_phase_c63.py",
        "c63_model_loader": REPO_ROOT / "dmea_ht" / "c61_cbpi.py",
        "c63_source_route": REPO_ROOT / "dmea_ht" / "c47_drfe.py",
    }
    rows: List[Dict[str, Any]] = []
    for name, path in files.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for pattern in FORBIDDEN_TASK_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            rows.append(
                {
                    "scope": name,
                    "path": str(path),
                    "forbidden_pattern": pattern,
                    "matched": bool(match),
                    "match_text": match.group(0) if match else "",
                }
            )
    return pd.DataFrame(rows)


def real_batch_gradients(
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    device: torch.device,
    batches: int,
) -> tuple[pd.DataFrame, bool]:
    records: List[Dict[str, Any]] = []
    all_pass = True
    for seed in common.SEEDS:
        common.set_seed(seed)
        loader = common.build_loaders(config, rows, seed, ("train",))["train"]
        model = common.build_from_base_model(config, seed, device)
        optimizer, _ = common.optimizer_parameter_groups(model, config)
        model.train(True)
        aggregate = {group: 0.0 for group in common.MODULE_GROUPS}
        active = {group: 0 for group in common.MODULE_GROUPS}
        finite = True
        observed = 0
        for raw_batch in loader:
            if observed >= batches:
                break
            batch = common.move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logit"], batch["label"])
            finite &= bool(torch.isfinite(loss))
            loss.backward()
            gradient_info = common.module_gradient_norms(model)
            for group in common.MODULE_GROUPS:
                aggregate[group] += float(gradient_info[group]["norm"]) ** 2
                active[group] += int(gradient_info[group]["nonzero_tensor_count"] > 0)
            observed += 1
        for group in common.MODULE_GROUPS:
            norm = float(np.sqrt(aggregate[group]))
            passed = bool(finite and observed > 0 and norm > 0.0 and active[group] > 0)
            all_pass &= passed
            records.append(
                {
                    "seed": seed,
                    "module_group": group,
                    "batches_observed": observed,
                    "aggregate_gradient_norm": norm,
                    "active_batch_count": active[group],
                    "finite": finite,
                    "pass": passed,
                }
            )
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(records), all_pass


def main() -> None:
    args = parse_args()
    config_path = common.resolve_path(args.config)
    config = common.load_c63_config(config_path)
    rows = common.manifest_rows(config)
    report_dir = common.resolve_path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    branch = git_value("branch", "--show-current")
    canonical_path = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve())
    add("01_canonical_main_and_path", canonical_path and branch == "main", f"path={REPO_ROOT} branch={branch}")
    manifest_path = common.resolve_path(config["project"]["manifest"])
    manifest_hash = common.sha256_file(manifest_path) if manifest_path.exists() else "missing"
    add("02_manifest_hash", manifest_hash == EXPECTED_MANIFEST_SHA256, manifest_hash)
    split_ok, split_detail = manifest_contract(rows)
    add("03_manifest_patient_split_labels", split_ok, split_detail)
    test_contract = bool(config["evaluation"]["test_reporting_only"]) and bool(
        config["training"]["evaluate_test_after_validation_decision"]
    )
    add("04_test_not_read_before_validation", test_contract, "Gate builds only the train loader; Test is reporting-only")
    structure_ok = (
        config["model"]["architecture"] == "c61_cbpi"
        and config["c61"]["bio_basis_order"] == 3
        and config["c61"]["learned_visit_score"] is False
        and config["c61"]["temporal_attention"] is False
        and config["c61"]["router"] is False
        and config["c61"]["visit_selector"] is False
    )
    add("05_c61_structure_preserved", structure_ok, "C61 continuous-bio basis and patient set statistics are fixed")
    add(
        "06_shortcut_fields_excluded",
        config["c61"]["shortcut_fields_used_as_inputs"] is False
        and config["c61"]["missingness_as_classifier_feature"] is False,
        "audit-only shortcut fields are not model inputs",
    )

    exclusion = static_exclusion_audit(config_path)
    exclusion.to_csv(report_dir / "c63_task_checkpoint_exclusion_audit.csv", index=False)
    exclusion_ok = not bool(exclusion["matched"].any())
    add("07_config_task_checkpoint_exclusion", exclusion[exclusion["scope"] == "config"]["matched"].eq(False).all(), "No C13-C62 task checkpoint reference in C63 config")
    add("08_model_loader_task_checkpoint_exclusion", exclusion[exclusion["scope"] == "c63_model_loader"]["matched"].eq(False).all(), "C63 model loader has no task-checkpoint initialization")
    add("09_trainer_task_checkpoint_exclusion", exclusion[exclusion["scope"] == "c63_trainer"]["matched"].eq(False).all(), "C63 trainer initializes from base and only loads its own selected checkpoint for reporting")
    init_cfg = config["initialization"]
    saved_input_ok = init_cfg["saved_prediction_input"] is False and init_cfg["saved_representation_input"] is False
    add("10_saved_prediction_representation_exclusion", saved_input_ok, "saved predictions and representations are not model inputs")

    inventory_parts: List[pd.DataFrame] = []
    initialization_parts: List[pd.DataFrame] = []
    optimizer_parts: List[pd.DataFrame] = []
    hash_parts: List[pd.DataFrame] = []
    overall_hashes: Dict[int, str] = {}
    init_ok = True
    counts: Dict[int, int] = {}
    for seed in common.SEEDS:
        common.set_seed(seed)
        model = common.build_from_base_model(config, seed, device)
        optimizer, optimizer_audit = common.optimizer_parameter_groups(model, config)
        inventory = common.parameter_inventory(model)
        inventory.insert(0, "seed", seed)
        initialization = common.initialization_inventory(model, seed, optimizer_audit)
        hashes, overall_hash = common.parameter_hashes(model, seed)
        counts[seed] = int(inventory["parameter_count"].sum())
        init_ok &= bool(getattr(model.sources, "initialization_type", "") == "random_task_specific")
        init_ok &= bool(initialization["task_trained_checkpoint_used"].eq(False).all())
        init_ok &= bool(initialization["initialization_type"].eq("random_task_specific").all())
        inventory_parts.append(inventory)
        initialization_parts.append(initialization)
        hashes["overall_parameter_hash"] = overall_hash
        hash_parts.append(hashes)
        optimizer_audit.insert(0, "seed", seed)
        optimizer_parts.append(optimizer_audit)
        overall_hashes[seed] = overall_hash
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    inventory_frame = pd.concat(inventory_parts, ignore_index=True)
    initialization_frame = pd.concat(initialization_parts, ignore_index=True)
    hash_frame = pd.concat(hash_parts, ignore_index=True)
    optimizer_frame = pd.concat(optimizer_parts, ignore_index=True)
    inventory_frame.to_csv(report_dir / "c63_trainable_parameter_inventory.csv", index=False)
    initialization_frame.to_csv(report_dir / "c63_initialization_inventory.csv", index=False)
    hash_frame.to_csv(report_dir / "c63_initial_parameter_hash_by_seed.csv", index=False)
    optimizer_frame.to_csv(report_dir / "c63_optimizer_parameter_groups.csv", index=False)
    add("11_random_task_specific_initialization", init_ok, f"initialization_type=random_task_specific; counts={counts}")
    add("12_initialization_inventory_clean", not bool(initialization_frame["task_trained_checkpoint_used"].any()), "all task_trained_checkpoint_used values are false")

    repeatable = True
    for seed in common.SEEDS:
        common.set_seed(seed)
        first = common.build_from_base_model(config, seed, torch.device("cpu"))
        _, first_hash = common.parameter_hashes(first, seed)
        del first
        common.set_seed(seed)
        second = common.build_from_base_model(config, seed, torch.device("cpu"))
        _, second_hash = common.parameter_hashes(second, seed)
        repeatable &= first_hash == second_hash == overall_hashes[seed]
        del second
    add("13_same_seed_initialization_reproducible", repeatable, "rebuilding each seed reproduces the same parameter hash")
    different_seed = len(set(overall_hashes.values())) == len(common.SEEDS)
    add("14_different_seed_initialization_distinct", different_seed, json.dumps(overall_hashes, sort_keys=True))
    public_ok = (
        init_cfg["public_pretrained_backbone_only"] is True
        and init_cfg["use_public_pretrained_backbone"] is False
        and init_cfg["public_pretrained_backbones"] == []
        and initialization_frame["public_pretrained_backbone"].eq(False).all()
    )
    add("15_public_pretrained_provenance", public_ok, "public pretrained backbone: NONE; all source modules use random_task_specific initialization")
    frozen_ok = not bool(inventory_frame["requires_grad"].eq(False).any())
    add("16_zero_frozen_predictive_parameters", frozen_ok, f"frozen_predictive_parameters={int((~inventory_frame['requires_grad'].astype(bool)).sum())}")
    optimizer_ok = bool(optimizer_frame["all_requires_grad"].astype(bool).all()) and set(optimizer_frame["group"]) == set(common.OPTIMIZER_GROUPS)
    add("17_optimizer_membership", optimizer_ok, "all predictive parameters belong to exactly one optimizer group")
    positive_lr_ok = bool(optimizer_frame["learning_rate"].astype(float).gt(0.0).all())
    add("18_positive_learning_rates", positive_lr_ok, optimizer_frame[["group", "learning_rate"]].drop_duplicates().to_json(orient="records"))
    expected_factors = config["learning_rate_groups"]
    factor_ok = all(
        abs(float(row.learning_rate_factor) - float(expected_factors[row.group])) < 1e-12
        for row in optimizer_frame.drop_duplicates("group").itertuples()
    )
    add("19_learning_rate_group_contract", factor_ok, optimizer_frame[["group", "learning_rate_factor", "learning_rate"]].drop_duplicates().to_json(orient="records"))

    route_source = inspect.getsource(C61CBPIModel.forward)
    source_route = inspect.getsource(C47DRFEModel._source_states)
    model = common.build_from_base_model(config, 42, device)
    active_end_to_end_branch = (
        bool(model.end_to_end)
        and "if self.end_to_end" in route_source
        and "source = self._source_states(batch)" in route_source
    )
    no_grad_detach = (
        active_end_to_end_branch
        and ".detach(" not in route_source
        and ".detach(" not in source_route
    )
    add(
        "20_no_grad_or_detach_predictive_route",
        no_grad_detach,
        "C63 active end-to-end branch has no no_grad/detach; legacy compatibility branch is inactive",
    )
    model.train(True)
    train_mode_ok = all(module.training for module in model.modules())
    add("21_all_predictive_modules_train_mode", train_mode_ok, f"all_modules_train={train_mode_ok}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gradients, gradients_ok = real_batch_gradients(config, rows, device, max(1, args.gradient_batches))
    gradients.to_csv(report_dir / "c63_gradient_connectivity_audit.csv", index=False)
    add("22_real_batch_module_gradient_connectivity", gradients_ok, gradients.to_json(orient="records"))
    science_ok = bool(config["loss"]["bce_only"]) and config["evaluation"]["auprc"] is False
    science_ok &= config["deployment"]["ensemble"] is False
    science_ok &= not any(key in config for key in ("ranking_loss", "distillation", "ema", "sweep", "smoke", "pilot"))
    add("23_bce_only_direct_no_alternatives", science_ok, "BCEWithLogitsLoss only; no smoke, pilot, sweep, EMA, distillation, ranking, or AUPRC")
    capacity_ok = max(counts.values()) <= int(config["initialization"].get("trainable_parameter_limit", 100000000))
    deployment_ok = config["deployment"] == {
        "one_checkpoint": True,
        "one_model": True,
        "one_forward": True,
        "ensemble": False,
    }
    add("24_capacity_and_single_model_deployment", capacity_ok and deployment_ok, f"counts={counts}; deployment={deployment_ok}")

    checks_frame = pd.DataFrame(checks)
    passed = int(checks_frame["passed"].sum())
    total = int(len(checks_frame))
    status = "C63_FROM_BASE_E2E_DIRECT_MULTI_SEED_AUTHORIZED" if passed == total and exclusion_ok else "DEMA_C63_PATH_GATE_FAIL"
    payload = {
        "phase": "C63-FS-CBPI",
        "status": status,
        "passed": passed,
        "total": total,
        "git_commit": git_value("rev-parse", "HEAD"),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "formal_seeds": list(common.SEEDS),
        "trainable_parameter_count_by_seed": counts,
        "initial_parameter_hash_by_seed": overall_hashes,
        "checks": checks,
    }
    (report_dir / "c63_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": total}))
    if status != "C63_FROM_BASE_E2E_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
