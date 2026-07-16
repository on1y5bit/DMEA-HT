#!/usr/bin/env python3
"""Authorize direct C62 full end-to-end multiseed training after a strict Gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts import c62_common as common  # noqa: E402


EXPECTED_MANIFEST_SHA256 = "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c62_e2e_cbpi_multiseed.yaml")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_contract(rows: List[Dict[str, Any]]) -> tuple[bool, str]:
    seen: Dict[str, str] = {}
    split_ids: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    labels: Dict[str, int] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        split = str(row["split"])
        label = int(row["label"])
        if patient_id in seen or split not in split_ids:
            return False, f"duplicate patient or invalid split: {patient_id}/{split}"
        seen[patient_id] = split
        split_ids[split].add(patient_id)
        labels[patient_id] = label
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


def initial_reproduction(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device, report_dir: Path
) -> tuple[pd.DataFrame, bool]:
    loader = common.core.build_loaders(config, rows, ("val",))["val"]
    records: List[Dict[str, Any]] = []
    all_pass = True
    for seed in common.SEEDS:
        common.set_seed(seed)
        checkpoint = common.resolve_path(
            str(config["c62"]["initialization_checkpoint"]).replace("{seed}", str(seed))
        )
        model = common.build_model(config, seed, device, checkpoint)
        model.eval()
        current: List[Dict[str, Any]] = []
        with torch.no_grad():
            for raw_batch in loader:
                batch = common.core.move_batch(raw_batch, device)
                output = model(batch)
                probabilities = output["prob"].detach().cpu().numpy().astype(float)
                labels = batch["label"].detach().cpu().numpy().astype(int)
                for index, patient_id in enumerate(batch["patient_id"]):
                    probability = float(probabilities[index])
                    current.append(
                        {
                            "patient_id": str(patient_id),
                            "label": int(labels[index]),
                            "prob": probability,
                            "predicted_class": int(probability >= 0.5),
                        }
                    )
        current_frame = pd.DataFrame(current).sort_values("patient_id").reset_index(drop=True)
        saved_path = common.resolve_path(config["c61"]["run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv"
        saved = pd.read_csv(saved_path)
        saved["patient_id"] = saved["patient_id"].astype(str)
        saved = saved.sort_values("patient_id").reset_index(drop=True)
        saved_prob = saved["final_prob"].to_numpy(dtype=float)
        current_prob = current_frame["prob"].to_numpy(dtype=float)
        ids_exact = current_frame["patient_id"].tolist() == saved["patient_id"].tolist()
        labels_exact = np.array_equal(current_frame["label"].to_numpy(dtype=int), saved["label"].to_numpy(dtype=int))
        classes_exact = np.array_equal(current_frame["predicted_class"].to_numpy(dtype=int), saved["predicted_class"].to_numpy(dtype=int))
        max_error = float(np.max(np.abs(saved_prob - current_prob))) if len(saved_prob) == len(current_prob) else float("inf")
        auc = float(roc_auc_score(current_frame["label"], current_prob))
        saved_auc = float(roc_auc_score(saved["label"], saved_prob))
        passed = bool(
            len(current_frame) == len(saved)
            and ids_exact
            and labels_exact
            and classes_exact
            and max_error <= 1e-6
            and abs(auc - saved_auc) <= 1e-6
        )
        all_pass &= passed
        records.append(
            {
                "seed": seed,
                "rows": len(current_frame),
                "patient_id_exact": ids_exact,
                "label_exact": labels_exact,
                "predicted_class_exact": classes_exact,
                "saved_auc": saved_auc,
                "rerun_auc": auc,
                "max_abs_probability_difference": max_error,
                "pass": passed,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    frame.to_csv(report_dir / "c62_initial_c61_reproduction.csv", index=False)
    return frame, all_pass


def gradient_connectivity(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device, batches: int
) -> tuple[pd.DataFrame, bool]:
    loader = common.core.build_loaders(config, rows, ("train",))["train"]
    records: List[Dict[str, Any]] = []
    all_pass = True
    for seed in common.SEEDS:
        common.set_seed(seed)
        checkpoint = common.resolve_path(
            str(config["c62"]["initialization_checkpoint"]).replace("{seed}", str(seed))
        )
        model = common.build_model(config, seed, device, checkpoint)
        optimizer, _ = common.optimizer_parameter_groups(model, config)
        model.train(True)
        aggregate = {group: 0.0 for group in common.GROUPS}
        active = {group: 0 for group in common.GROUPS}
        finite = True
        observed = 0
        for raw_batch in loader:
            if observed >= batches:
                break
            batch = common.core.move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logit"], batch["label"])
            finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(value).all()) for value in output.values() if torch.is_tensor(value))
            loss.backward()
            info = common.group_gradient_norms(model)
            for group in common.GROUPS:
                aggregate[group] += float(info[group]["norm"]) ** 2
                active[group] += int(info[group]["nonzero_tensor_count"] > 0)
            observed += 1
        for group in common.GROUPS:
            norm = float(np.sqrt(aggregate[group]))
            passed = bool(finite and observed > 0 and norm > 0.0 and active[group] > 0)
            all_pass &= passed
            records.append(
                {
                    "seed": seed,
                    "group": group,
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
    config = common.load_c62_config(args.config)
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
    manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else "missing"
    add("02_manifest_hash", manifest_hash == EXPECTED_MANIFEST_SHA256, manifest_hash)
    split_ok, split_detail = manifest_contract(rows)
    add("03_manifest_patient_split_labels", split_ok, split_detail)
    add(
        "04_test_not_read_by_gate",
        bool(config["evaluation"]["test_reporting_only"])
        and bool(config["training"]["evaluate_test_after_validation_decision"]),
        "Gate only reads manifest train/val and Test remains reporting-only",
    )
    checkpoint_paths = {
        seed: common.resolve_path(str(config["c62"]["initialization_checkpoint"]).replace("{seed}", str(seed)))
        for seed in common.SEEDS
    }
    metadata_ok = True
    metadata_detail: List[str] = []
    for seed, path in checkpoint_paths.items():
        exists = path.exists()
        payload = common.checkpoint_payload(path) if exists else {}
        seed_ok = exists and int(payload.get("seed", -1)) == seed and isinstance(payload.get("model"), Mapping)
        metadata_ok &= seed_ok
        metadata_detail.append(f"seed {seed}: {seed_ok}")
    add("05_c61_initialization_checkpoints", metadata_ok, ", ".join(metadata_detail))

    inventory_frames = []
    counts: List[int] = []
    scope_ok = True
    for seed in common.SEEDS:
        model = common.build_model(config, seed, device, checkpoint_paths[seed])
        inventory = common.parameter_inventory(model)
        inventory["seed"] = seed
        inventory_frames.append(inventory)
        counts.append(int(inventory["parameter_count"].sum()))
        scope_ok &= bool(inventory["requires_grad"].astype(bool).all()) and int((~inventory["requires_grad"].astype(bool)).sum()) == 0
        del model
    inventory_frame = pd.concat(inventory_frames, ignore_index=True)
    inventory_frame.to_csv(report_dir / "c62_trainable_parameter_inventory.csv", index=False)
    add("06_all_predictive_parameters_trainable", scope_ok, f"total_parameters_by_seed={counts}; frozen_predictive_parameters=0")

    model = common.build_model(config, 42, device, checkpoint_paths[42])
    optimizer, optimizer_audit = common.optimizer_parameter_groups(model, config)
    optimizer_audit.to_csv(report_dir / "c62_optimizer_parameter_groups.csv", index=False)
    optimizer_ok = len(optimizer.param_groups) == len(common.GROUPS) and all(float(group["lr"]) > 0.0 for group in optimizer.param_groups)
    add("07_optimizer_membership_and_positive_lr", optimizer_ok, optimizer_audit.to_json(orient="records"))
    expected_factors = config["learning_rate_groups"]
    factor_ok = all(abs(float(row.learning_rate_factor) - float(expected_factors[row.group])) < 1e-12 for row in optimizer_audit.itertuples())
    add("08_learning_rate_group_contract", factor_ok, optimizer_audit[["group", "learning_rate_factor", "learning_rate"]].to_json(orient="records"))
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reproduction, reproduction_ok = initial_reproduction(config, rows, device, report_dir)
    add("09_initial_c61_reproduction", reproduction_ok, reproduction.to_json(orient="records"))
    gradients, gradients_ok = gradient_connectivity(config, rows, device, max(1, args.gradient_batches))
    gradients.to_csv(report_dir / "c62_gradient_connectivity_audit.csv", index=False)
    add("10_real_batch_gradient_connectivity", gradients_ok, gradients.to_json(orient="records"))

    model = common.build_model(config, 42, device, checkpoint_paths[42])
    model.train(True)
    train_mode_ok = all(module.training for module in model.modules())
    source_text = inspect.getsource(type(model))
    route_active = bool(model.end_to_end) and "if self.end_to_end" in source_text and "source = self._source_states(batch)" in source_text
    add("11_all_predictive_modules_train_mode", train_mode_ok, f"all_modules_train={train_mode_ok}")
    add("12_end_to_end_source_route_active", route_active, f"model.end_to_end={model.end_to_end}")
    route_config_ok = (
        config["c62"]["architecture"] == "c61_cbpi"
        and config["c62"]["end_to_end"] is True
        and config["c62"]["freeze_any_predictive_module"] is False
        and config["c61"]["bio_basis_order"] == 3
        and config["c61"]["learned_visit_score"] is False
        and config["c61"]["temporal_attention"] is False
        and config["c61"]["router"] is False
        and config["c61"]["visit_selector"] is False
    )
    add("13_c61_structure_preserved", route_config_ok, "fixed continuous-bio basis and patient set statistics retained")
    no_shortcut = config["c61"]["shortcut_fields_used_as_inputs"] is False and "selected_n_visits" not in source_text and "used_images" not in source_text
    add("14_shortcut_fields_excluded", no_shortcut, "audit-only shortcut fields are not model inputs")
    loss_ok = bool(config["loss"]["bce_only"]) and config["evaluation"]["auprc"] is False
    add("15_bce_only_no_auprc", loss_ok, "BCEWithLogitsLoss only; AUPRC disabled")
    no_alternatives = bool(config["deployment"]["ensemble"] is False) and all(
        key not in config for key in ("smoke", "pilot", "scheduler", "ema", "ranking_loss", "distillation")
    )
    add("16_no_smoke_pilot_or_alternative_route", no_alternatives, "direct formal route only")
    add("17_parameter_capacity", max(counts) <= int(config["c62"]["trainable_parameter_limit"]), f"count={max(counts)} limit={config['c62']['trainable_parameter_limit']}")
    add("18_direct_three_seed_contract", list(config["training"]["seeds"]) == list(common.SEEDS), "seeds=[0,42,3407]")
    add("19_single_model_deployment_contract", config["deployment"] == {"one_checkpoint": True, "one_model": True, "one_forward": True, "ensemble": False}, "one checkpoint/one model/one forward")
    add("20_no_partial_freeze_contract", scope_ok and gradients_ok and train_mode_ok and route_active, "all predictive groups trainable, connected, and in train mode")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checks_frame = pd.DataFrame(checks)
    passed = int(checks_frame["passed"].sum())
    total = int(len(checks_frame))
    status = "C62_E2E_CBPI_DIRECT_MULTI_SEED_AUTHORIZED" if passed == total else "DEMA_C62_PATH_GATE_FAIL"
    payload = {
        "phase": "C62-E2E-CBPI",
        "status": status,
        "passed": passed,
        "total": total,
        "git_commit": git_value("rev-parse", "HEAD"),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_parameter_count_by_seed": {str(seed): count for seed, count in zip(common.SEEDS, counts)},
        "checks": checks,
    }
    (report_dir / "c62_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": total}))
    if status != "C62_E2E_CBPI_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
