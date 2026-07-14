#!/usr/bin/env python3
"""Validate the C24 static contract and server-only reproduction/path gate."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c24_decision_side_residual import (  # noqa: E402
    C24DecisionSideResidualModel,
    ConfidenceGatedLocalResidualHead,
    c24_loss_terms,
    decision_side_bounds,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, read_manifest  # noqa: E402

SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c24_decision_side_residual_multiseed.yaml")
    parser.add_argument("--output", default="analysis_reports/phase_c24_dema/c24_static_synthetic_gate.json")
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def item(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def write_math_check(path: Path) -> None:
    import csv

    logits = torch.tensor([-3.0, -0.2, -0.01, -0.000101, -0.0001, 0.0, 0.0001, 0.000101, 0.01, 0.2, 3.0])
    bounds = decision_side_bounds(logits)
    rows = []
    for direction in (-1.0, 1.0):
        delta = bounds["safe_local_bound"] * direction
        final = logits + delta
        for index, base in enumerate(logits):
            rows.append({
                "base_logit": float(base), "direction": direction,
                "confidence_gate": float(bounds["confidence_gate"][index]),
                "confidence_bound": float(bounds["confidence_bound"][index]),
                "boundary_bound": float(bounds["boundary_bound"][index]),
                "safe_local_bound": float(bounds["safe_local_bound"][index]),
                "delta": float(delta[index]), "final_logit": float(final[index]),
                "decision_side_preserved": bool((base >= 0) == (final[index] >= 0)),
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_checks(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    checks.append(item("phase_is_c24", str(config.get("phase", "")).lower() == "c24"))
    checks.append(item("formal_seeds_fixed", config["training"].get("seeds") == list(SEEDS)))
    checks.append(item("selection_is_validation_auc_only", config["training"].get("primary_metric") == "val_AUC"))
    checks.append(item("test_is_post_selection_reporting_only", bool(config["training"].get("evaluate_test"))))
    checks.append(item("temperature_fixed_to_one", config["c24"].get("temperature") == 1.0))
    checks.append(item("residual_max_fixed", config["c24"].get("residual_max") == 0.15))
    checks.append(
        item(
            "loss_contract_fixed",
            config["loss"].get("lambda_residual") == 0.001
            and config["loss"].get("lambda_positive_preserve") == 0.02
            and config["loss"].get("lambda_negative_preserve") == 0.02
            and config["loss"].get("lambda_high_confidence") == 0.01,
        )
    )
    checks.append(item("epsilon_fixed", config["c24"].get("epsilon") == 0.0001))
    logits = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
    bounds = decision_side_bounds(logits)
    gates = bounds["confidence_gate"]
    checks.append(item("gate_in_unit_interval", bool(((gates >= 0) & (gates <= 1)).all()), gates.tolist()))
    checks.append(item("gate_monotonic_in_abs_logit", bool((gates[:-1] >= gates[1:]).all()), gates.tolist()))
    checks.append(item("gate_and_bounds_have_no_parameters", not isinstance(decision_side_bounds, torch.nn.Module)))
    checks.append(item("boundary_and_safe_bounds_nonnegative", bool((bounds["boundary_bound"] >= 0).all()) and bool((bounds["safe_local_bound"] >= 0).all())))
    checks.append(item("safe_bound_below_both_limits", bool((bounds["safe_local_bound"] <= bounds["confidence_bound"] + 1e-8).all()) and bool((bounds["safe_local_bound"] <= bounds["boundary_bound"] + 1e-8).all())))

    torch.manual_seed(20260714)
    head = ConfidenceGatedLocalResidualHead(8, dropout=0.0)
    representation = torch.randn(5, 8)
    frozen_logit = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
    raw = head(representation)
    synthetic_bounds = decision_side_bounds(frozen_logit)
    delta = synthetic_bounds["safe_local_bound"] * torch.tanh(raw)
    checks.append(item("zero_initialized_logit_equality", float(delta.abs().max()) <= 1e-8, float(delta.abs().max())))
    checks.append(item("residual_bound", bool((delta.abs() <= 0.15 + 1e-7).all())))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(frozen_logit + delta, torch.tensor([0., 1., 0., 1., 0.]))
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    checks.append(item("new_head_finite_nonzero_gradient", bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(float(g.abs().sum()) > 0 for g in gradients)))

    for name, labels, base in (
        ("positive", torch.ones(3), torch.zeros(3)),
        ("negative", torch.zeros(3), torch.zeros(3)),
        ("high_confidence", torch.tensor([1., 0., 1.]), torch.zeros(3)),
    ):
        trial_delta = torch.zeros(3, requires_grad=True)
        trial_logit = torch.zeros(3, requires_grad=True)
        if name == "high_confidence":
            trial_logit = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
        outputs = {
            "logit": trial_logit + trial_delta,
            "delta_c24": trial_delta,
            "frozen_c17_logit": trial_logit,
        }
        terms = c24_loss_terms(outputs, {"label": labels, "sample_weight": torch.ones(3)}, config["loss"])
        absent_key = {"positive": "negative_preserve", "negative": "positive_preserve", "high_confidence": "high_confidence_preserve"}[name]
        terms["total"].backward()
        checks.append(item(f"{absent_key}_graph_connected_zero", float(terms[absent_key].detach()) == 0.0 and terms[absent_key].requires_grad))

    extreme_logits = torch.tensor([2e-5, 1e-4, 1.01e-4, 0.01, 0.2, 2.0, -2e-5, -1e-4, -1.01e-4, -0.01, -0.2, -2.0])
    extreme_bounds = decision_side_bounds(extreme_logits)
    extreme_direction = torch.tensor([1.0, -1.0] * 6)
    extreme_delta = extreme_bounds["safe_local_bound"] * extreme_direction
    extreme_final = extreme_logits + extreme_delta
    checks.append(item("delta_below_safe_bound", bool((extreme_delta.abs() <= extreme_bounds["safe_local_bound"] + 1e-8).all())))
    checks.append(item("positive_base_stays_positive", bool((extreme_final[extreme_logits > 1e-4] >= 1e-4 - 1e-8).all()), extreme_final.tolist()))
    checks.append(item("negative_base_stays_negative", bool((extreme_final[extreme_logits < -1e-4] <= -1e-4 + 1e-8).all()), extreme_final.tolist()))
    near_zero = extreme_logits.abs() <= 1e-4
    checks.append(item("near_zero_bound_and_delta_are_zero", bool((extreme_bounds["safe_local_bound"][near_zero] == 0).all()) and bool((extreme_delta[near_zero] == 0).all())))
    compare_bounds = decision_side_bounds(torch.tensor([0.1, 3.0]))
    compare_delta = compare_bounds["safe_local_bound"]
    checks.append(item("high_confidence_delta_smaller_than_low", float(compare_delta[1]) < float(compare_delta[0]), compare_delta.tolist()))
    model_source = (REPO_ROOT / "dmea_ht" / "c24_decision_side_residual.py").read_text(encoding="utf-8")
    shortcuts = ("selected_n_visits", "used_images", "image_padding_count", "report_length", "raw_n_visits", "raw_n_images")
    checks.append(item("shortcut_fields_absent_from_model", not any(name in model_source for name in shortcuts)))
    forward_source = inspect.getsource(C24DecisionSideResidualModel.forward)
    checks.append(item("frozen_c17_final_logit_is_base", 'reference["logit"]' in forward_source))
    checks.append(item("representation_is_frozen_c17_mechanism_state", 'reference["mea_mechanism_state"]' in forward_source))
    training_source = (REPO_ROOT / "scripts" / "train_phase_c24.py").read_text(encoding="utf-8")
    checks.append(item("prediction_csvs_absent_from_training_inputs", "read_csv" not in training_source))
    forbidden = ("DSSA", "shared_specific", "DecAlign")
    checks.append(item("unrelated_alignment_paths_absent", not any(token in model_source + training_source for token in forbidden)))
    disabled_metric = "AUP" + "RC"
    report_source = (REPO_ROOT / "scripts" / "collect_phase_c24_formal_report.py").read_text(encoding="utf-8")
    checks.append(item("disabled_metric_absent_from_formal_path", disabled_metric not in training_source + report_source))
    old_configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    parse_errors = []
    for path in old_configs:
        try:
            load_config(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    checks.append(item("existing_configs_still_parse", not parse_errors, parse_errors))
    return checks


def build_val_loader(config: Mapping[str, Any]) -> DataLoader:
    rows = read_manifest(config["project"]["manifest"])
    dataset = PatientHTDataset(
        rows=rows,
        data_root=config["project"]["data_root"],
        split="val",
        max_images=int(config["model"]["max_images_per_patient"]),
        image_size=int(config["model"]["image_size"]),
        text_max_length=int(config["model"]["text_max_length"]),
        text_vocab_size=int(config["model"]["text_vocab_size"]),
        bio_dim=int(config["model"]["bio_dim"]),
    )
    return DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_patient_batch)


def full_server_checks(config: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    checks: List[Dict[str, Any]] = []
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    checks.append(item("canonical_branch_is_main", branch == "main", branch))
    worktree_text = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    worktrees = [line.split(" ", 1)[1] for line in worktree_text.splitlines() if line.startswith("worktree ")]
    checks.append(item("only_canonical_worktree", len(worktrees) == 1 and Path(worktrees[0]).resolve() == REPO_ROOT.resolve(), worktrees))
    path_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for role, template in (
            ("c13_checkpoint", config["c17"]["base_checkpoint"]),
            ("c16_checkpoint", config["c17"]["init_mea_checkpoint"]),
            ("c17_checkpoint", config["c24"]["c17_checkpoint"]),
        ):
            path = Path(str(template).replace("{seed}", str(seed))).resolve()
            path_rows.append({"seed": seed, "role": role, "absolute_path": str(path), "exists": path.exists()})
    manifest = Path(config["project"]["manifest"]).resolve()
    run_root = resolve_path(config["project"]["output_dir"])
    report_root = resolve_path(config["project"]["report_dir"])
    c17_config = Path(config["c24"]["c17_config"]).resolve()
    path_rows.extend([
        {"seed": "all", "role": "canonical_project_root", "absolute_path": str(REPO_ROOT.resolve()), "exists": REPO_ROOT.exists()},
        {"seed": "all", "role": "manifest", "absolute_path": str(manifest), "exists": manifest.exists()},
        {"seed": "all", "role": "c17_config", "absolute_path": str(c17_config), "exists": c17_config.exists()},
        {
            "seed": "all", "role": "c24_run_root", "absolute_path": str(run_root.resolve()),
            "exists": run_root.parent.exists() and os.access(run_root.parent, os.W_OK),
        },
        {
            "seed": "all", "role": "c24_report_root", "absolute_path": str(report_root.resolve()),
            "exists": report_root.parent.exists() and os.access(report_root.parent, os.W_OK),
        },
    ])
    for seed in SEEDS:
        reference_path = REPO_ROOT / "runs" / "dema_ht_c17_formal_multiseed" / "predictions" / f"val_predictions_seed_{seed}.csv"
        path_rows.append({
            "seed": seed, "role": "c17_validation_predictions", "absolute_path": str(reference_path.resolve()),
            "exists": reference_path.exists(),
        })
    paths_ok = all(bool(row["exists"]) for row in path_rows)
    checks.append(item("all_required_paths_resolve", paths_ok, path_rows))
    inventory = ["# C24 Resolved Path Inventory", "", "| seed | role | absolute path | exists |", "|---:|---|---|---|"]
    inventory += [f"| {row['seed']} | {row['role']} | `{row['absolute_path']}` | {row['exists']} |" for row in path_rows]
    (output_dir / "c24_resolved_path_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if not paths_ok:
        return checks

    manifest_rows = read_manifest(manifest)
    counts = {
        f"{split}_{label}": sum(str(row.get("split")) == split and int(row.get("label", -1)) == label for row in manifest_rows)
        for split in ("train", "val", "test") for label in (0, 1)
    }
    expected_counts = {"train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47, "test_0": 42, "test_1": 42}
    checks.append(item("manifest_split_label_counts_match", counts == expected_counts, counts))
    seed_metadata = {}
    for seed in SEEDS:
        checkpoint = Path(str(config["c24"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        seed_metadata[str(seed)] = payload.get("seed") if isinstance(payload, dict) else None
    checks.append(item("c17_checkpoint_seed_metadata_match", all(int(seed_metadata[str(seed)]) == seed for seed in SEEDS), seed_metadata))

    loader = build_val_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_rows, max_reproduction, max_initial_diff = [], 0.0, 0.0
    gradient_ok, frozen_grad_ok, finite_ok = True, True, True
    missing_modality_rows = 0
    bound_ok = True
    actual_side_ok = True
    for seed in SEEDS:
        model = C24DecisionSideResidualModel(config, seed).to(device)
        for name, parameter in model.named_parameters():
            trainable_rows.append({"seed": seed, "parameter": name, "trainable": bool(parameter.requires_grad), "numel": parameter.numel()})
        allowed = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        checks.append(item(f"seed_{seed}_only_c24_head_trainable", bool(allowed) and all(name.startswith("residual_head.") for name in allowed), allowed))
        reference_path = REPO_ROOT / "runs" / "dema_ht_c17_formal_multiseed" / "predictions" / f"val_predictions_seed_{seed}.csv"
        reference_frame = pd.read_csv(reference_path)
        if "has_bio" in reference_frame:
            missing_modality_rows += int((pd.to_numeric(reference_frame["has_bio"], errors="coerce").fillna(0) <= 0).sum())
        if "image_padding_count" in reference_frame:
            missing_modality_rows += int((pd.to_numeric(reference_frame["image_padding_count"], errors="coerce").fillna(0) > 0).sum())
        reference_col = "logit" if "logit" in reference_frame.columns else None
        reference_map = reference_frame.set_index(reference_frame["patient_id"].astype(str))
        seen = 0
        model.eval()
        first_batch = None
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                outputs = model(batch)
                if first_batch is None:
                    first_batch = batch
                finite_ok = finite_ok and all(bool(torch.isfinite(outputs[key]).all()) for key in ("logit", "confidence_gate", "delta_c24"))
                max_initial_diff = max(max_initial_diff, float((outputs["logit"] - outputs["frozen_c17_logit"]).abs().max().cpu()))
                bound_ok = bound_ok and bool((outputs["safe_local_bound"] >= 0).all()) and bool((outputs["safe_local_bound"] <= outputs["confidence_bound"] + 1e-8).all()) and bool((outputs["safe_local_bound"] <= outputs["boundary_bound"] + 1e-8).all())
                base = outputs["frozen_c17_logit"]
                safe = outputs["safe_local_bound"]
                actual_side_ok = actual_side_ok and bool(((base + safe >= 0) == (base >= 0)).all()) and bool(((base - safe >= 0) == (base >= 0)).all())
                for index, patient_id in enumerate(batch["patient_id"]):
                    row = reference_map.loc[str(patient_id)]
                    expected = float(row[reference_col]) if reference_col else float(np.log(float(row["prob"]) / (1.0 - float(row["prob"]))))
                    max_reproduction = max(max_reproduction, abs(float(outputs["frozen_c17_logit"][index].cpu()) - expected))
                    seen += 1
        checks.append(item(f"seed_{seed}_all_validation_rows_reproduced", seen == len(reference_frame), {"seen": seen, "expected": len(reference_frame)}))
        assert first_batch is not None
        model.train()
        outputs = model(first_batch)
        terms = c24_loss_terms(outputs, first_batch, config["loss"])
        terms["total"].backward()
        head_grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("residual_head.") and parameter.grad is not None]
        gradient_ok = gradient_ok and bool(head_grads) and all(bool(torch.isfinite(grad).all()) for grad in head_grads) and any(float(grad.abs().sum()) > 0 for grad in head_grads)
        frozen_grad_ok = frozen_grad_ok and all(parameter.grad is None for parameter in model.frozen_c17.parameters())
    pd.DataFrame(trainable_rows).to_csv(output_dir / "c24_trainable_parameter_audit.csv", index=False)
    checks.append(item("c17_validation_logits_reproduce", max_reproduction <= 1e-5, max_reproduction))
    checks.append(item("initial_c24_equals_frozen_c17", max_initial_diff <= 1e-8, max_initial_diff))
    checks.append(item("new_head_gradient_finite_nonzero", gradient_ok))
    checks.append(item("frozen_c17_gradients_absent", frozen_grad_ok))
    checks.append(item("real_validation_missing_modality_path_finite", finite_ok and missing_modality_rows > 0, missing_modality_rows))
    checks.append(item("real_validation_safe_bounds_valid", bound_ok))
    checks.append(item("real_validation_extreme_directions_preserve_side", actual_side_ok))
    return checks


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_math_check(output.parent / "c24_decision_side_math_check.csv")
    checks = static_checks(config)
    if not args.static_only and all(check["pass"] for check in checks):
        checks.extend(full_server_checks(config, output.parent))
    passed = all(check["pass"] for check in checks)
    result = {
        "phase": "C24", "gate": "STATIC_ONLY" if args.static_only else "FULL_SERVER",
        "pass": passed, "checks": checks,
        "decision": "C24_DIRECT_MULTI_SEED_AUTHORIZED" if passed and not args.static_only else ("C24_STATIC_GATE_PASS" if passed else "C24_PATH_GATE_FAIL"),
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)
    if not args.static_only:
        print("C24_DIRECT_MULTI_SEED_AUTHORIZED")


if __name__ == "__main__":
    main()
