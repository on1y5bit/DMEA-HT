#!/usr/bin/env python3
"""Validate the C26-SM static, synthetic, capacity, path, and reproduction contracts."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c26sm_stable_mechanism_mixer import (  # noqa: E402
    MECHANISM_NAMES,
    RELATION_NAMES,
    C26SMResidualMLP,
    C26SMStableMechanismModel,
    StableMechanismMixer,
    c26sm_loss_terms,
    propagation_capacity,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, read_manifest  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402

SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c26sm_stable_mechanism_mixer_multiseed.yaml")
    parser.add_argument("--output", default="analysis_reports/phase_c26sm_dema/c26sm_static_synthetic_gate.json")
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def item(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def call_names(source: str) -> List[str]:
    names: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def static_checks(config: Mapping[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    model_source = (REPO_ROOT / "dmea_ht" / "c26sm_stable_mechanism_mixer.py").read_text(encoding="utf-8")
    train_source = (REPO_ROOT / "scripts" / "train_phase_c26sm.py").read_text(encoding="utf-8")
    collect_source = (REPO_ROOT / "scripts" / "collect_phase_c26sm_formal_report.py").read_text(encoding="utf-8")
    model_calls, train_calls = call_names(model_source), call_names(train_source)
    checks.append(item("phase_is_c26sm", str(config.get("phase", "")).lower() == "c26sm"))
    checks.append(item("formal_seeds_fixed", config["training"]["seeds"] == list(SEEDS)))
    checks.append(item("validation_auc_only", config["training"]["primary_metric"] == "val_AUC"))
    checks.append(item("test_reporting_only", bool(config["training"]["evaluate_test"])))
    checks.append(item("rho_fixed_nontrainable", config["c26sm"]["rho"] == 0.10 and "self.rho = float(rho)" in model_source))
    checks.append(item("residual_and_loss_contract_fixed", config["c26sm"]["residual_max"] == 0.50 and config["loss"]["lambda_residual"] == 0.001 and config["loss"]["lambda_positive_preserve"] == 0.02))
    checks.append(item("c26e_withdrawn_no_files", not list(REPO_ROOT.rglob("*c26e*"))))
    checks.append(item("no_checkpoint_averaging_calls", not any(name in model_calls + train_calls for name in ("average", "mean_state_dict", "load_multiple_checkpoints"))))
    checks.append(item("one_c17_checkpoint_template", model_source.count('phase_cfg["c17_checkpoint"]') == 1))
    checks.append(item("no_multihead_attention_in_new_model", "MultiheadAttention" not in model_source))
    checks.append(item("one_shared_message_projection", model_source.count("self.shared_message = nn.Linear") == 1))
    checks.append(item("relation_parameters_are_scalars", "torch.zeros(())" in model_source and len(RELATION_NAMES) == 6))
    checks.append(item("single_update_step", model_source.count("source_mean + self.rho * message") == 1))
    checks.append(item("shared_five_node_scorer", model_source.count("self.score_hidden = nn.Linear") == 1 and model_source.count("self.score_output = nn.Linear") == 1))
    checks.append(item("no_edge_specific_high_dimensional_module_dict", "ModuleDict" not in inspect.getsource(StableMechanismMixer)))
    checks.append(item("no_mechanism_to_mechanism_edges", not any(f"M{left}_to_M{right}" in model_source for left in range(1, 6) for right in range(1, 6))))
    checks.append(item("real_topology_sources_present", all(name in model_source for name in RELATION_NAMES)))
    checks.append(item("context_order_is_text_then_bio_then_norm", model_source.find("context = text_context") < model_source.find("context = context + bio_context") < model_source.find('disease_norm(mixed["mechanism_core"] + context)')))
    checks.append(item("one_residual_prediction_path", model_source.count("class C26SMResidualMLP") == 1 and model_source.count("self.residual_mlp =") == 1))
    checks.append(item("no_ranking_loss", "pairwise_rank" not in model_source and "ranking_loss" not in model_source))
    checks.append(item("no_forbidden_alignment_design", not any(token in (model_source + train_source).lower() for token in ("shared_specific", "shared-specific", "decalign", "common/private"))))
    shortcut_fields = ("selected_n_visits", "used_images", "image_padding_count", "report_length", "raw_n_visits", "raw_n_images")
    checks.append(item("shortcut_fields_absent_from_predictor_and_loss", not any(field in model_source for field in shortcut_fields)))
    checks.append(item("audit_fields_absent_from_predictor", not any(token in model_source for token in ("c14", "c20", "c21", "hard_patient", "inversion_count"))))
    checks.append(item("saved_prediction_csv_not_training_input", train_source.count("pd.read_csv(") == 1 and "pd.read_csv(metrics_path)" in train_source))
    checks.append(item("optimizer_only_in_training_entry", "Optimizer" in train_source and "Optimizer" not in model_source))
    checks.append(item("validation_decision_precedes_test_stage", "validation decision must be frozen before reporting-only test" in train_source))
    disabled_metric = "AUP" + "RC"
    checks.append(item("disabled_metric_absent", disabled_metric not in model_source + train_source + collect_source))

    torch.manual_seed(20260714)
    mixer = StableMechanismMixer(16, rho=0.10)
    batch = 4
    image_nodes, text_nodes, bio_nodes = torch.randn(batch, 5, 16), torch.randn(batch, 6, 16), torch.randn(batch, 3, 16)
    image_valid = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    text_valid = torch.tensor([[1] * 6, [1] * 6, [0] * 6, [0] * 6], dtype=torch.bool)
    bio_valid = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)
    mixed = mixer(image_nodes, image_valid, text_nodes, text_valid, bio_nodes, bio_valid)
    checks.append(item("synthetic_outputs_finite", all(bool(torch.isfinite(value).all()) for value in mixed.values())))
    checks.append(item("masked_softmax_normalized", bool(torch.allclose(mixed["node_weights"].sum(dim=1), torch.ones(batch), atol=1e-7))))
    checks.append(item("initial_node_weights_uniform", float((mixed["node_weights"] - 0.2).abs().max()) <= 1e-8, float((mixed["node_weights"] - 0.2).abs().max())))
    checks.append(item("initial_relation_gates_half", float((mixed["relation_gates"] - 0.5).abs().max()) <= 1e-8))
    checks.append(item("empty_slot_fallback_finite", bool(mixed["empty_slot_mask"][-1].all()) and bool(torch.isfinite(mixed["mechanism_nodes"][-1]).all())))
    checks.append(item("empty_slot_not_encoded_as_opposition", "opposition" not in inspect.getsource(StableMechanismMixer._slot)))
    residual = C26SMResidualMLP(16, dropout=0.0, residual_max=0.50)
    raw, delta = residual(mixed["mechanism_core"])
    checks.append(item("zero_initialized_residual_output", float(delta.abs().max()) <= 1e-8))
    checks.append(item("residual_bound_holds", float(delta.abs().max()) <= 0.50 + 1e-7))
    synthetic_outputs = {"logit": torch.randn(batch) + delta, "delta_logit": delta}
    mixed_terms = c26sm_loss_terms(synthetic_outputs, {"label": torch.tensor([0., 1., 0., 1.])}, config["loss"])
    mixed_terms["total"].backward()
    gradients = [parameter.grad for parameter in residual.parameters() if parameter.grad is not None]
    checks.append(item("initial_residual_gradient_finite_nonzero", bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(float(g.abs().sum()) > 0 for g in gradients)))
    negative_delta = torch.zeros(3, requires_grad=True)
    negative_terms = c26sm_loss_terms({"logit": negative_delta, "delta_logit": negative_delta}, {"label": torch.zeros(3)}, config["loss"])
    negative_terms["positive_preservation"].backward()
    checks.append(item("all_negative_positive_preserve_graph_zero", float(negative_terms["positive_preservation"]) == 0.0 and negative_delta.grad is not None))

    c26_configs = sorted((REPO_ROOT / "configs").glob("*c26sm*.yaml"))
    checks.append(item("one_formal_config_only", len(c26_configs) == 1, [path.name for path in c26_configs]))
    checks.append(item("no_smoke_pilot_fallback_config", not any(any(token in path.name.lower() for token in ("smoke", "pilot", "fallback", "seed0")) for path in c26_configs)))
    parse_errors = []
    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        try:
            load_config(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    checks.append(item("legacy_configs_parse", not parse_errors, parse_errors))
    topology = [
        "# C26-SM Fixed Topology", "",
        "| frozen source | stable slot | relation parameter |",
        "|---|---|---|",
        "| masked mean of all five image evidence slots | M1 morphology | scalar `image_morphology` |",
        "| masked mean of text support and nonspecific slots | M1 morphology | scalar `text_morphology` |",
        "| bio immune observed | M2 immune | scalar `bio_immune` |",
        "| bio function observed | M3 function | scalar `bio_function` |",
        "| text opposition | M4 opposition | scalar `text_opposition` |",
        "| text temporal | M5 temporal | scalar `text_temporal` |",
        "", "Frozen context order: text-global projection, then bio-other projection addition, then frozen C17 disease norm.",
        "No mechanism-to-mechanism edge is present.",
    ]
    (output_dir / "c26sm_topology_inventory.md").write_text("\n".join(topology) + "\n", encoding="utf-8")
    return checks


def build_loader(config: Mapping[str, Any], split: str, shuffle: bool = False) -> DataLoader:
    dataset = PatientHTDataset(
        rows=read_manifest(config["project"]["manifest"]), data_root=config["project"]["data_root"], split=split,
        max_images=int(config["model"]["max_images_per_patient"]), image_size=int(config["model"]["image_size"]),
        text_max_length=int(config["model"]["text_max_length"]), text_vocab_size=int(config["model"]["text_vocab_size"]),
        bio_dim=int(config["model"]["bio_dim"]),
    )
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, generator=torch.Generator().manual_seed(20260714), num_workers=0, collate_fn=collate_patient_batch)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def full_server_checks(config: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    checks.append(item("canonical_branch_main", branch == "main", branch))
    worktree_text = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    worktrees = [line.split(" ", 1)[1] for line in worktree_text.splitlines() if line.startswith("worktree ")]
    checks.append(item("only_canonical_worktree", len(worktrees) == 1 and Path(worktrees[0]).resolve() == REPO_ROOT.resolve(), worktrees))
    sibling = [str(path) for path in REPO_ROOT.parent.glob("*c26sm*") if path.is_dir()]
    checks.append(item("no_c26sm_project_copy", not sibling, sibling))

    path_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for role, template in (("c13_checkpoint", config["c17"]["base_checkpoint"]), ("c16_checkpoint", config["c17"]["init_mea_checkpoint"]), ("c17_checkpoint", config["c26sm"]["c17_checkpoint"])):
            path = Path(str(template).replace("{seed}", str(seed))).resolve()
            path_rows.append({"seed": seed, "role": role, "absolute_path": str(path), "exists": path.exists()})
        reference = Path(config["c26sm"]["c17_run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv"
        path_rows.append({"seed": seed, "role": "c17_validation_predictions", "absolute_path": str(reference.resolve()), "exists": reference.exists()})
    manifest = Path(config["project"]["manifest"]).resolve()
    path_rows.append({"seed": "all", "role": "manifest", "absolute_path": str(manifest), "exists": manifest.exists()})
    run_root, report_root = resolve_path(config["project"]["output_dir"]), resolve_path(config["project"]["report_dir"])
    for role, root in (("c26sm_run_root", run_root), ("c26sm_report_root", report_root)):
        path_rows.append({"seed": "all", "role": role, "absolute_path": str(root.resolve()), "exists": root.parent.exists() and os.access(root.parent, os.W_OK)})
    paths_ok = all(bool(row["exists"]) for row in path_rows)
    checks.append(item("all_required_paths_resolve", paths_ok, path_rows))
    inventory = ["# C26-SM Resolved Path Inventory", "", "| seed | role | absolute path | exists |", "|---:|---|---|---|"]
    inventory.extend(f"| {row['seed']} | {row['role']} | `{row['absolute_path']}` | {row['exists']} |" for row in path_rows)
    (output_dir / "c26sm_resolved_path_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if not paths_ok:
        return checks

    rows = read_manifest(manifest)
    counts = {f"{split}_{label}": sum(str(row.get("split")) == split and int(row.get("label", -1)) == label for row in rows) for split in ("train", "val", "test") for label in (0, 1)}
    expected = {"train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47, "test_0": 42, "test_1": 42}
    checks.append(item("manifest_split_label_counts", counts == expected, counts))
    val_loader = build_loader(config, "val")
    train_loader = build_loader(config, "train", shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    max_c17_logit_error = max_c17_prob_error = max_initial_error = max_context_error = 0.0
    alignment_ok = finite_ok = weights_ok = frozen_grad_ok = initial_grad_ok = warm_mixer_grad_ok = True
    missing_modality_rows = 0

    for seed in SEEDS:
        checkpoint = Path(str(config["c26sm"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        payload_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        metadata_ok = (
            isinstance(payload, dict) and int(payload.get("seed", -1)) == seed
            and str(payload_config.get("phase", "")).lower() == "c17"
            and payload_config.get("c17", {}).get("variant") == "positive_preserve"
            and payload_config.get("training", {}).get("primary_metric") == "val_AUC"
            and payload_config.get("project", {}).get("manifest") == config["project"]["manifest"]
        )
        checks.append(item(f"seed_{seed}_c17_checkpoint_contract", metadata_ok, {"seed": payload.get("seed"), "best_epoch": payload.get("best_epoch")}))
        model = C26SMStableMechanismModel(config, seed).to(device)
        allowed = []
        for name, parameter in model.named_parameters():
            trainable_rows.append({"seed": seed, "parameter": name, "trainable": bool(parameter.requires_grad), "numel": parameter.numel()})
            if parameter.requires_grad:
                allowed.append(name)
        scope_ok = bool(allowed) and all(name.startswith(("mixer.", "residual_mlp.")) for name in allowed)
        checks.append(item(f"seed_{seed}_trainable_scope", scope_ok, allowed))
        capacity = {"seed": seed, **propagation_capacity(model)}
        capacity_rows.append(capacity)
        reference = pd.read_csv(Path(config["c26sm"]["c17_run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv")
        reference["patient_id"] = reference["patient_id"].astype(str)
        reference_map = reference.set_index("patient_id")
        seen: List[str] = []
        first_batch = None
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                outputs = model(batch)
                c17_outputs = model.frozen_c17(batch)
                if first_batch is None:
                    first_batch = batch
                finite_ok = finite_ok and all(bool(torch.isfinite(outputs[key]).all()) for key in ("logit", "mechanism_nodes", "mechanism_node_weights", "mechanism_state", "delta_logit"))
                weights_ok = weights_ok and bool(torch.allclose(outputs["mechanism_node_weights"].sum(dim=1), torch.ones(len(batch["patient_id"]), device=device), atol=1e-6))
                max_initial_error = max(max_initial_error, float((outputs["logit"] - outputs["base_logit"]).abs().max().cpu()))
                for index, patient_id in enumerate(batch["patient_id"]):
                    pid = str(patient_id)
                    seen.append(pid)
                    if pid not in reference_map.index:
                        alignment_ok = False
                        continue
                    ref = reference_map.loc[pid]
                    alignment_ok = alignment_ok and int(ref["label"]) == int(batch["label"][index].cpu())
                    max_c17_logit_error = max(max_c17_logit_error, abs(float(c17_outputs["logit"][index].cpu()) - float(ref["logit"])))
                    max_c17_prob_error = max(max_c17_prob_error, abs(float(c17_outputs["prob"][index].cpu()) - float(ref["prob"])))
                    shortcut = batch["shortcuts"][index]
                    if float(shortcut.get("has_bio", 1) or 0) <= 0 or float(shortcut.get("image_padding_count", shortcut.get("padding_count", 0)) or 0) > 0:
                        missing_modality_rows += 1
        alignment_ok = alignment_ok and set(seen) == set(reference["patient_id"]) and len(seen) == 94
        assert first_batch is not None
        with torch.no_grad():
            frozen = model.frozen_c17
            mea = frozen.mechanism_evidence_alignment
            encoded = frozen.base_model.encode_modalities(first_batch)
            text_masks = {key: first_batch[key] for key in TEXT_MASK_KEYS}
            image = mea.image(encoded["image_tokens"], first_batch["image_mask"])
            text = mea.text(encoded["text_tokens"], first_batch["report_attention_mask"], text_masks)
            bio = mea.bio(encoded["bio_tokens"], first_batch["bio_missing_mask"])
            mixed = model.mixer(image["nodes"], image["valid"], text["nodes"], text["valid"], bio["nodes"], bio["valid"])
            text_context = mea.mechanisms.relations["text_global"](text["nodes"][:, 5]) * text["valid"][:, 5].unsqueeze(-1)
            bio_context = mea.mechanisms.relations["bio_other"](bio["nodes"][:, 0]) * bio["valid"][:, 0].unsqueeze(-1)
            expected_state = mea.mechanisms.disease_norm(mixed["mechanism_core"] + text_context + bio_context)
            actual_state = model(first_batch)["mechanism_state"]
            max_context_error = max(max_context_error, float((expected_state - actual_state).abs().max().cpu()))

        mixed_batch = None
        for batch in train_loader:
            if len(torch.unique(batch["label"])) == 2:
                mixed_batch = move_batch(batch, device)
                break
        assert mixed_batch is not None
        model.train()
        model.zero_grad(set_to_none=True)
        terms = c26sm_loss_terms(model(mixed_batch), mixed_batch, config["loss"])
        terms["total"].backward()
        initial_grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("residual_mlp.") and parameter.grad is not None]
        initial_grad_ok = initial_grad_ok and bool(initial_grads) and all(bool(torch.isfinite(grad).all()) for grad in initial_grads) and any(float(grad.abs().sum()) > 0 for grad in initial_grads)
        frozen_grad_ok = frozen_grad_ok and all(parameter.grad is None for parameter in model.frozen_c17.parameters())
        with torch.no_grad():
            output_layer = model.residual_mlp.mlp[-1]
            assert isinstance(output_layer, torch.nn.Linear)
            output_layer.weight.fill_(0.01)
        model.zero_grad(set_to_none=True)
        warm_terms = c26sm_loss_terms(model(mixed_batch), mixed_batch, config["loss"])
        warm_terms["total"].backward()
        mixer_grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("mixer.") and parameter.grad is not None]
        warm_mixer_grad_ok = warm_mixer_grad_ok and bool(mixer_grads) and all(bool(torch.isfinite(grad).all()) for grad in mixer_grads) and any(float(grad.abs().sum()) > 0 for grad in mixer_grads)
        frozen_grad_ok = frozen_grad_ok and all(parameter.grad is None for parameter in model.frozen_c17.parameters())

    pd.DataFrame(trainable_rows).to_csv(output_dir / "c26sm_trainable_parameter_audit.csv", index=False)
    capacity_frame = pd.DataFrame(capacity_rows)
    capacity_frame.to_csv(output_dir / "c26sm_capacity_comparison.csv", index=False)
    checks.append(item("c17_validation_patient_label_alignment", alignment_ok))
    checks.append(item("c17_validation_logits_reproduce", max_c17_logit_error <= 1e-5, max_c17_logit_error))
    checks.append(item("c17_validation_probabilities_reproduce", max_c17_prob_error <= 1e-6, max_c17_prob_error))
    checks.append(item("initial_c26sm_equals_c13", max_initial_error <= 1e-8, max_initial_error))
    checks.append(item("real_context_order_reproduced", max_context_error <= 1e-8, max_context_error))
    checks.append(item("real_missing_modality_outputs_finite", finite_ok and missing_modality_rows > 0, missing_modality_rows))
    checks.append(item("real_node_weights_normalized", weights_ok))
    checks.append(item("initial_trainable_gradient_finite_nonzero", initial_grad_ok))
    checks.append(item("warm_mixer_gradient_finite_nonzero", warm_mixer_grad_ok))
    checks.append(item("frozen_modules_receive_no_gradient", frozen_grad_ok))
    ratio = float(capacity_frame["stable_over_original_ratio"].max())
    checks.append(item("stable_mixer_capacity_at_most_half", ratio <= 0.50, ratio))
    return checks


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = static_checks(config, output.parent)
    if not args.static_only and all(check["pass"] for check in checks):
        checks.extend(full_server_checks(config, output.parent))
    passed = all(check["pass"] for check in checks)
    decision = "C26SM_DIRECT_MULTI_SEED_AUTHORIZED" if passed and not args.static_only else ("C26SM_STATIC_GATE_PASS" if passed else "DEMA_C26SM_PATH_GATE_FAIL")
    result = {"phase": "C26-SM", "gate": "STATIC_ONLY" if args.static_only else "FULL_SERVER", "pass": passed, "passed": sum(int(check["pass"]) for check in checks), "total": len(checks), "checks": checks, "decision": decision}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)
    if not args.static_only:
        print("C26SM_DIRECT_MULTI_SEED_AUTHORIZED")


if __name__ == "__main__":
    main()
