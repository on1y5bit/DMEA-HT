#!/usr/bin/env python3
"""Run the C27-VTME static, synthetic, reconstruction, and server gate."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c17_residual import C17ResidualModel  # noqa: E402
from dmea_ht.c27_vtme import (  # noqa: E402
    C27VTMEModel,
    MECHANISM_NAMES,
    VisitTemporalMechanismCore,
    trainable_parameter_count,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, read_manifest  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402


SEEDS = (0, 42, 3407)
STARTING_COMMIT = "3462c4e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="analysis_reports/phase_c27_dema/c27_static_synthetic_gate.json")
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def item(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def call_names(source: str) -> List[str]:
    tree = ast.parse(source)
    names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.append(function.id)
        elif isinstance(function, ast.Attribute):
            names.append(function.attr)
    return names


def static_checks(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    model_path = REPO_ROOT / "dmea_ht" / "c27_vtme.py"
    data_path = REPO_ROOT / "dmea_ht" / "visit_data.py"
    train_path = REPO_ROOT / "scripts" / "train_phase_c27.py"
    collect_path = REPO_ROOT / "scripts" / "collect_phase_c27_formal_report.py"
    model_source = model_path.read_text(encoding="utf-8")
    data_source = data_path.read_text(encoding="utf-8")
    train_source = train_path.read_text(encoding="utf-8")
    collect_source = collect_path.read_text(encoding="utf-8")
    model_calls = call_names(model_source)
    train_calls = call_names(train_source)

    checks.append(item("phase_is_c27", str(config.get("phase", "")).lower() == "c27"))
    checks.append(item("formal_seeds_fixed", config["training"]["seeds"] == list(SEEDS)))
    checks.append(item("validation_auc_only", config["training"]["primary_metric"] == "val_AUC"))
    checks.append(item("test_reporting_only_enabled", bool(config["training"]["evaluate_test"])))
    checks.append(item("bce_only_config", config["loss"] == {"bce_only": True}, config["loss"]))
    checks.append(item("optimizer_family_adamw", "torch.optim.AdamW" in train_source))
    checks.append(
        item(
            "effective_batch_size_preserved",
            int(config["training"]["batch_size"]) * int(config["training"]["gradient_accumulation_steps"])
            == int(config["training"]["effective_batch_size"]),
        )
    )
    checks.append(
        item(
            "recency_prior_fixed_log_two",
            math.isclose(float(config["c27"]["recency_prior_log_odds"]), math.log(2.0), rel_tol=0.0, abs_tol=1e-15),
        )
    )
    checks.append(item("text_length_per_visit_fixed_256", int(config["model"]["text_max_length"]) == 256))
    checks.append(item("capacity_limit_fixed", int(config["c27"]["trainable_parameter_limit"]) == 1_000_000))
    checks.append(item("no_artificial_max_visits_config", "max_visits" not in json.dumps(config)))
    checks.append(item("dynamic_batch_visit_padding", "max_visits = max(len(item[\"visits\"])" in data_source))
    checks.append(item("all_manifest_visits_retained", "visits = list(row.get(\"visits\", []))" in data_source and "visits[:" not in data_source))
    checks.append(item("oldest_to_latest_manifest_order", "visit_rank" in data_source and "oldest_to_latest" in data_source))
    checks.append(item("absolute_date_absent_from_predictor", "visit_dates" not in model_source))
    checks.append(item("visit_count_scalar_absent_from_predictor", "selected_n_visits" not in model_source and "visit_count" not in model_source))
    checks.append(item("image_count_scalar_absent_from_predictor", "used_images" not in model_source and "n_images" not in model_source))
    checks.append(item("report_length_absent_from_predictor", "report_length" not in model_source))
    checks.append(item("missing_count_absent_from_predictor", "missing_count" not in model_source))
    checks.append(item("label_and_patient_id_absent_from_predictor", "patient_id" not in model_source and "label" not in model_source))
    shortcut_fields = (
        "selected_n_visits", "raw_n_visits", "used_images", "raw_n_images", "report_length",
        "bio_missing_count", "has_bio", "image_padding_count", "source_folder",
    )
    checks.append(item("shortcut_fields_absent_from_predictor", not any(field in model_source for field in shortcut_fields)))
    checks.append(item("frozen_source_backbone_is_source_only", "FrozenC17EvidenceBackbone" in model_source))
    forbidden_c17_modules = (
        "EvidenceRoleScorer", "HTMechanismRelationLayer", "EvidenceConflictAggregator",
        "MechanismEvidenceAggregationHead", "MechanismResidualCorrectionHead", "C17ResidualModel",
    )
    checks.append(item("no_post_propagation_c17_module_loaded", not any(name in model_source for name in forbidden_c17_modules)))
    checks.append(
        item(
            "only_pre_propagation_projectors_loaded",
            all(name in model_source for name in ("ImageMorphologyEvidenceProjector", "TextEvidenceRoleProjector", "BioEvidenceProjector")),
        )
    )
    checks.append(
        item(
            "patient_fallback_bio_concatenated_once",
            model_source.count("[mechanism_states.flatten(start_dim=1), conflicts, fallback_bio_context]") == 1,
        )
    )
    checks.append(item("frozen_source_parameters_disabled", "parameter.requires_grad = False" in model_source))
    checks.append(item("frozen_source_eval_enforced", "self.frozen_sources.eval()" in model_source))
    checks.append(item("five_mechanism_slots_fixed", len(MECHANISM_NAMES) == 5 and "len(MECHANISM_NAMES)" in model_source))
    checks.append(item("visit_modalities_encoded_without_cross_visit_mixing", "batch_size * visits" in model_source and "flatten(0, 1)" in model_source))
    checks.append(item("one_empty_token_per_slot", "self.empty_slot_tokens" in model_source and "len(MECHANISM_NAMES), hidden_dim" in model_source))
    checks.append(item("shared_temporal_scorer_single_instance", model_source.count("self.temporal_linear =") == 1 and model_source.count("self.temporal_output =") == 1))
    checks.append(item("no_visit_specific_parameters", "ParameterList" not in model_source and "ModuleList" not in inspect.getsource(VisitTemporalMechanismCore)))
    checks.append(item("no_mechanism_specific_temporal_network", "ModuleDict" not in inspect.getsource(VisitTemporalMechanismCore)))
    checks.append(item("no_new_attention_recurrent_transformer", not any(token in model_source for token in ("MultiheadAttention", "Transformer", "nn.RNN", "nn.GRU", "nn.LSTM"))))
    checks.append(item("no_mechanism_to_mechanism_graph", "relations" not in model_source and "edge" not in model_source))
    checks.append(item("fixed_recency_not_parameter", "self.recency_prior_log_odds = float" in model_source))
    checks.append(item("masked_softmax_present", model_source.count("masked_softmax(scores, temporal_valid, dim=1)") == 1))
    checks.append(item("conflict_formula_content_only", "F.cosine_similarity" in inspect.getsource(VisitTemporalMechanismCore._conflicts) and "visit_count" not in inspect.getsource(VisitTemporalMechanismCore._conflicts)))
    checks.append(item("single_patient_projection", model_source.count("self.patient_projection =") == 1))
    checks.append(item("single_final_classifier", model_source.count("self.classifier =") == 1))
    checks.append(item("no_base_plus_correction_formula", "base_logit" not in model_source and "delta_logit" not in model_source))
    checks.append(item("one_c17_checkpoint_template", model_source.count('phase_cfg["c17_checkpoint"]') == 1))
    forbidden_inference_calls = ("average_checkpoints", "mean_state_dict", "load_multiple_checkpoints", "majority_vote", "stack_predictions")
    checks.append(item("no_checkpoint_or_prediction_combination_calls", not any(name in model_calls + train_calls for name in forbidden_inference_calls)))
    checks.append(item("optimizer_restricted_to_trainable_core", "[parameter for _, parameter in trainable]" in train_source and "name.startswith(\"core.\")" in train_source))
    checks.append(item("bce_is_only_training_loss", train_source.count("binary_cross_entropy_with_logits") == 1 and "ranking_loss" not in train_source and "auxiliary" not in train_source))
    checks.append(item("checkpoint_selected_only_by_val_auc", "if val_auc > best_auc" in train_source and "selected_by_val_auc" in train_source))
    checks.append(item("test_loader_created_only_in_reporting_stage", 'build_loaders(config, rows, ("test",))' in train_source and 'build_loaders(config, rows, ("train", "val"))' in train_source))
    checks.append(item("validation_decision_precedes_test", "validation decision must be frozen before reporting-only test" in train_source))
    checks.append(item("independent_seed_output_directories", 'out_dir / "seed_runs" / f"seed_{seed}"' in train_source))
    checks.append(item("representation_ids_are_unicode", "dtype=np.str_" in train_source))
    disabled_metric = "AUP" + "RC"
    checks.append(item("disabled_metric_absent", disabled_metric not in model_source + data_source + train_source + collect_source))
    tracked_files = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    checks.append(item("c26e_not_implemented", not any("c26e" in path.lower() for path in tracked_files)))
    historical = subprocess.run(
        ["git", "diff", "--name-only", STARTING_COMMIT, "--", "dmea_ht/c26sm_stable_mechanism_mixer.py", "scripts/train_phase_c26sm.py", "configs/dema_ht_c26sm_stable_mechanism_mixer_multiseed.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    checks.append(item("c26sm_route_unmodified", historical == "", historical))
    legacy_paths = (
        "dmea_ht/data.py", "dmea_ht/models.py", "dmea_ht/c17_residual.py", "train.py",
    )
    legacy_diff = subprocess.run(
        ["git", "diff", "--name-only", STARTING_COMMIT, "--", *legacy_paths],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    checks.append(item("legacy_paths_unchanged_use_vtme_false", legacy_diff == "", legacy_diff))

    torch.manual_seed(20260714)
    core = VisitTemporalMechanismCore(hidden_dim=16, dropout=0.0, recency_prior_log_odds=math.log(2.0))
    source_states = torch.randn(3, 3, 5, 16)
    source_valid = torch.tensor(
        [
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0]],
            [[0, 0, 0, 0, 0], [1, 0, 0, 1, 1], [0, 1, 1, 0, 1]],
        ],
        dtype=torch.bool,
    )
    visit_mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.bool)
    fallback = torch.randn(3, 16)
    outputs = core(source_states, source_valid, visit_mask, fallback)
    checks.append(item("synthetic_outputs_finite", all(bool(torch.isfinite(value).all()) for value in outputs.values())))
    checks.append(item("synthetic_five_slot_shapes", tuple(outputs["mechanism_states"].shape) == (3, 5, 16)))
    sums = outputs["temporal_weights"].sum(dim=1)
    checks.append(item("synthetic_temporal_weights_normalized", bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))))
    checks.append(item("synthetic_padding_weights_zero", float(outputs["temporal_weights"][0, 1:].abs().max()) == 0.0))
    checks.append(item("synthetic_single_visit_finite", bool(torch.isfinite(outputs["patient_state"][0]).all())))
    checks.append(item("synthetic_multi_visit_finite", bool(torch.isfinite(outputs["patient_state"][1:]).all())))
    checks.append(item("synthetic_recency_exact", bool(torch.allclose(outputs["recency"][0], torch.tensor([1.0, 0.0, 0.0])) and torch.allclose(outputs["recency"][1], torch.tensor([0.0, 0.5, 1.0])))))
    checks.append(item("synthetic_conflict_finite_range", bool(torch.isfinite(outputs["conflicts"]).all()) and float(outputs["conflicts"].min()) >= -1e-6 and float(outputs["conflicts"].max()) <= 2.0 + 1e-6))
    checks.append(item("synthetic_single_visit_conflict_zero", float(outputs["conflicts"][0].abs().max()) == 0.0))
    checks.append(item("synthetic_all_missing_slot_fallback_finite", bool(torch.isfinite(outputs["visit_states"][2, 0]).all())))
    mixed_loss = F.binary_cross_entropy_with_logits(outputs["logit"], torch.tensor([0.0, 1.0, 0.0]))
    one_class_loss = F.binary_cross_entropy_with_logits(outputs["logit"], torch.zeros(3))
    checks.append(item("synthetic_mixed_and_one_class_bce_finite", bool(torch.isfinite(mixed_loss)) and bool(torch.isfinite(one_class_loss))))
    mixed_loss.backward()
    gradients = [parameter.grad for parameter in core.parameters() if parameter.requires_grad]
    checks.append(item("synthetic_trainable_gradients_finite_nonzero", bool(gradients) and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients) and any(float(gradient.abs().sum()) > 0 for gradient in gradients if gradient is not None)))

    c27_configs = sorted((REPO_ROOT / "configs").glob("*c27*.yaml"))
    checks.append(item("one_formal_c27_config_only", len(c27_configs) == 1, [path.name for path in c27_configs]))
    checks.append(item("no_smoke_pilot_fallback_sweep_config", not any(any(token in path.name.lower() for token in ("smoke", "pilot", "fallback", "sweep", "seed0")) for path in c27_configs)))
    parse_errors = []
    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        try:
            load_config(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    checks.append(item("legacy_configs_parse", not parse_errors, parse_errors))
    checks.append(item("single_checkpoint_deployment_contract", "one_checkpoint_one_model_one_forward" in train_source + collect_source))
    return checks


def build_c17_loader(config: Mapping[str, Any]) -> DataLoader:
    dataset = PatientHTDataset(
        rows=read_manifest(config["project"]["base_manifest"]),
        data_root=config["project"]["data_root"],
        split="val",
        max_images=int(config["model"]["max_images_per_patient"]),
        image_size=int(config["model"]["image_size"]),
        text_max_length=int(config["model"]["text_max_length"]),
        text_vocab_size=int(config["model"]["text_vocab_size"]),
        bio_dim=int(config["model"]["bio_dim"]),
    )
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_patient_batch)


def build_real_visit_batch(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = VisitPatientDataset(
        rows=read_jsonl(config["project"]["manifest"]),
        data_root=config["project"]["data_root"],
        split="val",
        image_size=int(config["model"]["image_size"]),
        text_max_length=int(config["model"]["text_max_length"]),
        text_vocab_size=int(config["model"]["text_vocab_size"]),
        bio_dim=int(config["model"]["bio_dim"]),
        max_images_per_visit=int(config["model"]["max_images_per_visit"]),
    )
    chosen: List[int] = []
    requirements = ((1, 0), (2, 1), (2, 0), (1, 1))
    for minimum_visits, label in requirements:
        for index, row in enumerate(dataset.rows):
            if index in chosen:
                continue
            count = len(row.get("visits", []))
            if int(row["label"]) == label and ((minimum_visits == 1 and count == 1) or (minimum_visits == 2 and count > 1)):
                chosen.append(index)
                break
    if len(chosen) < 4:
        raise RuntimeError("Unable to build mixed single/multi C27 validation gate batch")
    batch = collate_visit_batch([dataset[index] for index in chosen])
    batch["images"][0, 0].zero_()
    batch["image_mask"][0, 0].zero_()
    batch["report_input_ids"][1, 0].zero_()
    batch["report_attention_mask"][1, 0].zero_()
    for key in TEXT_MASK_KEYS:
        batch[key][1, 0].zero_()
    batch["visit_text_valid"][1, 0] = False
    batch["bio_values"][2, 0].zero_()
    batch["bio_missing_mask"][2, 0].fill_(1)
    batch["bio_abnormal_flags"][2, 0].zero_()
    batch["images"][3, 0].zero_()
    batch["image_mask"][3, 0].zero_()
    batch["report_input_ids"][3, 0].zero_()
    batch["report_attention_mask"][3, 0].zero_()
    for key in TEXT_MASK_KEYS:
        batch[key][3, 0].zero_()
    batch["visit_text_valid"][3, 0] = False
    batch["bio_values"][3, 0].zero_()
    batch["bio_missing_mask"][3, 0].fill_(1)
    batch["bio_abnormal_flags"][3, 0].zero_()
    return batch


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint: {path}")
    return state


def full_server_checks(config: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    checks.append(item("canonical_branch_main", branch == "main", branch))
    worktree_text = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    worktrees = [line.split(" ", 1)[1] for line in worktree_text.splitlines() if line.startswith("worktree ")]
    checks.append(item("only_canonical_worktree", len(worktrees) == 1 and Path(worktrees[0]).resolve() == REPO_ROOT.resolve(), worktrees))
    siblings = [str(path) for path in REPO_ROOT.parent.glob("*c27*") if path.is_dir()]
    checks.append(item("no_c27_project_copy", not siblings, siblings))

    path_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for role, template in (
            ("c13_checkpoint", config["c17"]["base_checkpoint"]),
            ("c16_checkpoint", config["c17"]["init_mea_checkpoint"]),
            ("c17_checkpoint", config["c27"]["c17_checkpoint"]),
        ):
            path = Path(str(template).replace("{seed}", str(seed))).resolve()
            path_rows.append({"seed": seed, "role": role, "absolute_path": str(path), "exists": path.exists()})
        prediction = Path(config["c27"]["c17_run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv"
        path_rows.append({"seed": seed, "role": "c17_validation_predictions", "absolute_path": str(prediction.resolve()), "exists": prediction.exists()})
    for role, value in (
        ("c13_manifest", config["project"]["base_manifest"]),
        ("c27_visit_manifest", config["project"]["manifest"]),
        ("visit_invariance", Path(config["project"]["visit_design_dir"]) / "c27_visit_manifest_invariance.json"),
        ("visit_decision", Path(config["project"]["visit_design_dir"]) / "c27_visit_reconstruction_decision.json"),
    ):
        path = Path(value).resolve()
        path_rows.append({"seed": "all", "role": role, "absolute_path": str(path), "exists": path.exists()})
    run_root = resolve_path(config["project"]["output_dir"])
    report_root = resolve_path(config["project"]["report_dir"])
    for role, root in (("c27_run_root", run_root), ("c27_report_root", report_root)):
        path_rows.append(
            {
                "seed": "all",
                "role": role,
                "absolute_path": str(root.resolve()),
                "exists": root.parent.exists() and os.access(root.parent, os.W_OK),
            }
        )
    paths_ok = all(bool(row["exists"]) for row in path_rows)
    checks.append(item("all_required_paths_resolve", paths_ok, path_rows))
    inventory = ["# C27 Resolved Path Inventory", "", "| seed | role | absolute path | exists |", "|---:|---|---|---|"]
    inventory.extend(
        f"| {row['seed']} | {row['role']} | `{row['absolute_path']}` | {row['exists']} |" for row in path_rows
    )
    (output_dir / "c27_resolved_path_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if not paths_ok:
        return checks

    visit_rows = read_jsonl(config["project"]["manifest"])
    counts = {
        f"{split}_{label}": sum(str(row.get("split")) == split and int(row.get("label", -1)) == label for row in visit_rows)
        for split in ("train", "val", "test")
        for label in (0, 1)
    }
    expected = {"train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47, "test_0": 42, "test_1": 42}
    checks.append(item("visit_manifest_780_patients", len(visit_rows) == 780, len(visit_rows)))
    checks.append(item("visit_manifest_split_label_counts", counts == expected, counts))
    checks.append(item("visit_manifest_patient_ids_unique", len({str(row["patient_id"]) for row in visit_rows}) == 780))
    invariance = json.loads(
        (Path(config["project"]["visit_design_dir"]) / "c27_visit_manifest_invariance.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (Path(config["project"]["visit_design_dir"]) / "c27_visit_reconstruction_decision.json").read_text(encoding="utf-8")
    )
    hard = invariance["hard_checks"]
    checks.append(item("selected_visit_counts_match_c13", hard["selected_visit_counts_match"]))
    checks.append(item("image_grouping_matches_c13", hard["image_grouping_matches_c13"]))
    checks.append(item("selected_images_exist", hard["all_selected_image_paths_exist"]))
    checks.append(item("cross_patient_image_leakage_zero", hard["cross_patient_image_leakage_zero"]))
    checks.append(item("visit_date_order_reproducible", hard["visit_order_reproducible"]))
    checks.append(item("visit_reports_are_real_source", hard["visit_reports_match_real_source"]))
    checks.append(item("missing_reports_not_patient_text", hard["missing_reports_not_filled_from_patient_text"]))
    checks.append(item("dated_bio_exact_source_date", hard["dated_bio_matches_exact_source_date"]))
    fallback_once_ok = all(
        not bool(row.get("patient_bio_fallback", {}).get("valid", False))
        or not any(
            visit.get("dated_bio_row_id") is not None
            and any(int(value) == 0 for value in visit.get("bio_missing_mask_if_dated", []))
            for visit in row.get("visits", [])
        )
        for row in visit_rows
    )
    checks.append(item("patient_bio_fallback_not_copied_to_visits", fallback_once_ok))
    checks.append(item("history_cutoff_matches_c13", hard["history_cutoff_matches_c13"]))
    checks.append(item("test_not_used_for_reconstruction", hard["test_not_used_for_rule_design"] and not invariance["test_used_for_rule_design"]))
    checks.append(item("visit_report_coverage_gate", float(invariance["visit_report_coverage"]) >= 0.80, invariance["visit_report_coverage"]))
    checks.append(item("multi_visit_two_block_coverage_gate", float(invariance["multi_visit_validation_two_block_coverage"]) >= 0.70, invariance["multi_visit_validation_two_block_coverage"]))
    checks.append(item("visit_reconstruction_authorized", decision["decision"] == "C27_VISIT_RECONSTRUCTION_PASS", decision))
    distinct_report_patients = 0
    for row in visit_rows:
        reports = [str(visit.get("report_text", "")).strip() for visit in row.get("visits", []) if str(visit.get("report_text", "")).strip()]
        if len(reports) >= 2 and len(set(reports)) >= 2:
            distinct_report_patients += 1
    checks.append(item("multi_visit_text_not_globally_duplicated", distinct_report_patients > 0, distinct_report_patients))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c17_loader = build_c17_loader(config)
    real_batch_cpu = build_real_visit_batch(config)
    trainable_rows: List[Dict[str, Any]] = []
    max_c17_logit_error = 0.0
    max_c17_prob_error = 0.0
    max_c13_logit_error = 0.0
    max_c13_prob_error = 0.0
    patient_alignment_ok = True
    real_finite = True
    real_shape_ok = True
    real_weight_ok = True
    real_conflict_ok = True
    gradient_ok = True
    frozen_grad_ok = True
    frozen_eval_ok = True
    all_capacity_ok = True

    for seed in SEEDS:
        c17_path = Path(str(config["c27"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        try:
            payload = torch.load(c17_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(c17_path, map_location="cpu")
        payload_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        metadata_ok = (
            isinstance(payload, dict)
            and int(payload.get("seed", -1)) == seed
            and str(payload_config.get("phase", "")).lower() == "c17"
            and payload_config.get("c17", {}).get("variant") == "positive_preserve"
            and payload_config.get("training", {}).get("primary_metric") == "val_AUC"
            and payload_config.get("project", {}).get("manifest") == config["project"]["base_manifest"]
        )
        checks.append(item(f"seed_{seed}_c17_checkpoint_contract", metadata_ok, {"seed": payload.get("seed"), "best_epoch": payload.get("best_epoch")}))
        inherited_fields = ("batch_size", "lr", "weight_decay", "epochs", "patience")
        inherited_ok = all(
            payload_config.get("training", {}).get(field) == config["training"].get(field)
            for field in inherited_fields
        ) and payload_config.get("model", {}).get("dropout") == config["model"].get("dropout")
        checks.append(item(f"seed_{seed}_c17_training_hyperparameters_inherited", inherited_ok))

        c17_model = C17ResidualModel(config, seed).to(device)
        c17_model.load_state_dict(checkpoint_state(c17_path), strict=True)
        c17_model.eval()
        reference = pd.read_csv(
            Path(config["c27"]["c17_run_dir"]) / "predictions" / f"val_predictions_seed_{seed}.csv",
            dtype={"patient_id": str},
        )
        reference["patient_id"] = reference["patient_id"].astype(str)
        reference_map = reference.set_index("patient_id")
        seen: List[str] = []
        with torch.no_grad():
            for batch in c17_loader:
                batch = move_batch(batch, device)
                outputs = c17_model(batch)
                for index, patient_id in enumerate(batch["patient_id"]):
                    patient_id = str(patient_id)
                    seen.append(patient_id)
                    if patient_id not in reference_map.index:
                        patient_alignment_ok = False
                        continue
                    row = reference_map.loc[patient_id]
                    patient_alignment_ok = patient_alignment_ok and int(row["label"]) == int(batch["label"][index].cpu())
                    max_c17_logit_error = max(max_c17_logit_error, abs(float(outputs["logit"][index].cpu()) - float(row["logit"])))
                    max_c17_prob_error = max(max_c17_prob_error, abs(float(outputs["prob"][index].cpu()) - float(row["prob"])))
                    max_c13_logit_error = max(max_c13_logit_error, abs(float(outputs["base_logit"][index].cpu()) - float(row["base_logit"])))
                    max_c13_prob_error = max(max_c13_prob_error, abs(float(outputs["base_prob"][index].cpu()) - float(row["base_prob"])))
        patient_alignment_ok = patient_alignment_ok and len(seen) == 94 and set(seen) == set(reference["patient_id"])
        del c17_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = C27VTMEModel(config, seed).to(device)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        count = trainable_parameter_count(model)
        all_capacity_ok = all_capacity_ok and count <= int(config["c27"]["trainable_parameter_limit"])
        allowed = bool(trainable) and all(name.startswith("core.") for name, _ in trainable)
        checks.append(item(f"seed_{seed}_trainable_scope", allowed, [name for name, _ in trainable]))
        checks.append(item(f"seed_{seed}_capacity_contract", count <= 1_000_000, count))
        for name, parameter in model.named_parameters():
            trainable_rows.append(
                {
                    "seed": seed,
                    "parameter": name,
                    "trainable": bool(parameter.requires_grad),
                    "numel": int(parameter.numel()),
                }
            )
        batch = move_batch(real_batch_cpu, device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        tensor_keys = (
            "logit", "prob", "visit_states", "mechanism_states", "patient_state", "temporal_weights",
            "temporal_latest_weights", "temporal_normalized_entropy", "conflicts",
        )
        real_finite = real_finite and all(bool(torch.isfinite(outputs[key]).all()) for key in tensor_keys)
        real_shape_ok = real_shape_ok and outputs["mechanism_states"].shape == (
            len(batch["label"]), len(MECHANISM_NAMES), int(config["model"]["hidden_dim"])
        )
        real_weight_ok = real_weight_ok and bool(
            torch.allclose(outputs["temporal_weights"].sum(dim=1), torch.ones_like(outputs["temporal_latest_weights"]), atol=1e-6)
        )
        real_conflict_ok = real_conflict_ok and float(outputs["conflicts"].min()) >= -1e-6 and float(outputs["conflicts"].max()) <= 2.0 + 1e-6
        model.train()
        frozen_eval_ok = frozen_eval_ok and not model.frozen_sources.training
        model.zero_grad(set_to_none=True)
        training_outputs = model(batch)
        loss = F.binary_cross_entropy_with_logits(training_outputs["logit"], batch["label"])
        loss.backward()
        trainable_gradients = [parameter.grad for _, parameter in trainable]
        gradient_ok = gradient_ok and all(
            gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0
            for gradient in trainable_gradients
        )
        frozen_grad_ok = frozen_grad_ok and all(parameter.grad is None for parameter in model.frozen_sources.parameters())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(trainable_rows).to_csv(output_dir / "c27_trainable_parameter_audit.csv", index=False)
    checks.append(item("c13_c17_validation_patient_alignment", patient_alignment_ok))
    checks.append(item("c17_validation_logits_reproduce", max_c17_logit_error <= 1e-5, max_c17_logit_error))
    checks.append(item("c17_validation_probabilities_reproduce", max_c17_prob_error <= 1e-6, max_c17_prob_error))
    checks.append(item("c13_validation_logits_reproduce", max_c13_logit_error <= 1e-5, max_c13_logit_error))
    checks.append(item("c13_validation_probabilities_reproduce", max_c13_prob_error <= 1e-6, max_c13_prob_error))
    checks.append(item("real_single_multi_missing_outputs_finite", real_finite))
    checks.append(item("real_empty_visit_text_forward_finite", real_finite, "gate batch row 1 visit 0"))
    checks.append(item("real_missing_image_visit_forward_finite", real_finite, "gate batch row 0 visit 0"))
    checks.append(item("real_missing_bio_visit_forward_finite", real_finite, "gate batch row 2 visit 0"))
    checks.append(item("real_all_missing_slot_forward_finite", real_finite, "gate batch row 3 visit 0"))
    checks.append(item("real_five_mechanism_slot_shape", real_shape_ok))
    checks.append(item("real_temporal_weights_normalized", real_weight_ok))
    checks.append(item("real_conflicts_finite_range", real_conflict_ok))
    checks.append(item("real_trainable_gradients_all_finite_nonzero", gradient_ok))
    checks.append(item("real_frozen_modules_receive_no_gradient", frozen_grad_ok))
    checks.append(item("real_frozen_modules_remain_eval", frozen_eval_ok))
    checks.append(item("all_seed_capacity_contracts_pass", all_capacity_ok))

    leakage_lines = [
        "# C27 Leakage Exclusion Audit",
        "",
        f"- patient-level split and label counts preserved: `{counts == expected}`",
        f"- cross-patient image leakage count: `{decision['cross_patient_image_leakage_count']}`",
        f"- selected visit grouping matches C13: `{hard['image_grouping_matches_c13']}`",
        f"- visit reports match exact patient-date source rows: `{hard['visit_reports_match_real_source']}`",
        f"- dated bio matches exact patient-date source rows: `{hard['dated_bio_matches_exact_source_date']}`",
        f"- test used for reconstruction design: `{invariance['test_used_for_rule_design']}`",
        "- labels, patient IDs, absolute dates, counts, and audit-only shortcuts are absent from the predictor input.",
        "- validation AUC is the only checkpoint and route-promotion metric; test remains reporting-only.",
    ]
    (output_dir / "c27_leakage_exclusion_audit.md").write_text("\n".join(leakage_lines) + "\n", encoding="utf-8")
    return checks


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = static_checks(config)
    if not args.static_only and all(check["pass"] for check in checks):
        checks.extend(full_server_checks(config, output.parent))
    passed = all(check["pass"] for check in checks)
    if passed and not args.static_only:
        decision = "C27_VTME_DIRECT_MULTI_SEED_AUTHORIZED"
    elif passed:
        decision = "C27_VTME_STATIC_GATE_PASS"
    else:
        decision = "DEMA_C27_PATH_GATE_FAIL"
    result = {
        "phase": "C27-VTME",
        "gate": "STATIC_ONLY" if args.static_only else "FULL_SERVER",
        "pass": passed,
        "passed": sum(int(check["pass"]) for check in checks),
        "total": len(checks),
        "checks": checks,
        "decision": decision,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)
    if not args.static_only:
        print("C27_VTME_DIRECT_MULTI_SEED_AUTHORIZED")


if __name__ == "__main__":
    main()
