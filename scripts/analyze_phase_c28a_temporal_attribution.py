#!/usr/bin/env python3
"""Run the frozen C28-A validation temporal-attribution audit."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, masked_softmax  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402


SEEDS = (0, 42, 3407)
VARIANTS = OrderedDict(
    (
        ("V0_official", "official learned content plus fixed recency"),
        ("V1_uniform", "uniform valid-visit aggregation"),
        ("V2_recency_only", "fixed recency prior only"),
        ("V3_content_only", "learned content score only"),
        ("V4_latest_only", "latest valid visit only"),
        ("V5_history_mean_only", "uniform pre-latest history only"),
    )
)
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits", "raw_n_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c27_vtme_multiseed.yaml")
    parser.add_argument("--c27-run-dir", default="runs/dema_ht_c27_vtme_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c28a_dema")
    parser.add_argument("--stage", choices=("gate", "analyze"), required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_validation_loader(config: Mapping[str, Any], rows: Sequence[Dict[str, Any]]) -> DataLoader:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    dataset = VisitPatientDataset(
        rows=rows,
        data_root=project["data_root"],
        split="val",
        image_size=int(model_cfg["image_size"]),
        text_max_length=int(model_cfg["text_max_length"]),
        text_vocab_size=int(model_cfg["text_vocab_size"]),
        bio_dim=int(model_cfg["bio_dim"]),
        max_images_per_visit=int(model_cfg["max_images_per_visit"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_visit_batch,
        pin_memory=torch.cuda.is_available(),
    )


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def binary_auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    return float(roc_auc_score(y, p))


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def static_contract_checks() -> List[Dict[str, Any]]:
    analyzer = Path(__file__).resolve()
    collector = REPO_ROOT / "scripts" / "collect_phase_c28a_report.py"
    calls = called_names(analyzer) | called_names(collector)
    non_auc_metric_calls = {
        name for name in calls if (name.endswith("_score") or name.endswith("_curve")) and name != "roc_auc_score"
    }
    worktree_lines = git_output("worktree", "list", "--porcelain").splitlines()
    worktrees = [Path(line.split(" ", 1)[1]).resolve() for line in worktree_lines if line.startswith("worktree ")]
    c28_configs = list((REPO_ROOT / "configs").glob("*c28*"))
    return [
        {"name": "active_branch_main", "pass": git_output("branch", "--show-current") == "main"},
        {
            "name": "canonical_worktree_only",
            "pass": len(worktrees) == 1 and worktrees[0] == REPO_ROOT.resolve(),
            "detail": [str(path) for path in worktrees],
        },
        {"name": "no_c28_training_config", "pass": not c28_configs, "detail": [str(path) for path in c28_configs]},
        {"name": "no_optimizer_construction", "pass": not ({"Adam", "AdamW", "SGD", "RMSprop"} & calls)},
        {"name": "no_backward_call", "pass": "backward" not in calls},
        {"name": "validation_loader_only", "pass": True, "detail": "VisitPatientDataset split is fixed to val"},
        {
            "name": "auc_only_metric_contract",
            "pass": not non_auc_metric_calls,
            "detail": sorted(non_auc_metric_calls),
        },
        {
            "name": "no_model_combination_call",
            "pass": not ({"vstack_predictions", "average_checkpoints", "weighted_vote"} & calls),
        },
        {"name": "no_checkpoint_write", "pass": "save" not in calls},
        {
            "name": "no_training_process_launch",
            "pass": "Popen" not in calls and "run" not in calls,
            "detail": "the only subprocess call is read-only git check_output",
        },
    ]


def temporal_variant_weights(
    core: torch.nn.Module,
    content_scores: torch.Tensor,
    recency: torch.Tensor,
    visit_mask: torch.Tensor,
    official_weights: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    valid = visit_mask.unsqueeze(-1).expand_as(content_scores)
    valid_float = valid.to(content_scores.dtype)
    counts = visit_mask.sum(dim=1)
    denominator = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform = valid_float / denominator
    prior_scores = core.recency_prior_log_odds * recency.unsqueeze(-1)
    recency_only = masked_softmax(prior_scores.expand_as(content_scores), valid, dim=1)
    content_only = masked_softmax(content_scores, valid, dim=1)

    latest_only = torch.zeros_like(content_scores)
    latest_indices = (counts - 1).clamp_min(0)
    latest_only[
        torch.arange(len(counts), device=counts.device), latest_indices
    ] = 1.0

    history_valid = valid.clone()
    history_valid[
        torch.arange(len(counts), device=counts.device), latest_indices
    ] = False
    history_float = history_valid.to(content_scores.dtype)
    history_only = history_float / history_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    history_available = counts > 1
    return {
        "V0_official": official_weights,
        "V1_uniform": uniform,
        "V2_recency_only": recency_only,
        "V3_content_only": content_only,
        "V4_latest_only": latest_only,
        "V5_history_mean_only": history_only,
    }, history_available


def counterfactual_forward(
    core: torch.nn.Module,
    visit_states: torch.Tensor,
    temporal_weights: torch.Tensor,
    conflicts: torch.Tensor,
    fallback_bio_context: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    mechanism_states = torch.einsum("bvk,bvkh->bkh", temporal_weights, visit_states)
    patient_input = torch.cat(
        [mechanism_states.flatten(start_dim=1), conflicts, fallback_bio_context], dim=-1
    )
    patient_state = core.patient_projection(patient_input)
    logit = core.classifier(patient_state).squeeze(-1)
    return {"logit": logit, "prob": torch.sigmoid(logit), "patient_state": patient_state}


def text_evidence_group(support: torch.Tensor, opposition: torch.Tensor, count: int) -> str:
    if count <= 1:
        return "single_visit"
    latest_support = bool(support[count - 1])
    latest_opposition = bool(opposition[count - 1])
    history_support = bool(support[: count - 1].any())
    history_opposition = bool(opposition[: count - 1].any())
    if latest_support and history_opposition:
        return "latest_positive_like_history_negative_like"
    if latest_opposition and history_support:
        return "latest_negative_like_history_positive_like"
    return "latest_history_mixed_or_uncertain"


def conflict_group(mean_conflict: float, count: int) -> str:
    if count <= 1:
        return "single_visit"
    if mean_conflict < 0.25:
        return "multi_visit_low_conflict"
    if mean_conflict < 0.75:
        return "multi_visit_medium_conflict"
    return "multi_visit_high_conflict"


def checkpoint_unchanged(model: torch.nn.Module, reference: Mapping[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    if set(current) != set(reference):
        return False
    return all(torch.equal(current[key].detach().cpu(), reference[key].detach().cpu()) for key in current)


def build_pairwise_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = predictions[predictions["seed"].astype(int) == seed]
        official = seed_frame[seed_frame["variant"].eq("V0_official")].set_index("patient_id")
        positives = official[official["label"].astype(int) == 1].sort_index()
        negatives = official[official["label"].astype(int) == 0].sort_index()
        for variant in VARIANTS:
            variant_frame = seed_frame[seed_frame["variant"].eq(variant)].set_index("patient_id")
            for positive_id, positive in positives.iterrows():
                for negative_id, negative in negatives.iterrows():
                    positive_variant = variant_frame.loc[positive_id]
                    negative_variant = variant_frame.loc[negative_id]
                    available = bool(positive_variant["available"] and negative_variant["available"])
                    official_margin = float(positive["final_prob"] - negative["final_prob"])
                    official_inversion = official_margin < 0.0
                    if available:
                        positive_score = float(positive_variant["final_prob"])
                        negative_score = float(negative_variant["final_prob"])
                        margin = positive_score - negative_score
                        inversion: float | int = int(margin < 0.0)
                        repaired: float | int = int(official_inversion and not bool(inversion))
                        introduced: float | int = int((not official_inversion) and bool(inversion))
                    else:
                        positive_score = negative_score = margin = float("nan")
                        inversion = repaired = introduced = float("nan")
                    rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "positive_patient_id": positive_id,
                            "negative_patient_id": negative_id,
                            "available": available,
                            "positive_score": positive_score,
                            "negative_score": negative_score,
                            "margin": margin,
                            "inversion": inversion,
                            "official_margin": official_margin,
                            "official_inversion": int(official_inversion),
                            "repaired": repaired,
                            "introduced": introduced,
                        }
                    )
    return pd.DataFrame(rows)


def build_positive_damage_exports(
    predictions: pd.DataFrame, baseline: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    official = predictions[predictions["variant"].eq("V0_official")].copy()
    positive = official[official["label"].astype(int) == 1].copy()
    positive["material_positive_damage"] = (
        ((positive["c17_predicted_class"].astype(int) == 1) & (positive["predicted_class"].astype(int) == 0))
        | ((positive["final_prob"] - positive["c17_prob"]) <= -0.05)
    )
    positive["severe_positive_damage"] = (positive["final_prob"] - positive["c17_prob"]) <= -0.10
    material = positive[positive["material_positive_damage"]].copy()

    pivot_prob = predictions.pivot(index=["seed", "patient_id"], columns="variant", values="final_prob")
    pivot_pred = predictions.pivot(index=["seed", "patient_id"], columns="variant", values="predicted_class")
    for variant in VARIANTS:
        material[f"{variant}_prob"] = [
            pivot_prob.loc[(int(row.seed), str(row.patient_id)), variant] for row in material.itertuples()
        ]
        material[f"{variant}_predicted_class"] = [
            pivot_pred.loc[(int(row.seed), str(row.patient_id)), variant] for row in material.itertuples()
        ]

    baseline_wide = baseline.pivot_table(
        index=["seed", "patient_id"],
        columns="mechanism",
        values=["actual_latest_weight", "baseline_latest_weight", "latest_weight_excess"],
        aggfunc="first",
    )
    baseline_wide.columns = [f"{metric}_{mechanism}" for metric, mechanism in baseline_wide.columns]
    baseline_wide = baseline_wide.reset_index()
    material = material.merge(baseline_wide, on=["seed", "patient_id"], how="left", validate="one_to_one")

    response_rows: List[Dict[str, Any]] = []
    for row in material.itertuples(index=False):
        for variant in VARIANTS:
            probability = getattr(row, f"{variant}_prob")
            predicted = getattr(row, f"{variant}_predicted_class")
            available = bool(np.isfinite(probability))
            variant_damage = bool(
                available
                and ((int(row.c17_predicted_class) == 1 and int(predicted) == 0) or probability - row.c17_prob <= -0.05)
            )
            response_rows.append(
                {
                    "seed": int(row.seed),
                    "patient_id": str(row.patient_id),
                    "variant": variant,
                    "available": available,
                    "variant_probability": probability,
                    "difference_vs_official": probability - row.final_prob if available else float("nan"),
                    "difference_vs_c17": probability - row.c17_prob if available else float("nan"),
                    "variant_predicted_class": predicted,
                    "material_damage_under_variant": variant_damage if available else float("nan"),
                    "rescued_official_material_damage": bool(available and not variant_damage),
                    "rescued_official_c17_tp_to_fn": bool(
                        available
                        and int(row.c17_predicted_class) == 1
                        and int(row.predicted_class) == 0
                        and int(predicted) == 1
                    ),
                }
            )
    return material, pd.DataFrame(response_rows)


def run_inference(
    config: Dict[str, Any], c27_run: Path, c17_run: Path, device: torch.device, collect: bool
) -> Dict[str, Any]:
    rows = read_jsonl(config["project"]["manifest"])
    loader = build_validation_loader(config, rows)
    reproduction_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []
    runtime = {
        "checkpoint_paths_exist": True,
        "checkpoint_seed_metadata": True,
        "patient_label_alignment": True,
        "official_reproduction": True,
        "temporal_weight_reproduction": True,
        "visit_states_reused": True,
        "projection_classifier_reused": True,
        "conflicts_reused": True,
        "only_temporal_weights_change": True,
        "uniform_normalized": True,
        "recency_only_normalized": True,
        "content_only_normalized": True,
        "latest_only_normalized": True,
        "history_single_unavailable": True,
        "baseline_real_mask_exact": True,
        "normalization_values_finite": True,
        "variant_outputs_finite": True,
        "checkpoint_state_unchanged": True,
        "shortcut_fields_excluded": True,
        "pair_count_contract": True,
        "normalization_max_abs_sum_error": {variant: 0.0 for variant in VARIANTS},
    }

    for seed in SEEDS:
        checkpoint_path = c27_run / "checkpoints" / f"seed_{seed}_best.pt"
        c27_path = c27_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        c17_path = c17_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        if not all(path.exists() for path in (checkpoint_path, c27_path, c17_path)):
            runtime["checkpoint_paths_exist"] = False
            raise RuntimeError(f"C28A required C27/C17 artifact missing for seed {seed}")
        payload = load_checkpoint(checkpoint_path)
        if int(payload.get("seed", -1)) != seed:
            runtime["checkpoint_seed_metadata"] = False
            raise RuntimeError(f"C28A C27 checkpoint seed mismatch for seed {seed}")
        model = C27VTMEModel(config, seed).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        c27_saved = read_prediction(c27_path)
        c17_saved = read_prediction(c17_path)
        c27_map = c27_saved.set_index("patient_id")
        c17_map = c17_saved.set_index("patient_id")
        if len(c27_saved) != 94 or len(c17_saved) != 94 or set(c27_map.index) != set(c17_map.index):
            runtime["patient_label_alignment"] = False
            raise RuntimeError(f"C28A validation patient contract failed for seed {seed}")

        seen_ids: List[str] = []
        seen_labels: List[int] = []
        official_logits: List[float] = []
        official_probs: List[float] = []
        max_logit_error = max_probability_error = 0.0
        max_latest_error = max_weight_formula_error = max_v0_logit_error = 0.0
        threshold_mismatch = label_mismatch = 0
        content_min, content_max = float("inf"), -float("inf")
        combined_min, combined_max = float("inf"), -float("inf")
        max_visits = 0

        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                captured: Dict[str, torch.Tensor] = {}

                def capture_core_inputs(_module: torch.nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
                    captured["source_states"] = args[0]
                    captured["source_valid"] = args[1]
                    captured["visit_mask"] = args[2]
                    captured["fallback_bio_context"] = args[3]

                handle = model.core.register_forward_pre_hook(capture_core_inputs)
                official = model(batch)
                handle.remove()
                if set(captured) != {"source_states", "source_valid", "visit_mask", "fallback_bio_context"}:
                    raise RuntimeError("C28A forward hook failed to capture frozen core inputs")

                visit_states = official["visit_states"]
                conflicts = official["conflicts"]
                fallback = captured["fallback_bio_context"]
                visit_mask = captured["visit_mask"]
                source_valid = captured["source_valid"]
                visit_snapshot = visit_states.clone()
                conflict_snapshot = conflicts.clone()
                fallback_snapshot = fallback.clone()
                core_snapshot = {
                    name: parameter.detach().clone()
                    for name, parameter in model.core.named_parameters()
                    if name.startswith("patient_projection.") or name.startswith("classifier.")
                }

                content_scores = model.core.temporal_output(
                    torch.tanh(model.core.temporal_linear(model.core.temporal_norm(visit_states)))
                ).squeeze(-1)
                recency = official["recency"]
                valid = visit_mask.unsqueeze(-1).expand_as(content_scores)
                combined_scores = content_scores + model.core.recency_prior_log_odds * recency.unsqueeze(-1)
                rebuilt_official = masked_softmax(combined_scores, valid, dim=1)
                max_weight_formula_error = max(
                    max_weight_formula_error,
                    float((rebuilt_official - official["temporal_weights"]).abs().max().cpu()),
                )
                weights_by_variant, history_available = temporal_variant_weights(
                    model.core, content_scores, recency, visit_mask, official["temporal_weights"]
                )
                prior_scores = model.core.recency_prior_log_odds * recency.unsqueeze(-1)
                prior_numerator = torch.exp(prior_scores.expand_as(content_scores)) * valid.to(content_scores.dtype)
                manual_baseline = prior_numerator / prior_numerator.sum(dim=1, keepdim=True).clamp_min(1e-8)
                runtime["baseline_real_mask_exact"] = bool(
                    runtime["baseline_real_mask_exact"]
                    and torch.allclose(
                        manual_baseline,
                        weights_by_variant["V2_recency_only"],
                        atol=1e-7,
                        rtol=0.0,
                    )
                )
                variant_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
                for variant, weights in weights_by_variant.items():
                    variant_outputs[variant] = counterfactual_forward(
                        model.core, visit_states, weights, conflicts, fallback
                    )
                    eligible = history_available if variant == "V5_history_mean_only" else torch.ones_like(history_available)
                    sums = weights.sum(dim=1)
                    sum_error = (
                        float((sums[eligible] - 1.0).abs().max().cpu()) if bool(eligible.any()) else 0.0
                    )
                    runtime["normalization_max_abs_sum_error"][variant] = max(
                        float(runtime["normalization_max_abs_sum_error"][variant]), sum_error
                    )
                    normalized = bool(torch.isfinite(weights).all()) and sum_error <= 1e-6
                    padded_zero = bool((weights * (~valid).to(weights.dtype)).abs().max().cpu() <= 1e-8)
                    key = {
                        "V1_uniform": "uniform_normalized",
                        "V2_recency_only": "recency_only_normalized",
                        "V3_content_only": "content_only_normalized",
                        "V4_latest_only": "latest_only_normalized",
                    }.get(variant)
                    if key is not None:
                        runtime[key] = bool(runtime[key] and normalized and padded_zero)
                    runtime["variant_outputs_finite"] = bool(
                        runtime["variant_outputs_finite"]
                        and torch.isfinite(variant_outputs[variant]["logit"]).all()
                        and torch.isfinite(variant_outputs[variant]["prob"]).all()
                    )
                runtime["history_single_unavailable"] = bool(
                    runtime["history_single_unavailable"]
                    and torch.equal(history_available, visit_mask.sum(dim=1) > 1)
                    and torch.all(weights_by_variant["V5_history_mean_only"][~history_available] == 0)
                )
                max_v0_logit_error = max(
                    max_v0_logit_error,
                    float((variant_outputs["V0_official"]["logit"] - official["logit"]).abs().max().cpu()),
                )
                runtime["visit_states_reused"] = bool(
                    runtime["visit_states_reused"] and torch.equal(visit_states, visit_snapshot)
                )
                runtime["conflicts_reused"] = bool(
                    runtime["conflicts_reused"] and torch.equal(conflicts, conflict_snapshot)
                )
                runtime["only_temporal_weights_change"] = bool(
                    runtime["only_temporal_weights_change"]
                    and torch.equal(fallback, fallback_snapshot)
                    and all(
                        torch.equal(dict(model.core.named_parameters())[name], value)
                        for name, value in core_snapshot.items()
                    )
                )
                runtime["projection_classifier_reused"] = runtime["only_temporal_weights_change"]

                content_min = min(content_min, float(content_scores[valid].min().cpu()))
                content_max = max(content_max, float(content_scores[valid].max().cpu()))
                combined_min = min(combined_min, float(combined_scores[valid].min().cpu()))
                combined_max = max(combined_max, float(combined_scores[valid].max().cpu()))
                max_visits = max(max_visits, int(visit_mask.sum(dim=1).max().cpu()))

                support = batch["visit_support_present"].detach().cpu()
                opposition = batch["visit_opposition_present"].detach().cpu()
                labels = batch["label"].detach().cpu().numpy().astype(int)
                counts = visit_mask.sum(dim=1).detach().cpu().numpy().astype(int)
                official_cpu = {key: value.detach().cpu().numpy() for key, value in official.items() if torch.is_tensor(value)}
                source_valid_cpu = source_valid.detach().cpu().numpy().astype(bool)
                variant_cpu = {
                    variant: {key: value.detach().cpu().numpy() for key, value in output.items()}
                    for variant, output in variant_outputs.items()
                }
                weights_cpu = {variant: value.detach().cpu().numpy() for variant, value in weights_by_variant.items()}

                for index, patient_id_value in enumerate(batch["patient_id"]):
                    patient_id = str(patient_id_value)
                    if patient_id not in c27_map.index or patient_id not in c17_map.index:
                        runtime["patient_label_alignment"] = False
                        raise RuntimeError(f"C28A unknown validation patient {patient_id}")
                    saved = c27_map.loc[patient_id]
                    c17 = c17_map.loc[patient_id]
                    label = int(labels[index])
                    if label != int(saved["label"]) or label != int(c17["label"]):
                        label_mismatch += 1
                    logit = float(official_cpu["logit"][index])
                    probability = float(official_cpu["prob"][index])
                    max_logit_error = max(max_logit_error, abs(logit - float(saved["final_logit"])))
                    max_probability_error = max(max_probability_error, abs(probability - float(saved["final_prob"])))
                    threshold_mismatch += int((probability >= 0.5) != bool(int(saved["predicted_class"])))
                    latest_index = counts[index] - 1
                    actual_latest = official_cpu["temporal_weights"][index, latest_index]
                    baseline_latest = weights_cpu["V2_recency_only"][index, latest_index]
                    for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                        max_latest_error = max(
                            max_latest_error,
                            abs(float(actual_latest[mechanism_index]) - float(saved[f"temporal_weight_latest_{mechanism}"])),
                        )
                        actual = float(actual_latest[mechanism_index])
                        baseline_value = float(baseline_latest[mechanism_index])
                        excess = actual - baseline_value
                        ratio = actual / max(baseline_value, 1e-8)
                        log_ratio = float(np.log(max(actual, 1e-8) / max(baseline_value, 1e-8)))
                        runtime["normalization_values_finite"] = bool(
                            runtime["normalization_values_finite"]
                            and np.isfinite([actual, baseline_value, excess, ratio, log_ratio]).all()
                        )
                        baseline_rows.append(
                            {
                                "seed": seed,
                                "patient_id": patient_id,
                                "label": label,
                                "mechanism": mechanism,
                                "visit_count_audit_only": counts[index],
                                "selected_n_visits_audit_only": saved.get("selected_n_visits", counts[index]),
                                "temporal_mask_count": counts[index],
                                "source_valid_visit_count": int(source_valid_cpu[index, : counts[index], mechanism_index].sum()),
                                "actual_latest_weight": actual,
                                "baseline_latest_weight": baseline_value,
                                "latest_weight_excess": excess,
                                "latest_weight_ratio": ratio,
                                "latest_weight_log_ratio": log_ratio,
                            }
                        )

                    mean_conflict = float(official_cpu["conflicts"][index].mean())
                    common: Dict[str, Any] = {
                        "seed": seed,
                        "patient_id": patient_id,
                        "label": label,
                        "c17_logit": float(c17["final_logit"]),
                        "c17_prob": float(c17["final_prob"]),
                        "c17_predicted_class": int(float(c17["final_prob"]) >= 0.5),
                        "official_c27_logit": logit,
                        "official_c27_prob": probability,
                        "official_c27_predicted_class": int(probability >= 0.5),
                        "visit_count_audit_only": counts[index],
                        "reconstructable_visit_count_audit_only": saved.get("reconstructable_visit_count", ""),
                        "mean_conflict": mean_conflict,
                        "max_conflict": float(official_cpu["conflicts"][index].max()),
                        "conflict_group": conflict_group(mean_conflict, counts[index]),
                        "text_evidence_group": text_evidence_group(support[index], opposition[index], counts[index]),
                    }
                    for field in (*SELECTED_SHORTCUT_FIELDS, *RAW_SHORTCUT_FIELDS):
                        common[f"{field}_audit_only"] = saved.get(field, "")
                    for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                        common[f"conflict_{mechanism}"] = float(official_cpu["conflicts"][index, mechanism_index])
                        common[f"official_latest_weight_{mechanism}"] = float(actual_latest[mechanism_index])
                        common[f"baseline_latest_weight_{mechanism}"] = float(baseline_latest[mechanism_index])
                        common[f"latest_weight_excess_{mechanism}"] = float(
                            actual_latest[mechanism_index] - baseline_latest[mechanism_index]
                        )
                    if collect:
                        for variant in VARIANTS:
                            available = variant != "V5_history_mean_only" or counts[index] > 1
                            variant_logit = float(variant_cpu[variant]["logit"][index]) if available else float("nan")
                            variant_prob = float(variant_cpu[variant]["prob"][index]) if available else float("nan")
                            prediction_rows.append(
                                {
                                    **common,
                                    "variant": variant,
                                    "available": available,
                                    "final_logit": variant_logit,
                                    "final_prob": variant_prob,
                                    "predicted_class": int(variant_prob >= 0.5) if available else float("nan"),
                                }
                            )
                    seen_ids.append(patient_id)
                    seen_labels.append(label)
                    official_logits.append(logit)
                    official_probs.append(probability)

        id_alignment = set(seen_ids) == set(c27_map.index) and len(seen_ids) == len(set(seen_ids)) == 94
        label_alignment = label_mismatch == 0
        run_auc = binary_auc(seen_labels, official_probs)
        saved_auc = binary_auc(c27_saved["label"].astype(int), c27_saved["final_prob"].astype(float))
        state_unchanged = checkpoint_unchanged(model, payload["model"])
        runtime["patient_label_alignment"] = bool(runtime["patient_label_alignment"] and id_alignment and label_alignment)
        runtime["official_reproduction"] = bool(
            runtime["official_reproduction"]
            and max_logit_error <= 1e-6
            and max_probability_error <= 1e-7
            and abs(run_auc - saved_auc) <= 1e-12
            and threshold_mismatch == 0
        )
        runtime["temporal_weight_reproduction"] = bool(
            runtime["temporal_weight_reproduction"]
            and max_latest_error <= 1e-7
            and max_weight_formula_error <= 1e-7
            and max_v0_logit_error <= 1e-7
        )
        runtime["checkpoint_state_unchanged"] = bool(runtime["checkpoint_state_unchanged"] and state_unchanged)
        positive_count = int((np.asarray(seen_labels) == 1).sum())
        negative_count = int((np.asarray(seen_labels) == 0).sum())
        runtime["pair_count_contract"] = bool(
            runtime["pair_count_contract"] and positive_count == 47 and negative_count == 47
        )
        reproduction_rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "best_epoch": int(payload["best_epoch"]),
                "patient_count": len(seen_ids),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "patient_id_alignment": id_alignment,
                "label_alignment": label_alignment,
                "max_abs_logit_error": max_logit_error,
                "max_abs_probability_error": max_probability_error,
                "validation_auc": run_auc,
                "saved_validation_auc": saved_auc,
                "abs_auc_error": abs(run_auc - saved_auc),
                "threshold_prediction_mismatch_count": threshold_mismatch,
                "max_abs_latest_weight_error": max_latest_error,
                "max_abs_full_temporal_weight_formula_error": max_weight_formula_error,
                "max_abs_v0_counterfactual_logit_error": max_v0_logit_error,
                "checkpoint_state_unchanged": state_unchanged,
            }
        )
        inventory_rows.append(
            {
                "seed": seed,
                "patients": len(seen_ids),
                "max_visits": max_visits,
                "mechanism_slots": len(MECHANISM_NAMES),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "content_score_min": content_min,
                "content_score_max": content_max,
                "combined_score_min": combined_min,
                "combined_score_max": combined_max,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runtime["shortcut_fields_excluded"] = "shortcuts" not in counterfactual_forward.__code__.co_varnames
    return {
        "reproduction": pd.DataFrame(reproduction_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "baseline": pd.DataFrame(baseline_rows),
        "inventory": pd.DataFrame(inventory_rows),
        "runtime": runtime,
    }


def gate_payload(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    checks = static_contract_checks()
    runtime_names = (
        "checkpoint_paths_exist",
        "checkpoint_seed_metadata",
        "patient_label_alignment",
        "official_reproduction",
        "temporal_weight_reproduction",
        "visit_states_reused",
        "projection_classifier_reused",
        "conflicts_reused",
        "only_temporal_weights_change",
        "uniform_normalized",
        "recency_only_normalized",
        "content_only_normalized",
        "latest_only_normalized",
        "history_single_unavailable",
        "baseline_real_mask_exact",
        "normalization_values_finite",
        "variant_outputs_finite",
        "checkpoint_state_unchanged",
        "shortcut_fields_excluded",
        "pair_count_contract",
    )
    normalization_variants = {
        "uniform_normalized": "V1_uniform",
        "recency_only_normalized": "V2_recency_only",
        "content_only_normalized": "V3_content_only",
        "latest_only_normalized": "V4_latest_only",
    }
    for name in runtime_names:
        item: Dict[str, Any] = {"name": name, "pass": bool(runtime[name])}
        if name in normalization_variants:
            variant = normalization_variants[name]
            item["detail"] = {
                "variant": variant,
                "max_abs_weight_sum_error": runtime["normalization_max_abs_sum_error"][variant],
                "float32_tolerance": 1e-6,
            }
        checks.append(item)
    if len(checks) != 30:
        raise RuntimeError(f"C28A gate definition must contain exactly 30 checks, found {len(checks)}")
    passed = all(bool(item["pass"]) for item in checks)
    return {
        "phase": "C28-A",
        "status": "C28A_ANALYSIS_AUTHORIZED" if passed else "C28A_ANALYSIS_INVALID",
        "pass": passed,
        "passed_checks": sum(bool(item["pass"]) for item in checks),
        "total_checks": len(checks),
        "checks": checks,
        "seeds": list(SEEDS),
        "data_scope": "validation_only",
        "parameter_updates": False,
    }


def write_inventory(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        "# C28-A Temporal Intermediate Inventory",
        "",
        "- Intermediates were captured with a temporary forward pre-hook on the frozen C27 core.",
        "- The hook captured the exact source states, source-valid mask, visit mask, and fallback bio context used by the official forward.",
        "- Visit mechanism states, content scores, ordinal recency, combined scores, temporal weights, and conflicts were obtained under inference mode.",
        "- Counterfactuals reused the same visit-state, conflict, fallback-context, patient-projection, and classifier tensors; only temporal weights changed.",
        "- Raw images, token tensors, and full visit-state tensors were not exported.",
        "",
        "| seed | patients | max visits | slots | hidden | content range | combined range |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.patients} | {row.max_visits} | {row.mechanism_slots} | {row.hidden_dim} | "
            f"[{row.content_score_min:.6f}, {row.content_score_max:.6f}] | "
            f"[{row.combined_score_min:.6f}, {row.combined_score_max:.6f}] |"
        )
    (output / "c28a_temporal_intermediate_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c27":
        raise RuntimeError("C28A must audit the frozen C27 config")
    if tuple(int(seed) for seed in config["training"]["seeds"]) != SEEDS:
        raise RuntimeError("C28A formal seeds must remain [0, 42, 3407]")
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    c27_run = resolve_path(args.c27_run_dir)
    c17_run = resolve_path(args.c17_run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "gate":
        result = run_inference(config, c27_run, c17_run, device, collect=False)
        result["reproduction"].to_csv(output / "c28a_reproduction_by_seed.csv", index=False)
        payload = gate_payload(result["runtime"])
        (output / "c28a_runtime_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}"}))
        if not payload["pass"]:
            raise RuntimeError("C28A_ANALYSIS_INVALID")
        return

    gate_path = output / "c28a_runtime_gate.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("pass", False):
        raise RuntimeError("C28A analysis requires a passing reproduction/runtime gate")
    result = run_inference(config, c27_run, c17_run, device, collect=True)
    post_payload = gate_payload(result["runtime"])
    if not post_payload["pass"]:
        raise RuntimeError("C28A_ANALYSIS_INVALID: full pass no longer satisfies the gate")
    result["reproduction"].to_csv(output / "c28a_reproduction_by_seed.csv", index=False)
    result["baseline"].to_csv(output / "c28a_temporal_baseline_by_patient_slot.csv", index=False)
    result["predictions"].to_csv(output / "c28a_counterfactual_predictions_val.csv", index=False)
    pairwise = build_pairwise_table(result["predictions"])
    expected_pairs = len(SEEDS) * len(VARIANTS) * 2209
    if len(pairwise) != expected_pairs:
        raise RuntimeError(f"C28A pairwise row contract failed: {len(pairwise)} != {expected_pairs}")
    pairwise.to_csv(output / "c28a_pairwise_ranking_by_variant.csv", index=False)
    positive, response = build_positive_damage_exports(result["predictions"], result["baseline"])
    positive.to_csv(output / "c28a_positive_damage_patients.csv", index=False)
    response.to_csv(output / "c28a_positive_damage_variant_response.csv", index=False)
    write_inventory(result["inventory"], output)
    print(
        json.dumps(
            {
                "status": "C28A_VALIDATION_ANALYSIS_COMPLETE",
                "seeds": list(SEEDS),
                "patients_per_seed": 94,
                "variants": list(VARIANTS),
                "pairwise_rows": len(pairwise),
            }
        )
    )


if __name__ == "__main__":
    main()
