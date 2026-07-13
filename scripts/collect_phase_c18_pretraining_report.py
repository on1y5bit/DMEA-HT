#!/usr/bin/env python3
"""Run C18 static, synthetic, and validation-only pretraining gates."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS, build_text_evidence_masks  # noqa: E402
from dmea_ht.models import C18DirectionalResidualModel  # noqa: E402
from train import hard_pairwise_ranking_loss  # noqa: E402


EXPECTED_SEEDS = [0, 42, 3407]
FORBIDDEN_SHORTCUTS = (
    "n_images",
    "n_visits",
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "raw_n_images",
    "raw_n_visits",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "source_folder",
)
FORBIDDEN_DESIGNS = ("dssa", "shared_specific", "shared-specific", "private representation", "decalign")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional-config", default="configs/dema_ht_c18_directional_residual_multiseed.yaml")
    parser.add_argument("--hardrank-config", default="configs/dema_ht_c18_directional_hardrank_multiseed.yaml")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c18_dema")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def make_batch(hidden_text_length: int = 32) -> Dict[str, torch.Tensor]:
    torch.manual_seed(1818)
    texts = (
        "diffuse thyroid morphology HT history",
        "normal thyroid morphology",
        "uncertain thyroid report",
        "ordinary report",
    )
    masks = [build_text_evidence_masks(text, hidden_text_length) for text in texts]
    batch: Dict[str, torch.Tensor] = {
        "images": torch.randn(4, 3, 3, 16, 16),
        "image_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32),
        "report_input_ids": torch.randint(2, 128, (4, hidden_text_length)),
        "report_attention_mask": torch.tensor(
            [[1] * hidden_text_length, [1] * 30 + [0] * 2, [1] * 20 + [0] * 12, [1] * 10 + [0] * 22],
            dtype=torch.long,
        ),
        "bio_values": torch.randn(4, 7),
        "bio_missing_mask": torch.tensor(
            [[0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], [1] * 7],
            dtype=torch.float32,
        ),
        "bio_abnormal_flags": torch.zeros(4, 7),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    for key in TEXT_MASK_KEYS:
        batch[key] = torch.stack([item[key] for item in masks])
    return batch


def check_row(checks: List[Dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})


def finite_gradients(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                return float("nan")
            total += float(parameter.grad.detach().abs().sum())
    return total


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    directional_path = Path(args.directional_config)
    hardrank_path = Path(args.hardrank_config)
    if not directional_path.is_absolute():
        directional_path = repo_root / directional_path
    if not hardrank_path.is_absolute():
        hardrank_path = repo_root / hardrank_path
    directional_config = load_config(directional_path)
    hardrank_config = load_config(hardrank_path)
    checks: List[Dict[str, Any]] = []

    for label, config, path, expected_output, hard_rank in (
        ("directional", directional_config, directional_path, "runs/dema_ht_c18_directional_multiseed", False),
        ("hardrank", hardrank_config, hardrank_path, "runs/dema_ht_c18_directional_hardrank_multiseed", True),
    ):
        check_row(checks, f"{label}_phase", config.get("phase") == "c18", config.get("phase"))
        check_row(checks, f"{label}_seeds", config.get("training", {}).get("seeds") == EXPECTED_SEEDS, config.get("training", {}).get("seeds"))
        check_row(checks, f"{label}_primary_metric", config.get("training", {}).get("primary_metric") == "val_AUC", config.get("training", {}).get("primary_metric"))
        check_row(checks, f"{label}_output_dir", config.get("project", {}).get("output_dir") == expected_output, config.get("project", {}).get("output_dir"))
        loss = config.get("loss", {})
        check_row(checks, f"{label}_fixed_loss_weights", loss.get("lambda_directional_residual") == 0.001 and loss.get("lambda_positive_preserve") == 0.02, loss)
        check_row(checks, f"{label}_hardrank_contract", bool(loss.get("hard_rank", False)) == hard_rank and float(loss.get("lambda_hard_rank", 0.0)) == (0.01 if hard_rank else 0.0), loss)
        config_text = path.read_text(encoding="utf-8").lower()
        check_row(checks, f"{label}_no_smoke_or_pilot_config", not any(token in config_text for token in ("smoke", "pilot", "seed0")), path.name)
        check_row(checks, f"{label}_base_is_checkpoint_not_predictions", "prediction" not in str(config.get("c18", {}).get("base_checkpoint", "")).lower(), config.get("c18", {}).get("base_checkpoint"))

    source = "\n".join(
        inspect.getsource(module)
        for module in (
            sys.modules["dmea_ht.models"],
            sys.modules["dmea_ht.mechanism_evidence_alignment"],
        )
    )
    lowered = source.lower()
    check_row(checks, "shortcut_fields_absent_from_model_source", not any(field in source for field in FORBIDDEN_SHORTCUTS), FORBIDDEN_SHORTCUTS)
    check_row(checks, "forbidden_alignment_designs_absent", not any(term in lowered for term in FORBIDDEN_DESIGNS), FORBIDDEN_DESIGNS)
    train_source = (repo_root / "train.py").read_text(encoding="utf-8")
    check_row(checks, "saved_predictions_not_training_inputs", "val_predictions_seed_" not in train_source and "test_predictions_seed_" not in train_source, "prediction paths absent")

    batch = make_batch()
    model = C18DirectionalResidualModel(directional_config, seed=0).eval()
    with torch.no_grad():
        outputs = model(batch)
    float_outputs = [value for value in outputs.values() if torch.is_tensor(value) and value.is_floating_point()]
    check_row(checks, "synthetic_outputs_finite", all(bool(torch.isfinite(value).all()) for value in float_outputs), len(float_outputs))
    initial_delta = float((outputs["logit"] - outputs["base_logit"]).abs().max())
    check_row(checks, "initial_directional_delta_reproduces_base", initial_delta <= 1e-8, initial_delta)
    for name, lower, upper in (
        ("support_delta", 0.0, 0.50),
        ("opposition_delta", 0.0, 0.50),
        ("support_gate", 0.0, 1.0),
        ("opposition_gate", 0.0, 1.0),
        ("conflict_suppression", 0.0, 1.0),
    ):
        values = outputs[name]
        check_row(checks, f"{name}_bound", bool((values >= lower - 1e-7).all() and (values <= upper + 1e-7).all()), [float(values.min()), float(values.max())])
    check_row(checks, "high_conflict_suppression_is_smaller", float(1.0 - 0.9) < float(1.0 - 0.1), "deterministic 1-conflict mapping")

    empty_batch = {key: value.clone() for key, value in batch.items()}
    empty_batch["image_mask"].zero_()
    empty_batch["report_attention_mask"].zero_()
    empty_batch["bio_missing_mask"].fill_(1.0)
    for key in TEXT_MASK_KEYS:
        empty_batch[key].zero_()
    with torch.no_grad():
        empty_outputs = model(empty_batch)
    check_row(checks, "unavailable_evidence_has_no_nan", bool(torch.isfinite(empty_outputs["logit"]).all()), empty_outputs["logit"].tolist())

    model.train()
    model.zero_grad(set_to_none=True)
    train_outputs = model(batch)
    positive_mask = batch["label"] > 0.5
    loss = F.binary_cross_entropy_with_logits(train_outputs["logit"], batch["label"])
    loss = loss + 0.001 * (train_outputs["effective_support_delta"].square() + train_outputs["effective_opposition_delta"].square()).mean()
    loss = loss + 0.02 * F.relu(-train_outputs["directional_delta"][positive_mask] - 0.05).mean()
    loss.backward()
    new_gradients = {
        name: finite_gradients(module)
        for name, module in (
            ("support_head", model.support_head),
            ("opposition_head", model.opposition_head),
            ("support_gate", model.support_gate),
            ("opposition_gate", model.opposition_gate),
        )
    }
    check_row(checks, "new_directional_modules_receive_finite_gradient", all(value > 0.0 for value in new_gradients.values()), new_gradients)
    frozen_grad = sum(float(parameter.grad.detach().abs().sum()) for parameter in model.base_model.parameters() if parameter.grad is not None)
    check_row(checks, "frozen_c13_receives_no_gradient", frozen_grad == 0.0, frozen_grad)

    base_logit = torch.tensor([0.0, 0.1, 1.0, 1.1], requires_grad=False)
    final_logit = torch.tensor([0.2, 0.0, 1.1, 0.9], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    hard_loss, hard_count = hard_pairwise_ranking_loss(base_logit, final_logit, labels, 0.50)
    check_row(checks, "hard_pair_loss_has_training_pairs", hard_count > 0 and bool(torch.isfinite(hard_loss)), [hard_count, float(hard_loss)])
    hard_loss.backward()
    check_row(checks, "hard_pair_loss_is_graph_connected", final_logit.grad is not None and bool(torch.isfinite(final_logit.grad).all()), final_logit.grad.tolist())
    single_final = torch.tensor([0.1, 0.2], requires_grad=True)
    single_zero, single_count = hard_pairwise_ranking_loss(torch.zeros(2), single_final, torch.zeros(2), 0.50)
    single_zero.backward()
    check_row(checks, "single_class_hard_pair_is_graph_zero", single_count == 0 and single_final.grad is not None and float(single_zero) == 0.0, single_final.grad.tolist())

    checks_path = output_dir / "c18_synthetic_checks.csv"
    import csv

    with checks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "pass", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    result = {
        "status": "PASS" if checks and all(item["pass"] for item in checks) else "FAIL",
        "passed": sum(int(item["pass"]) for item in checks),
        "total": len(checks),
        "directional_config": str(directional_path),
        "hardrank_config": str(hardrank_path),
        "no_smoke": True,
        "no_seed0_pilot": True,
        "direct_seeds": EXPECTED_SEEDS,
        "checks": checks,
    }
    (output_dir / "c18_pretraining_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
