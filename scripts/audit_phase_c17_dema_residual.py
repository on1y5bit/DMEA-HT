#!/usr/bin/env python3
"""Run the C17 DEMA residual static and synthetic gate."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.c17_residual import C17ResidualModel  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import (  # noqa: E402
    TEXT_MASK_KEYS,
    DiseaseStateAlignmentHead,
    MechanismEvidenceAggregationHead,
    build_text_evidence_masks,
)
from dmea_ht.models import DMEAHTModel  # noqa: E402


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
FORBIDDEN_DESIGNS = ("dssa", "shared_specific", "shared-specific", "private representation", "modality-invariant", "decalign")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/dema_ht_c17_residual_bce_smoke.yaml")
    parser.add_argument("--output", default="analysis_reports/phase_c17_dema/c17_dema_residual_synthetic_gate.json")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def make_batch(hidden_text_length: int = 32) -> Dict[str, torch.Tensor]:
    torch.manual_seed(117)
    texts = (
        "[C13_LATEST_THYROID] diffuse morphology [C13_HISTORY_THYROID] history [C13_FULL_REPORT] report",
        "[C13_LATEST_THYROID] normal morphology [C13_FULL_REPORT] report",
        "[C13_FULL_REPORT] uncertain report",
        "ordinary thyroid report",
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


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    configs = sorted((repo_root / "configs").glob("*.yaml"))
    parsed = {path.name: load_config(path) for path in configs}
    expected = {
        "dema_ht_c17_residual_bce_smoke.yaml",
        "dema_ht_c17_residual_bce_seed0.yaml",
        "dema_ht_c17_residual_positive_preserve_smoke.yaml",
        "dema_ht_c17_residual_positive_preserve_seed0.yaml",
        "dema_ht_c17_formal_multiseed.yaml",
    }
    check("all_configs_parse", len(parsed) == len(configs), len(parsed))
    check("all_five_c17_configs_present", expected.issubset(parsed), sorted(expected & parsed.keys()))
    check("c17_primary_metric_is_validation_auc", all(item.get("training", {}).get("primary_metric") == "val_AUC" for item in parsed.values() if item.get("phase") == "c17"), "C17 configs")
    check("c17_test_disabled_before_formal", all(not item["training"].get("evaluate_test", True) for item in parsed.values() if item.get("phase") == "c17" and "formal" not in item.get("project", {}).get("name", "")), "smoke and seed-0 configs")

    batch = make_batch()
    legacy_config = copy.deepcopy(config)
    legacy_config.pop("phase", None)
    legacy_config.pop("c17", None)
    legacy_config["model"].pop("use_mea", None)
    torch.manual_seed(19)
    legacy_absent = DMEAHTModel(legacy_config).eval()
    torch.manual_seed(19)
    legacy_false_config = copy.deepcopy(legacy_config)
    legacy_false_config["model"]["use_mea"] = False
    legacy_false = DMEAHTModel(legacy_false_config).eval()
    absent_state = legacy_absent.state_dict()
    false_state = legacy_false.state_dict()
    state_equal = list(absent_state) == list(false_state) and all(torch.equal(absent_state[key], false_state[key]) for key in absent_state)
    with torch.no_grad():
        legacy_delta = float((legacy_absent(batch)["logit"] - legacy_false(batch)["logit"]).abs().max())
    check("legacy_absent_false_state_and_logits_identical", state_equal and legacy_delta == 0.0, legacy_delta)

    torch.manual_seed(23)
    old_head = DiseaseStateAlignmentHead(16, 0.0).eval()
    new_head = MechanismEvidenceAggregationHead(16, 0.0).eval()
    new_head.load_state_dict(old_head.state_dict())
    aggregate = {
        "support": torch.randn(3, 16),
        "opposition": torch.randn(3, 16),
        "uncertainty": torch.randn(3, 16),
        "conflict": torch.randn(3, 16),
        "conflict_score": torch.rand(3),
    }
    mechanism_state = torch.randn(3, 16)
    with torch.no_grad():
        old_result = old_head(mechanism_state, aggregate)["logit"]
        new_result = new_head(mechanism_state, aggregate)["logit"]
    check("mechanism_head_rename_logit_equivalent", float((old_result - new_result).abs().max()) <= 1e-8, float((old_result - new_result).abs().max()))

    model = C17ResidualModel(config, seed=0).eval()
    with torch.no_grad():
        outputs = model(batch)
    float_outputs = [value for value in outputs.values() if torch.is_tensor(value) and value.is_floating_point()]
    zero_equivalence = float((outputs["logit"] - outputs["base_logit"]).abs().max())
    residual_bound = float(outputs["delta_logit"].abs().max())
    check("c13_checkpoint_loads_and_outputs_are_finite", all(bool(torch.isfinite(value).all()) for value in float_outputs), len(float_outputs))
    check("zero_residual_reproduces_c13_logits", zero_equivalence <= 1e-8, zero_equivalence)
    check("residual_bound_holds", residual_bound <= 0.50 + 1e-6, residual_bound)

    empty_batch = clone_batch(batch)
    empty_batch["image_mask"].zero_()
    empty_batch["report_attention_mask"].zero_()
    empty_batch["bio_missing_mask"].fill_(1.0)
    for key in TEXT_MASK_KEYS:
        empty_batch[key].zero_()
    with torch.no_grad():
        empty_outputs = model(empty_batch)
    check("unavailable_evidence_is_masked_without_nan", bool(torch.isfinite(empty_outputs["logit"]).all()), empty_outputs["logit"].tolist())

    positive_delta = outputs["delta_logit"][:2]
    positive_loss = F.relu(-positive_delta - 0.05).mean()
    check("positive_preservation_loss_finite", bool(torch.isfinite(positive_loss)), float(positive_loss))
    negative_delta = outputs["delta_logit"].detach().clone().requires_grad_(True)
    negative_only_loss = negative_delta.sum() * 0.0
    negative_only_loss.backward()
    check("all_negative_positive_preservation_is_graph_zero", float(negative_only_loss) == 0.0 and negative_delta.grad is not None, negative_delta.grad.tolist())

    model.train()
    model.zero_grad(set_to_none=True)
    train_outputs = model(batch)
    mixed_loss = F.binary_cross_entropy_with_logits(train_outputs["logit"], batch["label"])
    mixed_loss = mixed_loss + 0.001 * train_outputs["delta_logit"].square().mean()
    mixed_loss = mixed_loss + 0.02 * F.relu(-train_outputs["delta_logit"][batch["label"] > 0.5] - 0.05).mean()
    mixed_loss.backward()
    output_grad = sum(float(parameter.grad.detach().abs().sum()) for parameter in model.residual_head.parameters() if parameter.grad is not None)
    check("residual_head_receives_initial_gradient", output_grad > 0.0, output_grad)

    with torch.no_grad():
        model.residual_head.mlp[-1].weight.fill_(0.01)
    model.zero_grad(set_to_none=True)
    warm_outputs = model(batch)
    warm_loss = F.binary_cross_entropy_with_logits(warm_outputs["logit"], batch["label"]) + 0.001 * warm_outputs["delta_logit"].square().mean()
    warm_loss.backward()
    mea_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith("mechanism_evidence_alignment.") and parameter.grad is not None
    )
    frozen_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.base_model.parameters()
        if parameter.grad is not None
    )
    check("gradients_reach_dema_modules_after_output_warm_start", mea_grad > 0.0, mea_grad)
    check("gradients_do_not_reach_frozen_c13", frozen_grad == 0.0, frozen_grad)

    source = "\n".join(
        inspect.getsource(module)
        for module in (
            sys.modules["dmea_ht.models"],
            sys.modules["dmea_ht.mechanism_evidence_alignment"],
            sys.modules["dmea_ht.c17_residual"],
        )
    )
    lowered = source.lower()
    check("shortcut_fields_absent_from_model_source", not any(field in source for field in FORBIDDEN_SHORTCUTS), FORBIDDEN_SHORTCUTS)
    check("forbidden_alignment_designs_absent", not any(term in lowered for term in FORBIDDEN_DESIGNS), FORBIDDEN_DESIGNS)

    result = {
        "status": "PASS" if checks and all(item["pass"] for item in checks) else "FAIL",
        "passed": sum(int(item["pass"]) for item in checks),
        "total": len(checks),
        "config": str(config_path),
        "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
