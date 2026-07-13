#!/usr/bin/env python3
"""Run the pre-training static and synthetic gate for C16-MEA."""

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
from dmea_ht.mechanism_evidence_alignment import (  # noqa: E402
    TEXT_MASK_KEYS,
    BioEvidenceProjector,
    TextEvidenceRoleProjector,
    build_text_evidence_masks,
)
from dmea_ht.mea_losses import mea_loss_weights_for_epoch, pairwise_ranking_loss, state_margin_loss  # noqa: E402
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
FORBIDDEN_DESIGNS = ("dssa", "shared_specific", "shared-specific", "private representation", "modality-invariant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="analysis_reports/phase_c16_mea/c16_mea_synthetic_gate.json")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def tiny_config(use_mea: bool | None) -> Dict[str, Any]:
    model: Dict[str, Any] = {
        "variant": "dmea",
        "hidden_dim": 32,
        "dropout": 0.0,
        "text_vocab_size": 128,
        "text_max_length": 32,
        "bio_dim": 7,
        "max_images_per_patient": 3,
        "image_size": 16,
    }
    if use_mea is not None:
        model["use_mea"] = use_mea
    return {"model": model}


def make_batch() -> Dict[str, torch.Tensor]:
    torch.manual_seed(101)
    texts = (
        "[C13_LATEST_THYROID]弥漫性改变[C13_HISTORY_THYROID]回声均匀[C13_FULL_REPORT]桥本",
        "[C13_LATEST_THYROID]甲状腺未见明显异常[C13_FULL_REPORT]结节",
        "[C13_FULL_REPORT]考虑回声不均",
        "普通甲状腺超声报告",
    )
    masks = [build_text_evidence_masks(text, 32) for text in texts]
    batch: Dict[str, torch.Tensor] = {
        "images": torch.randn(4, 3, 3, 16, 16),
        "image_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32),
        "report_input_ids": torch.randint(2, 128, (4, 32)),
        "report_attention_mask": torch.tensor(
            [[1] * 32, [1] * 30 + [0] * 2, [1] * 20 + [0] * 12, [1] * 10 + [0] * 22], dtype=torch.long
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


def normalize_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    out["project"]["name"] = "normalized"
    out["project"]["output_dir"] = "normalized"
    out["loss"]["pairwise_ranking_weight"] = 0.0
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    configs = sorted((repo_root / "configs").glob("*.yaml"))
    parsed = {path.name: load_config(path) for path in configs}
    check("all_previous_and_c16_configs_parse", len(parsed) == len(configs), len(parsed))
    c16_names = {
        "dmea_ht_v2_c16_mea_core_smoke.yaml",
        "dmea_ht_v2_c16_mea_core_seed0.yaml",
        "dmea_ht_v2_c16_mea_rank_smoke.yaml",
        "dmea_ht_v2_c16_mea_rank_seed0.yaml",
        "dmea_ht_v2_c16_mea_formal_multiseed.yaml",
    }
    check("all_five_c16_configs_present", c16_names.issubset(parsed), sorted(c16_names & parsed.keys()))
    core = parsed["dmea_ht_v2_c16_mea_core_seed0.yaml"]
    rank = parsed["dmea_ht_v2_c16_mea_rank_seed0.yaml"]
    check("core_and_rank_only_differ_by_route_identity_and_rank_weight", normalize_variant(core) == normalize_variant(rank), "normalized configs")
    check("fixed_c16_loss_targets", [core["loss"][key] for key in ("state_margin_weight", "mechanism_alignment_weight", "role_separation_weight", "pairwise_ranking_weight")] == [0.03, 0.02, 0.005, 0.0] and rank["loss"]["pairwise_ranking_weight"] == 0.02, "fixed targets")
    check("formal_seed_contract", parsed["dmea_ht_v2_c16_mea_formal_multiseed.yaml"]["training"]["seeds"] == [0, 42, 3407], "formal seeds")

    torch.manual_seed(17)
    legacy_absent = DMEAHTModel(tiny_config(None)).eval()
    torch.manual_seed(17)
    legacy_false = DMEAHTModel(tiny_config(False)).eval()
    absent_state = legacy_absent.state_dict()
    false_state = legacy_false.state_dict()
    keys_equal = list(absent_state) == list(false_state)
    values_equal = keys_equal and all(torch.equal(absent_state[key], false_state[key]) for key in absent_state)
    batch = make_batch()
    with torch.no_grad():
        absent_logits = legacy_absent(batch)["logit"]
        false_logits = legacy_false(batch)["logit"]
    max_legacy_delta = float((absent_logits - false_logits).abs().max())
    check("legacy_absent_false_state_dict_identical", keys_equal and values_equal, len(absent_state))
    check("legacy_absent_false_logits_identical", max_legacy_delta == 0.0, max_legacy_delta)

    model = DMEAHTModel(tiny_config(True))
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    check("mea_output_shape", tuple(outputs["logit"].shape) == (4,), tuple(outputs["logit"].shape))
    check("evidence_role_shape", tuple(outputs["mea_role_probs"].shape) == (4, 14, 3), tuple(outputs["mea_role_probs"].shape))
    float_outputs = [value for value in outputs.values() if torch.is_tensor(value) and value.is_floating_point()]
    check("all_mea_outputs_finite", all(bool(torch.isfinite(value).all()) for value in float_outputs), len(float_outputs))
    prob_error = float((outputs["mea_role_probs"].sum(dim=-1) - 1.0).abs().max())
    check("role_probabilities_sum_to_one", prob_error <= 1e-6, prob_error)
    check("conflict_score_finite", bool(torch.isfinite(outputs["patient_conflict_score"]).all()), outputs["patient_conflict_score"].tolist())

    text_projector = TextEvidenceRoleProjector(32, 0.0).eval()
    empty_masks = {key: torch.zeros(2, 12) for key in TEXT_MASK_KEYS}
    with torch.no_grad():
        text_result = text_projector(torch.randn(2, 12, 32), torch.ones(2, 12), empty_masks)
    check("empty_text_masks_use_fallback", bool(text_result["valid"].all()) and not bool(text_result["guidance_present"][:, :5].any()) and float(text_result["nodes"].abs().sum()) > 0.0, text_result["valid"].tolist())

    bio_projector = BioEvidenceProjector(32, 0.0).eval()
    missing_bio = torch.zeros(2, 7)
    missing_bio[0, [2, 5]] = 1.0
    with torch.no_grad():
        bio_result = bio_projector(torch.randn(2, 7, 32), missing_bio)
    check("unavailable_bio_group_skipped", not bool(bio_result["valid"][0, 1]) and float(bio_result["nodes"][0, 1].abs().sum()) == 0.0, bio_result["valid"].tolist())

    empty_batch = {key: value.clone() for key, value in batch.items()}
    empty_batch["image_mask"].zero_()
    empty_batch["report_attention_mask"].zero_()
    empty_batch["bio_missing_mask"].fill_(1.0)
    for key in TEXT_MASK_KEYS:
        empty_batch[key].zero_()
    with torch.no_grad():
        empty_outputs = model(empty_batch)
    check("mechanism_graph_handles_all_missing_nodes", bool(torch.isfinite(empty_outputs["logit"]).all()), empty_outputs["logit"].tolist())

    mixed_rank = pairwise_ranking_loss(torch.tensor([1.0, -1.0, 0.5], requires_grad=True), torch.tensor([1.0, 0.0, 1.0]))
    one_class_logits = torch.tensor([0.2, 0.3], requires_grad=True)
    one_class_rank = pairwise_ranking_loss(one_class_logits, torch.ones(2))
    one_class_rank.backward()
    check("mixed_pairwise_rank_finite", bool(torch.isfinite(mixed_rank)), float(mixed_rank.detach()))
    check("single_class_rank_graph_zero", float(one_class_rank) == 0.0 and one_class_logits.grad is not None, one_class_logits.grad.tolist())

    model.train()
    train_outputs = model(batch)
    total = (
        F.binary_cross_entropy_with_logits(train_outputs["logit"], batch["label"])
        + 0.03 * state_margin_loss(train_outputs["state_margin"], batch["label"])
        + 0.02 * train_outputs["mea_mechanism_alignment_loss"]
        + 0.005 * train_outputs["mea_role_separation_loss"]
        + 0.02 * pairwise_ranking_loss(train_outputs["logit"], batch["label"])
    )
    total.backward()
    gradient_prefixes = {
        "image_projector": "mechanism_evidence_alignment.image.",
        "text_projector": "mechanism_evidence_alignment.text.",
        "bio_projector": "mechanism_evidence_alignment.bio.",
        "role_scorer": "mechanism_evidence_alignment.role_scorer.",
        "mechanism_layer": "mechanism_evidence_alignment.mechanisms.",
        "conflict_aggregator": "mechanism_evidence_alignment.aggregator.",
        "disease_state_head": "mechanism_evidence_alignment.head.",
    }
    gradient_sums: Dict[str, float] = {}
    for name, prefix in gradient_prefixes.items():
        gradient_sums[name] = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter_name, parameter in model.named_parameters()
            if parameter_name.startswith(prefix) and parameter.grad is not None
        )
    check("gradients_reach_all_new_modules", all(value > 0.0 for value in gradient_sums.values()), gradient_sums)

    warmup = core["loss"]
    scales = {epoch: mea_loss_weights_for_epoch(warmup, epoch)["mea_auxiliary_scale"] for epoch in (1, 3, 4, 8, 9)}
    check("warmup_and_ramp_contract", scales == {1: 0.0, 3: 0.0, 4: 0.2, 8: 1.0, 9: 1.0}, scales)

    model_source = inspect.getsource(sys.modules["dmea_ht.models"]) + inspect.getsource(sys.modules["dmea_ht.mechanism_evidence_alignment"])
    lowered_source = model_source.lower()
    check("shortcut_fields_absent_from_model_source", not any(field in model_source for field in FORBIDDEN_SHORTCUTS), FORBIDDEN_SHORTCUTS)
    check("dssa_and_shared_specific_absent", not any(term in lowered_source for term in FORBIDDEN_DESIGNS), FORBIDDEN_DESIGNS)

    result = {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "passed": sum(int(item["pass"]) for item in checks),
        "total": len(checks),
        "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
