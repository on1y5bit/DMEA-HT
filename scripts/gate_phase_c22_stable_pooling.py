#!/usr/bin/env python3
"""Run the C22 static and synthetic contract gate without loading project data."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c22_stable_pooling import (  # noqa: E402
    C22StableEvidencePoolingModel,
    StableEvidencePoolingResidualHead,
    c22_loss_terms,
    stable_evidence_pool,
)
from dmea_ht.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c22_stable_evidence_pooling_multiseed.yaml")
    parser.add_argument("--output", default="analysis_reports/phase_c22_dema/c22_static_synthetic_gate.json")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def check(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    checks: List[Dict[str, Any]] = []
    checks.append(check("phase_is_c22", str(config.get("phase", "")).lower() == "c22"))
    checks.append(check("formal_seeds_are_0_42_3407", config.get("training", {}).get("seeds") == [0, 42, 3407]))
    checks.append(check("primary_metric_is_validation_auc", config.get("training", {}).get("primary_metric") == "val_AUC"))
    checks.append(
        check(
            "fixed_loss_contract",
            config.get("loss", {}).get("lambda_residual") == 0.001
            and config.get("loss", {}).get("lambda_positive_preserve") == 0.02
            and config.get("loss", {}).get("allowed_negative_delta") == 0.05,
        )
    )

    forward_source = inspect.getsource(C22StableEvidencePoolingModel.forward)
    forbidden = (r"role_scorer", r"\.mechanisms\s*\(", r"\.aggregator\s*\(", r"\.head\s*\(")
    violations = [pattern for pattern in forbidden if re.search(pattern, forward_source)]
    checks.append(check("forward_bypasses_propagation_and_aggregation", not violations, violations))
    model_source = (REPO_ROOT / "dmea_ht" / "c22_stable_pooling.py").read_text(encoding="utf-8")
    shortcut_names = (
        "selected_n_visits",
        "used_images",
        "image_padding_count",
        "has_bio",
        "bio_missing_count",
        "report_length",
        "raw_n_visits",
        "raw_n_images",
    )
    shortcut_hits = [name for name in shortcut_names if name in model_source]
    checks.append(check("shortcut_fields_absent_from_model", not shortcut_hits, shortcut_hits))

    torch.manual_seed(20260714)
    hidden_dim = 8
    head = StableEvidencePoolingResidualHead(hidden_dim, dropout=0.0, delta_max=0.50)
    stable = torch.randn(3, hidden_dim)
    head_output = head(stable)
    max_initial_delta = float(head_output["delta_c22"].abs().max())
    checks.append(check("zero_initialized_residual", max_initial_delta <= 1e-8, max_initial_delta))
    checks.append(check("residual_bound", bool((head_output["delta_c22"].abs() <= 0.50 + 1e-7).all()), max_initial_delta))

    nodes = torch.randn(2, 14, hidden_dim)
    valid = torch.tensor(
        [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]],
        dtype=torch.bool,
    )
    pooled_a, count_a = stable_evidence_pool(nodes, valid)
    permutation = torch.tensor([13, 0, 7, 3, 11, 1, 9, 5, 2, 12, 4, 10, 6, 8])
    pooled_b, count_b = stable_evidence_pool(nodes[:, permutation], valid[:, permutation])
    altered = nodes + (~valid).unsqueeze(-1).to(nodes.dtype) * 10000.0
    pooled_c, count_c = stable_evidence_pool(altered, valid)
    checks.append(check("valid_mask_pool_permutation_invariant", bool(torch.allclose(pooled_a, pooled_b)) and bool(torch.equal(count_a, count_b))))
    checks.append(check("invalid_nodes_excluded", bool(torch.allclose(pooled_a, pooled_c)) and bool(torch.equal(count_a, count_c))))

    delta = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    logits = torch.tensor([0.0, 0.2, -0.1], requires_grad=True)
    all_negative_batch = {
        "label": torch.zeros(3),
        "sample_weight": torch.ones(3),
    }
    terms = c22_loss_terms(
        {"logit": logits, "delta_c22": delta},
        all_negative_batch,
        config.get("loss", {}),
    )
    terms["total"].backward()
    checks.append(
        check(
            "all_negative_positive_term_is_graph_connected_zero",
            float(terms["positive_preserve"].detach()) == 0.0 and terms["positive_preserve"].requires_grad and delta.grad is not None,
        )
    )
    checks.append(
        check(
            "loss_components_are_finite",
            all(bool(torch.isfinite(value).all()) for value in terms.values()),
        )
    )

    passed = all(item["pass"] for item in checks)
    result = {
        "phase": "C22",
        "gate": "STATIC_SYNTHETIC",
        "pass": passed,
        "training_authorized_after_server_gate": passed,
        "checks": checks,
        "contract": {
            "frozen_reference": "C13 base plus C17 evidence projector state",
            "trainable_scope": "residual_head_only",
            "pooling": "valid_mask_mean over 14 real pre-propagation projector nodes",
            "selection": "validation AUC only",
            "test_role": "reporting only",
        },
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
