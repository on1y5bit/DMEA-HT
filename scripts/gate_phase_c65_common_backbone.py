#!/usr/bin/env python3
"""Gate C65-B common-backbone scope and real-batch update connectivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c63_common as c63  # noqa: E402
from scripts import c64_common as c64  # noqa: E402
from scripts import c65a_common as c65a  # noqa: E402
from scripts import c65b_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65b.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = common.load_c65b_config(args.config)
    report_dir = common.report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = common.development_rows(config)
    assignments = common.fold_assignments(config)
    expected_ids = {str(row["patient_id"]) for row in rows}
    if set(assignments) != expected_ids or len(assignments) != c65a.DEVELOPMENT_COUNT:
        raise RuntimeError("C65-B fold assignments do not cover exactly the development pool")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows = []
    passed = True
    for seed in common.SEEDS:
        model, payload, checkpoint = common.build_common_backbone_model(config, seed, device)
        source_params = [parameter for name, parameter in model.named_parameters() if name.startswith("sources.")]
        head_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("sources.")]
        source_frozen = all(not parameter.requires_grad for parameter in source_params)
        head_trainable = all(parameter.requires_grad for parameter in head_params)
        optimizer, optimizer_audit = c64.optimizer_parameter_groups(model, config, common.CANDIDATE)
        optimizer_groups_pass = set(optimizer_audit.loc[optimizer_audit["included_in_optimizer"].astype(bool), "group"]) == {"c61_task_path"}
        fold_rows = c64.fold_rows(rows, assignments, 0)
        loader = c63.build_loaders(config, fold_rows, seed, ("train",))["train"]
        batch = c63.move_batch(next(iter(loader)), device)
        model.train(True)
        model.sources.eval()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        finite_loss = bool(torch.isfinite(loss).item())
        if finite_loss:
            loss.backward()
        head_grad_norm = 0.0
        source_grad_none = True
        for name, parameter in model.named_parameters():
            if name.startswith("sources."):
                source_grad_none &= parameter.grad is None
            elif parameter.grad is not None:
                head_grad_norm += float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
        before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        if finite_loss:
            optimizer.step()
        update_norm = 0.0
        for name, parameter in model.named_parameters():
            if name in before:
                update_norm += float(torch.linalg.vector_norm(parameter.detach().cpu() - before[name]))
        row = {
            "seed": seed,
            "backbone_seed": 42,
            "common_checkpoint_seed": int(payload.get("seed", -1)),
            "checkpoint": str(checkpoint),
            "source_parameter_count": int(sum(parameter.numel() for parameter in source_params)),
            "head_parameter_count": int(sum(parameter.numel() for parameter in head_params)),
            "source_frozen_pass": source_frozen,
            "head_trainable_pass": head_trainable,
            "optimizer_groups_pass": optimizer_groups_pass,
            "finite_real_batch_loss_pass": finite_loss,
            "source_gradient_none_pass": source_grad_none,
            "head_gradient_norm": head_grad_norm,
            "head_gradient_pass": bool(head_grad_norm > 0.0),
            "head_update_norm": update_norm,
            "head_update_pass": bool(update_norm > 0.0),
            "test_loaded": False,
        }
        seed_rows.append(row)
        passed &= all(bool(row[key]) for key in ("source_frozen_pass", "head_trainable_pass", "optimizer_groups_pass", "finite_real_batch_loss_pass", "source_gradient_none_pass", "head_gradient_pass", "head_update_pass"))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decision = {
        "phase": "C65-VACS",
        "stage": "common_backbone_gate",
        "status": "C65B_COMMON_BACKBONE_CV_AUTHORIZED" if passed else "C65B_COMMON_BACKBONE_GATE_FAIL",
        "seed_rows": seed_rows,
        "formal_seed_count": len(seed_rows),
        "fold_reused": True,
        "development_patient_count": c65a.DEVELOPMENT_COUNT,
        "common_backbone_seed": 42,
        "test_loaded": False,
        "ensemble": False,
        "prediction_averaging": False,
    }
    common.write_json(report_dir / "c65b_gate.json", decision)
    print(json.dumps({"status": decision["status"], "formal_seed_count": len(seed_rows), "test_loaded": False}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
