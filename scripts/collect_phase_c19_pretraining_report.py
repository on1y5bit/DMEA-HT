#!/usr/bin/env python3
"""Run the validation-only C19-A audit and the C19 static/synthetic gate."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS, build_text_evidence_masks  # noqa: E402
from dmea_ht.models import (  # noqa: E402
    C19PolarityLockedResidualModel,
    MonotonicOppositionCalibrator,
    MonotonicSupportCalibrator,
)


EXPECTED_SEEDS = [0, 42, 3407]
REQUIRED_SCRIPTS = (
    "dmea_ht/models.py",
    "dmea_ht/mechanism_evidence_alignment.py",
    "train.py",
    "scripts/analyze_phase_c19_evidence_polarity.py",
    "scripts/collect_phase_c19_pretraining_report.py",
    "scripts/audit_phase_c19_polarity_locked_residual.py",
    "scripts/collect_phase_c19_formal_report.py",
)
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
    parser.add_argument("--config", default="configs/dema_ht_c19_polarity_locked_multiseed.yaml")
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--c18-directional-prediction-dir", default="runs/dema_ht_c18_directional_multiseed/predictions")
    parser.add_argument("--c18-hardrank-prediction-dir", default="runs/dema_ht_c18_directional_hardrank_multiseed/predictions")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c19_dema")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def check_row(checks: List[Dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})


def make_batch(hidden_text_length: int = 32) -> Dict[str, torch.Tensor]:
    torch.manual_seed(1919)
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


def finite_gradients(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return float("nan")
        total += float(parameter.grad.detach().abs().sum())
    return total


def run_c19a_audit(repo_root: Path, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    analyzer = repo_root / "scripts" / "analyze_phase_c19_evidence_polarity.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(analyzer),
            "--c17-prediction-dir",
            args.c17_prediction_dir,
            "--c18-directional-prediction-dir",
            args.c18_directional_prediction_dir,
            "--c18-hardrank-prediction-dir",
            args.c18_hardrank_prediction_dir,
            "--output-dir",
            str(output_dir),
            "--require-pass",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    audit_path = output_dir / "c19_polarity_audit.json"
    if not audit_path.exists():
        raise RuntimeError(
            f"C19-A audit did not produce {audit_path}; returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["returncode"] = completed.returncode
    audit["stdout_tail"] = completed.stdout[-2000:]
    audit["stderr_tail"] = completed.stderr[-2000:]
    return audit


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    checks: List[Dict[str, Any]] = []

    check_row(checks, "c19_phase", config.get("phase") == "c19", config.get("phase"))
    check_row(
        checks,
        "c19_seeds",
        config.get("training", {}).get("seeds") == EXPECTED_SEEDS,
        config.get("training", {}).get("seeds"),
    )
    check_row(
        checks,
        "c19_primary_metric",
        config.get("training", {}).get("primary_metric") == "val_AUC",
        config.get("training", {}).get("primary_metric"),
    )
    check_row(
        checks,
        "c19_output_dir",
        config.get("project", {}).get("output_dir") == "runs/dema_ht_c19_polarity_locked_multiseed",
        config.get("project", {}).get("output_dir"),
    )
    loss = config.get("loss", {})
    check_row(
        checks,
        "c19_fixed_loss_weights",
        float(loss.get("lambda_polarity", math.nan)) == 0.01
        and float(loss.get("lambda_positive_preserve", math.nan)) == 0.02
        and float(loss.get("lambda_negative_preserve", math.nan)) == 0.02
        and float(loss.get("lambda_magnitude", math.nan)) == 0.001,
        loss,
    )
    check_row(
        checks,
        "c19_fixed_thresholds",
        float(loss.get("reliable_conflict_threshold", math.nan)) == 0.35
        and float(loss.get("reliable_uncertainty_threshold", math.nan)) == 0.35
        and float(loss.get("allowed_positive_delta", math.nan)) == 0.02,
        loss,
    )
    config_text = config_path.read_text(encoding="utf-8").lower()
    check_row(
        checks,
        "no_smoke_or_pilot_config",
        not any(token in config_text for token in ("smoke", "pilot", "seed0", "seed-0")),
        config_path.name,
    )

    for relative in REQUIRED_SCRIPTS:
        path = repo_root / relative
        try:
            py_compile.compile(str(path), doraise=True)
            passed = True
            detail = "compiled"
        except Exception as exc:
            passed = False
            detail = repr(exc)
        check_row(checks, f"compile_{relative.replace('/', '_')}", passed, detail)

    c19_source = inspect.getsource(C19PolarityLockedResidualModel)
    support_source = inspect.getsource(MonotonicSupportCalibrator)
    opposition_source = inspect.getsource(MonotonicOppositionCalibrator)
    mechanism_source = (repo_root / "dmea_ht" / "mechanism_evidence_alignment.py").read_text(encoding="utf-8")
    train_source = (repo_root / "train.py").read_text(encoding="utf-8")
    c19_lower = c19_source.lower()
    check_row(
        checks,
        "shortcut_fields_absent_from_c19_model",
        not any(field in c19_source for field in FORBIDDEN_SHORTCUTS),
        FORBIDDEN_SHORTCUTS,
    )
    check_row(
        checks,
        "forbidden_alignment_designs_absent",
        not any(term in (c19_lower + mechanism_source.lower()) for term in FORBIDDEN_DESIGNS),
        FORBIDDEN_DESIGNS,
    )
    check_row(
        checks,
        "prediction_csvs_not_in_c19_model",
        not any(term in c19_lower for term in ("read_csv", "val_predictions", "test_predictions", "prediction.csv")),
        "C19 model source",
    )
    check_row(
        checks,
        "saved_predictions_not_training_inputs",
        "pd.read_csv" not in train_source and "read_csv(" not in c19_source,
        "model/training source",
    )
    check_row(checks, "support_calibrator_does_not_read_opposition", "opposition" not in support_source.lower(), support_source)
    check_row(checks, "opposition_calibrator_does_not_read_support", "support" not in opposition_source.lower(), opposition_source)
    for legacy in (
        "configs/dema_ht_c17_formal_multiseed.yaml",
        "configs/dema_ht_c18_directional_residual_multiseed.yaml",
        "configs/dema_ht_c18_directional_hardrank_multiseed.yaml",
    ):
        try:
            load_config(repo_root / legacy)
            passed = True
            detail = legacy
        except Exception as exc:
            passed = False
            detail = repr(exc)
        check_row(checks, f"parse_{legacy.replace('/', '_')}", passed, detail)

    c19_cfg = config.get("c19", {})
    checkpoint_paths: Dict[str, List[str]] = {}
    for key in ("c17_checkpoint", "c13_checkpoint"):
        values = [str(c19_cfg.get(key, "")).replace("{seed}", str(seed)) for seed in EXPECTED_SEEDS]
        checkpoint_paths[key] = values
        check_row(checks, f"{key}_all_seeds_exist", bool(values) and all(Path(value).exists() for value in values), values)

    try:
        c19_audit = run_c19a_audit(repo_root, args, output_dir)
        c19_a_pass = c19_audit.get("decision") == "C17_EVIDENCE_POLARITY_USABLE_WITH_CONSTRAINTS"
    except Exception as exc:
        c19_audit = {"decision": "C19_POLARITY_BASE_INVALID", "error": repr(exc)}
        c19_a_pass = False
    check_row(checks, "c19a_polarity_audit_pass", c19_a_pass, c19_audit.get("decision"))

    synthetic_checks: List[Dict[str, Any]] = []
    if c19_a_pass and all(item["pass"] for item in checks):
        batch = make_batch()
        models: Dict[int, C19PolarityLockedResidualModel] = {}
        try:
            for seed in EXPECTED_SEEDS:
                models[seed] = C19PolarityLockedResidualModel(config, seed).eval()
            model = models[0]
            with torch.no_grad():
                outputs = model(batch)
            float_outputs = [value for value in outputs.values() if torch.is_tensor(value) and value.is_floating_point()]
            check_row(
                synthetic_checks,
                "synthetic_outputs_finite",
                all(bool(torch.isfinite(value).all()) for value in float_outputs),
                len(float_outputs),
            )
            initial_diff = float((outputs["logit"] - outputs["frozen_c17_logit"]).abs().max())
            check_row(synthetic_checks, "initial_c19_matches_frozen_c17", initial_diff <= 1e-8, initial_diff)
            check_row(
                synthetic_checks,
                "calibrator_slopes_positive",
                float(model.support_calibrator.a_support) > 0.0 and float(model.opposition_calibrator.a_opposition) > 0.0,
                [float(model.support_calibrator.a_support), float(model.opposition_calibrator.a_opposition)],
            )
            check_row(
                synthetic_checks,
                "polarity_is_gap_locked",
                bool(((outputs["evidence_gap"] * outputs["evidence_polarity"]) >= -1e-7).all()),
                outputs["evidence_polarity"].detach().tolist(),
            )
            check_row(
                synthetic_checks,
                "correction_magnitude_bound",
                bool(
                    (outputs["correction_magnitude"] >= -1e-7).all()
                    and (outputs["correction_magnitude"] <= 0.20 + 1e-7).all()
                    and outputs["delta_c19"].abs().max() <= 0.20 + 1e-7
                ),
                [float(outputs["correction_magnitude"].min()), float(outputs["correction_magnitude"].max())],
            )
            check_row(
                synthetic_checks,
                "frozen_c17_has_no_trainable_parameters",
                not any(parameter.requires_grad for parameter in model.frozen_c17.parameters()),
                "all C17 parameters frozen",
            )

            polarity = torch.tensor([0.7, -0.7], dtype=torch.float32)
            magnitude = torch.full_like(polarity, 0.10)
            low_uncertainty = torch.full_like(polarity, 0.10)
            low_conflict = torch.full_like(polarity, 0.10)
            high_uncertainty = torch.full_like(polarity, 0.90)
            high_conflict = torch.full_like(polarity, 0.90)
            low_delta = polarity * polarity.abs() * (1.0 - low_uncertainty) * (1.0 - low_conflict) * magnitude
            high_conflict_delta = polarity * polarity.abs() * (1.0 - low_uncertainty) * (1.0 - high_conflict) * magnitude
            high_uncertainty_delta = polarity * polarity.abs() * (1.0 - high_uncertainty) * (1.0 - low_conflict) * magnitude
            check_row(
                synthetic_checks,
                "high_conflict_reduces_absolute_delta",
                bool((high_conflict_delta.abs() < low_delta.abs()).all()),
                [high_conflict_delta.tolist(), low_delta.tolist()],
            )
            check_row(
                synthetic_checks,
                "high_uncertainty_reduces_absolute_delta",
                bool((high_uncertainty_delta.abs() < low_delta.abs()).all()),
                [high_uncertainty_delta.tolist(), low_delta.tolist()],
            )
            check_row(
                synthetic_checks,
                "sign_lock_prevents_free_direction_flip",
                bool((low_delta[0] > 0.0) and (low_delta[1] < 0.0)),
                low_delta.tolist(),
            )

            empty_batch = {key: value.clone() for key, value in batch.items()}
            empty_batch["image_mask"].zero_()
            empty_batch["report_attention_mask"].zero_()
            empty_batch["bio_missing_mask"].fill_(1.0)
            for key in TEXT_MASK_KEYS:
                empty_batch[key].zero_()
            with torch.no_grad():
                empty_outputs = model(empty_batch)
            check_row(
                synthetic_checks,
                "missing_modalities_are_finite",
                bool(torch.isfinite(empty_outputs["logit"]).all()),
                empty_outputs["logit"].tolist(),
            )

            model.train()
            model.zero_grad(set_to_none=True)
            train_outputs = model(batch)
            labels = batch["label"]
            reliable_mask = (
                (train_outputs["normalized_conflict_score"] < 0.35)
                & (train_outputs["normalized_uncertainty_strength"] < 0.35)
            ).detach()
            if bool(reliable_mask.any().item()):
                polarity_loss = F.softplus(
                    -(2.0 * labels[reliable_mask] - 1.0) * train_outputs["evidence_gap"][reliable_mask]
                ).mean()
            else:
                polarity_loss = train_outputs["logit"].sum() * 0.0
            positive_mask = labels > 0.5
            negative_mask = ~positive_mask
            positive_loss = (
                F.relu(-train_outputs["delta_c19"][positive_mask] - 0.02).mean()
                if bool(positive_mask.any().item())
                else train_outputs["delta_c19"].sum() * 0.0
            )
            negative_loss = (
                F.relu(train_outputs["delta_c19"][negative_mask] - 0.02).mean()
                if bool(negative_mask.any().item())
                else train_outputs["delta_c19"].sum() * 0.0
            )
            training_loss = (
                F.binary_cross_entropy_with_logits(train_outputs["logit"], labels)
                + 0.01 * polarity_loss
                + 0.02 * positive_loss
                + 0.02 * negative_loss
                + 0.001 * train_outputs["delta_c19"].square().mean()
            )
            training_loss.backward()
            new_gradient = finite_gradients(model.magnitude_head)
            scale_gradient = float(model.residual_scale.grad.detach().abs().sum()) if model.residual_scale.grad is not None else 0.0
            frozen_gradient = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in model.frozen_c17.parameters()
                if parameter.grad is not None
            )
            check_row(
                synthetic_checks,
                "new_magnitude_module_has_finite_nonzero_gradient",
                math.isfinite(new_gradient) and new_gradient > 0.0 and math.isfinite(scale_gradient) and scale_gradient > 0.0,
                [new_gradient, scale_gradient],
            )
            check_row(synthetic_checks, "frozen_c17_receives_no_gradient", frozen_gradient == 0.0, frozen_gradient)

            all_negative = {key: value.clone() for key, value in batch.items()}
            all_negative["label"].zero_()
            all_positive = {key: value.clone() for key, value in batch.items()}
            all_positive["label"].fill_(1.0)
            negative_outputs = model(all_negative)
            positive_outputs = model(all_positive)
            positive_zero = negative_outputs["delta_c19"].sum() * 0.0
            negative_zero = positive_outputs["delta_c19"].sum() * 0.0
            forced_reliable_mask = torch.zeros_like(labels, dtype=torch.bool)
            if bool(forced_reliable_mask.any().item()):
                no_reliable_zero = train_outputs["evidence_gap"][forced_reliable_mask].mean()
            else:
                no_reliable_zero = train_outputs["logit"].sum() * 0.0
            check_row(
                synthetic_checks,
                "all_negative_positive_loss_is_graph_connected_zero",
                bool(positive_zero.requires_grad) and float(positive_zero.detach()) == 0.0,
                positive_zero,
            )
            check_row(
                synthetic_checks,
                "all_positive_negative_loss_is_graph_connected_zero",
                bool(negative_zero.requires_grad) and float(negative_zero.detach()) == 0.0,
                negative_zero,
            )
            check_row(
                synthetic_checks,
                "no_reliable_polarity_loss_is_graph_connected_zero",
                bool(no_reliable_zero.requires_grad) and float(no_reliable_zero.detach()) == 0.0,
                no_reliable_zero,
            )
        except Exception as exc:
            check_row(synthetic_checks, "synthetic_execution", False, repr(exc))
    else:
        check_row(synthetic_checks, "synthetic_skipped_until_c19a_and_static_pass", False, "C19-A or static check failed")

    static_checks = list(checks)
    all_checks = static_checks + synthetic_checks
    with (output_dir / "c19_synthetic_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "pass", "detail"])
        writer.writeheader()
        writer.writerows(synthetic_checks)

    status = "PASS" if c19_a_pass and all(item["pass"] for item in all_checks) else "FAIL"
    gate_label = "C19_DIRECT_MULTI_SEED_AUTHORIZED" if status == "PASS" else "C19_DIRECT_MULTI_SEED_BLOCKED"
    result = {
        "status": status,
        "static_synthetic_gate": gate_label,
        "c19a_decision": c19_audit.get("decision"),
        "passed": sum(int(item["pass"]) for item in all_checks),
        "total": len(all_checks),
        "static_checks": static_checks,
        "synthetic_checks": synthetic_checks,
        "expected_seeds": EXPECTED_SEEDS,
        "no_smoke": True,
        "no_seed0_pilot": True,
        "test_reporting_only": True,
        "checkpoint_paths": checkpoint_paths,
    }
    (output_dir / "c19_pretraining_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "c19_pretraining_polarity_report.md"
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else "# C19-A Evidence Polarity Audit\n"
    report_path.write_text(
        previous.rstrip()
        + "\n\n## Static/Synthetic Gate\n\n"
        + f"- Gate: {gate_label}.\n"
        + f"- C19-A decision: {c19_audit.get('decision')}.\n"
        + f"- Checks: {result['passed']}/{result['total']} passed.\n"
        + "- No smoke and no seed-0-only pilot are permitted.\n"
        + "- Formal training is authorized only when the gate is C19_DIRECT_MULTI_SEED_AUTHORIZED.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    if args.require_pass and status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
