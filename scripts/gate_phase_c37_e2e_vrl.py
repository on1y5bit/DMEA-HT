#!/usr/bin/env python3
"""Authorize direct C37-E2E-VRL formal execution with exactly 16 checks."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, trainable_parameter_count  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import build_loaders, move_batch, resolve_path, set_seed  # noqa: E402
from scripts.train_phase_c37 import (  # noqa: E402
    MODULE_CATEGORIES,
    SEEDS,
    build_optimizer,
    parameter_category,
)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c37_e2e_vrl_multiseed.yaml")
    parser.add_argument("--expected-project", default="/home/linruixin/chen/project/DMEA-HT")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    # BatchNorm running statistics may update during a train-mode probe; the gate
    # checks that gradient evaluation does not mutate parameter tensors.
    for name, tensor in sorted(model.named_parameters()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_contract(rows: List[Dict[str, Any]]) -> bool:
    expected = {
        "train": (602, 301, 301),
        "val": (94, 47, 47),
        "test": (84, 42, 42),
    }
    ids_by_split: Dict[str, set[str]] = {}
    for split, (count, positives, negatives) in expected.items():
        selected = [row for row in rows if str(row.get("split")) == split]
        labels = np.asarray([int(row["label"]) for row in selected], dtype=int)
        ids = {str(row["patient_id"]) for row in selected}
        if (
            len(selected) != count
            or len(ids) != count
            or int((labels == 1).sum()) != positives
            or int((labels == 0).sum()) != negatives
        ):
            return False
        ids_by_split[split] = ids
    return not (
        ids_by_split["train"] & ids_by_split["val"]
        or ids_by_split["train"] & ids_by_split["test"]
        or ids_by_split["val"] & ids_by_split["test"]
    )


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def logit_column(frame: pd.DataFrame) -> str:
    for name in ("final_logit", "logit"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No logit column in {list(frame.columns)}")


def clone_tensor_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def missing_modality_batches(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_missing = clone_tensor_batch(batch)
    image_missing["image_mask"].zero_()

    text_missing = clone_tensor_batch(batch)
    text_missing["report_input_ids"].zero_()
    text_missing["report_attention_mask"].zero_()
    text_missing["visit_text_valid"].fill_(False)
    for key in TEXT_MASK_KEYS:
        text_missing[key].zero_()

    bio_missing = clone_tensor_batch(batch)
    bio_missing["bio_values"].zero_()
    bio_missing["bio_missing_mask"].fill_(1.0)
    bio_missing["bio_abnormal_flags"].zero_()
    return [image_missing, text_missing, bio_missing]


def initial_reproduction(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device
) -> tuple[pd.DataFrame, bool]:
    loader = build_loaders(config, rows, ("val",))["val"]
    report_rows: List[Dict[str, Any]] = []
    all_pass = True
    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(
            str(config["c27"]["c27_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = checkpoint_payload(checkpoint_path)
        model = C27VTMEModel(config, seed).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        ids: List[str] = []
        labels: List[int] = []
        logits: List[float] = []
        probabilities: List[float] = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(batch)
                for index, patient_id in enumerate(batch["patient_id"]):
                    ids.append(str(patient_id))
                    labels.append(int(batch["label"][index].cpu()))
                    logits.append(float(output["logit"][index].cpu()))
                    probabilities.append(float(output["prob"][index].cpu()))
        reference = pd.read_csv(
            Path(config["c27"]["c27_run_dir"])
            / "predictions"
            / f"val_predictions_seed_{seed}.csv",
            dtype={"patient_id": str},
        )
        reference["patient_id"] = reference["patient_id"].astype(str)
        order = np.argsort(np.asarray(ids, dtype=str))
        model_ids = np.asarray(ids, dtype=str)[order]
        model_labels = np.asarray(labels, dtype=int)[order]
        model_logits = np.asarray(logits, dtype=float)[order]
        model_probabilities = np.asarray(probabilities, dtype=float)[order]
        reference = reference.sort_values("patient_id").reset_index(drop=True)
        reference_ids = reference["patient_id"].to_numpy(dtype=str)
        reference_labels = reference["label"].to_numpy(dtype=int)
        reference_logits = reference[logit_column(reference)].to_numpy(dtype=float)
        reference_probabilities = reference[probability_column(reference)].to_numpy(dtype=float)
        max_logit_error = float(np.max(np.abs(model_logits - reference_logits)))
        max_prob_error = float(np.max(np.abs(model_probabilities - reference_probabilities)))
        model_auc = float(roc_auc_score(model_labels, model_probabilities))
        reference_auc = float(roc_auc_score(reference_labels, reference_probabilities))
        class_exact = np.array_equal(model_probabilities >= 0.5, reference_probabilities >= 0.5)
        row_pass = bool(
            np.array_equal(model_ids, reference_ids)
            and np.array_equal(model_labels, reference_labels)
            and max_logit_error <= float(config["c37"]["initial_reproduction_logit_tolerance"])
            and max_prob_error <= float(config["c37"]["initial_reproduction_logit_tolerance"])
            and abs(model_auc - reference_auc) <= 1e-12
            and class_exact
        )
        all_pass &= row_pass
        report_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_best_epoch": int(payload.get("best_epoch", -1)),
                "max_abs_logit_error": max_logit_error,
                "max_abs_probability_error": max_prob_error,
                "model_auc": model_auc,
                "reference_auc": reference_auc,
                "auc_exact": abs(model_auc - reference_auc) <= 1e-12,
                "patient_id_exact": bool(np.array_equal(model_ids, reference_ids)),
                "label_exact": bool(np.array_equal(model_labels, reference_labels)),
                "threshold_class_exact": class_exact,
                "reproduction_pass": row_pass,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(report_rows), all_pass


def gradient_and_missing_checks(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device
) -> tuple[Dict[str, bool], Dict[str, Any], List[Dict[str, Any]]]:
    runtime = {
        "encoder_gradients": True,
        "projector_gradients": True,
        "prediction_path_gradients": True,
        "missing_modalities": True,
        "trainable_scope": True,
        "lr_groups": True,
        "recency_prior": True,
        "capacity": True,
    }
    details: Dict[str, Any] = {
        "gradient_norms_by_seed": {},
        "learning_rate_groups": {},
        "trainable_parameter_count_by_seed": {},
        "frozen_parameter_count_by_seed": {},
    }
    audit_rows: List[Dict[str, Any]] = []
    train_loader = build_loaders(config, rows, ("train",))["train"]
    train_batch = move_batch(next(iter(train_loader)), device)
    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(
            str(config["c27"]["c27_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = checkpoint_payload(checkpoint_path)
        model = C27VTMEModel(config, seed).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        runtime["trainable_scope"] &= bool(trainable_names) and all(
            parameter_category(name) in MODULE_CATEGORIES for name in trainable_names
        ) and all(parameter.requires_grad for _, parameter in model.named_parameters())
        count = trainable_parameter_count(model)
        details["trainable_parameter_count_by_seed"][str(seed)] = count
        details["frozen_parameter_count_by_seed"][str(seed)] = int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        )
        runtime["capacity"] &= count <= int(config["c37"]["trainable_parameter_limit"])
        for name, parameter in model.named_parameters():
            category = parameter_category(name)
            audit_rows.append(
                {
                    "seed": seed,
                    "parameter": name,
                    "category": category or "unexpected_or_frozen",
                    "trainable": bool(parameter.requires_grad),
                    "parameter_count": int(parameter.numel()),
                }
            )

        optimizer = build_optimizer(config, model)
        expected_scales = {
            "encoder_lr_scale": float(config["c37"]["encoder_lr_scale"]),
            "projector_lr_scale": float(config["c37"]["projector_lr_scale"]),
            "prediction_path_lr_scale": float(config["c37"]["prediction_path_lr_scale"]),
        }
        group_rows: List[Dict[str, Any]] = []
        for group in optimizer.param_groups:
            category = str(group.get("category", ""))
            if category in ("image_encoder", "text_encoder", "bio_encoder"):
                expected_lr = float(config["training"]["lr"]) * expected_scales["encoder_lr_scale"]
            elif category in ("image_projector", "text_projector", "bio_projector"):
                expected_lr = float(config["training"]["lr"]) * expected_scales["projector_lr_scale"]
            else:
                expected_lr = float(config["training"]["lr"]) * expected_scales["prediction_path_lr_scale"]
            row_pass = abs(float(group["lr"]) - expected_lr) <= 1e-12
            runtime["lr_groups"] &= row_pass and category in MODULE_CATEGORIES
            group_rows.append({"category": category, "lr": float(group["lr"]), "expected_lr": expected_lr, "pass": row_pass})
        details["learning_rate_groups"][str(seed)] = group_rows

        runtime["recency_prior"] &= (
            abs(float(model.core.recency_prior_log_odds) - math.log(2.0)) <= 1e-12
            and not any("recency_prior" in name for name, _ in model.named_parameters())
        )
        before = parameter_digest(model)
        max_norms = {category: 0.0 for category in MODULE_CATEGORIES}
        model.train(True)
        for probe_batch in [train_batch, *missing_modality_batches(train_batch)]:
            model.zero_grad(set_to_none=True)
            output = model(probe_batch)
            loss = F.binary_cross_entropy_with_logits(output["logit"], probe_batch["label"])
            if not bool(torch.isfinite(loss)):
                runtime["missing_modalities"] = False
                continue
            loss.backward()
            norms = {}
            for category, value in {
                category: 0.0 for category in MODULE_CATEGORIES
            }.items():
                norms[category] = 0.0
            for name, parameter in model.named_parameters():
                category = parameter_category(name)
                if category is not None and parameter.grad is not None:
                    norms[category] += float(parameter.grad.detach().float().pow(2).sum().cpu())
                if parameter.grad is not None:
                    runtime["missing_modalities"] &= bool(torch.isfinite(parameter.grad).all())
            for category in norms:
                max_norms[category] = max(max_norms[category], float(np.sqrt(norms[category])))
            runtime["missing_modalities"] &= all(
                (not value.is_floating_point() and not value.is_complex())
                or bool(torch.isfinite(value).all())
                for value in output.values()
                if torch.is_tensor(value)
            )
        details["gradient_norms_by_seed"][str(seed)] = max_norms
        runtime["encoder_gradients"] &= all(
            np.isfinite(max_norms[category]) and max_norms[category] > 0.0
            for category in ("image_encoder", "text_encoder", "bio_encoder")
        )
        runtime["projector_gradients"] &= all(
            np.isfinite(max_norms[category]) and max_norms[category] > 0.0
            for category in ("image_projector", "text_projector", "bio_projector")
        )
        runtime["prediction_path_gradients"] &= all(
            np.isfinite(max_norms[category]) and max_norms[category] > 0.0
            for category in ("temporal_path", "patient_projection", "classifier")
        )
        if before != parameter_digest(model):
            raise RuntimeError("C37 gate changed model state without an optimizer step")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return runtime, details, audit_rows


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    rows = read_jsonl(config["project"]["manifest"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reproduction, reproduction_pass = initial_reproduction(config, rows, device)
    runtime, details, audit_rows = gradient_and_missing_checks(config, rows, device)

    canonical = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve())
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain", "--untracked-files=no")
    train_source = (REPO_ROOT / "scripts" / "train_phase_c37.py").read_text(encoding="utf-8")
    collector_path = REPO_ROOT / "scripts" / "collect_phase_c37_report.py"
    collector_source = collector_path.read_text(encoding="utf-8")
    model_source = inspect.getsource(C27VTMEModel) + inspect.getsource(
        __import__("dmea_ht.c27_vtme", fromlist=["VisitTemporalMechanismCore"]).VisitTemporalMechanismCore
    )
    disabled_metric = "AUP" + "RC"
    decision_path = resolve_path(config["project"]["report_dir"]) / "c37_validation_decision.json"
    split_ok = split_contract(rows)
    manifest_ok = file_sha256(resolve_path(config["project"]["manifest"])) == str(
        "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
    )
    checkpoint_ok = True
    checkpoint_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        path = Path(str(config["c27"]["c27_checkpoint"]).replace("{seed}", str(seed)))
        payload = checkpoint_payload(path)
        state = payload.get("model", {})
        row_ok = bool(
            path.exists()
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("best_epoch", -1)) >= 0
            and any(str(key).startswith("frozen_sources.") for key in state)
            and any(str(key).startswith("core.") for key in state)
        )
        checkpoint_ok &= row_ok
        checkpoint_rows.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "exists": path.exists(),
                "payload_seed_matches": int(payload.get("seed", -1)) == seed,
                "best_epoch": int(payload.get("best_epoch", -1)),
                "contract_pass": row_ok,
            }
        )
    no_shortcut_fields = not any(
        field in model_source
        for field in (
            "shortcuts",
            "selected_n_visits",
            "used_images",
            "image_padding_count",
            "report_length",
            "patient_id",
            "raw_n_visits",
            "raw_n_images",
            "bio_missing_count",
        )
    )
    expected_core_modules = {
        "core.empty_slot_tokens",
        "core.temporal_norm.weight",
        "core.temporal_norm.bias",
        "core.temporal_linear.weight",
        "core.temporal_linear.bias",
        "core.temporal_output.weight",
        "core.temporal_output.bias",
        "core.patient_projection.0.weight",
        "core.patient_projection.0.bias",
        "core.patient_projection.2.weight",
        "core.patient_projection.2.bias",
        "core.classifier.1.weight",
        "core.classifier.1.bias",
    }
    core_architecture_unchanged = expected_core_modules.issubset(
        {name for name, _ in C27VTMEModel(config, 0).named_parameters()}
    )
    test_blocked = (
        not decision_path.exists()
        and "C37 validation decision must be frozen before reporting-only test" in train_source
        and "validation_decision_frozen_before_test" in train_source
    )
    checks = [
        ("01_canonical_main_clean", canonical and branch == "main" and clean),
        ("02_manifest_split_label_contract", manifest_ok and split_ok),
        ("03_c27_seed_checkpoints_contract", checkpoint_ok),
        ("04_initial_c27_reproduction_exact", reproduction_pass),
        ("05_full_e2e_trainable_scope_and_capacity", runtime["trainable_scope"] and runtime["capacity"]),
        ("06_fixed_recency_prior_nontrainable", runtime["recency_prior"]),
        ("07_shortcut_fields_do_not_enter_forward", no_shortcut_fields),
        ("08_encoder_gradients_finite_nonzero", runtime["encoder_gradients"]),
        ("09_projector_gradients_finite_nonzero", runtime["projector_gradients"]),
        ("10_prediction_path_gradients_finite_nonzero", runtime["prediction_path_gradients"]),
        ("11_missing_modality_outputs_finite", runtime["missing_modalities"]),
        ("12_learning_rate_groups_exact", runtime["lr_groups"]),
        ("13_bce_only_no_secondary_metric_or_scheduler", bool(config["loss"]["bce_only"]) and train_source.count("binary_cross_entropy_with_logits") == 1 and "scheduler" not in train_source.lower() and "auxiliary_loss" not in train_source and disabled_metric not in train_source and disabled_metric not in collector_source and disabled_metric not in config_path.read_text(encoding="utf-8")),
        ("14_test_blocked_before_validation_decision", test_blocked),
        ("15_direct_independent_multiseed_single_model_contract", config["deployment"] == {"one_checkpoint": True, "one_model": True, "one_forward": True, "ensemble": False} and "subprocess.Popen" in train_source and 'f"seed_{seed}_best.pt"' in train_source and len(SEEDS) == 3),
        ("16_c27_architecture_and_no_new_module", core_architecture_unchanged and "anchor" not in model_source.lower() and "prototype" not in model_source.lower() and "transformer" not in model_source.lower()),
    ]
    if len(checks) != 16:
        raise RuntimeError(f"C37 gate must contain exactly 16 checks, found {len(checks)}")
    report_dir = resolve_path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(report_dir / "c37_trainable_parameter_audit.csv", index=False)
    reproduction.to_csv(report_dir / "c37_initial_c27_reproduction.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(report_dir / "c37_c27_checkpoint_audit.csv", index=False)
    passed = sum(bool(value) for _, value in checks)
    status = "C37_E2E_VRL_DIRECT_MULTI_SEED_AUTHORIZED" if passed == len(checks) else "DEMA_C37_PATH_GATE_FAIL"
    audit = pd.DataFrame(audit_rows)
    payload = {
        "phase": "C37-E2E-VRL",
        "status": status,
        "passed": passed,
        "total": len(checks),
        "git_commit": git_output("rev-parse", "HEAD"),
        "project": str(REPO_ROOT.resolve()),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_parameter_count": int(audit.loc[audit["trainable"].astype(bool), "parameter_count"].sum()),
        "frozen_parameter_count": int(audit.loc[~audit["trainable"].astype(bool), "parameter_count"].sum()),
        "learning_rate_scales": {
            "encoder": float(config["c37"]["encoder_lr_scale"]),
            "projector": float(config["c37"]["projector_lr_scale"]),
            "prediction_path": float(config["c37"]["prediction_path_lr_scale"]),
        },
        "recency_prior_log_odds": float(config["c27"]["recency_prior_log_odds"]),
        "mechanism_names": list(MECHANISM_NAMES),
        "initial_reproduction": reproduction.to_dict(orient="records"),
        "gradient_norms_by_seed": details["gradient_norms_by_seed"],
        "learning_rate_groups": details["learning_rate_groups"],
        "trainable_parameter_count_by_seed": details["trainable_parameter_count_by_seed"],
        "frozen_parameter_count_by_seed": details["frozen_parameter_count_by_seed"],
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
    }
    (report_dir / "c37_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C37_E2E_VRL_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
