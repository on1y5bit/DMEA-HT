#!/usr/bin/env python3
"""Authorize direct C35-MTSA formal execution with exactly 18 checks."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c35_mtsa import (  # noqa: E402
    EVIDENCE_SOURCE_NAMES,
    MECHANISM_LABELS,
    MECHANISM_NAMES,
    MECHANISM_SOURCE_MAP,
    TRAINABLE_MODULES,
    C35MTSAModel,
    FrozenC17EvidenceRepresentation,
    parameter_audit,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import build_loaders, move_batch, resolve_path, set_seed  # noqa: E402
from scripts.train_phase_c35 import (  # noqa: E402
    SEEDS,
    trainable_gradient_norms,
)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c35_mtsa_multiseed.yaml")
    parser.add_argument("--expected-project", default="/home/linruixin/chen/project/DMEA-HT")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


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


def clone_tensor_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def gradient_probe_batches(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    partial_immune = clone_tensor_batch(batch)
    partial_immune["image_mask"].zero_()
    partial_immune["report_attention_mask"].fill_(1)
    for key in TEXT_MASK_KEYS:
        partial_immune[key].zero_()
    partial_immune["text_support_mask"].fill_(1)
    partial_immune["bio_missing_mask"][..., 2] = 1.0
    partial_immune["bio_missing_mask"][..., 5] = 1.0
    partial_immune["bio_missing_mask"][..., 3] = 0.0
    partial_immune["bio_missing_mask"][..., 4] = 0.0
    partial_immune["bio_missing_mask"][..., 6] = 0.0

    partial_function = clone_tensor_batch(batch)
    partial_function["report_attention_mask"].fill_(1)
    for key in TEXT_MASK_KEYS:
        partial_function[key].zero_()
    partial_function["text_support_mask"].fill_(1)
    partial_function["bio_missing_mask"][..., 2] = 0.0
    partial_function["bio_missing_mask"][..., 5] = 0.0
    partial_function["bio_missing_mask"][..., 3] = 1.0
    partial_function["bio_missing_mask"][..., 4] = 1.0
    partial_function["bio_missing_mask"][..., 6] = 1.0

    empty_evidence = clone_tensor_batch(batch)
    empty_evidence["image_mask"].zero_()
    empty_evidence["report_attention_mask"].zero_()
    empty_evidence["bio_values"].zero_()
    empty_evidence["bio_abnormal_flags"].zero_()
    empty_evidence["bio_missing_mask"].fill_(1.0)
    for key in TEXT_MASK_KEYS:
        empty_evidence[key].zero_()
    return [partial_immune, partial_function, empty_evidence]


def run_runtime_checks(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, bool], Dict[str, Any]]:
    audit_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    runtime = {
        "encoder_projector_checkpoints": True,
        "frozen_representation": True,
        "trainable_scope": True,
        "mechanism_mapping": True,
        "visit_state_shape": True,
        "trajectory_shape": True,
        "coordinate_shape": True,
        "anchor_contract": True,
        "fallback_finite": True,
        "gradient_contract": True,
    }
    details: Dict[str, Any] = {
        "single_visit_seen": False,
        "multi_visit_seen": False,
        "gradient_norms_by_seed": {},
        "probe_gradient_norms_by_seed": {},
        "fallback_gradient_norms_by_seed": {},
        "anchor_center_gradient_norms_by_seed": {},
    }

    expected_mapping = {
        "M1": ("image_morphology", "text_support", "text_nonspecific_morphology"),
        "M2": ("bio_immune",),
        "M3": ("bio_function",),
        "M4": ("text_opposition",),
        "M5": ("text_temporal",),
    }
    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(
            str(config["c35"]["c17_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = checkpoint_payload(checkpoint_path)
        state = payload.get("model", {})
        prefixes = (
            "base_model.image_encoder.",
            "base_model.text_encoder.",
            "base_model.bio_encoder.",
            "mechanism_evidence_alignment.image.",
            "mechanism_evidence_alignment.text.",
            "mechanism_evidence_alignment.bio.",
        )
        checkpoint_ok = (
            checkpoint_path.exists()
            and int(payload.get("seed", -1)) == seed
            and all(any(str(key).startswith(prefix) for key in state) for prefix in prefixes)
        )
        runtime["encoder_projector_checkpoints"] &= checkpoint_ok
        checkpoint_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "exists": checkpoint_path.exists(),
                "payload_seed_matches": int(payload.get("seed", -1)) == seed,
                **{
                    prefix.rstrip(".").replace(".", "_"): any(
                        str(key).startswith(prefix) for key in state
                    )
                    for prefix in prefixes
                },
                "contract_pass": checkpoint_ok,
            }
        )

        model = C35MTSAModel(config, seed).to(device)
        audit_rows.extend(parameter_audit(model))
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        runtime["trainable_scope"] &= bool(trainable_names) and all(
            C35MTSAModel.is_trainable_parameter(name) for name in trainable_names
        )
        runtime["frozen_representation"] &= all(
            not parameter.requires_grad
            for parameter in model.frozen_sources.parameters()
        )
        projector_ids = [
            tuple(id(parameter) for parameter in model.mechanism_projectors[name].parameters())
            for name in MECHANISM_NAMES
        ]
        head_ids = [
            tuple(id(parameter) for parameter in model.trajectory_coordinate_heads[name].parameters())
            for name in MECHANISM_NAMES
        ]
        runtime["mechanism_mapping"] &= (
            tuple(MECHANISM_LABELS)
            == (
                "M1_morphology",
                "M2_immune",
                "M3_function",
                "M4_opposition",
                "M5_temporal_text",
            )
        )
        runtime["mechanism_mapping"] &= MECHANISM_SOURCE_MAP == expected_mapping
        runtime["mechanism_mapping"] &= len(set(projector_ids)) == len(MECHANISM_NAMES)
        runtime["mechanism_mapping"] &= len(set(head_ids)) == len(MECHANISM_NAMES)

        loader = build_loaders(config, rows, ("val",))["val"]
        model.eval()
        first_batch = move_batch(next(iter(loader)), device)
        with torch.inference_mode():
            first_output = model(first_batch)
        source_states = first_output["mechanism_source_states"]
        source_valid = first_output["mechanism_source_valid"]
        visit_state = first_output["mechanism_visit_state"]
        trajectory = first_output["mechanism_trajectory"]
        coordinates = first_output["patient_coordinate"]
        runtime["mechanism_mapping"] &= source_states.shape[-2:] == (5, 256)
        runtime["mechanism_mapping"] &= source_valid.shape[-1] == 5
        runtime["visit_state_shape"] &= (
            visit_state.ndim == 4
            and visit_state.shape[0] == first_batch["visit_mask"].shape[0]
            and visit_state.shape[1] == first_batch["visit_mask"].shape[1]
            and tuple(visit_state.shape[2:]) == (5, 32)
            and bool(torch.isfinite(visit_state).all())
        )
        runtime["trajectory_shape"] &= (
            trajectory.ndim == 3
            and tuple(trajectory.shape[1:]) == (5, 96)
            and bool(torch.isfinite(trajectory).all())
        )
        runtime["coordinate_shape"] &= (
            coordinates.ndim == 2
            and coordinates.shape[1] == 5
            and bool(torch.isfinite(coordinates).all())
            and bool((coordinates.abs() <= 1.0).all())
        )

        single_seen = False
        multi_seen = False
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(batch)
                counts = batch["visit_mask"].sum(dim=1)
                single_seen |= bool((counts == 1).any())
                multi_seen |= bool((counts > 1).any())
                runtime["trajectory_shape"] &= (
                    tuple(output["mechanism_trajectory"].shape[1:]) == (5, 96)
                    and bool(torch.isfinite(output["mechanism_trajectory"]).all())
                )
        details["single_visit_seen"] |= single_seen
        details["multi_visit_seen"] |= multi_seen
        runtime["trajectory_shape"] &= single_seen and multi_seen

        fallback_batch = clone_tensor_batch(first_batch)
        fallback_batch["image_mask"].zero_()
        fallback_batch["report_attention_mask"].zero_()
        fallback_batch["bio_values"].zero_()
        fallback_batch["bio_abnormal_flags"].zero_()
        fallback_batch["bio_missing_mask"].fill_(1.0)
        for key in TEXT_MASK_KEYS:
            fallback_batch[key].zero_()
        with torch.inference_mode():
            fallback_output = model(fallback_batch)
        runtime["fallback_finite"] &= bool(
            torch.isfinite(fallback_output["mechanism_visit_state"]).all()
            and torch.isfinite(fallback_output["mechanism_trajectory"]).all()
            and torch.isfinite(fallback_output["patient_coordinate"]).all()
            and torch.isfinite(fallback_output["logit"]).all()
        )

        anchor_center = first_output["anchor_center"]
        anchor_direction = first_output["anchor_direction"]
        anchor_non_ht = first_output["anchor_non_ht"]
        anchor_ht = first_output["anchor_ht"]
        runtime["anchor_contract"] &= bool(
            torch.isfinite(anchor_center).all()
            and torch.isfinite(anchor_direction).all()
            and torch.isfinite(anchor_non_ht).all()
            and torch.isfinite(anchor_ht).all()
            and torch.isfinite(first_output["anchor_distance"])
            and float(first_output["anchor_distance"]) > 0.0
            and torch.allclose(
                (anchor_non_ht + anchor_ht) / 2.0, anchor_center, atol=1e-6
            )
            and torch.allclose(
                torch.linalg.vector_norm(anchor_direction),
                torch.ones((), device=device),
                atol=1e-5,
            )
        )

        train_batch = move_batch(
            next(iter(build_loaders(config, rows, ("train",))["train"])), device
        )
        before = state_digest(model)
        model.train(True)
        model.zero_grad(set_to_none=True)
        probe_batches = [train_batch, *gradient_probe_batches(train_batch)]
        probe_details: List[Dict[str, Any]] = []
        for probe_index, probe_batch in enumerate(probe_batches):
            output = model(probe_batch)
            loss = F.binary_cross_entropy_with_logits(output["logit"], probe_batch["label"])
            if not bool(torch.isfinite(loss)):
                runtime["gradient_contract"] = False
                probe_details.append(
                    {"probe": probe_index, "loss": float("nan"), "norms": trainable_gradient_norms(model)}
                )
                continue
            loss.backward()
            probe_details.append(
                {
                    "probe": probe_index,
                    "loss": float(loss.detach().cpu()),
                    "norms": trainable_gradient_norms(model),
                }
            )
        norms = trainable_gradient_norms(model)
        details["gradient_norms_by_seed"][str(seed)] = norms
        details["probe_gradient_norms_by_seed"][str(seed)] = probe_details
        details["fallback_gradient_norms_by_seed"][str(seed)] = {
            name: norms[f"mechanism_fallbacks"] for name in MECHANISM_NAMES
        }
        details["anchor_center_gradient_norms_by_seed"][str(seed)] = norms["anchor_center"]
        required_categories = (
            "mechanism_projectors",
            "trajectory_coordinate_heads",
            "anchor_direction",
        )
        runtime["gradient_contract"] &= all(
            np.isfinite(norms[category]) and norms[category] > 0.0
            for category in required_categories
        )
        runtime["gradient_contract"] &= all(
            parameter.grad is None
            for parameter in model.frozen_sources.parameters()
        )
        runtime["gradient_contract"] &= all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if before != state_digest(model):
            raise RuntimeError("C35 gate changed model state without an optimizer step")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(audit_rows), pd.DataFrame(checkpoint_rows), runtime, details


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    rows = read_jsonl(config["project"]["manifest"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit, checkpoints, runtime, details = run_runtime_checks(config, rows, device)

    canonical = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve())
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain", "--untracked-files=no")
    train_source = (REPO_ROOT / "scripts" / "train_phase_c35.py").read_text(
        encoding="utf-8"
    )
    collector_path = REPO_ROOT / "scripts" / "collect_phase_c35_report.py"
    collector_source = collector_path.read_text(encoding="utf-8")
    model_source = inspect.getsource(C35MTSAModel) + inspect.getsource(
        FrozenC17EvidenceRepresentation
    )
    disabled_metric = "AUP" + "RC"
    decision_path = resolve_path(config["project"]["report_dir"]) / "c35_validation_decision.json"
    forbidden_fields = (
        "shortcuts",
        "selected_n_visits",
        "used_images",
        "image_padding_count",
        "report_length",
        "patient_id",
        "visit_dates",
        "raw_n_visits",
        "raw_n_images",
        "bio_missing_count",
    )
    no_forbidden_fields = not any(field in model_source for field in forbidden_fields)
    test_blocked = (
        not decision_path.exists()
        and "C35 validation decision must be frozen before reporting-only test" in train_source
        and "validation_decision_frozen_before_test" in train_source
    )
    checks = [
        ("01_canonical_main_clean", canonical and branch == "main" and clean),
        (
            "02_manifest_sha256_exact",
            file_sha256(resolve_path(config["project"]["manifest"]))
            == str(config["c35"]["manifest_sha256"]),
        ),
        ("03_patient_split_label_contract", split_contract(rows)),
        ("04_c17_encoder_projector_checkpoints_correct", runtime["encoder_projector_checkpoints"]),
        ("05_c17_encoders_and_projectors_frozen", runtime["frozen_representation"]),
        ("06_only_c35_scope_trainable", runtime["trainable_scope"]),
        ("07_five_mechanisms_use_exact_real_nodes", runtime["mechanism_mapping"]),
        ("08_mechanism_visit_state_shape_bv532", runtime["visit_state_shape"]),
        ("09_fixed_single_multi_trajectory_shape", runtime["trajectory_shape"]),
        ("10_patient_coordinate_exactly_five", runtime["coordinate_shape"]),
        ("11_symmetric_anchor_contract", runtime["anchor_contract"]),
        ("12_missing_mechanism_fallback_finite", runtime["fallback_finite"]),
        ("13_real_activated_gradients_finite_nonzero", runtime["gradient_contract"]),
        (
            "14_trainable_parameter_count_within_limit",
            int(audit.loc[audit["trainable"].astype(bool), "parameter_count"].sum())
            <= int(config["c35"]["trainable_parameter_limit"]),
        ),
        ("15_audit_fields_do_not_enter_model", no_forbidden_fields),
        ("16_test_blocked_before_validation_decision", test_blocked),
        (
            "17_bce_only_and_no_secondary_metric",
            bool(config["loss"]["bce_only"])
            and train_source.count("binary_cross_entropy_with_logits") == 1
            and "positive_preservation_loss" not in train_source
            and "pairwise_ranking_loss" not in train_source
            and "auxiliary_loss" not in train_source
            and disabled_metric not in train_source
            and disabled_metric not in collector_source
            and disabled_metric not in config_path.read_text(encoding="utf-8"),
        ),
        (
            "18_single_model_independent_direct_multiseed",
            config["deployment"]
            == {
                "one_checkpoint": True,
                "one_model": True,
                "one_forward": True,
                "ensemble": False,
            }
            and "seed_runs" in train_source
            and "subprocess.Popen" in train_source
            and 'f"seed_{seed}_best.pt"' in train_source
            and len(SEEDS) == 3,
        ),
    ]
    if len(checks) != 18:
        raise RuntimeError(f"C35 gate must contain exactly 18 checks, found {len(checks)}")
    report_dir = resolve_path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(report_dir / "c35_trainable_parameter_audit.csv", index=False)
    checkpoints.to_csv(report_dir / "c35_c17_checkpoint_audit.csv", index=False)
    passed = sum(bool(value) for _, value in checks)
    status = (
        "C35_MTSA_DIRECT_MULTI_SEED_AUTHORIZED"
        if passed == len(checks)
        else "DEMA_C35_PATH_GATE_FAIL"
    )
    trainable = audit[audit["trainable"].astype(bool)].copy()
    payload = {
        "phase": "C35-MTSA",
        "status": status,
        "passed": passed,
        "total": len(checks),
        "git_commit": git_output("rev-parse", "HEAD"),
        "project": str(REPO_ROOT.resolve()),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_parameter_count": int(trainable["parameter_count"].sum()),
        "frozen_parameter_count": int(
            audit.loc[~audit["trainable"].astype(bool), "parameter_count"].sum()
        ),
        "mechanism_names": list(MECHANISM_NAMES),
        "mechanism_labels": list(MECHANISM_LABELS),
        "mechanism_source_map": MECHANISM_SOURCE_MAP,
        "evidence_source_names": list(EVIDENCE_SOURCE_NAMES),
        "mechanism_visit_state_shape": ["B", "V", 5, 32],
        "mechanism_trajectory_shape": ["B", 5, 96],
        "patient_coordinate_shape": ["B", 5],
        "single_visit_seen": bool(details["single_visit_seen"]),
        "multi_visit_seen": bool(details["multi_visit_seen"]),
        "gradient_norms_by_seed": details["gradient_norms_by_seed"],
        "probe_gradient_norms_by_seed": details["probe_gradient_norms_by_seed"],
        "fallback_gradient_norms_by_seed": details["fallback_gradient_norms_by_seed"],
        "anchor_center_gradient_norms_by_seed": details["anchor_center_gradient_norms_by_seed"],
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
    }
    (report_dir / "c35_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C35_MTSA_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
