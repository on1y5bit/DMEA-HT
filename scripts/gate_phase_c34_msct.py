#!/usr/bin/env python3
"""Authorize direct C34-MSCT formal execution with 18 focused checks."""

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

from dmea_ht.c34_msct import (  # noqa: E402
    SOURCE_NAMES,
    TEXT_MASK_KEYS,
    TRAINABLE_MODULES,
    C34MSCTModel,
    parameter_audit,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import build_loaders, move_batch, resolve_path, set_seed  # noqa: E402
from scripts.train_phase_c34 import (  # noqa: E402
    SEEDS,
    trainable_gradient_norms,
)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c34_msct_multiseed.yaml")
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
    """Create deterministic BCE probes for observed and fallback branches."""
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

    empty_visit = clone_tensor_batch(batch)
    empty_visit["visit_mask"].zero_()
    empty_visit["image_mask"].zero_()
    empty_visit["report_attention_mask"].zero_()
    empty_visit["bio_missing_mask"].fill_(1.0)
    for key in TEXT_MASK_KEYS:
        empty_visit[key].zero_()
    return [partial_immune, partial_function, empty_visit]


def run_runtime_checks(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, bool], Dict[str, Any]]:
    audit_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    runtime = {
        "encoder_checkpoints": True,
        "trainable_scope": True,
        "encoders_frozen": True,
        "source_shapes": True,
        "fallback_finite": True,
        "trajectory_finite": True,
        "trajectory_single_multi": True,
        "gradient_contract": True,
        "independent_heads": True,
    }
    details: Dict[str, Any] = {
        "single_visit_seen": False,
        "multi_visit_seen": False,
        "gradient_norms_by_seed": {},
    }

    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(
            str(config["c34"]["encoder_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = checkpoint_payload(checkpoint_path)
        state = payload.get("model", {})
        prefix_ok = all(
            any(str(key).startswith(prefix) for key in state)
            for prefix in (
                "base_model.image_encoder.",
                "base_model.text_encoder.",
                "base_model.bio_encoder.",
            )
        )
        checkpoint_ok = (
            checkpoint_path.exists()
            and int(payload.get("seed", -1)) == seed
            and prefix_ok
        )
        runtime["encoder_checkpoints"] &= checkpoint_ok
        checkpoint_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "exists": checkpoint_path.exists(),
                "payload_seed_matches": int(payload.get("seed", -1)) == seed,
                "image_encoder_prefix": any(
                    str(key).startswith("base_model.image_encoder.") for key in state
                ),
                "text_encoder_prefix": any(
                    str(key).startswith("base_model.text_encoder.") for key in state
                ),
                "bio_encoder_prefix": any(
                    str(key).startswith("base_model.bio_encoder.") for key in state
                ),
                "contract_pass": checkpoint_ok,
            }
        )

        model = C34MSCTModel(config, seed).to(device)
        audit_rows.extend(parameter_audit(model))
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        runtime["trainable_scope"] &= bool(trainable_names) and all(
            C34MSCTModel.is_trainable_parameter(name) for name in trainable_names
        )
        runtime["encoders_frozen"] &= all(
            not parameter.requires_grad
            for encoder in (
                model.image_encoder,
                model.text_encoder,
                model.bio_encoder,
            )
            for parameter in encoder.parameters()
        )
        head_parameter_ids = [
            tuple(id(parameter) for parameter in model.source_heads[name].parameters())
            for name in SOURCE_NAMES
        ]
        runtime["independent_heads"] &= len(set(head_parameter_ids)) == len(SOURCE_NAMES)
        runtime["independent_heads"] &= all(
            tuple(parameter.shape for parameter in model.source_heads[name].parameters())
            == tuple(parameter.shape for parameter in model.source_heads[SOURCE_NAMES[0]].parameters())
            for name in SOURCE_NAMES
        )

        loader = build_loaders(config, rows, ("val",))["val"]
        model.eval()
        first_batch = next(iter(loader))
        first_batch = move_batch(first_batch, device)
        with torch.inference_mode():
            first_output = model(first_batch)
        source_states = first_output["source_states"]
        source_valid = first_output["source_valid"]
        source_evidence = first_output["source_evidence_valid"]
        trajectory = first_output["trajectory"]
        runtime["source_shapes"] &= (
            source_states.ndim == 3
            and source_states.shape[-1] == len(SOURCE_NAMES)
            and source_valid.shape == source_states.shape
            and source_evidence.shape == source_states.shape
            and trajectory.ndim == 2
            and trajectory.shape[-1] == 5
        )
        runtime["source_shapes"] &= bool(torch.isfinite(source_states).all())
        runtime["source_shapes"] &= bool(
            ((source_states >= 0.0) & (source_states <= 1.0)).all()
        )
        runtime["trajectory_finite"] &= bool(torch.isfinite(trajectory).all())

        single_seen = False
        multi_seen = False
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(batch)
                counts = batch["visit_mask"].sum(dim=1)
                single_seen |= bool((counts == 1).any())
                multi_seen |= bool((counts > 1).any())
                runtime["trajectory_finite"] &= bool(
                    torch.isfinite(output["trajectory"]).all()
                )
                runtime["trajectory_finite"] &= output["trajectory"].shape[-1] == 5
        details["single_visit_seen"] |= single_seen
        details["multi_visit_seen"] |= multi_seen
        runtime["trajectory_single_multi"] &= single_seen and multi_seen

        missing_batch = clone_tensor_batch(first_batch)
        missing_batch["image_mask"].zero_()
        missing_batch["report_attention_mask"].zero_()
        missing_batch["bio_values"].zero_()
        missing_batch["bio_missing_mask"].fill_(1.0)
        for key in TEXT_MASK_KEYS:
            missing_batch[key].zero_()
        with torch.inference_mode():
            missing_output = model(missing_batch)
        runtime["fallback_finite"] &= bool(
            torch.isfinite(missing_output["source_states"]).all()
            and torch.isfinite(missing_output["visit_state"]).all()
            and torch.isfinite(missing_output["trajectory"]).all()
        )

        train_batch = next(iter(build_loaders(config, rows, ("train",))["train"]))
        train_batch = move_batch(train_batch, device)
        before = state_digest(model)
        model.train(True)
        model.zero_grad(set_to_none=True)
        probe_batches = [train_batch, *gradient_probe_batches(train_batch)]
        for probe_batch in probe_batches:
            output = model(probe_batch)
            loss = F.binary_cross_entropy_with_logits(output["logit"], probe_batch["label"])
            if not bool(torch.isfinite(loss)):
                runtime["gradient_contract"] = False
                continue
            loss.backward()
        norms = trainable_gradient_norms(model)
        details["gradient_norms_by_seed"][str(seed)] = norms
        runtime["gradient_contract"] &= all(
            np.isfinite(value) and value > 0.0 for value in norms.values()
        )
        runtime["gradient_contract"] &= all(
            parameter.grad is None
            for encoder in (model.image_encoder, model.text_encoder, model.bio_encoder)
            for parameter in encoder.parameters()
        )
        runtime["gradient_contract"] &= all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if before != state_digest(model):
            raise RuntimeError("C34 gate changed model state without an optimizer step")
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
    train_source = (REPO_ROOT / "scripts" / "train_phase_c34.py").read_text(
        encoding="utf-8"
    )
    collector_path = REPO_ROOT / "scripts" / "collect_phase_c34_report.py"
    collector_source = collector_path.read_text(encoding="utf-8")
    model_source = inspect.getsource(C34MSCTModel)
    disabled_metric = "AUP" + "RC"
    decision_path = resolve_path(config["project"]["report_dir"]) / "c34_validation_decision.json"
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
        "fallback_bio_values",
    )
    no_forbidden_fields = not any(field in model_source for field in forbidden_fields)
    test_blocked = (
        not decision_path.exists()
        and "validation decision must be frozen before reporting-only test" in train_source
        and "validation_decision_frozen_before_test" in train_source
    )
    checks = [
        ("01_canonical_main_clean", canonical and branch == "main" and clean),
        (
            "02_manifest_sha256_exact",
            file_sha256(resolve_path(config["project"]["manifest"]))
            == str(config["c34"]["manifest_sha256"]),
        ),
        ("03_patient_split_label_contract", split_contract(rows)),
        ("04_c17_encoder_checkpoints_correct", runtime["encoder_checkpoints"]),
        ("05_all_three_encoders_frozen", runtime["encoders_frozen"]),
        ("06_only_c34_scope_trainable", runtime["trainable_scope"]),
        ("07_five_source_states_shape_and_range", runtime["source_shapes"]),
        ("08_missing_source_fallback_finite", runtime["fallback_finite"]),
        (
            "09_single_and_multi_trajectory_finite",
            runtime["trajectory_finite"] and runtime["trajectory_single_multi"],
        ),
        ("10_trajectory_exactly_five_dimensions", runtime["source_shapes"]),
        ("11_audit_fields_do_not_enter_model", no_forbidden_fields),
        (
            "12_bce_is_the_only_loss",
            bool(config["loss"]["bce_only"])
            and train_source.count("binary_cross_entropy_with_logits") == 1
            and "positive_preservation_loss" not in train_source
            and "pairwise_ranking_loss" not in train_source
            and "auxiliary_loss" not in train_source,
        ),
        ("13_all_trainable_modules_have_finite_nonzero_gradients", runtime["gradient_contract"]),
        (
            "14_trainable_parameter_count_within_limit",
            int(audit.loc[audit["trainable"].astype(bool), "parameter_count"].sum())
            <= int(config["c34"]["trainable_parameter_limit"]),
        ),
        ("15_test_blocked_before_validation_decision", test_blocked),
        (
            "16_secondary_metric_absent",
            disabled_metric not in train_source
            and disabled_metric not in collector_source
            and disabled_metric not in config_path.read_text(encoding="utf-8"),
        ),
        (
            "17_no_ensemble_or_combined_predictions",
            config["deployment"]
            == {
                "one_checkpoint": True,
                "one_model": True,
                "one_forward": True,
                "ensemble": False,
            },
        ),
        (
            "18_independent_seed_checkpoints",
            "seed_runs" in train_source
            and "subprocess.Popen" in train_source
            and 'f"seed_{seed}_best.pt"' in train_source
            and len(SEEDS) == 3,
        ),
    ]
    if len(checks) != 18:
        raise RuntimeError(f"C34 gate must contain exactly 18 checks, found {len(checks)}")
    report_dir = resolve_path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(report_dir / "c34_trainable_parameter_audit.csv", index=False)
    checkpoints.to_csv(report_dir / "c34_encoder_checkpoint_audit.csv", index=False)
    passed = sum(bool(value) for _, value in checks)
    status = (
        "C34_MSCT_DIRECT_MULTI_SEED_AUTHORIZED"
        if passed == len(checks)
        else "DEMA_C34_PATH_GATE_FAIL"
    )
    trainable = audit[audit["trainable"].astype(bool)].copy()
    payload = {
        "phase": "C34-MSCT",
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
        "source_names": list(SOURCE_NAMES),
        "trajectory_dimension": 5,
        "single_visit_seen": bool(details["single_visit_seen"]),
        "multi_visit_seen": bool(details["multi_visit_seen"]),
        "gradient_norms_by_seed": details["gradient_norms_by_seed"],
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
    }
    (report_dir / "c34_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C34_MSCT_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
