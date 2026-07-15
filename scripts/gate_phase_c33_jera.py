#!/usr/bin/env python3
"""Authorize direct C33-JERA formal execution with 15 focused checks."""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c33_jera import (  # noqa: E402
    C33JERAModel,
    TRAINABLE_MODULES,
    parameter_audit,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import build_loaders, move_batch, resolve_path, set_seed  # noqa: E402
from scripts.train_phase_c33 import trainable_gradient_norms  # noqa: E402


SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dema_ht_c33_jera_multiseed.yaml"
    )
    parser.add_argument(
        "--expected-project", default="/home/linruixin/chen/project/DMEA-HT"
    )
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


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    return frame.sort_values("patient_id").reset_index(drop=True)


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def logit_column(frame: pd.DataFrame) -> str:
    for name in ("final_logit", "logit", "pred_logit"):
        if name in frame.columns:
            return name
    probabilities = frame[probability_column(frame)].to_numpy(dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    frame["_derived_logit"] = np.log(clipped / (1.0 - clipped))
    return "_derived_logit"


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return value


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


def run_runtime_checks(
    config: Dict[str, Any], rows: List[Dict[str, Any]], device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, bool]]:
    reproduction_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    runtime = {
        "checkpoints": True,
        "initial_reproduction": True,
        "trainable_scope": True,
        "encoders_no_grad": True,
        "temporal_readout_boundary": True,
        "authorized_gradients": True,
    }
    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(
            str(config["c33"]["c27_checkpoint"]).replace("{seed}", str(seed))
        )
        payload = checkpoint_payload(checkpoint_path)
        runtime["checkpoints"] &= checkpoint_path.exists() and int(
            payload.get("seed", -1)
        ) == seed
        model = C33JERAModel(config, seed).to(device)
        audit_rows.extend(parameter_audit(model))
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        runtime["trainable_scope"] &= bool(trainable_names) and all(
            C33JERAModel.is_trainable_parameter(name) for name in trainable_names
        )

        loader = build_loaders(config, rows, ("val",))["val"]
        ids: List[str] = []
        labels: List[int] = []
        logits: List[np.ndarray] = []
        probabilities: List[np.ndarray] = []
        all_finite = True
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(batch)
                ids.extend(str(value) for value in batch["patient_id"])
                labels.extend(int(value) for value in batch["label"].detach().cpu())
                logits.append(output["logit"].detach().cpu().numpy())
                probabilities.append(output["prob"].detach().cpu().numpy())
                all_finite &= bool(torch.isfinite(output["logit"]).all())
                all_finite &= bool(torch.isfinite(output["patient_state"]).all())
        actual = pd.DataFrame(
            {
                "patient_id": ids,
                "label": labels,
                "logit": np.concatenate(logits),
                "probability": np.concatenate(probabilities),
            }
        ).sort_values("patient_id").reset_index(drop=True)
        official = read_prediction(
            resolve_path(config["c33"]["c27_run_dir"])
            / "predictions"
            / f"val_predictions_seed_{seed}.csv"
        )
        ids_exact = np.array_equal(
            actual["patient_id"].to_numpy(dtype=str),
            official["patient_id"].to_numpy(dtype=str),
        )
        labels_exact = np.array_equal(
            actual["label"].to_numpy(dtype=int),
            official["label"].to_numpy(dtype=int),
        )
        official_prob = official[probability_column(official)].to_numpy(dtype=float)
        official_logit = official[logit_column(official)].to_numpy(dtype=float)
        actual_prob = actual["probability"].to_numpy(dtype=float)
        actual_logit = actual["logit"].to_numpy(dtype=float)
        logit_error = float(np.max(np.abs(actual_logit - official_logit)))
        probability_error = float(np.max(np.abs(actual_prob - official_prob)))
        auc_error = abs(
            float(roc_auc_score(actual["label"], actual_prob))
            - float(roc_auc_score(official["label"], official_prob))
        )
        class_mismatch = int(
            ((actual_prob >= 0.5) != (official_prob >= 0.5)).sum()
        )
        reproduction_pass = bool(
            ids_exact
            and labels_exact
            and all_finite
            and logit_error <= float(config["c33"]["initial_logit_tolerance"])
            and auc_error == 0.0
            and class_mismatch == 0
        )
        runtime["initial_reproduction"] &= reproduction_pass
        reproduction_rows.append(
            {
                "seed": seed,
                "patient_ids_exact": ids_exact,
                "labels_exact": labels_exact,
                "max_abs_logit_error": logit_error,
                "max_abs_probability_error": probability_error,
                "AUC_error": auc_error,
                "threshold_class_mismatch_count": class_mismatch,
                "all_outputs_finite": all_finite,
                "reproduction_pass": reproduction_pass,
            }
        )

        train_batch = next(iter(build_loaders(config, rows, ("train",))["train"]))
        train_batch = move_batch(train_batch, device)
        before = state_digest(model)
        model.train(True)
        output = model(train_batch)
        loss = F.binary_cross_entropy_with_logits(output["logit"], train_batch["label"])
        model.zero_grad(set_to_none=True)
        loss.backward()
        norms = trainable_gradient_norms(model)
        runtime["authorized_gradients"] &= all(
            np.isfinite(value) and value > 0.0 for value in norms.values()
        )
        runtime["encoders_no_grad"] &= all(
            parameter.grad is None
            for encoder in (
                model.c27.frozen_sources.image_encoder,
                model.c27.frozen_sources.text_encoder,
                model.c27.frozen_sources.bio_encoder,
            )
            for parameter in encoder.parameters()
        )
        temporal_modules = (
            model.c27.core.empty_slot_tokens,
            model.c27.core.temporal_norm,
            model.c27.core.temporal_linear,
            model.c27.core.temporal_output,
        )
        runtime["temporal_readout_boundary"] &= all(
            parameter.grad is None
            for module in temporal_modules
            for parameter in ([module] if isinstance(module, torch.nn.Parameter) else module.parameters())
        ) and float(model.c27.core.recency_prior_log_odds) == float(
            config["c27"]["recency_prior_log_odds"]
        )
        if before != state_digest(model):
            raise RuntimeError("C33 gate changed checkpoint state without an optimizer step")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(audit_rows), pd.DataFrame(reproduction_rows), runtime


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    rows = read_jsonl(config["project"]["manifest"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit, reproduction, runtime = run_runtime_checks(config, rows, device)

    canonical = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve())
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain", "--untracked-files=no")
    checkpoint_paths = [
        Path(str(config["c33"]["c27_checkpoint"]).replace("{seed}", str(seed)))
        for seed in SEEDS
    ]
    train_source = (REPO_ROOT / "scripts" / "train_phase_c33.py").read_text(
        encoding="utf-8"
    )
    collector_source = (
        REPO_ROOT / "scripts" / "collect_phase_c33_report.py"
    ).read_text(encoding="utf-8")
    model_source = inspect.getsource(C33JERAModel)
    disabled_metric = "AUP" + "RC"
    decision_path = resolve_path(config["project"]["report_dir"]) / "c33_validation_decision.json"
    test_blocked = (
        not decision_path.exists()
        and "validation decision must be frozen before reporting-only test" in train_source
        and "validation_decision_frozen_before_test" in train_source
    )
    checks = [
        ("01_canonical_main_clean", canonical and branch == "main" and clean),
        (
            "02_c27_checkpoints_exist_and_seed_match",
            runtime["checkpoints"] and len(checkpoint_paths) == 3,
        ),
        (
            "03_visit_manifest_sha256_exact",
            file_sha256(resolve_path(config["project"]["manifest"]))
            == str(config["c33"]["manifest_sha256"]),
        ),
        ("04_patient_split_label_contract", split_contract(rows)),
        ("05_initial_c27_predictions_reproduced", runtime["initial_reproduction"]),
        (
            "06_only_projectors_patient_projection_classifier_trainable",
            runtime["trainable_scope"],
        ),
        ("07_modality_encoders_have_no_gradients", runtime["encoders_no_grad"]),
        (
            "08_temporal_recency_conflict_path_frozen",
            runtime["temporal_readout_boundary"],
        ),
        (
            "09_all_authorized_modules_have_finite_nonzero_gradients",
            runtime["authorized_gradients"],
        ),
        (
            "10_bce_is_the_only_loss",
            bool(config["loss"]["bce_only"])
            and train_source.count("binary_cross_entropy_with_logits") == 1
            and "positive_preservation_loss" not in train_source
            and "pairwise_ranking_loss" not in train_source,
        ),
        ("11_shortcut_fields_do_not_enter_forward", "shortcuts" not in model_source),
        ("12_test_blocked_before_validation_decision", test_blocked),
        (
            "13_secondary_metric_absent",
            disabled_metric not in train_source
            and disabled_metric not in collector_source
            and disabled_metric not in config_path.read_text(encoding="utf-8"),
        ),
        (
            "14_single_model_no_ensemble",
            config["deployment"]
            == {
                "one_checkpoint": True,
                "one_model": True,
                "one_forward": True,
                "ensemble": False,
            },
        ),
        (
            "15_independent_seed_checkpoints",
            "seed_runs" in train_source
            and "subprocess.Popen" in train_source
            and 'f"seed_{seed}_best.pt"' in train_source,
        ),
    ]
    if len(checks) != 15:
        raise RuntimeError(f"C33 gate must contain exactly 15 checks, found {len(checks)}")
    report_dir = resolve_path(config["project"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    trainable = audit[audit["trainable"].astype(bool)].copy()
    audit.to_csv(report_dir / "c33_trainable_parameter_audit.csv", index=False)
    reproduction.to_csv(report_dir / "c33_initial_c27_reproduction.csv", index=False)
    passed = sum(bool(value) for _, value in checks)
    status = (
        "C33_JERA_DIRECT_MULTI_SEED_AUTHORIZED"
        if passed == len(checks)
        else "DEMA_C33_PATH_GATE_FAIL"
    )
    payload = {
        "phase": "C33-JERA",
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
        "trainable_modules": sorted(trainable["module_name"].unique().tolist()),
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
    }
    (report_dir / "c33_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C33_JERA_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
