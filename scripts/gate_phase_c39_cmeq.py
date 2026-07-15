#!/usr/bin/env python3
"""Authorize C39-CMEQ direct formal execution with a compact contract gate."""

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

from dmea_ht.c39_cmeq import C39CMEQModel, HEAD_PREFIXES, trainable_parameter_count  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c39 import SEEDS, build_loaders, move_batch, set_seed  # noqa: E402


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c39_cmeq_multiseed.yaml")
    parser.add_argument("--expected-project", default="/home/linruixin/chen/project/DMEA-HT")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def sha256_file(path: Path) -> str:
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
        raise RuntimeError(f"Invalid C39 checkpoint payload: {path}")
    return payload


def split_contract(rows: List[Dict[str, Any]]) -> bool:
    expected = {"train": (602, 301, 301), "val": (94, 47, 47), "test": (84, 42, 42)}
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


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def missing_batches(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_missing = clone_batch(batch)
    image_missing["image_mask"].zero_()
    text_missing = clone_batch(batch)
    text_missing["report_input_ids"].zero_()
    text_missing["report_attention_mask"].zero_()
    for key in TEXT_MASK_KEYS:
        text_missing[key].zero_()
    bio_missing = clone_batch(batch)
    bio_missing["bio_values"].zero_()
    bio_missing["bio_missing_mask"].fill_(1.0)
    bio_missing["bio_abnormal_flags"].zero_()
    return [image_missing, text_missing, bio_missing]


def finite_tensors(output: Mapping[str, Any]) -> bool:
    for value in output.values():
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    rows = read_jsonl(config["project"]["manifest"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loaders(config, rows, ("train",))["train"]
    train_batch = move_batch(next(iter(loader)), device)

    gradient_pass = True
    output_pass = True
    frozen_scope_pass = True
    trainable_counts: Dict[str, int] = {}
    gradient_rows: List[Dict[str, Any]] = []
    source_text = inspect.getsource(C39CMEQModel)
    for seed in SEEDS:
        set_seed(seed)
        model = C39CMEQModel(config, seed).to(device)
        trainable_counts[str(seed)] = trainable_parameter_count(model)
        frozen_scope_pass &= all(
            name.startswith(HEAD_PREFIXES)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        frozen_scope_pass &= not any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith("sources.")
        )
        model.train(True)
        for probe in [train_batch, *missing_batches(train_batch)]:
            model.zero_grad(set_to_none=True)
            output = model(probe)
            output_pass &= finite_tensors(output)
            loss = F.binary_cross_entropy_with_logits(output["logit"], probe["label"])
            output_pass &= bool(torch.isfinite(loss))
            loss.backward()
            finite_gradient_count = 0
            nonzero_gradient_count = 0
            source_gradient_count = 0
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if name.startswith("sources."):
                    source_gradient_count += 1
                    continue
                finite = bool(torch.isfinite(parameter.grad).all())
                norm = float(parameter.grad.detach().float().norm().cpu())
                finite_gradient_count += int(finite)
                nonzero_gradient_count += int(finite and norm > 0.0)
                gradient_pass &= finite
            gradient_pass &= source_gradient_count == 0
            gradient_rows.append(
                {
                    "seed": seed,
                    "finite_head_gradient_tensors": finite_gradient_count,
                    "nonzero_head_gradient_tensors": nonzero_gradient_count,
                    "source_gradient_tensors": source_gradient_count,
                }
            )
            gradient_pass &= nonzero_gradient_count >= 3
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainable_scope_pass = bool(trainable_counts) and max(trainable_counts.values()) <= int(config["c39"]["trainable_parameter_limit"])
    config_text = config_path.read_text(encoding="utf-8")
    train_source = (REPO_ROOT / "scripts" / "train_phase_c39.py").read_text(encoding="utf-8")
    collector_source = (REPO_ROOT / "scripts" / "collect_phase_c39_report.py").read_text(encoding="utf-8")
    disabled_metric = "AUP" + "RC"
    shortcut_pass = not any(
        field in source_text
        for field in (
            "patient_id",
            "selected_n_visits",
            "raw_n_visits",
            "used_images",
            "raw_n_images",
            "image_padding_count",
            "report_length",
            "source_folder",
        )
    ) and bool(config["c39"]["shortcut_fields_used_as_inputs"] is False)
    static_contract_pass = (
        "C27" not in source_text
        and "C37" not in source_text
        and "C38" not in source_text
        and "visit_score" not in source_text
        and "temporal_linear" not in source_text
        and "log2" in source_text
        and "pair_relations" in source_text
    )
    test_blocked = (
        "validation decision must be frozen before reporting-only test" in train_source
        and "set(metrics[\"split\"]) != {\"val\"}" in train_source
        and "c39_validation_decision.json" in train_source
    )
    direct_contract = (
        "subprocess.Popen" in train_source
        and "validation-seed" in train_source
        and 'f"seed_{seed}_best.pt"' in train_source
        and len(SEEDS) == 3
        and config["deployment"] == {"one_checkpoint": True, "one_model": True, "one_forward": True, "ensemble": False}
    )
    loss_contract = (
        bool(config["loss"]["bce_only"])
        and train_source.count("binary_cross_entropy_with_logits") == 1
        and "scheduler" not in train_source.lower()
        and disabled_metric not in train_source
        and disabled_metric not in collector_source
        and disabled_metric not in config_text
    )
    branch = git_output("branch", "--show-current")
    dirty_lines = [line for line in git_output("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    dirty_allowed = all(line.strip().endswith("DEVELOPMENT_LOG.md") for line in dirty_lines)
    canonical_main = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve()) and branch == "main" and dirty_allowed
    manifest_ok = sha256_file(Path(config["project"]["manifest"])) == "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
    checkpoint_ok = True
    checkpoint_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        path = Path(str(config["c17"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        payload = checkpoint_payload(path)
        state = payload.get("model", {})
        row_ok = bool(
            path.exists()
            and int(payload.get("seed", -1)) == seed
            and any(str(key).startswith("base_model.image_encoder.") for key in state)
            and any(str(key).startswith("mechanism_evidence_alignment.image.") for key in state)
            and any(str(key).startswith("mechanism_evidence_alignment.text.") for key in state)
            and any(str(key).startswith("mechanism_evidence_alignment.bio.") for key in state)
        )
        checkpoint_ok &= row_ok
        checkpoint_rows.append({"seed": seed, "checkpoint": str(path), "contract_pass": row_ok})
    checks = [
        ("01_canonical_main_and_path", canonical_main),
        ("02_manifest_patient_split_and_labels", manifest_ok and split_contract(rows)),
        ("03_c17_source_checkpoints", checkpoint_ok),
        ("04_new_head_scope_and_capacity", frozen_scope_pass and trainable_scope_pass),
        ("05_finite_outputs_and_missing_modality", output_pass),
        ("06_new_head_gradients_and_frozen_source", gradient_pass),
        ("07_shortcut_fields_excluded", shortcut_pass),
        ("08_fixed_modal_trajectory_cross_modal_relation", static_contract_pass and config["c39"]["fixed_recency_kernel"] == "inverse_log2_age" and config["c39"]["learned_visit_score"] is False),
        ("09_bce_only_and_no_secondary_metric", loss_contract),
        ("10_validation_test_isolation", test_blocked),
        ("11_direct_single_model_multiseed_contract", direct_contract),
    ]
    report_dir = Path(config["project"]["report_dir"])
    if not report_dir.is_absolute():
        report_dir = REPO_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gradient_rows).to_csv(report_dir / "c39_gradient_audit.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(report_dir / "c39_c17_checkpoint_audit.csv", index=False)
    passed = sum(bool(value) for _, value in checks)
    status = "C39_CMEQ_DIRECT_MULTI_SEED_AUTHORIZED" if passed == len(checks) else "C39_PATH_GATE_FAIL"
    payload = {
        "phase": "C39-CMEQ",
        "status": status,
        "passed": passed,
        "total": len(checks),
        "git_commit": git_output("rev-parse", "HEAD"),
        "branch": branch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_parameter_count_by_seed": trainable_counts,
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
    }
    (report_dir / "c39_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C39_CMEQ_DIRECT_MULTI_SEED_AUTHORIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
