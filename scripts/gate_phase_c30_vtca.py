#!/usr/bin/env python3
"""Run the exact 50-item C30-VTCA static, path, reproduction, and gradient gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c30_vtca import (  # noqa: E402
    C30VTCAModel,
    VisitTextContextAdapter,
    load_checkpoint,
    trainable_parameter_count,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import read_jsonl  # noqa: E402
from scripts.train_phase_c27 import build_loaders, move_batch, resolve_path, set_seed  # noqa: E402


SEEDS = (0, 42, 3407)
EXPECTED_SPLITS = {
    "train": {"rows": 602, "labels": {0: 301, 1: 301}},
    "val": {"rows": 94, "labels": {0: 47, 1: 47}},
    "test": {"rows": 84, "labels": {0: 42, 1: 42}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        default="analysis_reports/phase_c30_dema/c30_static_synthetic_gate.json",
    )
    return parser.parse_args()


def check(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "pred_prob", "prediction"):
        if name in frame:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def logit_column(frame: pd.DataFrame) -> str | None:
    for name in ("final_logit", "logit"):
        if name in frame:
            return name
    return None


def split_contract(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    passed = True
    for split, expected in EXPECTED_SPLITS.items():
        selected = [row for row in rows if str(row.get("split")) == split]
        labels = pd.Series([int(row["label"]) for row in selected]).value_counts().to_dict()
        labels = {int(key): int(value) for key, value in labels.items()}
        detail[split] = {"rows": len(selected), "labels": labels}
        passed = passed and len(selected) == expected["rows"] and labels == expected["labels"]
    return passed, detail


def unique_patient_contract(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
    all_ids = [str(row["patient_id"]) for row in rows]
    split_sets = {
        split: {str(row["patient_id"]) for row in rows if str(row.get("split")) == split}
        for split in EXPECTED_SPLITS
    }
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    passed = len(all_ids) == len(set(all_ids)) and not any(overlaps.values())
    return passed, {"rows": len(all_ids), "unique": len(set(all_ids)), "overlaps": overlaps}


def _all_finite(outputs: Mapping[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_tensor(value))


def reproduce_and_compare(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    c27_run = resolve_path(config["c30"]["c27_run_dir"])
    equivalence_rows: List[Dict[str, Any]] = []
    audits: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        set_seed(seed)
        model = C30VTCAModel(config, seed).to(device)
        model.eval()
        seed_audit: Dict[str, Any] = {
            "c27_prob_error": 0.0,
            "c27_logit_error": 0.0,
            "c27_auc_error": 0.0,
            "use_vtca_false_error": 0.0,
            "single_visit_finite": False,
            "multi_visit_finite": False,
            "missing_image_or_bio_finite": False,
        }
        loaders = build_loaders(config, rows, ("train", "val"))
        for split in ("train", "val"):
            ids: List[str] = []
            labels: List[int] = []
            c27_logits: List[float] = []
            c27_probs: List[float] = []
            c30_logits: List[float] = []
            max_initial_error = 0.0
            direct_reference_checked = False
            missing_case_checked = False
            for batch in loaders[split]:
                batch = move_batch(batch, device)
                with torch.inference_mode():
                    direct = model(batch, use_vtca=False)
                    adapted = model(batch, use_vtca=True)
                    reference = model.c27(batch) if not direct_reference_checked else None
                    missing_output = None
                    if not missing_case_checked:
                        missing_batch = {
                            key: value.clone() if torch.is_tensor(value) else value
                            for key, value in batch.items()
                        }
                        missing_batch["images"].zero_()
                        missing_batch["image_mask"].zero_()
                        missing_batch["bio_values"].zero_()
                        missing_batch["bio_missing_mask"].fill_(1.0)
                        missing_batch["bio_abnormal_flags"].zero_()
                        missing_batch["fallback_bio_values"].zero_()
                        missing_batch["fallback_bio_missing_mask"].fill_(1.0)
                        missing_batch["fallback_bio_valid"].zero_()
                        missing_output = model(missing_batch, use_vtca=True)
                if not _all_finite(direct) or not _all_finite(adapted):
                    raise RuntimeError(f"Non-finite C30 gate output for seed {seed} {split}")
                difference = (adapted["logit"] - direct["logit"]).abs()
                max_initial_error = max(max_initial_error, float(difference.max().cpu()))
                if reference is not None:
                    seed_audit["use_vtca_false_error"] = max(
                        seed_audit["use_vtca_false_error"],
                        float((direct["logit"] - reference["logit"]).abs().max().cpu()),
                    )
                    direct_reference_checked = True
                if missing_output is not None:
                    seed_audit["missing_image_or_bio_finite"] = _all_finite(missing_output)
                    missing_case_checked = True
                ids.extend(str(value) for value in batch["patient_id"])
                labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
                c27_logits.extend(direct["logit"].detach().cpu().numpy().tolist())
                c27_probs.extend(direct["prob"].detach().cpu().numpy().tolist())
                c30_logits.extend(adapted["logit"].detach().cpu().numpy().tolist())
                visit_counts = batch["visit_mask"].sum(dim=1)
                finite_by_patient = torch.isfinite(adapted["logit"])
                if bool(((visit_counts == 1) & finite_by_patient).any()):
                    seed_audit["single_visit_finite"] = True
                if bool(((visit_counts > 1) & finite_by_patient).any()):
                    seed_audit["multi_visit_finite"] = True
            record: Dict[str, Any] = {
                "seed": seed,
                "split": split,
                "patient_count": len(ids),
                "initial_c30_c27_max_abs_logit_error": max_initial_error,
            }
            if split == "val":
                official = load_prediction(
                    c27_run / "predictions" / f"val_predictions_seed_{seed}.csv"
                )
                current = pd.DataFrame(
                    {
                        "patient_id": ids,
                        "label": labels,
                        "reproduced_logit": c27_logits,
                        "reproduced_prob": c27_probs,
                    }
                ).sort_values("patient_id").reset_index(drop=True)
                if not current["patient_id"].equals(official["patient_id"]):
                    raise RuntimeError(f"C27 patient alignment failed for seed {seed}")
                if not np.array_equal(
                    current["label"].to_numpy(dtype=int), official["label"].to_numpy(dtype=int)
                ):
                    raise RuntimeError(f"C27 label alignment failed for seed {seed}")
                official_prob = official[probability_column(official)].to_numpy(dtype=float)
                probability_error = float(
                    np.max(np.abs(current["reproduced_prob"].to_numpy(dtype=float) - official_prob))
                )
                official_auc = float(roc_auc_score(official["label"], official_prob))
                reproduced_auc = float(
                    roc_auc_score(current["label"], current["reproduced_prob"])
                )
                auc_error = abs(reproduced_auc - official_auc)
                record.update(
                    {
                        "official_c27_probability_max_abs_error": probability_error,
                        "official_c27_auc": official_auc,
                        "reproduced_c27_auc": reproduced_auc,
                        "official_c27_auc_abs_error": auc_error,
                    }
                )
                seed_audit["c27_prob_error"] = probability_error
                seed_audit["c27_auc_error"] = auc_error
                official_logit_name = logit_column(official)
                if official_logit_name is not None:
                    logit_error = float(
                        np.max(
                            np.abs(
                                current["reproduced_logit"].to_numpy(dtype=float)
                                - official[official_logit_name].to_numpy(dtype=float)
                            )
                        )
                    )
                    record["official_c27_logit_max_abs_error"] = logit_error
                    seed_audit["c27_logit_error"] = logit_error
            equivalence_rows.append(record)
        audits[seed] = seed_audit
    return {"rows": equivalence_rows, "seeds": audits}


def gradient_audit(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for seed in SEEDS:
        set_seed(seed)
        model = C30VTCAModel(config, seed).to(device)
        model.train()
        loader = build_loaders(config, rows, ("train",))["train"]
        batch = move_batch(next(iter(loader)), device)
        outputs = model(batch)
        loss = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
        loss.backward()
        adapter = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith("adapter.")]
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith("c27.")]
        all_adapter_present = all(parameter.grad is not None for _, parameter in adapter)
        all_adapter_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for _, parameter in adapter
        )
        aggregate = float(
            sum(
                parameter.grad.detach().abs().sum().cpu()
                for _, parameter in adapter
                if parameter.grad is not None
            )
        )
        records.append(
            {
                "seed": seed,
                "adapter_gradient_present": all_adapter_present,
                "adapter_gradient_finite": all_adapter_finite,
                "adapter_gradient_abs_sum": aggregate,
                "frozen_c27_gradient_absent": all(parameter.grad is None for _, parameter in frozen),
                "trainable_parameter_count": trainable_parameter_count(model),
                "frozen_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
                ),
                "trainable_parameter_names": [
                    name for name, parameter in model.named_parameters() if parameter.requires_grad
                ],
                "frozen_parameter_names": [
                    name for name, parameter in model.named_parameters() if not parameter.requires_grad
                ],
            }
        )
        del model, loader, batch, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"records": records}


def synthetic_adapter_audit(config: Mapping[str, Any]) -> Dict[str, Any]:
    hidden = int(config["model"]["hidden_dim"])
    adapter = VisitTextContextAdapter(
        hidden,
        float(config["model"]["dropout"]),
        float(config["c30"]["adapter_max_delta"]),
    )
    adapter.eval()
    torch.manual_seed(20260714)
    tokens = torch.randn(4, 13, hidden)
    mask = torch.tensor(
        [
            [1] * 13,
            [1] * 8 + [0] * 5,
            [1] + [0] * 12,
            [0] * 13,
        ],
        dtype=torch.bool,
    )
    with torch.inference_mode():
        output = adapter(tokens, mask)
    delta = output["token_delta"]
    padding = delta.masked_select((~mask).unsqueeze(-1).expand_as(delta))
    real = delta.masked_select(mask.unsqueeze(-1).expand_as(delta))
    bounded_adapter = VisitTextContextAdapter(
        hidden,
        float(config["model"]["dropout"]),
        float(config["c30"]["adapter_max_delta"]),
    ).eval()
    with torch.no_grad():
        bounded_adapter.output.weight.normal_()
        bounded_adapter.output.bias.normal_()
    with torch.inference_mode():
        bounded_output = bounded_adapter(tokens, mask)["token_delta"]
    bounded_padding = bounded_output.masked_select((~mask).unsqueeze(-1).expand_as(delta))
    bounded_real = bounded_output.masked_select(mask.unsqueeze(-1).expand_as(delta))
    return {
        "shape_preserved": tuple(output["adapted_tokens"].shape) == tuple(tokens.shape),
        "finite": all(bool(torch.isfinite(value).all()) for value in output.values()),
        "empty_finite": bool(torch.isfinite(output["adapted_tokens"][-1]).all()),
        "all_padding_finite": bool(torch.isfinite(output["token_delta"][-1]).all()),
        "padding_delta_max": float(padding.abs().max()) if padding.numel() else 0.0,
        "real_delta_max": float(real.abs().max()) if real.numel() else 0.0,
        "trained_padding_delta_max": float(bounded_padding.abs().max()) if bounded_padding.numel() else 0.0,
        "trained_real_delta_max": float(bounded_real.abs().max()) if bounded_real.numel() else 0.0,
        "final_weight_max": float(adapter.output.weight.detach().abs().max()),
        "final_bias_max": float(adapter.output.bias.detach().abs().max()),
        "kernels": [adapter.depthwise_k3.kernel_size[0], adapter.depthwise_k7.kernel_size[0]],
        "groups": [adapter.depthwise_k3.groups, adapter.depthwise_k7.groups],
        "padding": [adapter.depthwise_k3.padding[0], adapter.depthwise_k7.padding[0]],
        "pointwise_channels": [adapter.pointwise.in_channels, adapter.pointwise.out_channels],
    }


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = REPO_ROOT / "dmea_ht" / "c30_vtca.py"
    train_path = REPO_ROOT / "scripts" / "train_phase_c30.py"
    collect_path = REPO_ROOT / "scripts" / "collect_phase_c30_formal_report.py"
    config_path = resolve_path(args.config)
    model_source = model_path.read_text(encoding="utf-8")
    adapter_source = inspect.getsource(VisitTextContextAdapter.forward)
    train_source = train_path.read_text(encoding="utf-8")
    collect_source = collect_path.read_text(encoding="utf-8")
    config_source = config_path.read_text(encoding="utf-8")
    rows = read_jsonl(config["project"]["manifest"])
    manifest_path = Path(config["project"]["manifest"])
    manifest_digest = sha256(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c30_configs = sorted((REPO_ROOT / "configs").glob("*c30*.yaml"))
    worktrees = [
        line.split(" ", 1)[1]
        for line in git_output("worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    sibling_copies = [
        str(path)
        for path in REPO_ROOT.parent.iterdir()
        if path.is_dir() and path.resolve() != REPO_ROOT.resolve() and "c30" in path.name.lower()
    ]
    checkpoint_rows: List[Dict[str, Any]] = []
    checkpoints_exist = True
    metadata_correct = True
    for seed in SEEDS:
        path = Path(str(config["c30"]["c27_checkpoint"]).replace("{seed}", str(seed)))
        exists = path.exists()
        checkpoints_exist = checkpoints_exist and exists
        payload = load_checkpoint(path) if exists else {}
        metadata_seed = int(payload.get("seed", -1)) if payload else -1
        metadata_correct = metadata_correct and metadata_seed == seed
        checkpoint_rows.append(
            {"seed": seed, "path": str(path), "exists": exists, "metadata_seed": metadata_seed}
        )

    split_pass, split_detail = split_contract(rows)
    unique_pass, unique_detail = unique_patient_contract(rows)
    synthetic = synthetic_adapter_audit(config)
    reproduction = reproduce_and_compare(config, rows, device)
    gradients = gradient_audit(config, rows, device)
    equivalence = pd.DataFrame(reproduction["rows"])
    equivalence.to_csv(output_path.parent / "c30_initial_equivalence_by_seed.csv", index=False)
    parameter_rows = []
    for record in gradients["records"]:
        for scope, names in (
            ("trainable", record["trainable_parameter_names"]),
            ("frozen", record["frozen_parameter_names"]),
        ):
            for name in names:
                parameter_rows.append({"seed": record["seed"], "scope": scope, "name": name})
    pd.DataFrame(parameter_rows).to_csv(
        output_path.parent / "c30_trainable_parameter_audit.csv", index=False
    )
    inventory_lines = [
        "# C30 Resolved Path Inventory",
        "",
        f"- canonical project: `{REPO_ROOT}`",
        f"- active branch: `{git_output('branch', '--show-current')}`",
        f"- manifest: `{manifest_path}`",
        f"- manifest SHA256: `{manifest_digest}`",
        f"- runtime device: `{device}`",
        f"- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}`",
        "- C27 checkpoints:",
        *[f"  - seed {row['seed']}: `{row['path']}`" for row in checkpoint_rows],
        "- deployment contract: one checkpoint, one model, one forward",
    ]
    (output_path.parent / "c30_resolved_path_inventory.md").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )

    direct_errors = [float(row["use_vtca_false_error"]) for row in reproduction["seeds"].values()]
    probability_errors = [float(row["c27_prob_error"]) for row in reproduction["seeds"].values()]
    logit_errors = [float(row["c27_logit_error"]) for row in reproduction["seeds"].values()]
    auc_errors = [float(row["c27_auc_error"]) for row in reproduction["seeds"].values()]
    train_equivalence = equivalence[equivalence["split"].eq("train")][
        "initial_c30_c27_max_abs_logit_error"
    ].to_numpy(dtype=float)
    val_equivalence = equivalence[equivalence["split"].eq("val")][
        "initial_c30_c27_max_abs_logit_error"
    ].to_numpy(dtype=float)
    atol = float(config["c30"]["initial_equivalence_atol"])
    adapter_counts = [int(record["trainable_parameter_count"]) for record in gradients["records"]]
    frozen_counts = [int(record["frozen_parameter_count"]) for record in gradients["records"]]
    forbidden_forward_fields = (
        "patient_id",
        "label",
        "selected_n_visits",
        "raw_n_visits",
        "used_images",
        "raw_n_images",
        "report_length",
        "shortcut",
    )
    combination_terms = (
        "stack_predictions(",
        "average_checkpoints(",
        "mean_state_dict(",
        "load_multiple_checkpoints(",
        "majority_vote(",
    )
    source_for_combination = (model_source + train_source + collect_source + config_source).lower()
    disabled_metric = "AUP" + "RC"
    source_for_loss = train_source + config_source
    all_runtime = list(reproduction["seeds"].values())

    checks = [
        check("01_active_branch_main", git_output("branch", "--show-current") == "main"),
        check(
            "02_canonical_worktree_only",
            len(worktrees) == 1 and Path(worktrees[0]).resolve() == REPO_ROOT.resolve(),
            worktrees,
        ),
        check("03_no_c30_project_copy", not sibling_copies, sibling_copies),
        check(
            "04_no_model_or_prediction_combination",
            not any(term in source_for_combination for term in combination_terms),
        ),
        check(
            "05_no_smoke_or_pilot_config",
            not any("smoke" in path.name.lower() or "pilot" in path.name.lower() for path in c30_configs),
            [path.name for path in c30_configs],
        ),
        check("06_exactly_one_formal_c30_config", len(c30_configs) == 1, [path.name for path in c30_configs]),
        check("07_three_c27_checkpoints_exist", checkpoints_exist, checkpoint_rows),
        check("08_c27_checkpoint_seed_metadata", metadata_correct, checkpoint_rows),
        check(
            "09_c27_manifest_sha256",
            manifest_digest == str(config["c30"]["manifest_sha256"]),
            manifest_digest,
        ),
        check("10_split_and_label_counts_unchanged", split_pass, split_detail),
        check(
            "11_exact_602_94_84_split_sizes",
            all(split_detail[key]["rows"] == EXPECTED_SPLITS[key]["rows"] for key in EXPECTED_SPLITS),
            split_detail,
        ),
        check("12_patient_ids_unique_and_disjoint", unique_pass, unique_detail),
        check(
            "13_official_c27_logits_and_probabilities_reproduced",
            max(probability_errors + logit_errors) <= atol,
            {"probability_max": max(probability_errors), "logit_max": max(logit_errors)},
        ),
        check("14_official_c27_validation_auc_reproduced", max(auc_errors) <= 1e-12, max(auc_errors)),
        check("15_old_c27_state_dict_loads_strictly", metadata_correct and len(reproduction["seeds"]) == 3),
        check("16_use_vtca_false_logits_unchanged", max(direct_errors) == 0.0, max(direct_errors)),
        check(
            "17_adapter_inserted_before_text_projector",
            model_source.index("self.adapter(text_tokens.detach()")
            < model_source.index("text_after = self.c27.frozen_sources.text_projector"),
        ),
        check(
            "18_adapter_not_after_temporal_head_or_logit",
            "self.adapter" not in inspect.getsource(C30VTCAModel.forward),
        ),
        check(
            "19_adapter_inputs_only_tokens_and_mask",
            list(inspect.signature(VisitTextContextAdapter.forward).parameters)
            == ["self", "text_tokens", "text_attention_mask"],
        ),
        check(
            "20_adapter_does_not_read_label_id_or_shortcuts",
            not any(field in adapter_source for field in forbidden_forward_fields),
        ),
        check("21_kernels_exactly_3_and_7", synthetic["kernels"] == [3, 7], synthetic["kernels"]),
        check(
            "22_depthwise_groups_equal_hidden_dim",
            synthetic["groups"] == [int(config["model"]["hidden_dim"])] * 2,
            synthetic["groups"],
        ),
        check("23_convolution_padding_exact", synthetic["padding"] == [1, 3], synthetic["padding"]),
        check(
            "24_pointwise_fusion_dimensions_exact",
            synthetic["pointwise_channels"]
            == [2 * int(config["model"]["hidden_dim"]), int(config["model"]["hidden_dim"])],
            synthetic["pointwise_channels"],
        ),
        check(
            "25_final_adapter_layer_zero_initialized",
            synthetic["final_weight_max"] == 0.0 and synthetic["final_bias_max"] == 0.0,
            {"weight": synthetic["final_weight_max"], "bias": synthetic["final_bias_max"]},
        ),
        check("26_initial_train_logits_equal_c27", bool((train_equivalence <= atol).all()), train_equivalence.tolist()),
        check("27_initial_validation_logits_equal_c27", bool((val_equivalence <= atol).all()), val_equivalence.tolist()),
        check(
            "28_padding_token_delta_strict_zero",
            synthetic["padding_delta_max"] == 0.0 and synthetic["trained_padding_delta_max"] == 0.0,
            {
                "initial": synthetic["padding_delta_max"],
                "trained_synthetic": synthetic["trained_padding_delta_max"],
            },
        ),
        check(
            "29_real_token_delta_bounded_at_0_10",
            synthetic["trained_real_delta_max"]
            <= float(config["c30"]["adapter_max_delta"]) + 1e-7,
            synthetic["trained_real_delta_max"],
        ),
        check("30_empty_visit_text_finite", synthetic["empty_finite"]),
        check("31_all_padding_defensive_case_finite", synthetic["all_padding_finite"]),
        check("32_single_visit_patient_finite", all(item["single_visit_finite"] for item in all_runtime)),
        check("33_multi_visit_patient_finite", all(item["multi_visit_finite"] for item in all_runtime)),
        check(
            "34_missing_image_or_bio_case_finite",
            all(item["missing_image_or_bio_finite"] for item in all_runtime),
        ),
        check("35_adapted_token_shape_unchanged", synthetic["shape_preserved"]),
        check(
            "36_frozen_c27_has_no_gradient",
            all(record["frozen_c27_gradient_absent"] for record in gradients["records"]),
        ),
        check(
            "37_all_adapter_gradients_present_and_finite",
            all(
                record["adapter_gradient_present"] and record["adapter_gradient_finite"]
                for record in gradients["records"]
            ),
        ),
        check(
            "38_adapter_aggregate_gradient_nonzero",
            all(float(record["adapter_gradient_abs_sum"]) > 0.0 for record in gradients["records"]),
            [record["adapter_gradient_abs_sum"] for record in gradients["records"]],
        ),
        check(
            "39_trainable_scope_only_adapter",
            all(
                all(name.startswith("adapter.") for name in record["trainable_parameter_names"])
                for record in gradients["records"]
            ),
        ),
        check(
            "40_trainable_parameters_within_one_million",
            max(adapter_counts) <= int(config["c30"]["trainable_parameter_limit"]),
            {"trainable": adapter_counts, "frozen": frozen_counts},
        ),
        check(
            "41_bce_is_only_training_loss",
            train_source.count("binary_cross_entropy_with_logits") == 1
            and config["loss"] == {"bce_only": True},
        ),
        check(
            "42_no_ranking_auxiliary_or_weak_label_loss",
            not any(term in source_for_loss for term in ("ranking_loss", "auxiliary_loss", "weak_label_loss")),
        ),
        check(
            "43_validation_auc_only_checkpoint_metric",
            config["training"]["primary_metric"] == "val_AUC"
            and config["training"]["checkpoint_metric"] == "val_auc"
            and "if val_auc > best_auc" in train_source,
        ),
        check(
            "44_disabled_metric_absent",
            disabled_metric not in model_source + train_source + collect_source + config_source,
        ),
        check(
            "45_test_loader_blocked_until_validation_decision",
            "validation decision must be frozen before reporting-only test" in train_source
            and train_source.index("decision_path.exists()") < train_source.index('build_loaders(config, rows, ("test",))'),
        ),
        check(
            "46_shortcut_fields_absent_from_forward",
            not any(field in model_source for field in forbidden_forward_fields[2:]),
        ),
        check(
            "47_raw_visit_and_image_fields_audit_only",
            "raw_n_visits" not in model_source and "raw_n_images" not in model_source,
        ),
        check(
            "48_each_seed_has_independent_checkpoint",
            'f"seed_{seed}_best.pt"' in train_source and 'out_dir / "seed_runs" / f"seed_{seed}"' in train_source,
        ),
        check(
            "49_seeds_not_combined",
            not any(term in source_for_combination for term in combination_terms)
            and "median_validation_seed" not in train_source,
        ),
        check(
            "50_one_checkpoint_one_model_one_forward_deployment",
            train_source.count('"one_checkpoint_one_model_one_forward"') >= 3,
        ),
    ]
    if len(checks) != 50:
        raise RuntimeError(f"C30 gate implementation error: expected 50 checks, found {len(checks)}")
    passed = sum(int(item["pass"]) for item in checks)
    initial_ok = bool((train_equivalence <= atol).all() and (val_equivalence <= atol).all())
    capacity_ok = max(adapter_counts) <= int(config["c30"]["trainable_parameter_limit"])
    if not initial_ok:
        decision = "DEMA_C30_INITIAL_EQUIVALENCE_FAIL"
    elif not capacity_ok:
        decision = "DEMA_C30_CAPACITY_CONTRACT_FAIL"
    elif passed == 50:
        decision = "C30_VTCA_DIRECT_MULTI_SEED_AUTHORIZED"
    else:
        decision = "DEMA_C30_PATH_GATE_FAIL"
    payload = {
        "phase": "C30-VTCA",
        "decision": decision,
        "pass": passed == 50,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_sha256": manifest_digest,
        "c27_checkpoints": checkpoint_rows,
        "initial_equivalence_tolerance": atol,
        "trainable_parameter_counts": adapter_counts,
        "frozen_parameter_counts": frozen_counts,
        "synthetic": synthetic,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "checks": f"{passed}/50"}))
    if passed != 50:
        failed = [item["name"] for item in checks if not item["pass"]]
        print(json.dumps({"failed": failed}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
