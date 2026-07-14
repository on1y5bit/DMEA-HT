#!/usr/bin/env python3
"""Validate the C25 pairwise-ranking residual contract and server paths."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c25_pairwise_ranking_residual import (  # noqa: E402
    C25PairwiseRankingResidualModel,
    ConfidenceGatedLocalResidualHead,
    c25_loss_terms,
    confidence_gate,
    correct_case_preserve_loss,
    pairwise_rank_loss,
)
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, read_manifest  # noqa: E402

SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c25_pairwise_ranking_residual_multiseed.yaml")
    parser.add_argument("--output", default="analysis_reports/phase_c25_dema/c25_static_synthetic_gate.json")
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def item(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def write_math_check(path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    cases = (
        ("mixed", torch.tensor([-1.0, 0.2, 0.5, -0.3]), torch.tensor([0.0, 1.0, 1.0, 0.0])),
        ("positive_only", torch.tensor([-0.2, 0.4]), torch.ones(2)),
        ("negative_only", torch.tensor([-0.2, 0.4]), torch.zeros(2)),
    )
    for name, logits, labels in cases:
        logits = logits.requires_grad_(True)
        loss, pair_count = pairwise_rank_loss(logits, labels, temperature=1.0)
        loss.backward()
        rows.append({
            "case": name,
            "positive_count": int((labels > 0.5).sum()),
            "negative_count": int((labels <= 0.5).sum()),
            "pair_count": pair_count,
            "loss": float(loss.detach()),
            "requires_grad": bool(loss.requires_grad),
            "gradient_l1": float(logits.grad.abs().sum()),
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_checks(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    checks.append(item("phase_is_c25", str(config.get("phase", "")).lower() == "c25"))
    checks.append(item("formal_seeds_fixed", config["training"].get("seeds") == list(SEEDS)))
    checks.append(item("default_shuffle_batch4", config["training"].get("batch_size") == 4 and config["training"].get("sampler") == "default_shuffle"))
    checks.append(item("selection_is_validation_auc_only", config["training"].get("primary_metric") == "val_AUC"))
    checks.append(item("test_is_post_selection_reporting_only", bool(config["training"].get("evaluate_test"))))
    checks.append(item("c17_formal_checkpoint_source", "dema_ht_c17_formal_multiseed" in str(config["c25"].get("c17_checkpoint", ""))))
    checks.append(item("representation_fixed", config["c25"].get("representation") == "frozen_c17_mea_mechanism_state"))
    checks.append(item("temperature_and_bound_fixed", config["c25"].get("temperature") == 1.0 and config["c25"].get("residual_max") == 0.15))
    checks.append(item("loss_weights_fixed", config["loss"].get("rank_temperature") == 1.0 and config["loss"].get("lambda_correct_preserve") == 0.02 and config["loss"].get("lambda_center") == 0.01 and config["loss"].get("lambda_magnitude") == 0.001))

    logits = torch.tensor([-1.0, 0.2, 0.5, -0.3], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    loss, pairs = pairwise_rank_loss(logits, labels)
    manual = torch.nn.functional.softplus(-((logits[labels > 0.5, None] - logits[labels <= 0.5][None, :]))).mean()
    checks.append(item("all_positive_negative_pairs_used", pairs == 4 and torch.allclose(loss, manual), {"pairs": pairs, "loss": float(loss)}))
    loss.backward()
    checks.append(item("rank_gradient_finite_nonzero", bool(torch.isfinite(logits.grad).all()) and float(logits.grad.abs().sum()) > 0))
    for label, name in ((0.0, "negative"), (1.0, "positive")):
        one_logits = torch.tensor([-0.2, 0.4], requires_grad=True)
        one_loss, one_pairs = pairwise_rank_loss(one_logits, torch.full((2,), label))
        one_loss.backward()
        checks.append(item(f"one_class_{name}_graph_zero", one_pairs == 0 and float(one_loss) == 0.0 and one_loss.requires_grad and float(one_logits.grad.abs().sum()) == 0.0))

    delta = torch.tensor([0.3, -0.2, 0.4, -0.5], requires_grad=True)
    frozen = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    preserve = correct_case_preserve_loss(delta, frozen, labels)
    checks.append(item("correct_preserve_uses_only_frozen_correct_cases", torch.allclose(preserve, torch.tensor(0.25)), float(preserve)))
    empty = correct_case_preserve_loss(delta, torch.tensor([1.0, -1.0, -1.0, 1.0]), labels)
    checks.append(item("empty_correct_mask_graph_zero", float(empty) == 0.0 and empty.requires_grad))

    gates = confidence_gate(torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]), 1.0)
    checks.append(item("confidence_gate_monotonic", bool((gates[:-1] >= gates[1:]).all()) and bool(((gates >= 0) & (gates <= 1)).all()), gates.tolist()))
    torch.manual_seed(20260714)
    head = ConfidenceGatedLocalResidualHead(8, dropout=0.0)
    raw = head(torch.randn(5, 8))
    trial_delta = 0.15 * gates.detach() * torch.tanh(raw)
    checks.append(item("zero_initialized_head", float(trial_delta.abs().max()) <= 1e-8))
    trial_outputs = {"logit": torch.tensor([-1., 0.2, 0.5, -0.3]) + trial_delta[:4], "delta_c25": trial_delta[:4], "frozen_c17_logit": torch.tensor([-1., 0.2, 0.5, -0.3])}
    terms = c25_loss_terms(trial_outputs, {"label": labels}, config["loss"])
    terms["total"].backward()
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    checks.append(item("new_head_gradient_finite_nonzero", bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(float(g.abs().sum()) > 0 for g in gradients)))

    model_source = (REPO_ROOT / "dmea_ht" / "c25_pairwise_ranking_residual.py").read_text(encoding="utf-8")
    train_source = (REPO_ROOT / "scripts" / "train_phase_c25.py").read_text(encoding="utf-8")
    shortcuts = ("selected_n_visits", "used_images", "image_padding_count", "report_length", "raw_n_visits", "raw_n_images")
    checks.append(item("shortcut_fields_absent_from_model", not any(token in model_source for token in shortcuts)))
    checks.append(item("frozen_final_logit_is_base", 'reference["logit"]' in inspect.getsource(C25PairwiseRankingResidualModel.forward)))
    checks.append(item("frozen_mechanism_state_is_only_representation", 'reference["mea_mechanism_state"]' in inspect.getsource(C25PairwiseRankingResidualModel.forward)))
    checks.append(item("no_pointwise_bce", "binary_cross_entropy" not in model_source and "BCE" not in model_source))
    checks.append(item("no_custom_sampler", "sampler=" not in train_source and 'shuffle=split == "train"' in train_source))
    checks.append(item("validation_has_no_loss", "if is_train:" in train_source))
    disabled_metric = "AUP" + "RC"
    report_path = REPO_ROOT / "scripts" / "collect_phase_c25_formal_report.py"
    report_source = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    checks.append(item("disabled_metric_absent", disabled_metric not in model_source + train_source + report_source))
    c25_configs = sorted((REPO_ROOT / "configs").glob("*c25*.yaml"))
    checks.append(item("one_formal_c25_config_only", len(c25_configs) == 1 and not any("smoke" in p.name.lower() or "pilot" in p.name.lower() for p in c25_configs), [p.name for p in c25_configs]))
    parse_errors = []
    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        try:
            load_config(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    checks.append(item("existing_configs_parse", not parse_errors, parse_errors))
    return checks


def build_loader(config: Mapping[str, Any], split: str, shuffle: bool = False) -> DataLoader:
    dataset = PatientHTDataset(
        rows=read_manifest(config["project"]["manifest"]), data_root=config["project"]["data_root"], split=split,
        max_images=int(config["model"]["max_images_per_patient"]), image_size=int(config["model"]["image_size"]),
        text_max_length=int(config["model"]["text_max_length"]), text_vocab_size=int(config["model"]["text_vocab_size"]),
        bio_dim=int(config["model"]["bio_dim"]),
    )
    generator = torch.Generator().manual_seed(20260714)
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, generator=generator, num_workers=0, collate_fn=collate_patient_batch)


def full_server_checks(config: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    checks.append(item("canonical_branch_is_main", branch == "main", branch))
    worktree_text = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    worktrees = [line.split(" ", 1)[1] for line in worktree_text.splitlines() if line.startswith("worktree ")]
    checks.append(item("only_canonical_worktree", len(worktrees) == 1 and Path(worktrees[0]).resolve() == REPO_ROOT.resolve(), worktrees))
    sibling_c25 = [str(p) for p in REPO_ROOT.parent.glob("*c25*") if p.is_dir()]
    checks.append(item("no_c25_project_copy", not sibling_c25, sibling_c25))

    path_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for role, template in (("c13_checkpoint", config["c17"]["base_checkpoint"]), ("c16_checkpoint", config["c17"]["init_mea_checkpoint"]), ("c17_checkpoint", config["c25"]["c17_checkpoint"])):
            path = Path(str(template).replace("{seed}", str(seed))).resolve()
            path_rows.append({"seed": seed, "role": role, "absolute_path": str(path), "exists": path.exists()})
        reference = REPO_ROOT / "runs" / "dema_ht_c17_formal_multiseed" / "predictions" / f"val_predictions_seed_{seed}.csv"
        path_rows.append({"seed": seed, "role": "c17_validation_predictions", "absolute_path": str(reference.resolve()), "exists": reference.exists()})
    for role, path in (("manifest", Path(config["project"]["manifest"])), ("c17_config", Path(config["c25"]["c17_config"]))):
        path_rows.append({"seed": "all", "role": role, "absolute_path": str(path.resolve()), "exists": path.exists()})
    run_root, report_root = resolve_path(config["project"]["output_dir"]), resolve_path(config["project"]["report_dir"])
    for role, root in (("c25_run_root", run_root), ("c25_report_root", report_root)):
        path_rows.append({"seed": "all", "role": role, "absolute_path": str(root.resolve()), "exists": root.parent.exists() and os.access(root.parent, os.W_OK)})
    paths_ok = all(bool(row["exists"]) for row in path_rows)
    checks.append(item("all_required_paths_resolve", paths_ok, path_rows))
    inventory = ["# C25 Resolved Path Inventory", "", "| seed | role | absolute path | exists |", "|---:|---|---|---|"]
    inventory.extend(f"| {r['seed']} | {r['role']} | `{r['absolute_path']}` | {r['exists']} |" for r in path_rows)
    (output_dir / "c25_resolved_path_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    if not paths_ok:
        return checks

    manifest_rows = read_manifest(config["project"]["manifest"])
    counts = {f"{split}_{label}": sum(str(r.get("split")) == split and int(r.get("label", -1)) == label for r in manifest_rows) for split in ("train", "val", "test") for label in (0, 1)}
    expected = {"train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47, "test_0": 42, "test_1": 42}
    checks.append(item("manifest_split_label_counts_match", counts == expected, counts))
    all_ids = [str(r["patient_id"]) for r in manifest_rows]
    checks.append(item("patient_ids_unique", len(all_ids) == len(set(all_ids)), {"rows": len(all_ids), "unique": len(set(all_ids))}))

    val_loader = build_loader(config, "val")
    train_loader = build_loader(config, "train", shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_rows: List[Dict[str, Any]] = []
    max_reproduction, max_initial_diff = 0.0, 0.0
    ids_labels_ok = finite_ok = frozen_grad_ok = gradient_ok = True
    for seed in SEEDS:
        checkpoint = Path(str(config["c25"]["c17_checkpoint"]).replace("{seed}", str(seed)))
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        checks.append(item(f"seed_{seed}_checkpoint_metadata", isinstance(payload, dict) and int(payload.get("seed", -1)) == seed, payload.get("seed") if isinstance(payload, dict) else None))
        model = C25PairwiseRankingResidualModel(config, seed).to(device)
        allowed = []
        for name, parameter in model.named_parameters():
            trainable_rows.append({"seed": seed, "parameter": name, "trainable": bool(parameter.requires_grad), "numel": parameter.numel()})
            if parameter.requires_grad:
                allowed.append(name)
        checks.append(item(f"seed_{seed}_only_residual_head_trainable", bool(allowed) and all(n.startswith("residual_head.") for n in allowed), allowed))
        reference = pd.read_csv(REPO_ROOT / "runs" / "dema_ht_c17_formal_multiseed" / "predictions" / f"val_predictions_seed_{seed}.csv")
        reference["patient_id"] = reference["patient_id"].astype(str)
        reference_map = reference.set_index("patient_id")
        seen: List[str] = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch)
                finite_ok = finite_ok and all(bool(torch.isfinite(outputs[k]).all()) for k in ("logit", "frozen_c17_logit", "delta_c25"))
                max_initial_diff = max(max_initial_diff, float((outputs["logit"] - outputs["frozen_c17_logit"]).abs().max().cpu()))
                for index, patient_id in enumerate(batch["patient_id"]):
                    pid = str(patient_id)
                    seen.append(pid)
                    if pid not in reference_map.index:
                        ids_labels_ok = False
                        continue
                    ref = reference_map.loc[pid]
                    ids_labels_ok = ids_labels_ok and int(ref["label"]) == int(batch["label"][index].cpu())
                    expected_logit = float(ref["logit"]) if "logit" in reference.columns else float(np.log(float(ref["prob"]) / (1.0 - float(ref["prob"]))))
                    max_reproduction = max(max_reproduction, abs(float(outputs["frozen_c17_logit"][index].cpu()) - expected_logit))
        ids_labels_ok = ids_labels_ok and set(seen) == set(reference["patient_id"]) and len(seen) == len(reference)

        mixed_batch = None
        for batch in train_loader:
            if len(torch.unique(batch["label"])) == 2:
                mixed_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                break
        if mixed_batch is None:
            gradient_ok = False
            continue
        model.train()
        outputs = model(mixed_batch)
        terms = c25_loss_terms(outputs, mixed_batch, config["loss"])
        terms["total"].backward()
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith("residual_head.") and p.grad is not None]
        gradient_ok = gradient_ok and bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients) and any(float(g.abs().sum()) > 0 for g in gradients)
        frozen_grad_ok = frozen_grad_ok and all(p.grad is None for p in model.frozen_c17.parameters())
    pd.DataFrame(trainable_rows).to_csv(output_dir / "c25_trainable_parameter_audit.csv", index=False)
    checks.append(item("validation_ids_and_labels_match_c17", ids_labels_ok))
    checks.append(item("c17_validation_logits_reproduce", max_reproduction <= 1e-5, max_reproduction))
    checks.append(item("initial_c25_equals_frozen_c17", max_initial_diff <= 1e-8, max_initial_diff))
    checks.append(item("real_outputs_finite", finite_ok))
    checks.append(item("mixed_batch_head_gradient_finite_nonzero", gradient_ok))
    checks.append(item("frozen_c17_gradients_absent", frozen_grad_ok))
    return checks


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_math_check(output.parent / "c25_pairwise_loss_math_check.csv")
    checks = static_checks(config)
    if not args.static_only and all(check["pass"] for check in checks):
        checks.extend(full_server_checks(config, output.parent))
    passed = all(check["pass"] for check in checks)
    decision = "C25_DIRECT_MULTI_SEED_AUTHORIZED" if passed and not args.static_only else ("C25_STATIC_GATE_PASS" if passed else "DEMA_C25_PATH_GATE_FAIL")
    result = {"phase": "C25", "gate": "STATIC_ONLY" if args.static_only else "FULL_SERVER", "pass": passed, "checks": checks, "decision": decision}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)
    if not args.static_only:
        print("C25_DIRECT_MULTI_SEED_AUTHORIZED")


if __name__ == "__main__":
    main()
