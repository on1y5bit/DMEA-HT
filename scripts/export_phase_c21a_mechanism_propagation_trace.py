#!/usr/bin/env python3
"""Export the true C17 mechanism graph for the validation-only C21-A audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from phase_c21a_common import (
    ALL_EDGES,
    ATTENTION_EDGES,
    NODE_NAMES,
    RELATION_EDGES,
    build_loader,
    c17_forward_variant,
    flatten_tensor,
    load_config,
    load_model,
    load_rows,
    move_batch,
    read_validation_predictions,
    resolve_path,
    sha256_file,
)


SEEDS = (0, 42, 3407)
SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "raw_n_visits",
    "raw_n_images",
    "source_folder",
    "patient_id_encoding",
)
NODE_FIELDS = (
    "node_pre",
    "message_aggregate",
    "incoming_message_count",
    "incoming_message_sum",
    "incoming_message_mean",
    "incoming_message_norm",
    "node_after_update_before_norm",
    "node_after_norm",
    "node_valid",
)
EDGE_FIELDS = (
    "source_representation",
    "transformed_source",
    "raw_message",
    "message_norm",
    "edge_weight",
    "edge_gate",
    "effective_message",
)
TRACE_TENSOR_NAMES = (
    "mechanism_state",
    "mechanism_attention",
    "mechanism_pre",
    "mechanism_message_aggregate",
    "mechanism_after_norm",
    "role_logits",
    "role_probs",
    "aggregate_support",
    "aggregate_opposition",
    "aggregate_uncertainty",
    "aggregate_conflict",
    "aggregate_reliability",
    "aggregate_conflict_score",
    "aggregate_modality_weights",
    "aggregate_strengths",
    "base_role_nodes",
    "base_role_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c17_formal_multiseed.yaml")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def shortcut_value(item: Mapping[str, Any], field: str) -> Any:
    if field == "patient_id_encoding":
        return "EXCLUDED_PATIENT_ID_ALIGNMENT_ONLY"
    aliases = {
        "image_padding_count": ("image_padding_count", "padding_count"),
        "used_images": ("used_images", "n_images"),
    }
    for key in aliases.get(field, (field,)):
        if key in item and item[key] not in (None, ""):
            return item[key]
    return ""


def numeric_or_string(values: Sequence[Any]) -> np.ndarray:
    converted: List[float] = []
    for value in values:
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            return np.asarray([str(value) for value in values], dtype="<U256")
    return np.asarray(converted, dtype=np.float32)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_batch_trace(result: Mapping[str, Any]) -> Dict[str, Any]:
    trace = result["trace"]
    outputs = result["outputs"]
    tensors: Dict[str, np.ndarray] = {}
    for node, fields in trace["nodes"].items():
        for field, value in fields.items():
            if value is not None:
                tensors[f"node__{node}__{field}"] = flatten_tensor(value)
    for edge, fields in trace["edges"].items():
        for field, value in fields.items():
            if value is not None:
                tensors[f"edge__{edge}__{field}"] = flatten_tensor(value)
    for name, value in trace["tensors"].items():
        tensors[f"tensor__{name}"] = flatten_tensor(value)
    for name in ("base_logit", "delta_logit", "logit", "prob", "base_prob"):
        tensors[f"scalar__{name}"] = flatten_tensor(outputs[name])
    return {
        "patient_id": [str(value) for value in result["batch_patient_id"]],
        "labels": result["labels"],
        "tensors": tensors,
        "shortcuts": result["shortcuts"],
    }


def edge_type(edge: str) -> str:
    if edge in ATTENTION_EDGES:
        return "attention"
    if edge in RELATION_EDGES[:6]:
        return "relation"
    return "context"


def static_inventory(observed: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in NODE_NAMES:
        for field in NODE_FIELDS:
            name = f"node__{node}__{field}"
            rows.append(
                {
                    "tensor": name,
                    "category": "node",
                    "available": True,
                    "source": "HTMechanismRelationLayer forward path",
                    "notes": "incoming_message_mean is a diagnostic mean; node_after_update_before_norm aliases the real aggregate because no residual update exists",
                    **observed.get(name, {}),
                }
            )
        rows.append(
            {
                "tensor": f"node__{node}__explicit_residual_update",
                "category": "node",
                "available": False,
                "source": "HTMechanismRelationLayer forward path",
                "notes": "unavailable: the real layer has relation transforms followed by LayerNorm, with no explicit residual node update",
            }
        )
    for edge in ALL_EDGES:
        for field in EDGE_FIELDS:
            name = f"edge__{edge}__{field}"
            available = field != "edge_weight" or edge in ATTENTION_EDGES
            note = ""
            if field == "raw_message":
                note = "the real graph has no separate message operator; raw_message equals transformed_source"
            elif field == "edge_weight" and edge not in ATTENTION_EDGES:
                note = "unavailable: relation/context edges have no independent learned scalar edge weight"
            elif field == "edge_weight":
                note = "MultiheadAttention output weight for the node-to-final mechanism edge"
            rows.append(
                {
                    "tensor": name,
                    "category": f"edge_{edge_type(edge)}",
                    "available": available,
                    "source": "HTMechanismRelationLayer forward path",
                    "notes": note,
                    **observed.get(name, {}),
                }
            )
    for name in TRACE_TENSOR_NAMES:
        key = f"tensor__{name}"
        rows.append(
            {
                "tensor": key,
                "category": "diagnostic_tensor",
                "available": True,
                "source": "C21-A read-only forward trace",
                "notes": "",
                **observed.get(key, {}),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    run_dir = resolve_path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    rows = load_rows(config, manifest=args.manifest, data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive_payloads: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}}
    observed: MutableMapping[str, Dict[str, Any]] = OrderedDict()
    reproduction_rows: List[Dict[str, Any]] = []
    environment_rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        checkpoint = run_dir / "checkpoints" / f"seed_{seed}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"C17 checkpoint missing: {checkpoint}")
        model = load_model(config, run_dir, seed, device)
        for split in ("train", "val"):
            loader = build_loader(config, rows, split, args.batch_size, args.num_workers)
            ids: List[str] = []
            labels: List[np.ndarray] = []
            batches: MutableMapping[str, List[np.ndarray]] = {}
            shortcuts: MutableMapping[str, List[Any]] = {field: [] for field in SHORTCUT_FIELDS}
            with torch.no_grad():
                for batch in loader:
                    moved = move_batch(batch, device)
                    result = c17_forward_variant(model, moved)
                    extracted = collect_batch_trace(
                        {
                            **result,
                            "batch_patient_id": moved["patient_id"],
                            "labels": flatten_tensor(moved["label"]).reshape(-1).astype(np.int64),
                            "shortcuts": list(batch.get("shortcuts", [])),
                        }
                    )
                    ids.extend(extracted["patient_id"])
                    labels.append(extracted["labels"])
                    for name, value in extracted["tensors"].items():
                        batches.setdefault(name, []).append(value)
                        observed.setdefault(name, {})[f"shape_{split}"] = list(value.shape)
                    for item in extracted["shortcuts"]:
                        for field in SHORTCUT_FIELDS:
                            shortcuts[field].append(shortcut_value(item, field))
            split_labels = np.concatenate(labels).astype(np.int64, copy=False) if labels else np.empty(0, dtype=np.int64)
            split_tensors = {
                name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
                for name, values in batches.items()
            }
            for name, value in split_tensors.items():
                observed.setdefault(name, {})[f"shape_{split}"] = list(value.shape)
            split_payload: Dict[str, np.ndarray] = {
                f"seed_{seed}__patient_id": np.asarray(ids, dtype="<U256"),
                f"seed_{seed}__labels": split_labels,
            }
            split_payload.update({f"seed_{seed}__{name}": value for name, value in split_tensors.items()})
            split_payload.update(
                {
                    f"seed_{seed}__shortcut__{field}": numeric_or_string(values)
                    for field, values in shortcuts.items()
                }
            )
            archive_payloads[split].update(split_payload)

        val_path = run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"
        saved = read_validation_predictions(val_path)
        export_ids = archive_payloads["val"][f"seed_{seed}__patient_id"].astype(str)
        labels = archive_payloads["val"][f"seed_{seed}__labels"]
        id_match = len(saved) == len(export_ids) and set(saved) == set(export_ids)
        label_match = id_match and all(int(saved[pid]["label"]) == int(label) for pid, label in zip(export_ids, labels))
        exported = archive_payloads["val"][f"seed_{seed}__scalar__prob"].reshape(-1).astype(np.float64)
        saved_prob = np.asarray([saved[pid]["prob"] for pid in export_ids], dtype=np.float64) if id_match else np.empty(0)
        differences = np.abs(exported - saved_prob) if id_match else np.asarray([])
        reproduction_rows.append(
            {
                "seed": seed,
                "validation_patient_count": len(export_ids),
                "saved_prediction_count": len(saved),
                "patient_id_exact_match": id_match,
                "label_exact_match": label_match,
                "max_abs_prob_diff": float(differences.max()) if differences.size else float("nan"),
                "mean_abs_prob_diff": float(differences.mean()) if differences.size else float("nan"),
                "pass": bool(id_match and label_match and differences.size and differences.max() <= 1e-8 and differences.mean() <= 1e-9),
            }
        )
        environment_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "train_count": int(len(archive_payloads["train"][f"seed_{seed}__patient_id"])),
                "val_count": int(len(export_ids)),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for split in ("train", "val"):
        np.savez_compressed(output_dir / f"c21a_trace_{split}.npz", **archive_payloads[split])
    write_csv(output_dir / "c21a_reproduction_check_by_seed.csv", reproduction_rows)

    inventory_rows = static_inventory(observed)
    write_csv(output_dir / "c21a_trace_tensor_inventory.csv", inventory_rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        commit = "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    c20_paths = [
        output_dir.parent / "phase_c20_dema" / "c20_internal_representations_train.npz",
        output_dir.parent / "phase_c20_dema" / "c20_internal_representations_val.npz",
    ]
    lines = [
        "# C21-A Environment And Input Inventory",
        "",
        "- phase: C21-A mechanism propagation responsibility audit",
        f"- repository commit: `{commit}`",
        f"- runtime device: `{gpu}`",
        f"- config: `{config_path}`",
        f"- manifest: `{config['project']['manifest']}`",
        f"- data root: `{config['project']['data_root']}`",
        f"- C17 run directory: `{run_dir}`",
        "- seeds: `[0, 42, 3407]`",
        "- execution mode: `eval()` plus `torch.no_grad()`; no optimizer, backward, parameter update, or training loader was used",
        "- splits read: `train` and `val` only; test data and test prediction files were not read",
        "- intervention scope: inference-only graph traces and edge/node bypasses; labels, manifests, splits, and task definition are unchanged",
        "",
        "## Checkpoints",
        "",
        "| seed | checkpoint | sha256 | train n | val n |",
        "|---:|---|---|---:|---:|",
    ]
    for item in environment_rows:
        lines.append(
            f"| {item['seed']} | `{item['checkpoint']}` | `{item['checkpoint_sha256']}` | {item['train_count']} | {item['val_count']} |"
        )
    lines.extend(["", "## C20 Inputs", ""])
    for path in c20_paths:
        if path.exists():
            lines.append(f"- `{path}` sha256 `{sha256_file(path)}`; server-only prior audit artifact")
        else:
            lines.append(f"- `{path}` unavailable at export time")
    lines.extend(
        [
            "",
            "## Graph Contract",
            "",
            "- relation edges are the eight `HTMechanismRelationLayer.relations` paths present in source code",
            "- M1 receives image morphology and text morphology messages; M2-M5 receive their named single relation message",
            "- five mechanism states feed the final `MultiheadAttention`; text-global and bio-other are additive context edges",
            "- independent relation edge weights and explicit residual node updates are unavailable in the real implementation and are recorded as such in the tensor inventory",
            "- `raw_message` is retained as an alias of `transformed_source` because there is no separate message operator",
        ]
    )
    (output_dir / "c21a_environment_and_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = all(bool(row["pass"]) for row in reproduction_rows)
    report_lines = [
        "# C21-A Reproduction Check",
        "",
        f"- overall: `{'PASS' if passed else 'C21A_REPRODUCTION_GATE_FAIL'}`",
        "- comparison: C21-A trace forward probabilities versus saved C17 validation prediction CSVs by patient ID",
        "- thresholds: max absolute probability difference <= 1e-8; mean absolute probability difference <= 1e-9",
        "- test predictions were not read",
        "",
        "| seed | IDs | labels | max abs diff | mean abs diff | pass |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in reproduction_rows:
        report_lines.append(
            f"| {row['seed']} | {row['patient_id_exact_match']} | {row['label_exact_match']} | {row['max_abs_prob_diff']:.12g} | {row['mean_abs_prob_diff']:.12g} | {row['pass']} |"
        )
    (output_dir / "c21a_reproduction_check_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"reproduction_pass": passed, "output_dir": str(output_dir), "device": gpu}, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
