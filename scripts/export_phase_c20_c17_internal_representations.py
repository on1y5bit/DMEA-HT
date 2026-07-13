#!/usr/bin/env python3
"""Export frozen C17 internal representations for the validation-only C20 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c17_residual import C17ResidualModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import PatientHTDataset, collate_patient_batch, patient_split, read_manifest  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c17_formal_multiseed.yaml")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c20_dema")
    parser.add_argument("--run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"unsupported checkpoint payload: {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def stack_or_empty(values: Sequence[np.ndarray], width: int | None = None) -> np.ndarray:
    if values:
        return np.concatenate(values, axis=0).astype(np.float32, copy=False)
    return np.empty((0, int(width or 0)), dtype=np.float32)


def flatten_tensor(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 1:
        array = array[:, None]
    return array.reshape(array.shape[0], -1).astype(np.float32, copy=False)


def scalar(value: torch.Tensor) -> np.ndarray:
    return flatten_tensor(value)


def extract_internal(model: C17ResidualModel, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the frozen C17 forward path while retaining true intermediate tensors."""
    base = model.base_model
    encoded = base.encode_modalities(batch)
    base_outputs = base.forward_from_encoded(batch, encoded)

    representations: Dict[str, torch.Tensor] = {
        "raw_image_global": encoded["image_global"],
        "raw_text_global": encoded["text_global"],
        "raw_bio_global": encoded["bio_global"],
    }
    if hasattr(base, "evidence") and hasattr(base, "anchor"):
        evidence_tokens, evidence_scores, _role_loss = base.evidence(
            encoded["image_tokens"], encoded["text_tokens"], encoded["bio_tokens"]
        )
        representations["evidence_role_pooled"] = evidence_tokens
        representations["evidence_role_scores"] = torch.stack(
            [evidence_scores[role] for role in base.evidence.roles], dim=1
        )
        aux_outputs = base._auxiliary_outputs(
            encoded["image_global"],
            encoded["text_global"],
            encoded["text_tokens"],
            batch["report_attention_mask"],
        )
        token_parts = [encoded["image_tokens"], encoded["text_tokens"], encoded["bio_tokens"], evidence_tokens]
        if base.fuse_text_morphology_anchor and "text_morphology_anchor" in aux_outputs:
            token_parts.append(aux_outputs["text_morphology_anchor"].unsqueeze(1))
        representations["raw_patient_anchor"] = base.anchor(torch.cat(token_parts, dim=1))

    text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
    mea = model.mechanism_evidence_alignment
    mea_outputs = mea(
        image_tokens=encoded["image_tokens"],
        image_mask=batch["image_mask"],
        text_tokens=encoded["text_tokens"],
        text_attention_mask=batch["report_attention_mask"],
        bio_tokens=encoded["bio_tokens"],
        bio_missing_mask=batch["bio_missing_mask"],
        text_masks=text_masks,
    )

    # These projectors are evaluated again only to retain their named evidence nodes.
    image_evidence = mea.image(encoded["image_tokens"], batch["image_mask"])
    text_evidence = mea.text(encoded["text_tokens"], batch["report_attention_mask"], text_masks)
    bio_evidence = mea.bio(encoded["bio_tokens"], batch["bio_missing_mask"])
    representations.update(
        {
            "evidence_image_morphology": image_evidence["nodes"][:, 0],
            "evidence_text_support": text_evidence["nodes"][:, 0],
            "evidence_text_opposition": text_evidence["nodes"][:, 1],
            "evidence_text_uncertainty": text_evidence["nodes"][:, 2],
            "evidence_text_temporal": text_evidence["nodes"][:, 4],
            "evidence_bio_immune_observed": bio_evidence["nodes"][:, 1],
            "evidence_bio_function_observed": bio_evidence["nodes"][:, 2],
            "evidence_role_logits_per_evidence": mea_outputs["mea_role_logits"],
            "evidence_role_probabilities_per_evidence": mea_outputs["mea_role_probs"],
            "aggregate_support": mea_outputs["mea_support_state"],
            "aggregate_opposition": mea_outputs["mea_opposition_state"],
            "aggregate_uncertainty": mea_outputs["mea_uncertainty_state"],
            "aggregate_conflict": mea_outputs["mea_conflict_state"],
        }
    )

    mechanism_nodes = mea_outputs["mea_mechanism_nodes"]
    mechanism_names = ("morphology", "immune", "function", "opposition", "temporal")
    for index, name in enumerate(mechanism_names):
        representations[f"mechanism_{name}_node"] = mechanism_nodes[:, index]
    representations["mechanism_nodes_all"] = mechanism_nodes
    representations["mechanism_final_representation"] = mea_outputs["mea_mechanism_state"]
    representations.update(
        {
            "scalar_support_strength": mea_outputs["patient_support_strength"],
            "scalar_opposition_strength": mea_outputs["patient_opposition_strength"],
            "scalar_uncertainty_strength": mea_outputs["patient_uncertainty_strength"],
            "scalar_conflict_score": mea_outputs["patient_conflict_score"],
            "scalar_temporal_conflict_score": mea_outputs["evidence_temporal_conflict_score"],
            "scalar_morphology_alignment_cosine": mea_outputs["evidence_morphology_alignment_cosine"],
            "scalar_base_logit": base_outputs["logit"],
            "scalar_residual_logit": model.residual_head(model._mechanism_correction_features(mea_outputs))["delta_logit"],
        }
    )
    representations["scalar_final_logit"] = base_outputs["logit"] + representations["scalar_residual_logit"]
    representations["scalar_final_prob"] = torch.sigmoid(representations["scalar_final_logit"])

    return {
        "representations": {name: flatten_tensor(value) for name, value in representations.items()},
        "patient_id": [str(value) for value in batch["patient_id"]],
        "labels": flatten_tensor(batch["label"]).reshape(-1).astype(np.int64),
        "shortcuts": list(batch.get("shortcuts", [])),
    }


def shortcut_value(item: Mapping[str, Any], field: str, patient_id: str) -> Any:
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
    all_numeric = True
    for value in values:
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            all_numeric = False
            break
    if all_numeric:
        return np.asarray(converted, dtype=np.float32)
    return np.asarray([str(value) for value in values], dtype="<U256")


def build_loader(config: Mapping[str, Any], rows: List[Dict[str, Any]], split: str, args: argparse.Namespace) -> DataLoader:
    project = config["project"]
    model_cfg = config["model"]
    dataset = PatientHTDataset(
        rows=rows,
        data_root=project["data_root"],
        split=split,
        max_images=int(model_cfg.get("max_images_per_patient", 4)),
        image_size=int(model_cfg.get("image_size", 224)),
        text_max_length=int(model_cfg.get("text_max_length", 256)),
        text_vocab_size=int(model_cfg.get("text_vocab_size", 50000)),
        bio_dim=int(model_cfg.get("bio_dim", 32)),
    )
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_patient_batch,
        pin_memory=torch.cuda.is_available(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"patient_id", "label"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"validation prediction file lacks {required}: {path}")
    probability_column = next(
        (column for column in ("prob", "final_prob", "prediction", "y_prob") if column in frame.columns), None
    )
    if probability_column is None:
        raise RuntimeError(f"no probability column in validation prediction file: {path}")
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in frame.iterrows():
        result[str(row["patient_id"])] = {
            "label": int(float(row["label"])),
            "prob": float(row[probability_column]),
        }
    return result


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    config_path = resolve_path(repo_root, args.config)
    output_dir = resolve_path(repo_root, args.output_dir)
    run_dir = resolve_path(repo_root, args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if args.manifest:
        config["project"]["manifest"] = args.manifest
    if args.data_root:
        config["project"]["data_root"] = args.data_root
    rows = read_manifest(config["project"]["manifest"])
    if not all(str(row.get("split", "")).strip() for row in rows):
        splits = patient_split(rows, seed=42)
        for row, split in zip(rows, splits):
            row["split"] = split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_inventory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    reproduction_rows: List[Dict[str, Any]] = []
    environment_rows: List[Dict[str, Any]] = []
    archive_payloads: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}}

    for seed in SEEDS:
        checkpoint = run_dir / "checkpoints" / f"seed_{seed}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"C17 checkpoint missing: {checkpoint}")
        model = C17ResidualModel(config, seed).to(device)
        model.load_state_dict(checkpoint_state(checkpoint), strict=True)
        model.eval()
        split_results: Dict[str, Dict[str, Any]] = {}
        for split in ("train", "val"):
            loader = build_loader(config, rows, split, args)
            representation_batches: MutableMapping[str, List[np.ndarray]] = {}
            ids: List[str] = []
            labels: List[np.ndarray] = []
            shortcuts: MutableMapping[str, List[Any]] = {field: [] for field in SHORTCUT_FIELDS}
            with torch.no_grad():
                for batch in loader:
                    moved = move_batch(batch, device)
                    extracted = extract_internal(model, moved)
                    ids.extend(extracted["patient_id"])
                    labels.append(extracted["labels"])
                    for name, value in extracted["representations"].items():
                        representation_batches.setdefault(name, []).append(value)
                        all_inventory.setdefault(name, {"source": "C17 eval path", "available": True})
                    for patient_id, item in zip(extracted["patient_id"], extracted["shortcuts"]):
                        for field in SHORTCUT_FIELDS:
                            shortcuts[field].append(shortcut_value(item, field, patient_id))
            split_results[split] = {
                "patient_id": np.asarray(ids, dtype="<U256"),
                "labels": np.concatenate(labels).astype(np.int64, copy=False) if labels else np.empty(0, dtype=np.int64),
                "representations": {name: stack_or_empty(values) for name, values in representation_batches.items()},
                "shortcuts": {field: numeric_or_string(values) for field, values in shortcuts.items()},
            }
            for name, value in split_results[split]["representations"].items():
                entry = all_inventory.setdefault(name, {"source": "C17 eval path", "available": True})
                entry[f"shape_{split}"] = list(value.shape)

        val_result = split_results["val"]
        saved = read_validation_predictions(
            resolve_path(repo_root, run_dir / "predictions" / f"val_predictions_seed_{seed}.csv")
        )
        export_ids = val_result["patient_id"].astype(str)
        saved_ids = list(saved)
        id_match = len(saved_ids) == len(export_ids) and set(saved_ids) == set(export_ids)
        label_match = id_match and all(int(saved[pid]["label"]) == int(label) for pid, label in zip(export_ids, val_result["labels"]))
        exported_prob = val_result["representations"]["scalar_final_prob"].reshape(-1)
        saved_prob = np.asarray([saved[pid]["prob"] for pid in export_ids], dtype=np.float64) if id_match else np.empty(0)
        differences = np.abs(exported_prob.astype(np.float64) - saved_prob) if id_match else np.asarray([])
        reproduction_rows.append(
            {
                "seed": seed,
                "validation_patient_count": len(export_ids),
                "saved_prediction_count": len(saved_ids),
                "patient_id_exact_match": id_match,
                "label_exact_match": label_match,
                "max_abs_prob_diff": float(differences.max()) if differences.size else float("nan"),
                "mean_abs_prob_diff": float(differences.mean()) if differences.size else float("nan"),
                "pass": bool(id_match and label_match and differences.size and differences.max() <= 1e-8 and differences.mean() <= 1e-9),
            }
        )
        torch.cuda.empty_cache()

        environment_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "train_count": int(len(split_results["train"]["patient_id"])),
                "val_count": int(len(split_results["val"]["patient_id"])),
            }
        )

        # Keep arrays in memory only until the seed archive is written.
        for split, result in split_results.items():
            result["seed"] = seed
            result["payload"] = {
                f"seed_{seed}__patient_id": result["patient_id"],
                f"seed_{seed}__labels": result["labels"],
                **{f"seed_{seed}__layer__{name}": value for name, value in result["representations"].items()},
                **{f"seed_{seed}__shortcut__{field}": value for field, value in result["shortcuts"].items()},
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            archive_path = output_dir / f"c20_internal_representations_{split}.npz"
            archive_payloads.setdefault(split, {}).update(result["payload"])
    for split in ("train", "val"):
        np.savez_compressed(output_dir / f"c20_internal_representations_{split}.npz", **archive_payloads.get(split, {}))

    with (output_dir / "c20_reproduction_check_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reproduction_rows[0]))
        writer.writeheader()
        writer.writerows(reproduction_rows)

    config_commit = "unknown"
    try:
        config_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    inventory_path = output_dir / "c20_internal_representation_inventory.csv"
    fields = ["layer", "available", "source", "shape_train", "shape_val"]
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, item in all_inventory.items():
            writer.writerow(
                {
                    "layer": name,
                    "available": item.get("available", True),
                    "source": item.get("source", "C17 eval path"),
                    "shape_train": json.dumps(item.get("shape_train", [])),
                    "shape_val": json.dumps(item.get("shape_val", [])),
                }
            )

    lines = [
        "# C20 Environment And Input Inventory",
        "",
        "- phase: C20 validation-only pathological-mechanism evidence identifiability audit",
        f"- repository commit: `{config_commit}`",
        f"- runtime device: `{gpu}`",
        f"- config: `{config_path}`",
        f"- manifest: `{config['project']['manifest']}`",
        f"- data root: `{config['project']['data_root']}`",
        f"- C17 run directory: `{run_dir}`",
        "- seeds: `[0, 42, 3407]`",
        "- model mode: `eval()` with `torch.no_grad()`; no optimizer, backward, or new model was created",
        "- splits read: `train` and `val` only; test data and test prediction files were not read",
        "- saved C17 validation predictions are used only for reproduction checking, never as training inputs",
        "",
        "## Checkpoints",
        "",
        "| seed | checkpoint | sha256 | train n | val n |",
        "|---:|---|---|---:|---:|",
    ]
    for row in environment_rows:
        lines.append(
            f"| {row['seed']} | `{row['checkpoint']}` | `{row['checkpoint_sha256']}` | {row['train_count']} | {row['val_count']} |"
        )
    lines.extend(["", "## Large Artifacts", "", "- `c20_internal_representations_train.npz` and `c20_internal_representations_val.npz` are server-only artifacts."])
    (output_dir / "c20_environment_and_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = all(row["pass"] for row in reproduction_rows)
    report_lines = [
        "# C20 Reproduction Check",
        "",
        f"- overall: `{'PASS' if passed else 'C20_REPRODUCTION_GATE_FAIL'}`",
        "- comparison: exported C17 validation probabilities versus the saved validation prediction CSV for the same patient IDs",
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
    (output_dir / "c20_reproduction_check_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"reproduction_pass": passed, "output_dir": str(output_dir), "device": gpu}, ensure_ascii=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
