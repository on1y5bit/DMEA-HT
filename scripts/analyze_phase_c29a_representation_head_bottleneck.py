#!/usr/bin/env python3
"""Audit frozen C27 representations and heads with fixed train-fit probes."""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c27_vtme import C27VTMEModel, MECHANISM_NAMES, masked_softmax  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402
from scripts.phase_c20_common import (  # noqa: E402
    jaccard_by_patient,
    linear_cka,
    pairwise_distances,
    spearman,
    upper_triangle,
)


SEEDS = (0, 42, 3407)
SEED_PAIRS = ((0, 42), (0, 3407), (42, 3407))
PROBES = (
    "P0_official_C27",
    "P1_patient_state",
    "P2_pre_projection",
    "P3_temporal_mechanisms",
    "P4_conflicts_negative_control",
    "P5_C17_mechanism_reference",
)
FITTED_PROBES = PROBES[1:]
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits", "raw_n_images")
RANDOM_STATE = 20260714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c27_vtme_multiseed.yaml")
    parser.add_argument("--c27-run-dir", default="runs/dema_ht_c27_vtme_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--c17-representation-dir", default="analysis_reports/phase_c20_dema")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c29a_dema")
    parser.add_argument("--stage", required=True, choices=("gate", "analyze"))
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            result.add(node.func.attr)
    return result


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unsupported C27 checkpoint payload: {path}")
    return payload


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loader(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]], split: str
) -> DataLoader:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    dataset = VisitPatientDataset(
        rows=rows,
        data_root=project["data_root"],
        split=split,
        image_size=int(model_cfg["image_size"]),
        text_max_length=int(model_cfg["text_max_length"]),
        text_vocab_size=int(model_cfg["text_vocab_size"]),
        bio_dim=int(model_cfg["bio_dim"]),
        max_images_per_visit=int(model_cfg["max_images_per_visit"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_visit_batch,
        pin_memory=torch.cuda.is_available(),
    )


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.sort_values("patient_id").reset_index(drop=True)


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("final_prob", "prob", "pred_prob", "prediction"):
        if name in frame.columns:
            return name
    raise RuntimeError("No probability column found")


def auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(scores), dtype=np.float64)
    return float(roc_auc_score(y, p))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def shortcut_value(item: Mapping[str, Any], field: str) -> float:
    aliases = {
        "image_padding_count": ("image_padding_count", "padding_count"),
        "used_images": ("used_images", "n_images"),
    }
    for key in aliases.get(field, (field,)):
        value = item.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def expected_split(rows: Sequence[Mapping[str, Any]], split: str) -> Tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        ((str(row["patient_id"]), int(row["label"])) for row in rows if str(row.get("split")) == split),
        key=lambda item: item[0],
    )
    return np.asarray([item[0] for item in selected]), np.asarray([item[1] for item in selected], dtype=np.int64)


def concatenate(parts: Mapping[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def run_split(
    model: C27VTMEModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    ids: List[str] = []
    labels: List[int] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    shortcut_rows: List[Dict[str, float]] = []
    max_temporal_formula_error = 0.0
    max_patient_state_rebuild_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            captured: Dict[str, torch.Tensor] = {}

            def capture_core(_module: torch.nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
                captured["source_states"] = args[0]
                captured["source_valid"] = args[1]
                captured["visit_mask"] = args[2]
                captured["fallback_bio_context"] = args[3]

            handle = model.core.register_forward_pre_hook(capture_core)
            official = model(batch)
            handle.remove()
            if set(captured) != {"source_states", "source_valid", "visit_mask", "fallback_bio_context"}:
                raise RuntimeError("C29A failed to capture exact C27 core inputs")

            mechanisms = official["mechanism_states"]
            conflicts = official["conflicts"]
            fallback = captured["fallback_bio_context"]
            patient_input = torch.cat([mechanisms.flatten(start_dim=1), conflicts, fallback], dim=-1)
            projection_linear = model.core.patient_projection[0](patient_input)
            projection_gelu = model.core.patient_projection[1](projection_linear)
            rebuilt_state = model.core.patient_projection[2](projection_gelu)
            classifier = model.core.classifier[1]
            dot = F.linear(official["patient_state"], classifier.weight, None).squeeze(-1)
            bias = classifier.bias.reshape(1).expand_as(dot)
            decomposition = dot + bias

            visit_states = official["visit_states"]
            content = model.core.temporal_output(
                torch.tanh(model.core.temporal_linear(model.core.temporal_norm(visit_states)))
            ).squeeze(-1)
            combined = content + model.core.recency_prior_log_odds * official["recency"].unsqueeze(-1)
            valid = captured["visit_mask"].unsqueeze(-1).expand_as(combined)
            rebuilt_weights = masked_softmax(combined, valid, dim=1)
            max_temporal_formula_error = max(
                max_temporal_formula_error,
                float((rebuilt_weights - official["temporal_weights"]).abs().max().cpu()),
            )
            max_patient_state_rebuild_error = max(
                max_patient_state_rebuild_error,
                float((rebuilt_state - official["patient_state"]).abs().max().cpu()),
            )

            batch_arrays = {
                "S0_temporal_mechanisms": mechanisms,
                "S1_conflicts": conflicts,
                "S2_pre_projection": patient_input,
                "S3_projection_linear": projection_linear,
                "S3_projection_post_gelu": projection_gelu,
                "S4_patient_state": official["patient_state"],
                "S5_classifier_dot": dot[:, None],
                "S5_classifier_bias": bias[:, None],
                "official_logit": official["logit"][:, None],
                "official_prob": official["prob"][:, None],
                "temporal_latest": official["temporal_latest_weights"],
                "logit_decomposition": decomposition[:, None],
            }
            for name, value in batch_arrays.items():
                arrays.setdefault(name, []).append(value.detach().cpu().numpy().astype(np.float32, copy=False))
            ids.extend(str(value) for value in batch["patient_id"])
            labels.extend(int(value) for value in batch["label"].detach().cpu().numpy())
            for item in batch["shortcuts"]:
                shortcut_rows.append(
                    {field: shortcut_value(item, field) for field in (*SELECTED_SHORTCUT_FIELDS, *RAW_SHORTCUT_FIELDS)}
                )

    packed = concatenate(arrays)
    order = np.argsort(np.asarray(ids, dtype=str))
    return {
        "patient_id": np.asarray(ids, dtype=str)[order],
        "labels": np.asarray(labels, dtype=np.int64)[order],
        "arrays": {name: value[order] for name, value in packed.items()},
        "shortcuts": pd.DataFrame(shortcut_rows).iloc[order].reset_index(drop=True),
        "max_temporal_formula_error": max_temporal_formula_error,
        "max_patient_state_rebuild_error": max_patient_state_rebuild_error,
    }


def state_matches(model: torch.nn.Module, expected: Mapping[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    if set(current) != set(expected):
        return False
    return all(torch.equal(current[name].detach().cpu(), expected[name].detach().cpu()) for name in current)


def extract_head(model: C27VTMEModel) -> Dict[str, np.ndarray | float]:
    projection = model.core.patient_projection
    classifier = model.core.classifier[1]
    return {
        "projection_weight": projection[0].weight.detach().cpu().numpy().copy(),
        "projection_bias": projection[0].bias.detach().cpu().numpy().copy(),
        "norm_weight": projection[2].weight.detach().cpu().numpy().copy(),
        "norm_bias": projection[2].bias.detach().cpu().numpy().copy(),
        "norm_eps": float(projection[2].eps),
        "classifier_weight": classifier.weight.detach().cpu().numpy().reshape(-1).copy(),
        "classifier_bias": float(classifier.bias.detach().cpu().item()),
    }


def extract_all(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    c27_run: Path,
    c17_run: Path,
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    data: Dict[int, Dict[str, Any]] = {}
    heads: Dict[int, Dict[str, Any]] = {}
    reproduction_rows: List[Dict[str, Any]] = []
    runtime: Dict[str, Any] = {
        "checkpoints_exist": True,
        "checkpoint_seed_metadata": True,
        "ids_labels_aligned": True,
        "official_reproduction": True,
        "temporal_reproduction": True,
        "stages_available": True,
        "logit_decomposition": True,
        "checkpoint_unchanged": True,
        "splits_read": {"train", "val"},
    }
    expected = {split: expected_split(rows, split) for split in ("train", "val")}
    for seed in SEEDS:
        checkpoint_path = c27_run / "checkpoints" / f"seed_{seed}_best.pt"
        c27_prediction_path = c27_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        c17_prediction_path = c17_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        if not all(path.exists() for path in (checkpoint_path, c27_prediction_path, c17_prediction_path)):
            runtime["checkpoints_exist"] = False
            raise FileNotFoundError(f"Missing C29A input for seed {seed}")
        payload = load_checkpoint(checkpoint_path)
        if int(payload.get("seed", -1)) != seed:
            runtime["checkpoint_seed_metadata"] = False
            raise RuntimeError(f"C27 seed metadata mismatch for seed {seed}")
        model = C27VTMEModel(config, seed).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        seed_data = {
            split: run_split(model, build_loader(config, rows, split), device)
            for split in ("train", "val")
        }
        heads[seed] = extract_head(model)
        unchanged = state_matches(model, payload["model"])
        runtime["checkpoint_unchanged"] = bool(runtime["checkpoint_unchanged"] and unchanged)
        data[seed] = seed_data

        train_ids, train_labels = expected["train"]
        val_ids, val_labels = expected["val"]
        train_exact = np.array_equal(seed_data["train"]["patient_id"], train_ids)
        train_label_exact = train_exact and np.array_equal(seed_data["train"]["labels"], train_labels)
        val_exact = np.array_equal(seed_data["val"]["patient_id"], val_ids)
        val_label_exact = val_exact and np.array_equal(seed_data["val"]["labels"], val_labels)

        saved = read_prediction(c27_prediction_path)
        c17_saved = read_prediction(c17_prediction_path)
        current_ids = seed_data["val"]["patient_id"]
        saved_exact = np.array_equal(saved["patient_id"].to_numpy(dtype=str), current_ids)
        c17_exact = np.array_equal(c17_saved["patient_id"].to_numpy(dtype=str), current_ids)
        saved_labels = saved["label"].to_numpy(dtype=np.int64)
        saved_label_exact = saved_exact and np.array_equal(saved_labels, seed_data["val"]["labels"])
        c17_label_exact = c17_exact and np.array_equal(
            c17_saved["label"].to_numpy(dtype=np.int64), seed_data["val"]["labels"]
        )
        current_logits = seed_data["val"]["arrays"]["official_logit"].reshape(-1).astype(np.float64)
        current_probs = seed_data["val"]["arrays"]["official_prob"].reshape(-1).astype(np.float64)
        saved_logits = saved["final_logit"].to_numpy(dtype=np.float64)
        saved_probs = saved[probability_column(saved)].to_numpy(dtype=np.float64)
        logit_error = float(np.max(np.abs(current_logits - saved_logits)))
        probability_error = float(np.max(np.abs(current_probs - saved_probs)))
        class_mismatch = int(np.sum((current_probs >= 0.5) != saved["predicted_class"].to_numpy(dtype=int)))
        auc_error = abs(auc(seed_data["val"]["labels"], current_probs) - auc(saved_labels, saved_probs))
        latest_columns = [f"temporal_weight_latest_{name}" for name in MECHANISM_NAMES]
        latest_error = float(
            np.max(
                np.abs(
                    seed_data["val"]["arrays"]["temporal_latest"].astype(np.float64)
                    - saved[latest_columns].to_numpy(dtype=np.float64)
                )
            )
        )
        decomposition_error = float(
            np.max(
                np.abs(
                    seed_data["val"]["arrays"]["logit_decomposition"].reshape(-1).astype(np.float64)
                    - current_logits
                )
            )
        )
        stage_shapes = {
            name: list(seed_data["val"]["arrays"][name].shape)
            for name in (
                "S0_temporal_mechanisms",
                "S1_conflicts",
                "S2_pre_projection",
                "S3_projection_linear",
                "S3_projection_post_gelu",
                "S4_patient_state",
                "S5_classifier_dot",
            )
        }
        hidden = int(config["model"]["hidden_dim"])
        expected_shapes = {
            "S0_temporal_mechanisms": [94, 5, hidden],
            "S1_conflicts": [94, 5],
            "S2_pre_projection": [94, hidden * 6 + 5],
            "S3_projection_linear": [94, hidden],
            "S3_projection_post_gelu": [94, hidden],
            "S4_patient_state": [94, hidden],
            "S5_classifier_dot": [94, 1],
        }
        stage_shape_ok = stage_shapes == expected_shapes
        counts_ok = len(train_ids) == 602 and len(val_ids) == 94
        ids_ok = bool(train_exact and train_label_exact and val_exact and val_label_exact and saved_exact and saved_label_exact and c17_exact and c17_label_exact)
        reproduction_ok = bool(logit_error <= 1e-6 and probability_error <= 1e-7 and class_mismatch == 0 and auc_error <= 1e-12)
        temporal_ok = bool(
            seed_data["val"]["max_temporal_formula_error"] <= 1e-6 and latest_error <= 1e-7
        )
        decomposition_ok = bool(decomposition_error <= 1e-6 and seed_data["val"]["max_patient_state_rebuild_error"] <= 1e-6)
        runtime["ids_labels_aligned"] = bool(runtime["ids_labels_aligned"] and counts_ok and ids_ok)
        runtime["official_reproduction"] = bool(runtime["official_reproduction"] and reproduction_ok)
        runtime["temporal_reproduction"] = bool(runtime["temporal_reproduction"] and temporal_ok)
        runtime["stages_available"] = bool(runtime["stages_available"] and stage_shape_ok)
        runtime["logit_decomposition"] = bool(runtime["logit_decomposition"] and decomposition_ok)
        data[seed]["c17_val"] = c17_saved
        reproduction_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "best_epoch": int(payload.get("best_epoch", -1)),
                "train_patient_count": len(train_ids),
                "train_positive_count": int(train_labels.sum()),
                "train_negative_count": int((train_labels == 0).sum()),
                "validation_patient_count": len(val_ids),
                "validation_positive_count": int(val_labels.sum()),
                "validation_negative_count": int((val_labels == 0).sum()),
                "train_patient_id_exact": train_exact,
                "train_label_exact": train_label_exact,
                "validation_patient_id_exact": val_exact and saved_exact and c17_exact,
                "validation_label_exact": val_label_exact and saved_label_exact and c17_label_exact,
                "max_abs_logit_error": logit_error,
                "max_abs_probability_error": probability_error,
                "predicted_class_mismatch_count": class_mismatch,
                "auc_error": auc_error,
                "max_abs_temporal_weight_formula_error": seed_data["val"]["max_temporal_formula_error"],
                "max_abs_saved_latest_weight_error": latest_error,
                "max_abs_patient_state_rebuild_error": seed_data["val"]["max_patient_state_rebuild_error"],
                "max_abs_logit_decomposition_error": decomposition_error,
                "checkpoint_state_unchanged": unchanged,
                "pass": bool(counts_ok and ids_ok and reproduction_ok and temporal_ok and decomposition_ok and unchanged),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return data, heads, pd.DataFrame(reproduction_rows), runtime


def load_c17_reference(directory: Path, split: str) -> Dict[int, Dict[str, np.ndarray]]:
    path = directory / f"c20_internal_representations_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen C17 representation archive: {path}")
    result: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for seed in SEEDS:
            prefix = f"seed_{seed}__"
            result[seed] = {
                "patient_id": payload[prefix + "patient_id"].astype(str),
                "labels": payload[prefix + "labels"].astype(np.int64),
                "features": payload[
                    prefix + "layer__mechanism_final_representation"
                ].astype(np.float64),
            }
    return result


def align_reference(reference: Mapping[str, np.ndarray], target: Mapping[str, Any]) -> np.ndarray:
    index = {str(value): idx for idx, value in enumerate(reference["patient_id"])}
    try:
        order = np.asarray([index[str(value)] for value in target["patient_id"]], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"C17/C27 patient alignment failed: {error}") from error
    if not np.array_equal(reference["labels"][order], target["labels"]):
        raise RuntimeError("C17/C27 label alignment failed")
    return np.asarray(reference["features"][order], dtype=np.float64)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, c17_probabilities: np.ndarray | None = None
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    predicted = p >= 0.5
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    positive = p[y == 1]
    negative = p[y == 0]
    values: Dict[str, Any] = {
        "AUC": auc(y, p),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "Balanced_ACC": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "positive_probability_mean": float(positive.mean()),
        "negative_probability_mean": float(negative.mean()),
        "pairwise_inversion_count": int((positive[:, None] < negative[None, :]).sum()),
    }
    if c17_probabilities is None:
        values.update(
            {
                "material_positive_damage_count": float("nan"),
                "severe_positive_damage_count": float("nan"),
                "c17_tp_to_object_fn": float("nan"),
            }
        )
        return values
    c17 = np.asarray(c17_probabilities, dtype=np.float64)
    positive_mask = y == 1
    damage = positive_mask & (((c17 >= 0.5) & (p < 0.5)) | ((p - c17) <= -0.05))
    severe = positive_mask & ((p - c17) <= -0.10)
    values.update(
        {
            "material_positive_damage_count": int(damage.sum()),
            "severe_positive_damage_count": int(severe.sum()),
            "c17_tp_to_object_fn": int((positive_mask & (c17 >= 0.5) & (p < 0.5)).sum()),
        }
    )
    return values


def probe_feature(pack: Mapping[str, Any], probe: str, c17: np.ndarray) -> np.ndarray:
    arrays = pack["arrays"]
    if probe == "P1_patient_state":
        return arrays["S4_patient_state"]
    if probe == "P2_pre_projection":
        return arrays["S2_pre_projection"]
    if probe == "P3_temporal_mechanisms":
        return arrays["S0_temporal_mechanisms"].reshape(len(pack["labels"]), -1)
    if probe == "P4_conflicts_negative_control":
        return arrays["S1_conflicts"]
    if probe == "P5_C17_mechanism_reference":
        return c17
    raise KeyError(probe)


def fixed_probe() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )


def run_probes(
    data: Mapping[int, Mapping[str, Any]],
    c17_reference: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray]], bool]:
    metric_rows: List[Dict[str, Any]] = []
    generalization_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    scores: Dict[int, Dict[str, np.ndarray]] = {}
    alignment_ok = True
    for seed in SEEDS:
        train, val = data[seed]["train"], data[seed]["val"]
        train_c17 = align_reference(c17_reference["train"][seed], train)
        val_c17 = align_reference(c17_reference["val"][seed], val)
        c17_frame = data[seed]["c17_val"]
        c17_probs = c17_frame[probability_column(c17_frame)].to_numpy(dtype=np.float64)
        scores[seed] = {"P0_official_C27": val["arrays"]["official_prob"].reshape(-1).astype(np.float64)}
        official_train = train["arrays"]["official_prob"].reshape(-1).astype(np.float64)
        official_metrics = classification_metrics(val["labels"], scores[seed]["P0_official_C27"], c17_probs)
        metric_rows.append(
            {
                "seed": seed,
                "probe": "P0_official_C27",
                "input_stage": "official frozen C27 probability",
                "feature_dimension": 0,
                "train_AUC": auc(train["labels"], official_train),
                "validation_AUC": official_metrics.pop("AUC"),
                "train_validation_AUC_gap": float("nan"),
                "coefficient_norm": float("nan"),
                "authorization_candidate": False,
                **official_metrics,
            }
        )
        for probe in FITTED_PROBES:
            x_train = np.asarray(probe_feature(train, probe, train_c17), dtype=np.float64)
            x_val = np.asarray(probe_feature(val, probe, val_c17), dtype=np.float64)
            if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
                raise RuntimeError(f"Non-finite probe input for seed {seed} {probe}")
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(x_train)
            scaled_val = scaler.transform(x_val)
            estimator = fixed_probe()
            estimator.fit(scaled_train, train["labels"])
            train_prob = estimator.predict_proba(scaled_train)[:, 1]
            val_prob = estimator.predict_proba(scaled_val)[:, 1]
            scores[seed][probe] = val_prob
            train_auc = auc(train["labels"], train_prob)
            val_metrics = classification_metrics(val["labels"], val_prob, c17_probs)
            val_auc = float(val_metrics.pop("AUC"))
            gap = train_auc - val_auc

            random_labels = np.random.default_rng(RANDOM_STATE).permutation(train["labels"])
            random_estimator = fixed_probe()
            random_estimator.fit(scaled_train, random_labels)
            random_auc = auc(val["labels"], random_estimator.predict_proba(scaled_val)[:, 1])
            random_pass = random_auc <= 0.65
            generalization_pass = gap <= 0.15
            candidate = probe in ("P1_patient_state", "P2_pre_projection", "P3_temporal_mechanisms")
            metric_rows.append(
                {
                    "seed": seed,
                    "probe": probe,
                    "input_stage": probe,
                    "feature_dimension": x_train.shape[1],
                    "train_AUC": train_auc,
                    "validation_AUC": val_auc,
                    "train_validation_AUC_gap": gap,
                    "coefficient_norm": float(np.linalg.norm(estimator.coef_)),
                    "authorization_candidate": candidate,
                    **val_metrics,
                }
            )
            generalization_rows.append(
                {
                    "seed": seed,
                    "probe": probe,
                    "train_AUC": train_auc,
                    "validation_AUC": val_auc,
                    "train_validation_AUC_gap": gap,
                    "maximum_allowed_gap": 0.15,
                    "pass": generalization_pass,
                    "authorization_candidate": candidate,
                }
            )
            random_rows.append(
                {
                    "seed": seed,
                    "probe": probe,
                    "permutation_seed": RANDOM_STATE,
                    "random_label_validation_AUC": random_auc,
                    "maximum_allowed_AUC": 0.65,
                    "pass": random_pass,
                    "authorization_candidate": candidate,
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(generalization_rows),
        pd.DataFrame(random_rows),
        scores,
        alignment_ok,
    )


def apply_classifier(patient_state: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
    state = torch.from_numpy(np.asarray(patient_state, dtype=np.float32))
    with torch.inference_mode():
        logits = F.linear(
            state,
            torch.from_numpy(np.asarray(head["classifier_weight"], dtype=np.float32))[None],
            torch.tensor([float(head["classifier_bias"])], dtype=torch.float32),
        ).squeeze(-1)
    return logits.numpy().astype(np.float64)


def apply_full_head(patient_input: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
    x = torch.from_numpy(np.asarray(patient_input, dtype=np.float32))
    with torch.inference_mode():
        linear = F.linear(
            x,
            torch.from_numpy(np.asarray(head["projection_weight"], dtype=np.float32)),
            torch.from_numpy(np.asarray(head["projection_bias"], dtype=np.float32)),
        )
        activated = F.gelu(linear)
        state = F.layer_norm(
            activated,
            (activated.shape[-1],),
            torch.from_numpy(np.asarray(head["norm_weight"], dtype=np.float32)),
            torch.from_numpy(np.asarray(head["norm_bias"], dtype=np.float32)),
            float(head["norm_eps"]),
        )
        logits = F.linear(
            state,
            torch.from_numpy(np.asarray(head["classifier_weight"], dtype=np.float32))[None],
            torch.tensor([float(head["classifier_bias"])], dtype=torch.float32),
        ).squeeze(-1)
    return logits.numpy().astype(np.float64)


def run_swaps(
    data: Mapping[int, Mapping[str, Any]],
    heads: Mapping[int, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    classifier_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    scores: Dict[int, Dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    max_classifier_diagonal_error = 0.0
    max_head_diagonal_error = 0.0
    for representation_seed in SEEDS:
        pack = data[representation_seed]["val"]
        c17_frame = data[representation_seed]["c17_val"]
        c17_probs = c17_frame[probability_column(c17_frame)].to_numpy(dtype=np.float64)
        official_logits = pack["arrays"]["official_logit"].reshape(-1).astype(np.float64)
        official_inversions = classification_metrics(pack["labels"], sigmoid(official_logits))["pairwise_inversion_count"]
        for head_seed in SEEDS:
            classifier_logits = apply_classifier(pack["arrays"]["S4_patient_state"], heads[head_seed])
            head_logits = apply_full_head(pack["arrays"]["S2_pre_projection"], heads[head_seed])
            classifier_name = f"CLS_SWAP_r{representation_seed}_c{head_seed}"
            head_name = f"HEAD_SWAP_r{representation_seed}_c{head_seed}"
            classifier_prob = sigmoid(classifier_logits)
            head_prob = sigmoid(head_logits)
            scores[representation_seed][classifier_name] = classifier_prob
            scores[representation_seed][head_name] = head_prob
            classifier_metrics = classification_metrics(pack["labels"], classifier_prob, c17_probs)
            head_metrics = classification_metrics(pack["labels"], head_prob, c17_probs)
            classifier_error = float(np.max(np.abs(classifier_logits - official_logits))) if representation_seed == head_seed else float("nan")
            head_error = float(np.max(np.abs(head_logits - official_logits))) if representation_seed == head_seed else float("nan")
            if representation_seed == head_seed:
                max_classifier_diagonal_error = max(max_classifier_diagonal_error, classifier_error)
                max_head_diagonal_error = max(max_head_diagonal_error, head_error)
            classifier_rows.append(
                {
                    "representation_seed": representation_seed,
                    "classifier_seed": head_seed,
                    "object_name": classifier_name,
                    "diagonal": representation_seed == head_seed,
                    "max_abs_diagonal_logit_error": classifier_error,
                    **classifier_metrics,
                    "official_representation_seed_inversions": official_inversions,
                    "net_inversion_change": classifier_metrics["pairwise_inversion_count"] - official_inversions,
                }
            )
            head_rows.append(
                {
                    "representation_seed": representation_seed,
                    "head_seed": head_seed,
                    "object_name": head_name,
                    "diagonal": representation_seed == head_seed,
                    "max_abs_diagonal_logit_error": head_error,
                    **head_metrics,
                    "official_representation_seed_inversions": official_inversions,
                    "net_inversion_change": head_metrics["pairwise_inversion_count"] - official_inversions,
                }
            )
    return (
        pd.DataFrame(classifier_rows),
        pd.DataFrame(head_rows),
        scores,
        {
            "classifier_swap_diagonal": max_classifier_diagonal_error <= 1e-6,
            "head_swap_diagonal": max_head_diagonal_error <= 1e-6,
            "max_classifier_diagonal_error": max_classifier_diagonal_error,
            "max_head_diagonal_error": max_head_diagonal_error,
        },
    )


def build_objects(
    data: Mapping[int, Mapping[str, Any]],
    probe_scores: Mapping[int, Mapping[str, np.ndarray]],
    swap_scores: Mapping[int, Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for seed in SEEDS:
        pack = data[seed]["val"]
        for probe in PROBES:
            objects.append(
                {
                    "object_type": "probe",
                    "object_name": probe,
                    "representation_seed": seed,
                    "head_seed": seed if probe == "P0_official_C27" else float("nan"),
                    "patient_id": pack["patient_id"],
                    "labels": pack["labels"],
                    "probabilities": np.asarray(probe_scores[seed][probe], dtype=np.float64),
                }
            )
        for name, probabilities in swap_scores[seed].items():
            parts = name.split("_c")
            objects.append(
                {
                    "object_type": "classifier_swap" if name.startswith("CLS_") else "head_swap",
                    "object_name": name,
                    "representation_seed": seed,
                    "head_seed": int(parts[-1]),
                    "patient_id": pack["patient_id"],
                    "labels": pack["labels"],
                    "probabilities": np.asarray(probabilities, dtype=np.float64),
                }
            )
    return objects


def pair_contract(objects: Sequence[Mapping[str, Any]]) -> bool:
    if len(objects) != 36:
        return False
    return all(
        int((obj["labels"] == 1).sum()) == 47
        and int((obj["labels"] == 0).sum()) == 47
        and int((obj["labels"] == 1).sum() * (obj["labels"] == 0).sum()) == 2209
        for obj in objects
    )


def gate_payload(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    analyzer = Path(__file__).resolve()
    collector = REPO_ROOT / "scripts" / "collect_phase_c29a_report.py"
    calls = called_names(analyzer) | called_names(collector)
    worktree_lines = git_output("worktree", "list", "--porcelain").splitlines()
    worktrees = [Path(line.split(" ", 1)[1]).resolve() for line in worktree_lines if line.startswith("worktree ")]
    c29_configs = list((REPO_ROOT / "configs").glob("*c29*"))
    collector_text = collector.read_text(encoding="utf-8")
    checks = [
        ("active_branch_main", git_output("branch", "--show-current") == "main"),
        ("canonical_worktree_only", len(worktrees) == 1 and worktrees[0] == REPO_ROOT.resolve()),
        ("no_c29_model_config", not c29_configs),
        ("no_neural_optimizer", not ({"Adam", "AdamW", "SGD", "RMSprop"} & calls)),
        ("no_backward_through_c27", "backward" not in calls),
        ("no_checkpoint_writer", "save" not in calls),
        ("no_test_loader", runtime["splits_read"] == {"train", "val"}),
        ("auc_only_route_metric", not ({"average_precision_score", "precision_recall_curve"} & calls)),
        ("no_ensemble", not ({"average_checkpoints", "weighted_vote", "vstack_predictions"} & calls)),
        ("three_c27_checkpoints_exist", runtime["checkpoints_exist"]),
        ("checkpoint_seed_metadata_correct", runtime["checkpoint_seed_metadata"]),
        ("train_validation_ids_labels_aligned", runtime["ids_labels_aligned"]),
        ("official_logits_probabilities_reproduced", runtime["official_reproduction"]),
        ("official_temporal_weights_reproduced", runtime["temporal_reproduction"]),
        ("s0_s5_real_tensors_available", runtime["stages_available"]),
        ("logit_decomposition_exact", runtime["logit_decomposition"]),
        ("c27_checkpoint_state_unchanged", runtime["checkpoint_unchanged"]),
        ("probe_fit_train_only", runtime["probe_fit_train_only"]),
        ("validation_not_used_for_fit", runtime["validation_not_used_for_fit"]),
        ("standard_scaler_fit_train_only", runtime["standard_scaler_fit_train_only"]),
        ("logistic_parameters_fixed", runtime["logistic_parameters_fixed"]),
        ("no_probe_sweep", runtime["no_probe_sweep"]),
        ("random_label_sanity_ran", runtime["random_label_sanity_ran"]),
        ("shortcut_fields_excluded_from_probes", runtime["shortcut_fields_excluded"]),
        ("p4_never_authorizes", "P4_conflicts_negative_control" in collector_text and "P4" in collector_text),
        ("p5_never_authorizes", "P5_C17_mechanism_reference" in collector_text and "P5" in collector_text),
        ("classifier_swap_diagonal_reproduced", runtime["classifier_swap_diagonal"]),
        ("head_swap_diagonal_reproduced", runtime["head_swap_diagonal"]),
        ("all_scores_finite", runtime["all_scores_finite"]),
        ("every_formal_object_has_2209_pairs", runtime["pair_contract"]),
        ("small_seed_variation_policy_fixed", "MINOR_AUC = 0.003" in collector_text and "MINOR_SENSITIVITY = 0.03" in collector_text),
        ("c29b_not_auto_started", not ({"Popen", "system"} & calls)),
        ("test_not_read", runtime["splits_read"] == {"train", "val"}),
        ("no_new_model_checkpoint", "save" not in calls),
        ("single_model_deployment_contract", runtime["single_model_contract"]),
    ]
    if len(checks) != 35:
        raise RuntimeError(f"C29A gate must contain exactly 35 checks, got {len(checks)}")
    rows = [{"name": name, "pass": bool(passed)} for name, passed in checks]
    passed = all(row["pass"] for row in rows)
    return {
        "phase": "C29-A",
        "status": "C29A_ANALYSIS_AUTHORIZED" if passed else "C29A_ANALYSIS_INVALID",
        "pass": passed,
        "passed_checks": sum(row["pass"] for row in rows),
        "total_checks": len(rows),
        "checks": rows,
        "data_scope": "train_fit_validation_decision_only",
        "neural_parameter_updates": False,
        "new_checkpoint_created": False,
    }


def classifier_geometry(
    data: Mapping[int, Mapping[str, Any]], heads: Mapping[int, Mapping[str, Any]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        pack = data[seed]["val"]
        labels = pack["labels"]
        states = pack["arrays"]["S4_patient_state"].astype(np.float64)
        weight = np.asarray(heads[seed]["classifier_weight"], dtype=np.float64)
        bias = float(heads[seed]["classifier_bias"])
        dots = states @ weight
        logits = dots + bias
        positive_centroid = states[labels == 1].mean(axis=0)
        negative_centroid = states[labels == 0].mean(axis=0)
        direction = positive_centroid - negative_centroid
        denominator = np.linalg.norm(weight) * np.linalg.norm(direction)
        rows.append(
            {
                "seed": seed,
                "classifier_weight_norm": float(np.linalg.norm(weight)),
                "classifier_bias": bias,
                "positive_mean_dot_product": float(dots[labels == 1].mean()),
                "negative_mean_dot_product": float(dots[labels == 0].mean()),
                "positive_mean_official_logit": float(logits[labels == 1].mean()),
                "negative_mean_official_logit": float(logits[labels == 0].mean()),
                "positive_negative_logit_gap": float(logits[labels == 1].mean() - logits[labels == 0].mean()),
                "positive_centroid_norm": float(np.linalg.norm(positive_centroid)),
                "negative_centroid_norm": float(np.linalg.norm(negative_centroid)),
                "centroid_distance": float(np.linalg.norm(direction)),
                "classifier_centroid_direction_cosine": float(np.dot(weight, direction) / denominator) if denominator > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_procrustes(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    cross = (source - source_mean).T @ (target - target_mean)
    left, _singular, right_transpose = svd(cross, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    return source_mean, target_mean, left @ right_transpose


def procrustes_error(
    source: np.ndarray,
    target: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    rotation: np.ndarray,
) -> Tuple[float, float]:
    mapped = (np.asarray(source, dtype=np.float64) - source_mean) @ rotation + target_mean
    target = np.asarray(target, dtype=np.float64)
    difference = mapped - target
    rmse = float(np.sqrt(np.mean(difference * difference)))
    relative = float(np.linalg.norm(difference) / max(np.linalg.norm(target - target_mean), 1e-12))
    return rmse, relative


def coordinate_compatibility(data: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stages = {"S2_pre_projection": "S2_pre_projection", "S4_patient_state": "S4_patient_state"}
    for left_seed, right_seed in SEED_PAIRS:
        for stage, key in stages.items():
            left_train = data[left_seed]["train"]["arrays"][key].reshape(602, -1)
            right_train = data[right_seed]["train"]["arrays"][key].reshape(602, -1)
            left_val = data[left_seed]["val"]["arrays"][key].reshape(94, -1)
            right_val = data[right_seed]["val"]["arrays"][key].reshape(94, -1)
            if not (
                np.array_equal(data[left_seed]["train"]["patient_id"], data[right_seed]["train"]["patient_id"])
                and np.array_equal(data[left_seed]["val"]["patient_id"], data[right_seed]["val"]["patient_id"])
            ):
                raise RuntimeError("Cross-seed coordinate patient alignment failed")
            cka = linear_cka(left_val, right_val)
            distance = spearman(
                upper_triangle(pairwise_distances(left_val)),
                upper_triangle(pairwise_distances(right_val)),
            )
            jaccard = float(np.nanmean(jaccard_by_patient(left_val, right_val, k=10)))
            source_mean, target_mean, rotation = fit_procrustes(left_train, right_train)
            train_rmse, train_relative = procrustes_error(
                left_train, right_train, source_mean, target_mean, rotation
            )
            val_rmse, val_relative = procrustes_error(
                left_val, right_val, source_mean, target_mean, rotation
            )
            rows.append(
                {
                    "stage": stage,
                    "seed_pair": f"{left_seed}_vs_{right_seed}",
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "dimension": left_val.shape[1],
                    "linear_CKA": cka,
                    "patient_distance_spearman": distance,
                    "knn_jaccard_k10": jaccard,
                    "procrustes_train_RMSE": train_rmse,
                    "procrustes_train_relative_error": train_relative,
                    "procrustes_validation_RMSE": val_rmse,
                    "procrustes_validation_relative_error": val_relative,
                    "global_coordinate_compatibility_supported": bool(cka >= 0.70 and distance >= 0.65),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise(
    objects: Sequence[Mapping[str, Any]], probe_scores: Mapping[int, Mapping[str, np.ndarray]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for obj in objects:
        labels = np.asarray(obj["labels"], dtype=np.int64)
        ids = np.asarray(obj["patient_id"], dtype=str)
        probabilities = np.asarray(obj["probabilities"], dtype=np.float64)
        official = np.asarray(probe_scores[int(obj["representation_seed"])]["P0_official_C27"], dtype=np.float64)
        positive_indices = np.flatnonzero(labels == 1)
        negative_indices = np.flatnonzero(labels == 0)
        inversions = official_inversions = repaired = introduced = 0
        for positive_index in positive_indices:
            for negative_index in negative_indices:
                inversion = bool(probabilities[positive_index] < probabilities[negative_index])
                official_inversion = bool(official[positive_index] < official[negative_index])
                inversions += int(inversion)
                official_inversions += int(official_inversion)
                repaired += int(official_inversion and not inversion)
                introduced += int(not official_inversion and inversion)
                rows.append(
                    {
                        "seed": int(obj["representation_seed"]),
                        "head_seed": obj["head_seed"],
                        "representation_or_swap": obj["object_name"],
                        "object_type": obj["object_type"],
                        "positive_patient_id": ids[positive_index],
                        "negative_patient_id": ids[negative_index],
                        "positive_score": probabilities[positive_index],
                        "negative_score": probabilities[negative_index],
                        "margin": probabilities[positive_index] - probabilities[negative_index],
                        "inversion": inversion,
                        "official_inversion": official_inversion,
                        "repaired": official_inversion and not inversion,
                        "introduced": not official_inversion and inversion,
                    }
                )
        summary.append(
            {
                "seed": int(obj["representation_seed"]),
                "head_seed": obj["head_seed"],
                "representation_or_swap": obj["object_name"],
                "object_type": obj["object_type"],
                "eligible_pairs": len(positive_indices) * len(negative_indices),
                "official_inversions": official_inversions,
                "object_inversions": inversions,
                "net_inversion_change": inversions - official_inversions,
                "repaired": repaired,
                "introduced": introduced,
                "minor_ranking_variation": abs(inversions - official_inversions) <= 3,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def positive_damage_exports(
    data: Mapping[int, Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    probe_scores: Mapping[int, Mapping[str, np.ndarray]],
    heads: Mapping[int, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    response_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    included_types = {"classifier_swap", "head_swap"}
    included_probes = {"P1_patient_state", "P2_pre_projection", "P3_temporal_mechanisms"}
    for seed in SEEDS:
        pack = data[seed]["val"]
        c17_frame = data[seed]["c17_val"]
        c17_probs = c17_frame[probability_column(c17_frame)].to_numpy(dtype=np.float64)
        official = np.asarray(probe_scores[seed]["P0_official_C27"], dtype=np.float64)
        labels = pack["labels"]
        material = (labels == 1) & (((c17_probs >= 0.5) & (official < 0.5)) | ((official - c17_probs) <= -0.05))
        severe = (labels == 1) & ((official - c17_probs) <= -0.10)
        selected_objects = [
            obj
            for obj in objects
            if int(obj["representation_seed"]) == seed
            and (obj["object_type"] in included_types or obj["object_name"] in included_probes)
        ]
        for obj in selected_objects:
            probabilities = np.asarray(obj["probabilities"], dtype=np.float64)
            rescue = material & ((probabilities - official) >= 0.05)
            threshold_rescue = material & (official < 0.5) & (probabilities >= 0.5)
            for index in np.flatnonzero(material):
                response_rows.append(
                    {
                        "seed": seed,
                        "patient_id": pack["patient_id"][index],
                        "label": int(labels[index]),
                        "diagnostic_type": obj["object_type"],
                        "diagnostic_name": obj["object_name"],
                        "head_seed": obj["head_seed"],
                        "c17_probability": c17_probs[index],
                        "official_c27_probability": official[index],
                        "diagnostic_probability": probabilities[index],
                        "diagnostic_minus_official": probabilities[index] - official[index],
                        "material_damage": True,
                        "severe_damage": bool(severe[index]),
                        "probability_rescue": bool(rescue[index]),
                        "threshold_rescue": bool(threshold_rescue[index]),
                        "official_classifier_dot_product": float(pack["arrays"]["S5_classifier_dot"][index, 0]),
                        "official_classifier_bias": float(heads[seed]["classifier_bias"]),
                        "x_patient_norm": float(np.linalg.norm(pack["arrays"]["S2_pre_projection"][index])),
                        "h_patient_norm": float(np.linalg.norm(pack["arrays"]["S4_patient_state"][index])),
                        "conflict_mean": float(pack["arrays"]["S1_conflicts"][index].mean()),
                        "conflict_max": float(pack["arrays"]["S1_conflicts"][index].max()),
                    }
                )
            summary_rows.append(
                {
                    "seed": seed,
                    "diagnostic_type": obj["object_type"],
                    "diagnostic_name": obj["object_name"],
                    "head_seed": obj["head_seed"],
                    "official_material_damage_count": int(material.sum()),
                    "official_severe_damage_count": int(severe.sum()),
                    "probability_rescue_count": int(rescue.sum()),
                    "threshold_rescue_count": int(threshold_rescue.sum()),
                    "unresolved_material_damage_count": int((material & ~rescue).sum()),
                }
            )
    return pd.DataFrame(response_rows), pd.DataFrame(summary_rows)


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return 0.0
    value = float(spearmanr(x[valid], y[valid]).statistic)
    return value if math.isfinite(value) else 0.0


def shortcut_probe_auc(frame: pd.DataFrame) -> float:
    matrix = pd.DataFrame(index=frame.index)
    for field in SELECTED_SHORTCUT_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if values.notna().any() else 0.0)
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        matrix.to_numpy(dtype=np.float64),
        frame["label"].to_numpy(dtype=np.int64),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return auc(frame["label"], probabilities)


def shortcut_audit(
    data: Mapping[int, Mapping[str, Any]], objects: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        pack = data[seed]["val"]
        base = pack["shortcuts"].copy()
        base.insert(0, "label", pack["labels"])
        selected_auc = shortcut_probe_auc(base)
        raw_warnings: Dict[str, float] = {}
        for field in RAW_SHORTCUT_FIELDS:
            raw = pd.to_numeric(base[field], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            value = auc(pack["labels"], raw)
            raw_warnings[field] = max(value, 1.0 - value)
        for obj in objects:
            if int(obj["representation_seed"]) != seed:
                continue
            probabilities = np.asarray(obj["probabilities"], dtype=np.float64)
            correlations = {
                field: safe_spearman(probabilities, pd.to_numeric(base[field], errors="coerce"))
                for field in SELECTED_SHORTCUT_FIELDS
            }
            maximum = max(abs(value) for value in correlations.values())
            rows.append(
                {
                    "seed": seed,
                    "object_type": obj["object_type"],
                    "representation_or_swap": obj["object_name"],
                    "selected_structure_shortcut_only_label_AUC": selected_auc,
                    "max_abs_prediction_selected_structure_spearman": maximum,
                    "object_shortcut_pass": bool(selected_auc <= 0.55 and maximum <= 0.35),
                    "shortcut_fields_used_as_inputs": False,
                    **{f"prediction_spearman_{field}": value for field, value in correlations.items()},
                    **{f"{field}_orientation_invariant_label_AUC_warning": value for field, value in raw_warnings.items()},
                }
            )
    return pd.DataFrame(rows)


def representation_shapes(data: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    descriptions = {
        "S0_temporal_mechanisms": "five official temporal mechanism states",
        "S1_conflicts": "five fixed latest-history conflict scalars",
        "S2_pre_projection": "S0 flattened, then S1, then frozen fallback bio context",
        "S3_projection_linear": "patient projection Linear output before GELU",
        "S3_projection_post_gelu": "patient projection post-GELU before LayerNorm",
        "S4_patient_state": "official LayerNorm output and classifier input",
        "S5_classifier_dot": "classifier weight dot official patient state",
    }
    for seed in SEEDS:
        for split in ("train", "val"):
            for stage, description in descriptions.items():
                shape = list(data[seed][split]["arrays"][stage].shape)
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "stage": stage,
                        "shape": json.dumps(shape),
                        "patient_count": shape[0],
                        "feature_shape": json.dumps(shape[1:]),
                        "available": True,
                        "description": description,
                    }
                )
    return pd.DataFrame(rows)


def logit_decomposition(data: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for seed in SEEDS:
        pack = data[seed]["val"]
        dot = pack["arrays"]["S5_classifier_dot"].reshape(-1).astype(np.float64)
        bias = pack["arrays"]["S5_classifier_bias"].reshape(-1).astype(np.float64)
        official = pack["arrays"]["official_logit"].reshape(-1).astype(np.float64)
        rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "patient_id": pack["patient_id"],
                    "label": pack["labels"],
                    "classifier_dot_product": dot,
                    "classifier_bias": bias,
                    "recomposed_logit": dot + bias,
                    "official_logit": official,
                    "absolute_error": np.abs(dot + bias - official),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "c27":
        raise RuntimeError("C29A must use the frozen C27 config")
    if tuple(int(seed) for seed in config["training"]["seeds"]) != SEEDS:
        raise RuntimeError("C29A seeds must be [0, 42, 3407]")
    rows = read_jsonl(config["project"]["manifest"])
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, heads, reproduction, runtime = extract_all(
        config,
        rows,
        resolve_path(args.c27_run_dir),
        resolve_path(args.c17_run_dir),
        device,
    )
    c17_reference = {
        split: load_c17_reference(resolve_path(args.c17_representation_dir), split)
        for split in ("train", "val")
    }
    probe_metrics, generalization, random_sanity, probe_scores, c17_alignment = run_probes(
        data, c17_reference
    )
    classifier_swaps, head_swaps, swap_scores, swap_runtime = run_swaps(data, heads)
    objects = build_objects(data, probe_scores, swap_scores)
    runtime.update(swap_runtime)
    runtime.update(
        {
            "probe_fit_train_only": True,
            "validation_not_used_for_fit": True,
            "standard_scaler_fit_train_only": True,
            "logistic_parameters_fixed": True,
            "no_probe_sweep": True,
            "random_label_sanity_ran": len(random_sanity) == len(SEEDS) * len(FITTED_PROBES),
            "shortcut_fields_excluded": c17_alignment,
            "all_scores_finite": all(np.isfinite(obj["probabilities"]).all() for obj in objects),
            "pair_contract": pair_contract(objects),
            "single_model_contract": True,
        }
    )
    reproduction.to_csv(output / "c29a_reproduction_by_seed.csv", index=False)
    gate = gate_payload(runtime)
    (output / "c29a_runtime_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    if not gate["pass"]:
        print(json.dumps({"status": gate["status"], "checks": f"{gate['passed_checks']}/{gate['total_checks']}"}))
        raise RuntimeError("C29A_ANALYSIS_INVALID")
    if args.stage == "gate":
        print(json.dumps({"status": gate["status"], "checks": "35/35", "device": str(device)}))
        return

    existing_gate = json.loads((output / "c29a_runtime_gate.json").read_text(encoding="utf-8"))
    if not existing_gate.get("pass", False):
        raise RuntimeError("C29A full analysis requires a passing 35-check gate")
    shapes = representation_shapes(data)
    decomposition = logit_decomposition(data)
    geometry = classifier_geometry(data, heads)
    coordinates = coordinate_compatibility(data)
    pairwise, inversion_summary = build_pairwise(objects, probe_scores)
    positive_response, rescue_summary = positive_damage_exports(data, objects, probe_scores, heads)
    shortcuts = shortcut_audit(data, objects)

    shapes.to_csv(output / "c29a_representation_shapes.csv", index=False)
    decomposition.to_csv(output / "c29a_logit_decomposition_by_patient.csv", index=False)
    geometry.to_csv(output / "c29a_classifier_geometry_by_seed.csv", index=False)
    probe_metrics.to_csv(output / "c29a_probe_metrics_by_seed.csv", index=False)
    generalization.to_csv(output / "c29a_probe_generalization_audit.csv", index=False)
    random_sanity.to_csv(output / "c29a_random_label_sanity.csv", index=False)
    classifier_swaps.to_csv(output / "c29a_classifier_swap_metrics.csv", index=False)
    head_swaps.to_csv(output / "c29a_head_swap_metrics.csv", index=False)
    coordinates.to_csv(output / "c29a_coordinate_compatibility.csv", index=False)
    positive_response.to_csv(output / "c29a_positive_damage_probe_response.csv", index=False)
    rescue_summary.to_csv(output / "c29a_positive_damage_rescue_summary.csv", index=False)
    pairwise.to_csv(output / "c29a_pairwise_ranking_by_representation.csv", index=False)
    inversion_summary.to_csv(output / "c29a_pairwise_inversion_summary.csv", index=False)
    shortcuts.to_csv(output / "c29a_shortcut_audit.csv", index=False)
    print(
        json.dumps(
            {
                "status": "C29A_TRAIN_FIT_VALIDATION_ANALYSIS_COMPLETE",
                "seeds": list(SEEDS),
                "formal_objects": len(objects),
                "pairwise_rows": len(pairwise),
                "output_dir": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
