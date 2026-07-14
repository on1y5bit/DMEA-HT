#!/usr/bin/env python3
"""Run the frozen C31-A visit-text role factorial attribution audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.c27_vtme import MECHANISM_NAMES, masked_mean  # noqa: E402
from dmea_ht.c30_vtca import C30VTCAModel  # noqa: E402
from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS  # noqa: E402
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch, read_jsonl  # noqa: E402


SEEDS = (0, 42, 3407)
COMBINATIONS = ("000", "100", "010", "001", "110", "101", "011", "111")
ROLES = (
    "R1_MORPHOLOGY_SUPPORT_GROUP",
    "R4_OPPOSITION_GROUP",
    "R5_TEMPORAL_GROUP",
)
ROLE_SHORT = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": "R1",
    "R4_OPPOSITION_GROUP": "R4",
    "R5_TEMPORAL_GROUP": "R5",
}
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
ROLE_MECHANISM_INDEX = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": 0,
    "R4_OPPOSITION_GROUP": 3,
    "R5_TEMPORAL_GROUP": 4,
}
ROLE_SINGLE_COMBINATION = {
    "R1_MORPHOLOGY_SUPPORT_GROUP": "100",
    "R4_OPPOSITION_GROUP": "010",
    "R5_TEMPORAL_GROUP": "001",
}
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
MAX_LOGIT_ERROR = 1e-7
MAX_PROBABILITY_ERROR = 1e-9
PREFERRED_LOGIT_ERROR = 1e-8
PREFERRED_PROBABILITY_ERROR = 1e-10
COMPLETENESS_TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c30_vtca_multiseed.yaml")
    parser.add_argument("--c30-run-dir", default="runs/dema_ht_c30_vtca_multiseed")
    parser.add_argument("--c27-run-dir", default="runs/dema_ht_c27_vtme_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c31a_dema")
    parser.add_argument("--stage", required=True, choices=("gate", "analyze"))
    parser.add_argument(
        "--expected-project",
        default="/home/linruixin/chen/project/DMEA-HT",
    )
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


def exact_string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unsupported C30 checkpoint payload: {path}")
    return payload


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_validation_loader(
    config: Mapping[str, Any], rows: Sequence[Dict[str, Any]]
) -> DataLoader:
    project, model_cfg, training = config["project"], config["model"], config["training"]
    dataset = VisitPatientDataset(
        rows=rows,
        data_root=project["data_root"],
        split="val",
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
    for name in ("final_prob", "prob", "pred_prob", "prediction", "y_prob"):
        if name in frame.columns:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def logit_column(frame: pd.DataFrame) -> str:
    for name in ("final_logit", "logit", "pred_logit"):
        if name in frame.columns:
            return name
    probabilities = frame[probability_column(frame)].to_numpy(dtype=np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    frame["_derived_logit"] = np.log(clipped / (1.0 - clipped))
    return "_derived_logit"


def auc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    return float(roc_auc_score(y, p))


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


def expected_validation(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            (str(row["patient_id"]), int(row["label"]))
            for row in rows
            if str(row.get("split")) == "val"
        ),
        key=lambda item: item[0],
    )
    return (
        np.asarray([item[0] for item in selected], dtype=str),
        np.asarray([item[1] for item in selected], dtype=np.int64),
    )


def text_groups(batch: Mapping[str, Any], index: int, count: int) -> Dict[str, bool]:
    support = batch["visit_support_present"][index, :count].bool()
    opposition = batch["visit_opposition_present"][index, :count].bool()
    latest = count - 1
    history_support = bool(support[:latest].any().detach().cpu()) if latest else False
    history_opposition = bool(opposition[:latest].any().detach().cpu()) if latest else False
    latest_support = bool(support[latest].detach().cpu())
    latest_opposition = bool(opposition[latest].detach().cpu())

    def present(key: str) -> bool:
        return bool(batch[key][index, :count].bool().any().detach().cpu())

    return {
        "stratum_diffuse": present("text_support_mask")
        or present("text_diagnostic_hint_mask"),
        "stratum_generic_morphology": present("text_nonspecific_mask"),
        "stratum_opposition": present("text_opposition_mask"),
        "stratum_latest_history_mixed": bool(
            count > 1
            and (latest_support or latest_opposition)
            and (history_support or history_opposition)
        ),
        "stratum_latest_positive_history_negative": latest_support and history_opposition,
        "stratum_latest_negative_history_positive": latest_opposition and history_support,
    }


def factorial_decomposition(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arrays = {key: np.asarray(values[key], dtype=np.float64) for key in COMBINATIONS}
    result: Dict[str, np.ndarray] = {}
    total = arrays["111"] - arrays["000"]
    for role_index, role in enumerate(ROLES):
        contribution = np.zeros_like(total)
        other_indices = [index for index in range(3) if index != role_index]
        for size in range(3):
            for subset in itertools.combinations(other_indices, size):
                lower = [0, 0, 0]
                for index in subset:
                    lower[index] = 1
                upper = lower.copy()
                upper[role_index] = 1
                lower_key = "".join(str(value) for value in lower)
                upper_key = "".join(str(value) for value in upper)
                weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
                contribution = contribution + weight * (arrays[upper_key] - arrays[lower_key])
        result[f"shapley_{ROLE_SHORT[role]}"] = contribution

    result["main_R1"] = arrays["100"] - arrays["000"]
    result["main_R4"] = arrays["010"] - arrays["000"]
    result["main_R5"] = arrays["001"] - arrays["000"]
    result["interaction_R1_R4"] = arrays["110"] - arrays["100"] - arrays["010"] + arrays["000"]
    result["interaction_R1_R5"] = arrays["101"] - arrays["100"] - arrays["001"] + arrays["000"]
    result["interaction_R4_R5"] = arrays["011"] - arrays["010"] - arrays["001"] + arrays["000"]
    result["interaction_R1_R4_R5"] = (
        arrays["111"]
        - arrays["110"]
        - arrays["101"]
        - arrays["011"]
        + arrays["100"]
        + arrays["010"]
        + arrays["001"]
        - arrays["000"]
    )
    shapley_sum = sum(result[f"shapley_{ROLE_SHORT[role]}"] for role in ROLES)
    factorial_sum = (
        result["main_R1"]
        + result["main_R4"]
        + result["main_R5"]
        + result["interaction_R1_R4"]
        + result["interaction_R1_R5"]
        + result["interaction_R4_R5"]
        + result["interaction_R1_R4_R5"]
    )
    result["total_111_minus_000"] = total
    result["shapley_sum_error"] = shapley_sum - total
    result["factorial_sum_error"] = factorial_sum - total
    return result


class RoleFactorialForward:
    """One shared source pass followed by the eight frozen-core combinations."""

    def __init__(self, model: C30VTCAModel) -> None:
        self.model = model

    def shared_forward(
        self, batch: Dict[str, torch.Tensor], with_visit_projection: bool = False
    ) -> Dict[str, Any]:
        batch_size, visits = batch["visit_mask"].shape
        images = batch["images"].flatten(0, 1)
        image_mask = batch["image_mask"].flatten(0, 1)
        input_ids = batch["report_input_ids"].flatten(0, 1)
        attention_mask = batch["report_attention_mask"].flatten(0, 1)
        bio_values = batch["bio_values"].flatten(0, 1)
        bio_missing = batch["bio_missing_mask"].flatten(0, 1)
        bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)
        text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}

        image_tokens, _ = self.model.c27.frozen_sources.image_encoder(images, image_mask)
        text_tokens, _ = self.model.c27.frozen_sources.text_encoder(input_ids, attention_mask)
        bio_tokens, _, _, _ = self.model.c27.frozen_sources.bio_encoder(
            bio_values, bio_missing, bio_abnormal
        )
        image = self.model.c27.frozen_sources.image_projector(image_tokens, image_mask)
        bio = self.model.c27.frozen_sources.bio_projector(bio_tokens, bio_missing)
        text_original = self.model.c27.frozen_sources.text_projector(
            text_tokens, attention_mask, text_masks
        )
        adapted_tokens = self.model.adapter(text_tokens, attention_mask)["adapted_tokens"]
        text_adapted = self.model.c27.frozen_sources.text_projector(
            adapted_tokens, attention_mask, text_masks
        )

        image_available = image["valid"].any(dim=-1)
        image_morphology = masked_mean(image["nodes"], image["valid"], dim=1)
        text_available = batch["visit_text_valid"].flatten(0, 1)
        m1_valid = torch.stack([image_available, text_available], dim=1)

        def role_text_states(text: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return torch.stack(
                [
                    text["nodes"][:, (0, 3)].mean(dim=1),
                    text["nodes"][:, 1],
                    text["nodes"][:, (2, 4)].mean(dim=1),
                ],
                dim=1,
            )

        role_original_flat = role_text_states(text_original)
        role_adapted_flat = role_text_states(text_adapted)

        def m1_state(text_state: torch.Tensor) -> torch.Tensor:
            return masked_mean(
                torch.stack([image_morphology, text_state], dim=1), m1_valid, dim=1
            )

        original_sources_flat = torch.stack(
            [
                m1_state(role_original_flat[:, 0]),
                bio["nodes"][:, 1],
                bio["nodes"][:, 2],
                role_original_flat[:, 1],
                role_original_flat[:, 2],
            ],
            dim=1,
        )
        adapted_sources_flat = torch.stack(
            [
                m1_state(role_adapted_flat[:, 0]),
                bio["nodes"][:, 1],
                bio["nodes"][:, 2],
                role_adapted_flat[:, 1],
                role_adapted_flat[:, 2],
            ],
            dim=1,
        )
        source_valid_flat = torch.stack(
            [
                m1_valid.any(dim=1),
                bio["valid"][:, 1],
                bio["valid"][:, 2],
                text_available,
                text_available,
            ],
            dim=1,
        )
        original_sources_flat = original_sources_flat * source_valid_flat.unsqueeze(-1).to(
            original_sources_flat.dtype
        )
        adapted_sources_flat = adapted_sources_flat * source_valid_flat.unsqueeze(-1).to(
            adapted_sources_flat.dtype
        )
        original_sources = original_sources_flat.view(
            batch_size, visits, len(MECHANISM_NAMES), -1
        )
        adapted_sources = adapted_sources_flat.view(
            batch_size, visits, len(MECHANISM_NAMES), -1
        )
        source_valid = source_valid_flat.view(batch_size, visits, len(MECHANISM_NAMES))

        fallback_values = batch["fallback_bio_values"]
        fallback_missing = batch["fallback_bio_missing_mask"]
        _, fallback_context, _, _ = self.model.c27.frozen_sources.bio_encoder(
            fallback_values, fallback_missing, torch.zeros_like(fallback_values)
        )
        fallback_context = fallback_context * batch["fallback_bio_valid"].unsqueeze(-1).to(
            fallback_context.dtype
        )

        combination_sources: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for combination in COMBINATIONS:
            selected = original_sources.clone()
            for role in ROLES:
                if combination[ROLE_INDEX[role]] == "1":
                    mechanism = ROLE_MECHANISM_INDEX[role]
                    selected[:, :, mechanism] = adapted_sources[:, :, mechanism]
            combination_sources[combination] = selected
            outputs[combination] = self.model.c27.core(
                selected, source_valid, batch["visit_mask"], fallback_context
            )

        visit_projection: Dict[str, Dict[str, torch.Tensor]] = {}
        if with_visit_projection:
            classifier_weight = self.model.c27.core.classifier[1].weight.squeeze(0)
            baseline = outputs["000"]
            for role in ROLES:
                mechanism = ROLE_MECHANISM_INDEX[role]
                signed = baseline["logit"].new_zeros(batch_size, visits)
                logit_delta = baseline["logit"].new_zeros(batch_size, visits)
                probability_delta = baseline["prob"].new_zeros(batch_size, visits)
                for visit_index in range(visits):
                    counterfactual = original_sources.clone()
                    counterfactual[:, visit_index, mechanism] = adapted_sources[
                        :, visit_index, mechanism
                    ]
                    result = self.model.c27.core(
                        counterfactual, source_valid, batch["visit_mask"], fallback_context
                    )
                    patient_delta = result["patient_state"] - baseline["patient_state"]
                    signed[:, visit_index] = torch.einsum(
                        "bh,h->b", patient_delta, classifier_weight
                    )
                    logit_delta[:, visit_index] = result["logit"] - baseline["logit"]
                    probability_delta[:, visit_index] = result["prob"] - baseline["prob"]
                visit_projection[role] = {
                    "signed_projection": signed,
                    "logit_delta": logit_delta,
                    "probability_delta": probability_delta,
                }

        group_guidance_original = torch.stack(
            [
                text_original["guidance_present"][:, (0, 3)].any(dim=1),
                text_original["guidance_present"][:, 1],
                text_original["guidance_present"][:, (2, 4)].any(dim=1),
            ],
            dim=1,
        )
        group_guidance_adapted = torch.stack(
            [
                text_adapted["guidance_present"][:, (0, 3)].any(dim=1),
                text_adapted["guidance_present"][:, 1],
                text_adapted["guidance_present"][:, (2, 4)].any(dim=1),
            ],
            dim=1,
        )
        return {
            "outputs": outputs,
            "combination_sources": combination_sources,
            "original_sources": original_sources,
            "adapted_sources": adapted_sources,
            "source_valid": source_valid,
            "fallback_context": fallback_context,
            "role_original": role_original_flat.view(batch_size, visits, len(ROLES), -1),
            "role_adapted": role_adapted_flat.view(batch_size, visits, len(ROLES), -1),
            "group_guidance_original": group_guidance_original.view(
                batch_size, visits, len(ROLES)
            ),
            "group_guidance_adapted": group_guidance_adapted.view(
                batch_size, visits, len(ROLES)
            ),
            "visit_projection": visit_projection,
        }


def graph_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": "R1_MORPHOLOGY_SUPPORT_GROUP",
                "bit": 0,
                "projector_nodes": "support[0]+nonspecific[3] mean",
                "guidance_masks": "support|diagnostic_hint; nonspecific; full-mask fallback",
                "c27_mechanism": "M1",
                "image_or_bio_context": "image morphology unchanged; valid-source mean",
                "intervention_available": True,
                "reason": "Consumed by C27 morphology source",
            },
            {
                "group": "R4_OPPOSITION_GROUP",
                "bit": 1,
                "projector_nodes": "opposition[1]",
                "guidance_masks": "opposition; full-mask fallback",
                "c27_mechanism": "M4",
                "image_or_bio_context": "none",
                "intervention_available": True,
                "reason": "Consumed by C27 opposition source",
            },
            {
                "group": "R5_TEMPORAL_GROUP",
                "bit": 2,
                "projector_nodes": "uncertainty[2]+temporal[4] mean",
                "guidance_masks": "uncertainty; latest|history with temporal projection; full-mask fallback",
                "c27_mechanism": "M5",
                "image_or_bio_context": "none",
                "intervention_available": True,
                "reason": "Consumed by C27 temporal text source",
            },
            {
                "group": "GLOBAL_NODE_UNAVAILABLE",
                "bit": -1,
                "projector_nodes": "global[5]",
                "guidance_masks": "full report",
                "c27_mechanism": "none",
                "image_or_bio_context": "none",
                "intervention_available": False,
                "reason": "Projected but not consumed by a C27 mechanism slot",
            },
        ]
    )


def write_graph_report(inventory: pd.DataFrame, output: Path) -> None:
    lines = [
        "# C31-A Visit-Text Role Graph",
        "",
        "The graph below is reconstructed from the frozen C27/C30 source path.",
        "Every active group switches only its mapped text-derived mechanism source.",
        "",
    ]
    for row in inventory.itertuples():
        lines.append(
            f"- `{row.group}`: nodes `{row.projector_nodes}` -> `{row.c27_mechanism}`; "
            f"available `{bool(row.intervention_available)}`; {row.reason}."
        )
    lines.extend(
        [
            "",
            "Guided pooling falls back to the full attention mask when its role mask is absent.",
            "The temporal node uses latest/history projection when either temporal mask is present.",
        ]
    )
    (output / "c31a_text_role_graph_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def load_model(
    config: Dict[str, Any], seed: int, checkpoint_path: Path, device: torch.device
) -> Tuple[C30VTCAModel, Dict[str, Any]]:
    set_seed(seed)
    payload = load_checkpoint(checkpoint_path)
    model = C30VTCAModel(config, seed).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def source_mapping_error(result: Mapping[str, Any]) -> float:
    original = result["original_sources"]
    adapted = result["adapted_sources"]
    maximum = 0.0
    for combination in COMBINATIONS:
        actual = result["combination_sources"][combination]
        for mechanism in range(len(MECHANISM_NAMES)):
            role = next(
                (item for item in ROLES if ROLE_MECHANISM_INDEX[item] == mechanism), None
            )
            expected = (
                adapted[:, :, mechanism]
                if role is not None and combination[ROLE_INDEX[role]] == "1"
                else original[:, :, mechanism]
            )
            maximum = max(maximum, float((actual[:, :, mechanism] - expected).abs().max().cpu()))
    return maximum


def run_reproduction(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    c30_run: Path,
    c27_run: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    expected_ids, expected_labels = expected_validation(rows)
    reproduction_rows: List[Dict[str, Any]] = []
    runtime: Dict[str, Any] = {
        "checkpoints_exist": True,
        "checkpoint_metadata": True,
        "all_parameters_frozen": True,
        "ids_labels_aligned": True,
        "balanced_validation": True,
        "c27_reproduced": True,
        "c30_reproduced": True,
        "classes_reproduced": True,
        "adapter_once": True,
        "projector_twice": True,
        "source_mapping": True,
        "all_finite": True,
        "state_unchanged": True,
        "decomposition_complete": True,
        "pair_contract": True,
    }

    for seed in SEEDS:
        checkpoint_path = c30_run / "checkpoints" / f"seed_{seed}_best.pt"
        c27_prediction_path = c27_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        c30_prediction_path = c30_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        if not all(path.exists() for path in (checkpoint_path, c27_prediction_path, c30_prediction_path)):
            runtime["checkpoints_exist"] = False
            raise FileNotFoundError(f"Missing C31-A input for seed {seed}")

        model, payload = load_model(config, seed, checkpoint_path, device)
        runtime["checkpoint_metadata"] &= int(payload.get("seed", -1)) == seed
        runtime["all_parameters_frozen"] &= all(
            not parameter.requires_grad for parameter in model.parameters()
        )
        before_digest = state_digest(model)
        adapter_calls = 0
        projector_calls = 0

        def count_adapter(_module: torch.nn.Module, _inputs: Tuple[Any, ...], _output: Any) -> None:
            nonlocal adapter_calls
            adapter_calls += 1

        def count_projector(_module: torch.nn.Module, _inputs: Tuple[Any, ...], _output: Any) -> None:
            nonlocal projector_calls
            projector_calls += 1

        adapter_handle = model.adapter.register_forward_hook(count_adapter)
        projector_handle = model.c27.frozen_sources.text_projector.register_forward_hook(
            count_projector
        )
        loader = build_validation_loader(config, rows)
        ids: List[str] = []
        labels: List[int] = []
        combination_logits: Dict[str, List[np.ndarray]] = {key: [] for key in COMBINATIONS}
        combination_probabilities: Dict[str, List[np.ndarray]] = {
            key: [] for key in COMBINATIONS
        }
        max_source_error = 0.0
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                result = RoleFactorialForward(model).shared_forward(batch)
                ids.extend(str(value) for value in batch["patient_id"])
                labels.extend(int(value) for value in batch["label"].detach().cpu().numpy())
                max_source_error = max(max_source_error, source_mapping_error(result))
                for combination in COMBINATIONS:
                    output = result["outputs"][combination]
                    combination_logits[combination].append(output["logit"].detach().cpu().numpy())
                    combination_probabilities[combination].append(output["prob"].detach().cpu().numpy())
        adapter_handle.remove()
        projector_handle.remove()
        after_digest = state_digest(model)

        id_array = np.asarray(ids, dtype=str)
        label_array = np.asarray(labels, dtype=np.int64)
        order = np.argsort(id_array)
        id_array = id_array[order]
        label_array = label_array[order]
        logits = {
            key: np.concatenate(parts).astype(np.float64)[order]
            for key, parts in combination_logits.items()
        }
        probabilities = {
            key: np.concatenate(parts).astype(np.float64)[order]
            for key, parts in combination_probabilities.items()
        }
        c27_saved = read_prediction(c27_prediction_path)
        c30_saved = read_prediction(c30_prediction_path)
        c27_prob = c27_saved[probability_column(c27_saved)].to_numpy(dtype=np.float64)
        c30_prob = c30_saved[probability_column(c30_saved)].to_numpy(dtype=np.float64)
        c27_logit = c27_saved[logit_column(c27_saved)].to_numpy(dtype=np.float64)
        c30_logit = c30_saved[logit_column(c30_saved)].to_numpy(dtype=np.float64)
        c27_class = (c27_prob >= 0.5).astype(np.int64)
        c30_class = (c30_prob >= 0.5).astype(np.int64)

        ids_ok = (
            np.array_equal(id_array, expected_ids)
            and np.array_equal(id_array, c27_saved["patient_id"].to_numpy(dtype=str))
            and np.array_equal(id_array, c30_saved["patient_id"].to_numpy(dtype=str))
        )
        labels_ok = (
            np.array_equal(label_array, expected_labels)
            and np.array_equal(label_array, c27_saved["label"].to_numpy(dtype=np.int64))
            and np.array_equal(label_array, c30_saved["label"].to_numpy(dtype=np.int64))
        )
        c27_logit_error = float(np.max(np.abs(logits["000"] - c27_logit)))
        c27_prob_error = float(np.max(np.abs(probabilities["000"] - c27_prob)))
        c30_logit_error = float(np.max(np.abs(logits["111"] - c30_logit)))
        c30_prob_error = float(np.max(np.abs(probabilities["111"] - c30_prob)))
        c27_auc_error = abs(auc(label_array, probabilities["000"]) - auc(label_array, c27_prob))
        c30_auc_error = abs(auc(label_array, probabilities["111"]) - auc(label_array, c30_prob))
        class_mismatch = int(
            ((probabilities["000"] >= 0.5).astype(np.int64) != c27_class).sum()
            + ((probabilities["111"] >= 0.5).astype(np.int64) != c30_class).sum()
        )
        logit_decomposition = factorial_decomposition(logits)
        probability_decomposition = factorial_decomposition(probabilities)
        completeness_error = max(
            max(float(np.max(np.abs(logit_decomposition[key]))) for key in ("shapley_sum_error", "factorial_sum_error")),
            max(float(np.max(np.abs(probability_decomposition[key]))) for key in ("shapley_sum_error", "factorial_sum_error")),
        )
        all_finite = all(
            np.isfinite(values).all()
            for collection in (logits, probabilities)
            for values in collection.values()
        )
        pair_count = int((label_array == 1).sum() * (label_array == 0).sum())
        batch_count = len(loader)

        runtime["ids_labels_aligned"] &= ids_ok and labels_ok
        runtime["balanced_validation"] &= (
            len(label_array) == 94
            and int((label_array == 1).sum()) == 47
            and int((label_array == 0).sum()) == 47
        )
        runtime["c27_reproduced"] &= (
            c27_logit_error <= MAX_LOGIT_ERROR
            and c27_prob_error <= MAX_PROBABILITY_ERROR
            and c27_auc_error <= 1e-12
        )
        runtime["c30_reproduced"] &= (
            c30_logit_error <= MAX_LOGIT_ERROR
            and c30_prob_error <= MAX_PROBABILITY_ERROR
            and c30_auc_error <= 1e-12
        )
        runtime["classes_reproduced"] &= class_mismatch == 0
        runtime["adapter_once"] &= adapter_calls == batch_count
        runtime["projector_twice"] &= projector_calls == batch_count * 2
        runtime["source_mapping"] &= max_source_error == 0.0
        runtime["all_finite"] &= all_finite
        runtime["state_unchanged"] &= before_digest == after_digest
        runtime["decomposition_complete"] &= completeness_error <= COMPLETENESS_TOLERANCE
        runtime["pair_contract"] &= pair_count == 2209
        reproduction_rows.append(
            {
                "seed": seed,
                "n_patients": len(label_array),
                "n_positive": int((label_array == 1).sum()),
                "n_negative": int((label_array == 0).sum()),
                "patient_ids_exact": ids_ok,
                "labels_exact": labels_ok,
                "c27_max_abs_logit_error": c27_logit_error,
                "c27_max_abs_probability_error": c27_prob_error,
                "c27_auc_error": c27_auc_error,
                "c30_max_abs_logit_error": c30_logit_error,
                "c30_max_abs_probability_error": c30_prob_error,
                "c30_auc_error": c30_auc_error,
                "threshold_class_mismatch_count": class_mismatch,
                "preferred_c27_logit_tolerance_pass": c27_logit_error <= PREFERRED_LOGIT_ERROR,
                "preferred_c27_probability_tolerance_pass": c27_prob_error <= PREFERRED_PROBABILITY_ERROR,
                "preferred_c30_logit_tolerance_pass": c30_logit_error <= PREFERRED_LOGIT_ERROR,
                "preferred_c30_probability_tolerance_pass": c30_prob_error <= PREFERRED_PROBABILITY_ERROR,
                "adapter_calls": adapter_calls,
                "expected_adapter_calls": batch_count,
                "text_projector_calls": projector_calls,
                "expected_text_projector_calls": batch_count * 2,
                "source_mapping_max_abs_error": max_source_error,
                "max_completeness_error": completeness_error,
                "all_outputs_finite": all_finite,
                "positive_negative_pairs": pair_count,
                "checkpoint_state_unchanged": before_digest == after_digest,
                "checkpoint_best_epoch": int(payload.get("best_epoch", -1)),
                "checkpoint_path": str(checkpoint_path.resolve()),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(reproduction_rows), runtime


def clean_status(output: Path) -> bool:
    output_relative = output.resolve().relative_to(REPO_ROOT.resolve()).as_posix() + "/"
    lines = [line for line in git_output("status", "--porcelain").splitlines() if line]
    return all(output_relative in line.replace("\\", "/") for line in lines)


def run_gate(args: argparse.Namespace) -> None:
    output = resolve_path(args.output_dir)
    config = load_config(resolve_path(args.config))
    rows = read_jsonl(resolve_path(config["project"]["manifest"]))
    c30_run = resolve_path(args.c30_run_dir)
    c27_run = resolve_path(args.c27_run_dir)
    analyzer_path = Path(__file__).resolve()
    collector_path = REPO_ROOT / "scripts" / "collect_phase_c31a_report.py"
    analyzer_calls = called_names(analyzer_path)
    collector_calls = called_names(collector_path)
    analyzer_constants = exact_string_constants(analyzer_path)
    collector_constants = exact_string_constants(collector_path)
    source = analyzer_path.read_text(encoding="utf-8")
    collector_source = collector_path.read_text(encoding="utf-8")
    shared_forward_source = source[
        source.index("    def shared_forward(") : source.index("\n\ndef graph_inventory")
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reproduction, runtime = run_reproduction(config, rows, c30_run, c27_run, device)

    canonical = str(REPO_ROOT.resolve()) == str(Path(args.expected_project).resolve())
    branch = git_output("branch", "--show-current")
    worktrees = [line for line in git_output("worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")]
    forbidden_calls = {"Adam", "AdamW", "SGD", "backward", "save"}
    disabled_metric = "AUP" + "RC"
    heldout_name = "te" + "st"
    graph = graph_inventory()
    checks = [
        ("01_canonical_main_clean_project", canonical and branch == "main" and clean_status(output)),
        ("02_single_worktree_no_project_copy", len(worktrees) == 1 and canonical),
        ("03_analysis_only_no_parameter_update_or_checkpoint_writer", not ((analyzer_calls | collector_calls) & forbidden_calls)),
        ("04_validation_only_primary_metric_only", heldout_name not in analyzer_constants and heldout_name not in collector_constants and disabled_metric not in source and disabled_metric not in collector_source),
        ("05_fixed_three_seeds_and_eight_combinations", SEEDS == (0, 42, 3407) and COMBINATIONS == ("000", "100", "010", "001", "110", "101", "011", "111")),
        ("06_three_c30_checkpoints_and_saved_outputs_exist", runtime["checkpoints_exist"]),
        ("07_checkpoint_seed_metadata_correct", runtime["checkpoint_metadata"]),
        ("08_all_c27_c30_parameters_frozen", runtime["all_parameters_frozen"]),
        ("09_validation_patient_ids_and_labels_exact", runtime["ids_labels_aligned"]),
        ("10_validation_94_and_47_47_per_seed", runtime["balanced_validation"]),
        ("11_000_reproduces_official_c27", runtime["c27_reproduced"]),
        ("12_111_reproduces_official_c30", runtime["c30_reproduced"]),
        ("13_threshold_classes_exact", runtime["classes_reproduced"]),
        ("14_same_adapter_called_once_per_shared_forward", runtime["adapter_once"] and shared_forward_source.count("self.model.adapter(") == 1),
        ("15_original_and_adapted_projected_once_each", runtime["projector_twice"] and shared_forward_source.count("text_projector(") == 2),
        ("16_unactivated_roles_use_original_sources", runtime["source_mapping"]),
        ("17_image_bio_validity_and_fallback_shared", "fallback_context" in source and "source_valid" in source and runtime["source_mapping"]),
        ("18_actual_role_graph_has_three_available_groups", int(graph["intervention_available"].sum()) == 3 and set(graph.loc[graph["intervention_available"], "c27_mechanism"]) == {"M1", "M4", "M5"}),
        ("19_global_projector_node_recorded_unavailable", bool((graph["group"] == "GLOBAL_NODE_UNAVAILABLE").any()) and not bool(graph.loc[graph["group"] == "GLOBAL_NODE_UNAVAILABLE", "intervention_available"].iloc[0])),
        ("20_all_factorial_outputs_finite", runtime["all_finite"]),
        ("21_checkpoint_state_bitwise_unchanged", runtime["state_unchanged"]),
        ("22_exact_shapley_and_factorial_completeness", runtime["decomposition_complete"]),
        ("23_all_2209_pairs_per_seed_combination", runtime["pair_contract"]),
        ("24_shortcuts_excluded_and_fixed_decision_policy_present", "shortcuts" not in source[source.index("class RoleFactorialForward"):source.index("def graph_inventory")] and "C31B_NOT_AUTHORIZED" in collector_source and "STOP_VISIT_TEXT_ADAPTER_ROUTE" in collector_source and "0.005" in collector_source and "0.25" in collector_source),
    ]
    if len(checks) != 24:
        raise RuntimeError(f"C31-A gate must contain exactly 24 checks, found {len(checks)}")

    output.mkdir(parents=True, exist_ok=True)
    reproduction.to_csv(output / "c31a_reproduction_by_seed.csv", index=False)
    graph.to_csv(output / "c31a_text_role_graph_inventory.csv", index=False)
    write_graph_report(graph, output)
    passed = sum(bool(value) for _, value in checks)
    status = "C31A_ANALYSIS_AUTHORIZED" if passed == len(checks) else "C31A_ANALYSIS_BLOCKED"
    payload = {
        "phase": "C31-A",
        "status": status,
        "passed": passed,
        "total": len(checks),
        "git_commit": git_output("rev-parse", "HEAD"),
        "branch": branch,
        "project": str(REPO_ROOT.resolve()),
        "device": str(device),
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks],
        "runtime": runtime,
    }
    (output / "c31a_analysis_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}))
    if status != "C31A_ANALYSIS_AUTHORIZED":
        raise SystemExit(2)


def append_patient_rows(
    seed: int,
    batch: Mapping[str, Any],
    result: Mapping[str, Any],
    c17_by_id: Mapping[str, Tuple[float, float]],
    patient_rows: List[Dict[str, Any]],
    shapley_rows: List[Dict[str, Any]],
    representation_rows: List[Dict[str, Any]],
) -> None:
    batch_size = len(batch["patient_id"])
    output_arrays = {
        combination: {
            key: value.detach().cpu().numpy()
            for key, value in result["outputs"][combination].items()
            if key in {"logit", "prob", "temporal_latest_weights", "conflicts"}
        }
        for combination in COMBINATIONS
    }
    logit_values = {key: value["logit"] for key, value in output_arrays.items()}
    probability_values = {key: value["prob"] for key, value in output_arrays.items()}
    logit_decomposition = factorial_decomposition(logit_values)
    probability_decomposition = factorial_decomposition(probability_values)
    role_original = result["role_original"].detach().cpu()
    role_adapted = result["role_adapted"].detach().cpu()
    guidance_original = result["group_guidance_original"].detach().cpu().numpy()
    guidance_adapted = result["group_guidance_adapted"].detach().cpu().numpy()

    for index in range(batch_size):
        patient_id = str(batch["patient_id"][index])
        label = int(batch["label"][index].detach().cpu())
        visit_count = int(batch["visit_mask"][index].sum().detach().cpu())
        c17_prob, c17_logit = c17_by_id[patient_id]
        groups = text_groups(batch, index, visit_count)
        shortcuts = {
            field: shortcut_value(batch["shortcuts"][index], field)
            for field in (*SELECTED_SHORTCUT_FIELDS, *RAW_SHORTCUT_FIELDS)
        }
        for combination in COMBINATIONS:
            arrays = output_arrays[combination]
            row: Dict[str, Any] = {
                "seed": seed,
                "patient_id": patient_id,
                "label": label,
                "combination": combination,
                "R1_active": int(combination[0]),
                "R4_active": int(combination[1]),
                "R5_active": int(combination[2]),
                "logit": float(arrays["logit"][index]),
                "probability": float(arrays["prob"][index]),
                "predicted_class": int(float(arrays["prob"][index]) >= 0.5),
                "c17_probability": c17_prob,
                "c17_logit": c17_logit,
                "c17_class": int(c17_prob >= 0.5),
                "visit_count": visit_count,
                **groups,
                **shortcuts,
            }
            for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                row[f"temporal_latest_weight_{mechanism}"] = float(
                    arrays["temporal_latest_weights"][index, mechanism_index]
                )
                row[f"conflict_{mechanism}"] = float(
                    arrays["conflicts"][index, mechanism_index]
                )
            patient_rows.append(row)

        shapley_row: Dict[str, Any] = {
            "seed": seed,
            "patient_id": patient_id,
            "label": label,
            "c17_probability": c17_prob,
            "c17_class": int(c17_prob >= 0.5),
            **groups,
        }
        for combination in COMBINATIONS:
            shapley_row[f"logit_{combination}"] = float(logit_values[combination][index])
            shapley_row[f"probability_{combination}"] = float(
                probability_values[combination][index]
            )
        for name, values in logit_decomposition.items():
            shapley_row[f"logit_{name}"] = float(values[index])
        for name, values in probability_decomposition.items():
            shapley_row[f"probability_{name}"] = float(values[index])
        shapley_rows.append(shapley_row)

        for visit_index in range(visit_count):
            for role_index, role in enumerate(ROLES):
                before = role_original[index, visit_index, role_index]
                after = role_adapted[index, visit_index, role_index]
                delta = after - before
                projection = result["visit_projection"][role]
                signed = float(projection["signed_projection"][index, visit_index].detach().cpu())
                logit_delta = float(projection["logit_delta"][index, visit_index].detach().cpu())
                representation_rows.append(
                    {
                        "seed": seed,
                        "patient_id": patient_id,
                        "label": label,
                        "visit_rank": visit_index,
                        "visit_count": visit_count,
                        "is_latest_visit": visit_index == visit_count - 1,
                        "role_group": role,
                        "mapped_mechanism": MECHANISM_NAMES[ROLE_MECHANISM_INDEX[role]],
                        "text_available": bool(batch["visit_text_valid"][index, visit_index].detach().cpu()),
                        "guidance_present_original": bool(guidance_original[index, visit_index, role_index]),
                        "guidance_present_adapted": bool(guidance_adapted[index, visit_index, role_index]),
                        "original_state_norm": float(before.norm()),
                        "adapted_state_norm": float(after.norm()),
                        "original_adapted_cosine": float(F.cosine_similarity(before.unsqueeze(0), after.unsqueeze(0), dim=-1)),
                        "state_l2_delta": float(delta.norm()),
                        "signed_projection_toward_final_classifier": signed,
                        "single_visit_logit_delta": logit_delta,
                        "single_visit_probability_delta": float(projection["probability_delta"][index, visit_index].detach().cpu()),
                        "projection_logit_consistency_error": abs(signed - logit_delta),
                    }
                )


def build_pair_tables(
    predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: List[Dict[str, Any]] = []
    shapley_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = predictions[predictions["seed"].astype(int) == seed]
        matrices: Dict[str, pd.DataFrame] = {}
        for combination in COMBINATIONS:
            frame = seed_frame[seed_frame["combination"] == combination].sort_values(
                "patient_id"
            )
            matrices[combination] = frame.set_index("patient_id")
        baseline = matrices["000"]
        positive_ids = baseline[baseline["label"].astype(int) == 1].index.tolist()
        negative_ids = baseline[baseline["label"].astype(int) == 0].index.tolist()
        if len(positive_ids) * len(negative_ids) != 2209:
            raise RuntimeError(f"C31-A pair contract failed for seed {seed}")
        for positive_id in positive_ids:
            for negative_id in negative_ids:
                margins: Dict[str, np.ndarray] = {}
                row_base = {
                    "seed": seed,
                    "positive_patient_id": positive_id,
                    "negative_patient_id": negative_id,
                }
                for combination in COMBINATIONS:
                    frame = matrices[combination]
                    positive_probability = float(frame.at[positive_id, "probability"])
                    negative_probability = float(frame.at[negative_id, "probability"])
                    margin = positive_probability - negative_probability
                    margins[combination] = np.asarray(margin)
                    pair_rows.append(
                        {
                            **row_base,
                            "combination": combination,
                            "positive_probability": positive_probability,
                            "negative_probability": negative_probability,
                            "pair_margin": margin,
                            "inversion": margin < 0.0,
                            "c27_inversion": float(margins.get("000", np.asarray(0.0))) < 0.0,
                        }
                    )
                decomposition = factorial_decomposition(margins)
                shapley_row: Dict[str, Any] = dict(row_base)
                for combination in COMBINATIONS:
                    shapley_row[f"margin_{combination}"] = float(margins[combination])
                for name, value in decomposition.items():
                    shapley_row[f"margin_{name}"] = float(value)
                shapley_row["c27_inversion"] = float(margins["000"]) < 0.0
                shapley_row["c30_inversion"] = float(margins["111"]) < 0.0
                shapley_row["c30_introduced_inversion"] = (
                    float(margins["000"]) >= 0.0 and float(margins["111"]) < 0.0
                )
                shapley_row["c30_repaired_inversion"] = (
                    float(margins["000"]) < 0.0 and float(margins["111"]) >= 0.0
                )
                shapley_rows.append(shapley_row)
    pairs = pd.DataFrame(pair_rows)
    baseline_lookup = pairs[pairs["combination"] == "000"].set_index(
        ["seed", "positive_patient_id", "negative_patient_id"]
    )["inversion"]
    final_lookup = pairs[pairs["combination"] == "111"].set_index(
        ["seed", "positive_patient_id", "negative_patient_id"]
    )["inversion"]
    index = pd.MultiIndex.from_frame(
        pairs[["seed", "positive_patient_id", "negative_patient_id"]]
    )
    pairs["c27_inversion"] = baseline_lookup.reindex(index).to_numpy(dtype=bool)
    pairs["c30_inversion"] = final_lookup.reindex(index).to_numpy(dtype=bool)
    pairs["c30_introduced_inversion"] = ~pairs["c27_inversion"] & pairs["c30_inversion"]
    pairs["c30_repaired_inversion"] = pairs["c27_inversion"] & ~pairs["c30_inversion"]
    return pairs, pd.DataFrame(shapley_rows)


def run_analysis(args: argparse.Namespace) -> None:
    output = resolve_path(args.output_dir)
    gate_path = output / "c31a_analysis_gate.json"
    if not gate_path.exists():
        raise RuntimeError("C31-A analysis gate has not been run")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "C31A_ANALYSIS_AUTHORIZED" or int(gate.get("passed", 0)) != 24:
        raise RuntimeError("C31-A analysis is not authorized")
    if gate.get("git_commit") != git_output("rev-parse", "HEAD"):
        raise RuntimeError("C31-A gate commit differs from current commit")

    config = load_config(resolve_path(args.config))
    rows = read_jsonl(resolve_path(config["project"]["manifest"]))
    c30_run = resolve_path(args.c30_run_dir)
    c17_run = resolve_path(args.c17_run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_rows: List[Dict[str, Any]] = []
    shapley_rows: List[Dict[str, Any]] = []
    representation_rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        checkpoint_path = c30_run / "checkpoints" / f"seed_{seed}_best.pt"
        c17_path = c17_run / "predictions" / f"val_predictions_seed_{seed}.csv"
        model, _ = load_model(config, seed, checkpoint_path, device)
        c17 = read_prediction(c17_path)
        c17_probability_name = probability_column(c17)
        c17_logit_name = logit_column(c17)
        c17_by_id = {
            str(row["patient_id"]): (
                float(row[c17_probability_name]),
                float(row[c17_logit_name]),
            )
            for _, row in c17.iterrows()
        }
        loader = build_validation_loader(config, rows)
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                result = RoleFactorialForward(model).shared_forward(
                    batch, with_visit_projection=True
                )
                append_patient_rows(
                    seed,
                    batch,
                    result,
                    c17_by_id,
                    patient_rows,
                    shapley_rows,
                    representation_rows,
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = pd.DataFrame(patient_rows).sort_values(
        ["seed", "combination", "patient_id"]
    )
    patient_shapley = pd.DataFrame(shapley_rows).sort_values(["seed", "patient_id"])
    representation = pd.DataFrame(representation_rows).sort_values(
        ["seed", "patient_id", "visit_rank", "role_group"]
    )
    expected_prediction_rows = len(SEEDS) * len(COMBINATIONS) * 94
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"C31-A prediction row contract failed: {len(predictions)} != {expected_prediction_rows}"
        )
    pairs, pair_shapley = build_pair_tables(predictions)
    predictions.to_csv(output / "c31a_factorial_predictions_val.csv", index=False)
    patient_shapley.to_csv(output / "c31a_patient_role_shapley.csv", index=False)
    pairs.to_csv(output / "c31a_pairwise_ranking_by_combination.csv", index=False)
    pair_shapley.to_csv(output / "c31a_pair_role_shapley.csv", index=False)
    representation.to_csv(output / "c31a_role_representation_delta.csv", index=False)
    print(
        json.dumps(
            {
                "status": "C31A_FACTORIAL_ANALYSIS_COMPLETE",
                "patient_combination_rows": len(predictions),
                "pair_combination_rows": len(pairs),
                "patient_shapley_rows": len(patient_shapley),
                "pair_shapley_rows": len(pair_shapley),
                "representation_rows": len(representation),
            }
        )
    )


def main() -> None:
    args = parse_args()
    if args.stage == "gate":
        run_gate(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
