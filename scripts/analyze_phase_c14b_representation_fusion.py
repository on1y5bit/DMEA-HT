from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import PatientHTDataset, collate_patient_batch, read_manifest, tokenize_text  # noqa: E402
from dmea_ht.models import DMEAHTModel  # noqa: E402
from scripts.analyze_phase_c11_report_filter_hypotheses import (  # noqa: E402
    DIFFUSE_HT_CUES,
    NEGATIVE_THYROID_CUES,
    split_clauses,
)


SEEDS = (0, 42, 3407)
ABLATION_CONDITIONS = (
    "full_model",
    "mask_text",
    "mask_image",
    "mask_bio",
    "text_only_like",
    "image_only_like",
    "bio_only_like",
)
TEXT_VARIANTS = (
    "full_c13_text",
    "remove_diffuse_ht_like_clauses",
    "remove_negative_or_normal_thyroid_clauses",
    "thyroid_focus_prefix_only",
    "remove_thyroid_focus_prefix",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C14-B representation and fusion audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--seeds", default="0,42,3407")
    parser.add_argument("--include-test-reporting-only", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def read_predictions(run_dir: Path, split: str, input_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    paths = sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv"))
    if not paths:
        input_rows.append({"path": str(run_dir), "status": "missing", "notes": f"no {split} prediction CSVs"})
        return pd.DataFrame()
    for path in paths:
        try:
            frame = pd.read_csv(path)
            if "patient_id" not in frame.columns:
                raise ValueError("missing patient_id")
            frame = frame.copy()
            frame["patient_id"] = frame["patient_id"].astype(str)
            if "seed" in frame.columns:
                frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(seed_from_path(path)).astype(int)
            else:
                frame["seed"] = seed_from_path(path)
            prob_col = next((c for c in ("pred_prob", "prob", "prediction_prob", "score") if c in frame.columns), None)
            if prob_col is None:
                raise ValueError("missing probability column")
            frame["pred_prob"] = pd.to_numeric(frame[prob_col], errors="coerce")
            pred_col = next((c for c in ("pred_label", "prediction", "pred", "y_pred") if c in frame.columns), None)
            frame["pred_label"] = pd.to_numeric(frame[pred_col], errors="coerce") if pred_col else (frame["pred_prob"] >= 0.5).astype(int)
            frame["split"] = split
            frames.append(frame[["patient_id", "seed", "split", "pred_prob", "pred_label"]])
            input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows"})
        except Exception as exc:
            input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_c14a(output_dir: Path, input_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    path = output_dir.parent / "phase_c14a" / "c14a_positive_patient_token_exposure_val.csv"
    if not path.is_file():
        input_rows.append({"path": str(path), "status": "missing", "notes": "C14-A fields unavailable"})
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows; reused for audit stratification"})
        return frame
    except Exception as exc:
        input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
        return pd.DataFrame()


def build_manifest_frame(manifest: Path, input_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = read_manifest(manifest)
    input_rows.append({"path": str(manifest), "status": "loaded", "notes": f"{len(rows)} manifest rows"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype(int)
    frame["split"] = frame.get("split", "").astype(str)
    frame["report_text"] = [report_text(row) for row in rows]
    return frame


def report_text(row: Mapping[str, Any]) -> str:
    for key in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def build_cross_seed_groups(predictions: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive = manifest[(manifest["split"] == "val") & (manifest["label"] == 1)][["patient_id", "label"]].drop_duplicates()
    frame = predictions[predictions["split"] == "val"].merge(positive, on="patient_id", how="inner", suffixes=("", "_manifest"))
    frame["pred_label"] = frame["pred_label"].astype(int)
    frame["row_type"] = np.where(frame["pred_label"] == 1, "TP", "FN")
    counts = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        n_seed_rows=("seed", "nunique"),
        fn_count=("row_type", lambda x: int((x == "FN").sum())),
        tp_count=("row_type", lambda x: int((x == "TP").sum())),
        pred_prob_mean=("pred_prob", "mean"),
        pred_prob_std=("pred_prob", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
    )

    def group_name(row: pd.Series) -> str:
        if row["n_seed_rows"] != 3:
            return "incomplete_seed_rows"
        if row["fn_count"] == 3:
            return "all_seed_fn"
        if row["fn_count"] == 2:
            return "majority_fn"
        if row["fn_count"] == 1:
            return "seed_sensitive_positive"
        return "all_seed_tp"

    counts["cross_seed_group"] = counts.apply(group_name, axis=1)
    frame = frame.merge(counts, on="patient_id", suffixes=("", "_patient"))
    frame["cross_seed_group"] = frame["cross_seed_group"].fillna("incomplete_seed_rows")
    order = ["all_seed_fn", "majority_fn", "seed_sensitive_positive", "all_seed_tp", "incomplete_seed_rows"]
    summary = (
        counts.groupby("cross_seed_group", as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_rows=("patient_id", lambda x: int(len(x) * 3)),
            mean_pred_prob=("pred_prob_mean", "mean"),
            mean_pred_prob_std=("pred_prob_std", "mean"),
        )
    )
    summary["cross_seed_group"] = pd.Categorical(summary["cross_seed_group"], categories=order, ordered=True)
    summary = summary.sort_values("cross_seed_group").reset_index(drop=True)
    return frame.sort_values(["cross_seed_group", "patient_id", "seed"]), counts.sort_values(["cross_seed_group", "patient_id"]), summary


def load_checkpoint(path: Path, device: torch.device) -> tuple[DMEAHTModel, Dict[str, Any], Dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = DMEAHTModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, config, checkpoint


def make_loader(config: Mapping[str, Any], val_rows: Sequence[Mapping[str, Any]], batch_size: int) -> DataLoader:
    project_cfg = config.get("project", {})
    model_cfg = config.get("model", {})
    dataset = PatientHTDataset(
        rows=list(val_rows),
        data_root=str(project_cfg.get("data_root", "")),
        split="val",
        max_images=int(model_cfg.get("max_images_per_patient", 28)),
        image_size=int(model_cfg.get("image_size", 224)),
        text_max_length=int(model_cfg.get("text_max_length", 256)),
        text_vocab_size=int(model_cfg.get("text_vocab_size", 50000)),
        bio_dim=int(model_cfg.get("bio_dim", 32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_patient_batch)


def reproduce_seed(
    seed: int,
    model: DMEAHTModel,
    loader: DataLoader,
    manifest_by_patient: Mapping[str, Mapping[str, Any]],
    saved_predictions: Mapping[tuple[str, int], float],
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    reproduced: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            outputs = model(batch)
            probs = outputs["prob"].detach().cpu().tolist()
            labels = batch["label"].detach().cpu().tolist()
            for index, patient_id_raw in enumerate(batch["patient_id"]):
                patient_id = str(patient_id_raw)
                reproduced.append({"patient_id": patient_id, "label": int(labels[index]), "pred_prob": float(probs[index])})
    expected_ids = set(manifest_by_patient)
    saved_ids = {patient_id for (patient_id, saved_seed) in saved_predictions if saved_seed == seed}
    reproduced_ids = [row["patient_id"] for row in reproduced]
    reproduced_id_set = set(reproduced_ids)
    duplicate_reproduced = len(reproduced_ids) != len(reproduced_id_set)
    patient_id_match = expected_ids == saved_ids == reproduced_id_set and not duplicate_reproduced
    label_match = all(row["label"] == int(manifest_by_patient[row["patient_id"]].get("label", 0)) for row in reproduced)
    differences = [
        abs(row["pred_prob"] - saved_predictions[(row["patient_id"], seed)])
        for row in reproduced
        if (row["patient_id"], seed) in saved_predictions
    ]
    max_diff = max(differences) if differences else float("nan")
    mean_diff = float(np.mean(differences)) if differences else float("nan")
    model_cfg = config.get("model", {})
    notes = (
        f"tokenizer=character-level/text_max_length={model_cfg.get('text_max_length', 256)}; "
        f"image_size={model_cfg.get('image_size', 224)}; max_images={model_cfg.get('max_images_per_patient', 28)}; "
        f"duplicate_reproduced={duplicate_reproduced}"
    )
    return {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "saved_prediction_rows": int(len(saved_ids)),
        "reproduced_prediction_rows": int(len(reproduced)),
        "patient_id_match": int(patient_id_match),
        "label_match": int(label_match),
        "max_abs_prob_diff": float(max_diff),
        "mean_abs_prob_diff": float(mean_diff),
        "reproduction_pass": int(patient_id_match and label_match and math.isfinite(max_diff) and max_diff <= 1e-5 and mean_diff <= 1e-6),
        "notes": notes,
    }


def zero_like_pair(reference: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros(batch_size, reference.shape[1], reference.shape[2], device=reference.device, dtype=reference.dtype)
    global_token = torch.zeros(batch_size, reference.shape[2], device=reference.device, dtype=reference.dtype)
    return tokens, global_token


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)


def forward_with_diagnostics(model: DMEAHTModel, batch: Dict[str, Any], mask: str = "full_model") -> Dict[str, torch.Tensor]:
    batch_size = batch["report_input_ids"].shape[0]
    if mask in {"mask_image", "text_only_like", "bio_only_like"}:
        image_reference = batch["images"].new_zeros(batch_size, batch["images"].shape[1], model.classifier.e_img.in_features)
        image_tokens, image_global = zero_like_pair(image_reference, batch_size)
    else:
        image_tokens, image_global = model.image_encoder(batch["images"], batch["image_mask"])
    if mask in {"mask_text", "image_only_like", "bio_only_like"}:
        text_reference = batch["images"].new_zeros(batch_size, batch["report_input_ids"].shape[1], model.classifier.e_text.in_features)
        text_tokens, text_global = zero_like_pair(text_reference, batch_size)
    else:
        text_tokens, text_global = model.text_encoder(batch["report_input_ids"], batch["report_attention_mask"])
    if mask in {"mask_bio", "text_only_like", "image_only_like"}:
        hidden_dim = model.classifier.e_bio.in_features
        bio_tokens = torch.zeros(batch_size, batch["bio_values"].shape[1], hidden_dim, device=batch["bio_values"].device)
        bio_global = torch.zeros(batch_size, hidden_dim, device=batch["bio_values"].device)
        bio_medical = torch.zeros_like(bio_global)
    else:
        bio_tokens, bio_global, bio_medical, _bio_observation = model.bio_encoder(
            batch["bio_values"], batch["bio_missing_mask"], batch["bio_abnormal_flags"]
        )

    aux_outputs = model._auxiliary_outputs(image_global, text_global, text_tokens, batch["report_attention_mask"])
    if model.variant in {"image_only", "text_only", "bio_only", "concat"}:
        parts = {
            "image_only": [image_global, torch.zeros_like(text_global), torch.zeros_like(bio_global)],
            "text_only": [torch.zeros_like(image_global), text_global, torch.zeros_like(bio_global)],
            "bio_only": [torch.zeros_like(image_global), torch.zeros_like(text_global), bio_global],
            "concat": [image_global, text_global, bio_global],
        }[model.variant]
        logit = model.baseline_head(torch.cat(parts, dim=-1)).squeeze(-1)
        outputs: Dict[str, torch.Tensor] = {"logit": logit, "prob": torch.sigmoid(logit), **aux_outputs}
        z_patient = torch.zeros_like(text_global)
        evidence_scores: Dict[str, torch.Tensor] = {}
        discordance: Dict[str, torch.Tensor] = {}
    else:
        evidence_tokens, evidence_scores, role_loss = model.evidence(image_tokens, text_tokens, bio_tokens)
        token_parts = [image_tokens, text_tokens, bio_tokens, evidence_tokens]
        if model.fuse_text_morphology_anchor and "text_morphology_anchor" in aux_outputs:
            token_parts.append(aux_outputs["text_morphology_anchor"].unsqueeze(1))
        z_patient = model.anchor(torch.cat(token_parts, dim=1))
        discordance = model.discordance(image_global, text_global, bio_global)
        negative_token = evidence_tokens[:, model.evidence.roles.index("negative"), :]
        outputs = model.classifier(image_global, text_global, bio_medical, z_patient, negative_token)
        outputs.update(aux_outputs)
        outputs["role_alignment_loss"] = role_loss
        outputs.update({f"evidence_{key}": value for key, value in evidence_scores.items()})
        outputs.update({key: value.norm(dim=-1) for key, value in discordance.items()})

    outputs.update(
        {
            "text_embedding_norm": text_global.norm(dim=-1),
            "image_embedding_norm": image_global.norm(dim=-1),
            "bio_embedding_norm": bio_global.norm(dim=-1),
            "patient_anchor_norm": z_patient.norm(dim=-1),
            "fusion_feature_norm": z_patient.norm(dim=-1),
            "text_anchor_cosine": cosine(text_global, z_patient),
            "image_anchor_cosine": cosine(image_global, z_patient),
            "bio_anchor_cosine": cosine(bio_global, z_patient),
            "text_image_cosine": cosine(text_global, image_global),
            "text_bio_cosine": cosine(text_global, bio_global),
            "image_bio_cosine": cosine(image_global, bio_global),
        }
    )
    return outputs


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (value.clone() if torch.is_tensor(value) else value) for key, value in batch.items()}


def replace_text(batch: Dict[str, Any], text_by_patient: Mapping[str, str], max_length: int, vocab_size: int) -> Dict[str, Any]:
    out = clone_batch(batch)
    ids: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for patient_id in batch["patient_id"]:
        token_ids, attention_mask = tokenize_text(text_by_patient[str(patient_id)], max_length, vocab_size)
        ids.append(token_ids)
        masks.append(attention_mask)
    out["report_input_ids"] = torch.stack(ids).to(batch["report_input_ids"].device)
    out["report_attention_mask"] = torch.stack(masks).to(batch["report_attention_mask"].device)
    return out


def text_variants(text: str) -> Dict[str, str]:
    clauses = split_clauses(text)
    diffuse = [clause for clause in clauses if any(cue in clause for cue in DIFFUSE_HT_CUES)]
    negative = [clause for clause in clauses if any(cue in clause for cue in NEGATIVE_THYROID_CUES)]
    _ = diffuse, negative
    without_diffuse = "。".join(clause for clause in clauses if not any(cue in clause for cue in DIFFUSE_HT_CUES))
    without_negative = "。".join(clause for clause in clauses if not any(cue in clause for cue in NEGATIVE_THYROID_CUES))
    marker = "[C13_FULL_REPORT]"
    marker_pos = text.find(marker)
    if marker_pos >= 0:
        prefix = text[:marker_pos].strip()
        suffix = text[marker_pos + len(marker) :].lstrip(" \n\r。")
        prefix_only = prefix
        without_prefix = suffix
    else:
        prefix_only = text[:220]
        without_prefix = text
    return {
        "full_c13_text": text,
        "remove_diffuse_ht_like_clauses": without_diffuse,
        "remove_negative_or_normal_thyroid_clauses": without_negative,
        "thyroid_focus_prefix_only": prefix_only,
        "remove_thyroid_focus_prefix": without_prefix,
    }


def as_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def metadata_row(patient_id: str, manifest_by_patient: Mapping[str, Mapping[str, Any]], group_by_patient: Mapping[str, str], seed: int) -> Dict[str, Any]:
    row = manifest_by_patient[patient_id]
    shortcut = row.get("shortcuts", {}) if isinstance(row.get("shortcuts"), dict) else {}
    return {
        "patient_id": patient_id,
        "seed": int(seed),
        "label": int(row.get("label", 0)),
        "row_type": "positive_patient" if int(row.get("label", 0)) == 1 else "negative_patient",
        "cross_seed_group": group_by_patient.get(patient_id, "not_positive_group"),
        "report_length": row.get("report_length", len(report_text(row))),
        "selected_n_visits": row.get("selected_n_visits", ""),
        "used_images": row.get("used_images", row.get("n_images", "")),
        "has_bio": row.get("has_bio", ""),
        "bio_missing_count": row.get("bio_missing_count", ""),
    }


def run_seed(
    seed: int,
    model: DMEAHTModel,
    loader: DataLoader,
    manifest_by_patient: Mapping[str, Mapping[str, Any]],
    group_by_patient: Mapping[str, str],
    saved_predictions: Mapping[tuple[str, int], float],
    text_by_patient: Mapping[str, str],
    config: Mapping[str, Any],
    device: torch.device,
    input_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float]:
    model.eval()
    model_cfg = config.get("model", {})
    max_length = int(model_cfg.get("text_max_length", 256))
    vocab_size = int(model_cfg.get("text_vocab_size", 50000))
    feature_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    occlusion_rows: List[Dict[str, Any]] = []
    max_reproduction_error = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            full = forward_with_diagnostics(model, batch, "full_model")
            for index, patient_id_raw in enumerate(batch["patient_id"]):
                patient_id = str(patient_id_raw)
                pred_prob = float(full["prob"][index].detach().cpu())
                saved = saved_predictions.get((patient_id, seed), float("nan"))
                if math.isfinite(saved):
                    max_reproduction_error = max(max_reproduction_error, abs(pred_prob - saved))
                base = metadata_row(patient_id, manifest_by_patient, group_by_patient, seed)
                base.update({
                    "pred_prob": pred_prob,
                    "pred_label": int(pred_prob >= 0.5),
                    "full_model_reproduction_error": abs(pred_prob - saved) if math.isfinite(saved) else float("nan"),
                })
                for key, value in full.items():
                    if torch.is_tensor(value) and value.ndim == 1 and len(value) == len(batch["patient_id"]):
                        base[key] = float(value[index].detach().cpu())
                base["text_classifier_contribution"] = base.get("e_text", float("nan"))
                base["image_classifier_contribution"] = base.get("e_img", float("nan"))
                base["bio_classifier_contribution"] = base.get("e_bio", float("nan"))
                discordance_values = [base.get(key, float("nan")) for key in ("d_img_txt", "d_img_bio", "d_txt_bio")]
                base["discordance_feature_norm"] = float(np.nanmean(discordance_values)) if any(math.isfinite(value) for value in discordance_values) else float("nan")
                base["fusion_gate_or_attention_values"] = "unavailable"
                feature_rows.append(base)

            condition_outputs: Dict[str, torch.Tensor] = {"full_model": full["prob"]}
            for condition in ABLATION_CONDITIONS[1:]:
                condition_outputs[condition] = forward_with_diagnostics(model, batch, condition)["prob"]
            for index, patient_id_raw in enumerate(batch["patient_id"]):
                patient_id = str(patient_id_raw)
                row = metadata_row(patient_id, manifest_by_patient, group_by_patient, seed)
                for condition in ABLATION_CONDITIONS:
                    row[f"pred_{condition}"] = float(condition_outputs[condition][index].detach().cpu())
                row["full_prob"] = row["pred_full_model"]
                row["mask_text_prob"] = row["pred_mask_text"]
                row["mask_image_prob"] = row["pred_mask_image"]
                row["mask_bio_prob"] = row["pred_mask_bio"]
                row["text_only_like_prob"] = row["pred_text_only_like"]
                row["image_only_like_prob"] = row["pred_image_only_like"]
                row["bio_only_like_prob"] = row["pred_bio_only_like"]
                row["delta_mask_text"] = row["pred_mask_text"] - row["pred_full_model"]
                row["delta_mask_image"] = row["pred_mask_image"] - row["pred_full_model"]
                row["delta_mask_bio"] = row["pred_mask_bio"] - row["pred_full_model"]
                row["text_contribution"] = row["pred_full_model"] - row["pred_mask_text"]
                row["image_contribution"] = row["pred_full_model"] - row["pred_mask_image"]
                row["bio_contribution"] = row["pred_full_model"] - row["pred_mask_bio"]
                row["delta_text_only_like"] = row["pred_text_only_like"] - row["pred_full_model"]
                row["delta_image_only_like"] = row["pred_image_only_like"] - row["pred_full_model"]
                row["delta_bio_only_like"] = row["pred_bio_only_like"] - row["pred_full_model"]
                ablation_rows.append(row)

            positive_indices = [index for index, patient_id in enumerate(batch["patient_id"]) if int(manifest_by_patient[str(patient_id)].get("label", 0)) == 1]
            if positive_indices:
                variant_outputs: Dict[str, torch.Tensor] = {}
                for variant in TEXT_VARIANTS:
                    variant_text = {patient_id: text_variants(text_by_patient[patient_id])[variant] for patient_id in batch["patient_id"]}
                    variant_batch = replace_text(batch, variant_text, max_length, vocab_size)
                    variant_outputs[variant] = forward_with_diagnostics(model, variant_batch, "full_model")["prob"]
                for index in positive_indices:
                    patient_id = str(batch["patient_id"][index])
                    row = metadata_row(patient_id, manifest_by_patient, group_by_patient, seed)
                    for variant in TEXT_VARIANTS:
                        row[f"pred_{variant}"] = float(variant_outputs[variant][index].detach().cpu())
                    row["delta_remove_diffuse"] = row["pred_remove_diffuse_ht_like_clauses"] - row["pred_full_c13_text"]
                    row["delta_remove_negative"] = row["pred_remove_negative_or_normal_thyroid_clauses"] - row["pred_full_c13_text"]
                    row["delta_prefix_only"] = row["pred_thyroid_focus_prefix_only"] - row["pred_full_c13_text"]
                    row["delta_remove_prefix"] = row["pred_remove_thyroid_focus_prefix"] - row["pred_full_c13_text"]
                    row["full_prob"] = row["pred_full_c13_text"]
                    row["remove_diffuse_prob"] = row["pred_remove_diffuse_ht_like_clauses"]
                    row["remove_negative_prob"] = row["pred_remove_negative_or_normal_thyroid_clauses"]
                    row["prefix_only_prob"] = row["pred_thyroid_focus_prefix_only"]
                    row["remove_prefix_prob"] = row["pred_remove_thyroid_focus_prefix"]
                    occlusion_rows.append(row)
    input_rows.append({"path": "inference", "status": "loaded", "notes": f"seed {seed}; eval/no_grad; max full reproduction error {max_reproduction_error:.8g}"})
    return feature_rows, ablation_rows, occlusion_rows, max_reproduction_error


def finite_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def grouped_summary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["cross_seed_group", "n_patients", "n_rows", *columns])
    rows: List[Dict[str, Any]] = []
    for group, subset in frame.groupby("cross_seed_group", dropna=False):
        row: Dict[str, Any] = {"cross_seed_group": group, "n_patients": subset["patient_id"].nunique(), "n_rows": len(subset)}
        for column in columns:
            row[f"mean_{column}"] = finite_mean(subset, column)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cross_seed_group")


def write_ranked_cases(features: pd.DataFrame, ablation: pd.DataFrame, occlusion: pd.DataFrame, out_dir: Path) -> None:
    all_fn = features[features["cross_seed_group"] == "all_seed_fn"].copy()
    if all_fn.empty:
        pd.DataFrame().to_csv(out_dir / "c14b_all_seed_fn_ranked_cases_val.csv", index=False)
        return
    rows: List[pd.DataFrame] = []
    merged = all_fn.merge(ablation, on=["patient_id", "seed"], suffixes=("", "_ablation"), how="left")
    merged = merged.merge(occlusion, on=["patient_id", "seed"], suffixes=("", "_occlusion"), how="left")
    specs = [
        ("weakest_text_contribution", "text_contribution", False),
        ("strongest_image_suppression", "delta_mask_image", True),
        ("strongest_bio_suppression", "delta_mask_bio", True),
        ("minimal_diffuse_occlusion_effect", "delta_remove_diffuse", False),
        ("high_text_contribution_low_final_probability", "text_contribution", True),
    ]
    for rank_name, column, descending in specs:
        subset = merged.sort_values(column, ascending=not descending).head(10).copy()
        subset.insert(0, "rank_type", rank_name)
        rows.append(subset)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "c14b_all_seed_fn_ranked_cases_val.csv", index=False)


def corr_abs(frame: pd.DataFrame, left: str, right: str) -> float:
    if left not in frame or right not in frame:
        return float("nan")
    values = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(values[left].corr(values[right])) if len(values) >= 3 else float("nan")


def write_reports(
    out_dir: Path,
    group_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    occlusion_summary: pd.DataFrame,
    stability: pd.DataFrame,
    all_fn: pd.DataFrame,
    max_reproduction_error: float,
    missing: pd.DataFrame,
) -> str:
    fn = ablation_summary[ablation_summary["cross_seed_group"] == "all_seed_fn"] if not ablation_summary.empty else pd.DataFrame()
    tp = ablation_summary[ablation_summary["cross_seed_group"] == "all_seed_tp"] if not ablation_summary.empty else pd.DataFrame()
    fn_text = feature_summary[feature_summary["cross_seed_group"] == "all_seed_fn"] if not feature_summary.empty else pd.DataFrame()
    tp_text = feature_summary[feature_summary["cross_seed_group"] == "all_seed_tp"] if not feature_summary.empty else pd.DataFrame()
    fn_occ = occlusion_summary[occlusion_summary["cross_seed_group"] == "all_seed_fn"] if not occlusion_summary.empty else pd.DataFrame()

    text_gap = finite_mean(fn, "text_contribution") - finite_mean(tp, "text_contribution")
    fn_text_only_gap = finite_mean(fn, "pred_text_only_like") - finite_mean(fn, "pred_full_model")
    fn_image_suppression = finite_mean(fn, "delta_mask_image")
    fn_bio_suppression = finite_mean(fn, "delta_mask_bio")
    diffuse_effect = finite_mean(fn_occ, "delta_remove_diffuse")
    text_norm_gap = finite_mean(fn_text, "text_embedding_norm") - finite_mean(tp_text, "text_embedding_norm")

    support_fusion = int((fn_text_only_gap > 0.05) + (fn_image_suppression > 0.02) + (fn_bio_suppression > 0.02) + (text_gap >= -0.02))
    support_text = int((text_norm_gap < -0.1) + (abs(diffuse_effect) < 0.02) + (finite_mean(fn, "text_contribution") < 0.02) + (fn_text_only_gap <= 0.05))
    support_semantic = int((abs(diffuse_effect) >= 0.02) + (abs(text_gap) < 0.02) + (fn_image_suppression <= 0.02) + (fn_bio_suppression <= 0.02))
    scores = {"FUSION_SUPPRESSION": support_fusion, "TEXT_REPRESENTATION_FAILURE": support_text, "EVIDENCE_SEMANTIC_AMBIGUITY": support_semantic}
    top_score = max(scores.values()) if scores else 0
    top_labels = [label for label, score in scores.items() if score == top_score]
    route = top_labels[0] if top_score >= 3 and len(top_labels) == 1 else "MIXED_OR_INCONCLUSIVE"

    corr_pred_contrib = corr_abs(stability, "pred_prob_std", "contribution_std_mean")
    corr_pred_text = corr_abs(stability, "pred_prob_std", "text_contribution_std")
    lines = [
        "# Phase C14-B Representation and Multimodal Fusion Audit",
        "",
        "C14-B is analysis-only. No training, threshold tuning, label/split/task changes, manifest changes, or architecture changes were made.",
        "",
        "## Inputs And Checkpoint Reproduction",
        "",
        f"- Manifest: `{out_dir}` audit input record; current C13 temporal-focus manifest was used.",
        "- Run: `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds` with seeds `[0, 42, 3407]`.",
        f"- Full-model saved-prediction reproduction maximum absolute error: `{max_reproduction_error:.8g}`.",
        "- Inference used `model.eval()` and `torch.no_grad()`; no optimizer was constructed.",
        "- The modality masks are inference-only zero-equivalent representation interventions. They are diagnostics, not candidate models.",
        "",
        "## Corrected Cross-Seed Groups",
        "",
        "- `all_seed_fn`: FN in all three seeds.",
        "- `majority_fn`: FN in exactly two seeds.",
        "- `seed_sensitive_positive`: class differs across seeds, including 1-FN/2-TP and 2-FN/1-TP.",
        "- `all_seed_tp`: TP in all three seeds.",
        "",
        group_summary.to_markdown(index=False) if not group_summary.empty else "_No group rows._",
        "",
        "The expected all-seed FN count from C14-A was seven; the corrected grouping above is computed from predictions rather than hard-coded.",
        "",
        "## Representation And Fusion Findings",
        "",
        feature_summary.to_markdown(index=False) if not feature_summary.empty else "_No feature rows._",
        "",
        "## Modality Masking Findings",
        "",
        ablation_summary.to_markdown(index=False) if not ablation_summary.empty else "_No ablation rows._",
        "",
        f"- all-seed FN text-only-like minus full probability: `{fn_text_only_gap:.4f}`.",
        f"- all-seed FN image-removal delta: `{fn_image_suppression:.4f}`; bio-removal delta: `{fn_bio_suppression:.4f}`.",
        "- Positive removal deltas indicate that the removed modality was suppressing the positive prediction.",
        "",
        "## Text Occlusion Findings",
        "",
        occlusion_summary.to_markdown(index=False) if not occlusion_summary.empty else "_No occlusion rows._",
        "",
        f"- all-seed FN diffuse-clause removal delta: `{diffuse_effect:.4f}`.",
        "- Text variants were constructed from the C13 report text and used only for inference-time diagnostics.",
        "",
        "## Seed-Wise Fusion Stability",
        "",
        stability.to_markdown(index=False) if not stability.empty else "_No stability rows._",
        "",
        f"- Correlation of prediction variance with mean modality-contribution variance: `{corr_pred_contrib:.4f}`.",
        f"- Correlation of prediction variance with text-contribution variance: `{corr_pred_text:.4f}`.",
        "",
        "## Missing Diagnostics And Limitations",
        "",
        "- The current model exposes no learned gate tensor or attention weight as a stable public output; available native evidence scores, discordance norms, embedding norms, anchor cosines, and classifier contributions were exported.",
        "- `fusion_feature_norm` is the patient-anchor output norm because the current implementation has no separate named fusion tensor.",
        "- Zero-equivalent masking is a diagnostic intervention. It does not claim causal identifiability and must not be used as a trained model result.",
        f"- Missing/input audit is stored in `inputs_used_and_missing.csv` with `{len(missing)}` records.",
        "- Shortcut fields are retained only as audit metadata and were not supplied to the classifier or route scoring.",
        "",
        "## Final Route Decision",
        "",
        f"`{route}`.",
        "",
        f"Support scores used only to organize the validation audit: `{json.dumps(scores, ensure_ascii=False, sort_keys=True)}`.",
        f"Text representation norm gap FN minus TP: `{text_norm_gap:.4f}`; text contribution gap FN minus TP: `{text_gap:.4f}`.",
        "This route label is a validation-only audit conclusion and does not claim a model improvement.",
        "",
        "## Next-Step Gate",
        "",
        {"TEXT_REPRESENTATION_FAILURE": "Only one low-cost text representation pilot may be designed, such as thyroid-focused sentence pooling, local evidence-token pooling, or text encoder initialization.", "FUSION_SUPPRESSION": "Only one small fusion pilot may be designed, such as a residual text path, minimum text-preservation gate, or modality-dropout robustness.", "EVIDENCE_SEMANTIC_AMBIGUITY": "Do not modify the model; return to evidence-definition and manual clinical audit.", "MIXED_OR_INCONCLUSIVE": "Do not train; perform one narrower follow-up analysis."}[route],
        "",
        "The following remain prohibited: formal multi-seed training directly from C14-B; label, split, task, manifest, report-construction, or core training changes; shortcut variables as classifier inputs; test-based route selection; and revival of prior evidence/alignment/counterfactual losses.",
        "",
        "C13 remains the current strict best. C14-B is analysis-only. No new model improvement is claimed. Test metrics were not used for route selection.",
    ]
    (out_dir / "c14b_representation_fusion_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "phase_c14b_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return route


def write_reproduction_report(out_dir: Path, reproduction: pd.DataFrame) -> None:
    reproduction.to_csv(out_dir / "c14b_reproduction_check_by_seed.csv", index=False)
    passed = bool(not reproduction.empty and (reproduction["reproduction_pass"].astype(int) == 1).all())
    lines = [
        "# Phase C14-B Reproduction Check",
        "",
        "This is a mandatory pre-audit gate. Downstream representation, masking, and occlusion claims are valid only when every required seed passes.",
        "",
        reproduction.to_markdown(index=False) if not reproduction.empty else "_No reproduction rows._",
        "",
        f"Overall reproduction gate: `{'PASS' if passed else 'FAIL'}`.",
        "Required thresholds: max absolute probability difference <= 1e-5 and mean absolute probability difference <= 1e-6, with matching patient IDs and labels.",
    ]
    (out_dir / "c14b_reproduction_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def direction_summary(values: Sequence[float], threshold: float = 1e-6) -> tuple[int, int, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    positive = sum(value > threshold for value in finite)
    negative = sum(value < -threshold for value in finite)
    zero = len(finite) - positive - negative
    return positive, negative, zero


def write_seed_consistency(frame: pd.DataFrame, metrics: Sequence[str], out_path: Path, seeds: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        result = pd.DataFrame(columns=["cross_seed_group", "metric"])
        result.to_csv(out_path, index=False)
        return result
    for group, group_frame in frame.groupby("cross_seed_group", dropna=False):
        for metric in metrics:
            means = group_frame.groupby("seed")[metric].mean()
            values = [float(means.get(seed, float("nan"))) for seed in seeds]
            positive, negative, zero = direction_summary(values, threshold=1e-6)
            rows.append(
                {
                    "cross_seed_group": group,
                    "metric": metric,
                    **{f"seed_{seed}_mean": values[index] for index, seed in enumerate(seeds)},
                    "mean_across_seeds": float(np.nanmean(values)) if any(math.isfinite(value) for value in values) else float("nan"),
                    "std_across_seeds": float(np.nanstd(values)) if any(math.isfinite(value) for value in values) else float("nan"),
                    "positive_seed_count": positive,
                    "negative_seed_count": negative,
                    "zero_or_missing_seed_count": zero,
                    "direction_consistent_2_of_3": int(max(positive, negative) >= 2 and min(positive, negative) == 0),
                    "sign_change_across_seeds": int(positive > 0 and negative > 0),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    return result


def write_final_report(
    out_dir: Path,
    reproduction: pd.DataFrame,
    group_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    occlusion_summary: pd.DataFrame,
    modality_consistency: pd.DataFrame,
    occlusion_consistency: pd.DataFrame,
    stability: pd.DataFrame,
    missing: pd.DataFrame,
    route: str,
    allowed_next_step: str,
    route_basis: str,
) -> None:
    repro_pass = bool(not reproduction.empty and (reproduction["reproduction_pass"].astype(int) == 1).all())
    lines = [
        "# Phase C14-B Multi-Seed Representation And Fusion Audit",
        "",
        "C14-B is analysis-only. No training, optimizer construction, backward pass, threshold tuning, label/split/task changes, manifest changes, tokenizer changes, report-construction changes, or architecture changes were performed.",
        "",
        "## Inputs And Reproduction Gate",
        "",
        "- Run: `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds`.",
        "- Manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`.",
        "- Required seeds: `[0, 42, 3407]`.",
        f"- Reproduction gate: `{'PASS' if repro_pass else 'FAIL'}`.",
        "- All downstream route claims are validation-only; test outputs were not used for route selection.",
        "",
        reproduction.to_markdown(index=False) if not reproduction.empty else "_No reproduction rows._",
        "",
        "## Corrected Positive-Patient Groups",
        "",
        "- `all_seed_fn`: FN in all three seeds.",
        "- `majority_fn`: FN in exactly two seeds.",
        "- `seed_sensitive_positive`: both TP and FN across seeds.",
        "- `all_seed_tp`: TP in all three seeds.",
        "",
        group_summary.to_markdown(index=False) if not group_summary.empty else "_No group rows._",
        "",
        "## Representation Diagnostics",
        "",
        feature_summary.to_markdown(index=False) if not feature_summary.empty else "_No representation rows._",
        "",
        "## Modality Masking",
        "",
        ablation_summary.to_markdown(index=False) if not ablation_summary.empty else "_No masking rows._",
        "",
        modality_consistency.to_markdown(index=False) if not modality_consistency.empty else "_No modality consistency rows._",
        "",
        "Masking is a diagnostic distribution shift, not a formal ablation model or candidate model.",
        "",
        "## Text Occlusion",
        "",
        occlusion_summary.to_markdown(index=False) if not occlusion_summary.empty else "_No occlusion rows._",
        "",
        occlusion_consistency.to_markdown(index=False) if not occlusion_consistency.empty else "_No occlusion consistency rows._",
        "",
        "## Seed-Wise Fusion Stability",
        "",
        stability.to_markdown(index=False) if not stability.empty else "_No stability rows._",
        "",
        "## Unavailable Diagnostics And Limitations",
        "",
        "- The current model exposes classifier contributions, evidence-role scores, discordance norms, embedding norms, anchor cosines, and patient-anchor output norms.",
        "- Learned fusion gate or attention values are unavailable as stable public outputs and are marked `unavailable`; they were not invented or approximated.",
        "- Shortcut fields are retained only as audit metadata and were not fed into the classifier or route logic.",
        f"- Input and missing-field audit: `inputs_used_and_missing.csv` ({len(missing)} records).",
        "",
        "## Final Decision",
        "",
        f"`{route}`.",
        "",
        f"Decision basis: {route_basis}",
        "",
        f"Allowed next-step class: `{allowed_next_step}`.",
        "",
        "C13 remains the current strict best at validation AUC 0.8665 +/- 0.0077. C14-B claims no model improvement. No training was launched and no test result was used for route selection.",
    ]
    (out_dir / "phase_c14b_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "c14b_representation_fusion_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_group_means(frame: pd.DataFrame, group: str, metric: str, seeds: Sequence[int]) -> List[float]:
    if frame.empty or metric not in frame.columns:
        return [float("nan") for _ in seeds]
    subset = frame[frame["cross_seed_group"] == group]
    means = subset.groupby("seed")[metric].mean()
    return [float(means.get(seed, float("nan"))) for seed in seeds]


def at_least_two(values: Sequence[float], predicate: Any) -> bool:
    return sum(bool(math.isfinite(float(value)) and predicate(float(value))) for value in values) >= 2


def decide_route(
    reproduction: pd.DataFrame,
    features: pd.DataFrame,
    ablation: pd.DataFrame,
    occlusion: pd.DataFrame,
    seeds: Sequence[int],
) -> tuple[str, str, str]:
    gate_pass = bool(not reproduction.empty and (reproduction["reproduction_pass"].astype(int) == 1).all())
    if not gate_pass:
        return "MIXED_OR_INCONCLUSIVE", "MORE_ANALYSIS_ONLY", "The mandatory checkpoint reproduction gate failed for at least one required seed. No contribution route can be promoted."
    if features.empty or ablation.empty or occlusion.empty:
        return "MIXED_OR_INCONCLUSIVE", "MORE_ANALYSIS_ONLY", "One or more required diagnostic families produced no rows."
    fn = "all_seed_fn"
    tp = "all_seed_tp"
    fn_text_rescue = [a - b for a, b in zip(seed_group_means(ablation, fn, "text_only_like_prob", seeds), seed_group_means(ablation, fn, "full_prob", seeds))]
    tp_text_rescue = [a - b for a, b in zip(seed_group_means(ablation, tp, "text_only_like_prob", seeds), seed_group_means(ablation, tp, "full_prob", seeds))]
    fn_image_suppression = seed_group_means(ablation, fn, "delta_mask_image", seeds)
    tp_image_suppression = seed_group_means(ablation, tp, "delta_mask_image", seeds)
    fn_bio_suppression = seed_group_means(ablation, fn, "delta_mask_bio", seeds)
    tp_bio_suppression = seed_group_means(ablation, tp, "delta_mask_bio", seeds)
    fn_diffuse_effect = seed_group_means(occlusion, fn, "delta_remove_diffuse", seeds)
    fn_prefix_effect = seed_group_means(occlusion, fn, "delta_prefix_only", seeds)
    fn_text_norm = seed_group_means(features, fn, "text_embedding_norm", seeds)
    tp_text_norm = seed_group_means(features, tp, "text_embedding_norm", seeds)

    fusion_flags = {
        "text_only_rescue": at_least_two(fn_text_rescue, lambda value: value > 0.05),
        "image_suppression": at_least_two(fn_image_suppression, lambda value: value > 0.02) and sum(
            (math.isfinite(fn_value) and math.isfinite(tp_value) and fn_value > tp_value + 0.01)
            for fn_value, tp_value in zip(fn_image_suppression, tp_image_suppression)
        ) >= 1,
        "bio_suppression": at_least_two(fn_bio_suppression, lambda value: value > 0.02) and sum(
            (math.isfinite(fn_value) and math.isfinite(tp_value) and fn_value > tp_value + 0.01)
            for fn_value, tp_value in zip(fn_bio_suppression, tp_bio_suppression)
        ) >= 1,
        "text_norm_comparable": sum(
            math.isfinite(fn_value) and math.isfinite(tp_value) and abs(fn_value - tp_value) < 0.1
            for fn_value, tp_value in zip(fn_text_norm, tp_text_norm)
        ) >= 2,
    }
    fusion_support = int(fusion_flags["text_only_rescue"]) + int(fusion_flags["image_suppression"]) + int(fusion_flags["bio_suppression"])
    text_flags = {
        "diffuse_has_little_effect": at_least_two(fn_diffuse_effect, lambda value: abs(value) < 0.02),
        "prefix_does_not_rescue": at_least_two(fn_prefix_effect, lambda value: value <= 0.05),
        "text_only_remains_low": at_least_two(fn_text_rescue, lambda value: value <= 0.05),
        "text_norm_not_distinctive": fusion_flags["text_norm_comparable"],
    }
    text_support = sum(int(value) for value in text_flags.values())
    semantic_flags = {
        "diffuse_changes_prediction": at_least_two(fn_diffuse_effect, lambda value: abs(value) >= 0.02),
        "no_text_only_rescue": at_least_two(fn_text_rescue, lambda value: value <= 0.05),
        "no_consistent_modality_suppression": fusion_support == 0,
    }
    semantic_support = sum(int(value) for value in semantic_flags.values())
    if fusion_support >= 2 and fusion_flags["text_norm_comparable"]:
        return "FUSION_SUPPRESSION", "SMALL_RESIDUAL_OR_GATED_FUSION_PILOT", f"Fusion support flags={fusion_flags}; all-seed FN text-only rescue, image/bio masking, and TP comparison were checked across {len(seeds)} seeds."
    if text_support >= 3 and fusion_support == 0:
        return "TEXT_REPRESENTATION_FAILURE", "SMALL_TEXT_REPRESENTATION_PILOT", f"Text failure flags={text_flags}; effects were required in at least two seeds and no consistent fusion suppression was found."
    if semantic_support >= 2 and fusion_support == 0:
        return "EVIDENCE_SEMANTIC_AMBIGUITY", "HT_SPECIFIC_EVIDENCE_REDEFINITION_AUDIT", f"Semantic ambiguity flags={semantic_flags}; text occlusion affected prediction without a consistent modality suppression pattern."
    return "MIXED_OR_INCONCLUSIVE", "MORE_ANALYSIS_ONLY", f"Fusion flags={fusion_flags}; text flags={text_flags}; semantic flags={semantic_flags}. No single mechanism met the multi-seed strength gate."


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, Any]] = []
    manifest = build_manifest_frame(Path(args.manifest), input_rows)
    predictions = read_predictions(Path(args.run_dir), "val", input_rows)
    if manifest.empty or predictions.empty:
        raise RuntimeError("C14-B requires a non-empty manifest and validation predictions.")
    positive_groups, group_counts, group_summary = build_cross_seed_groups(predictions, manifest)
    probability_wide = positive_groups.pivot_table(index="patient_id", columns="seed", values="pred_prob", aggfunc="first").reset_index()
    probability_wide = probability_wide.rename(columns={seed: f"seed_{seed}_pred_prob" for seed in seeds})
    groups_out = group_counts.merge(probability_wide, on="patient_id", how="left")
    groups_out = groups_out.rename(columns={"cross_seed_group": "group", "pred_prob_mean": "cross_seed_pred_mean", "pred_prob_std": "cross_seed_pred_std"})
    groups_out.to_csv(out_dir / "c14b_cross_seed_positive_groups.csv", index=False)
    group_summary.to_csv(out_dir / "c14b_cross_seed_group_summary.csv", index=False)
    c14a = read_c14a(out_dir, input_rows)

    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    manifest_by_patient = {str(row["patient_id"]): row for row in val_rows}
    group_by_patient = dict(zip(group_counts["patient_id"].astype(str), group_counts["cross_seed_group"].astype(str)))
    text_by_patient = {patient_id: report_text(row) for patient_id, row in manifest_by_patient.items()}
    saved_predictions = {(str(row.patient_id), int(row.seed)): float(row.pred_prob) for row in predictions.itertuples()}
    input_rows.append({"path": "phase_c14a", "status": "available" if not c14a.empty else "missing", "notes": "reused for audit stratification; no label/prediction recomputation"})

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    input_rows.append({"path": "runtime", "status": "loaded", "notes": f"device={device}; eval/no_grad only; seeds={list(seeds)}"})
    loaded: List[tuple[int, DMEAHTModel, Dict[str, Any], DataLoader, Path, Dict[str, Any]]] = []
    reproduction_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = Path(args.run_dir) / "checkpoints" / f"seed_{seed}_best.pt"
        if not checkpoint_path.is_file():
            input_rows.append({"path": str(checkpoint_path), "status": "missing", "notes": "required C13 checkpoint"})
            raise FileNotFoundError(checkpoint_path)
        model, config, checkpoint = load_checkpoint(checkpoint_path, device)
        loader = make_loader(config, val_rows, args.batch_size)
        input_rows.append({"path": str(checkpoint_path), "status": "loaded", "notes": f"seed={seed}; best_epoch={checkpoint.get('best_epoch', '')}"})
        reproduction_rows.append(reproduce_seed(seed, model, loader, manifest_by_patient, saved_predictions, checkpoint_path, config, device))
        loaded.append((seed, model, config, loader, checkpoint_path, checkpoint))
    reproduction = pd.DataFrame(reproduction_rows)
    write_reproduction_report(out_dir, reproduction)
    gate_pass = bool(not reproduction.empty and (reproduction["reproduction_pass"].astype(int) == 1).all())
    if not gate_pass:
        empty = pd.DataFrame()
        for filename in (
            "c14b_representation_diagnostics_val.csv", "c14b_representation_group_summary.csv",
            "c14b_modality_masking_val.csv", "c14b_modality_masking_group_summary.csv", "c14b_modality_masking_seed_consistency.csv",
            "c14b_text_occlusion_val.csv", "c14b_text_occlusion_group_summary.csv", "c14b_text_occlusion_seed_consistency.csv",
            "c14b_seedwise_fusion_stability_val.csv",
        ):
            empty.to_csv(out_dir / filename, index=False)
        missing = pd.DataFrame(input_rows, columns=["path", "status", "notes"])
        missing.to_csv(out_dir / "c14b_inputs_used_and_missing.csv", index=False)
        missing.to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
        write_final_report(out_dir, reproduction, group_summary, empty, empty, empty, empty, empty, empty, missing, "MIXED_OR_INCONCLUSIVE", "MORE_ANALYSIS_ONLY", "The mandatory reproduction gate failed; downstream contribution analysis was stopped.")
        print(json.dumps({"output_dir": str(out_dir), "route": "MIXED_OR_INCONCLUSIVE", "reproduction_gate": "FAIL", "device": str(device)}, ensure_ascii=False))
        return

    feature_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    occlusion_rows: List[Dict[str, Any]] = []
    for seed, model, config, loader, _checkpoint_path, _checkpoint in loaded:
        seed_features, seed_ablation, seed_occlusion, _max_error = run_seed(
            seed, model, loader, manifest_by_patient, group_by_patient, saved_predictions, text_by_patient, config, device, input_rows
        )
        feature_rows.extend(seed_features)
        ablation_rows.extend(seed_ablation)
        occlusion_rows.extend(seed_occlusion)

    features = pd.DataFrame(feature_rows)
    ablation = pd.DataFrame(ablation_rows)
    occlusion = pd.DataFrame(occlusion_rows)
    if not c14a.empty:
        features = features.merge(c14a.drop_duplicates("patient_id"), on="patient_id", how="left", suffixes=("", "_c14a"))
    features.to_csv(out_dir / "c14b_representation_diagnostics_val.csv", index=False)
    ablation.to_csv(out_dir / "c14b_modality_masking_val.csv", index=False)
    occlusion.to_csv(out_dir / "c14b_text_occlusion_val.csv", index=False)

    feature_columns = [
        "text_embedding_norm", "image_embedding_norm", "bio_embedding_norm", "patient_anchor_norm", "fusion_feature_norm",
        "text_anchor_cosine", "image_anchor_cosine", "bio_anchor_cosine", "text_image_cosine", "text_bio_cosine", "image_bio_cosine",
        "text_classifier_contribution", "image_classifier_contribution", "bio_classifier_contribution", "discordance_feature_norm",
    ]
    feature_summary = grouped_summary(features[features["label"] == 1], feature_columns)
    ablation_columns = [
        "full_prob", "mask_text_prob", "mask_image_prob", "mask_bio_prob", "text_only_like_prob", "image_only_like_prob", "bio_only_like_prob",
        "delta_mask_text", "delta_mask_image", "delta_mask_bio", "delta_text_only_like", "delta_image_only_like", "delta_bio_only_like",
    ]
    ablation_summary = grouped_summary(ablation, ablation_columns)
    occlusion_columns = ["full_prob", "remove_diffuse_prob", "remove_negative_prob", "prefix_only_prob", "remove_prefix_prob", "delta_remove_diffuse", "delta_remove_negative", "delta_prefix_only", "delta_remove_prefix"]
    occlusion_summary = grouped_summary(occlusion, occlusion_columns)
    feature_summary.to_csv(out_dir / "c14b_representation_group_summary.csv", index=False)
    ablation_summary.to_csv(out_dir / "c14b_modality_masking_group_summary.csv", index=False)
    occlusion_summary.to_csv(out_dir / "c14b_text_occlusion_group_summary.csv", index=False)
    modality_consistency = write_seed_consistency(
        ablation[ablation["label"] == 1],
        ["delta_mask_text", "delta_mask_image", "delta_mask_bio", "delta_text_only_like", "delta_image_only_like", "delta_bio_only_like"],
        out_dir / "c14b_modality_masking_seed_consistency.csv",
        seeds,
    )
    occlusion_consistency = write_seed_consistency(
        occlusion,
        ["delta_remove_diffuse", "delta_remove_negative", "delta_prefix_only", "delta_remove_prefix"],
        out_dir / "c14b_text_occlusion_seed_consistency.csv",
        seeds,
    )

    stability_base = ablation[ablation["label"] == 1].copy()
    stability = stability_base.groupby("patient_id", as_index=False).agg(
        cross_seed_group=("cross_seed_group", "first"),
        n_seeds=("seed", "nunique"),
        full_prob_std=("full_prob", "std"),
        delta_mask_text_std=("delta_mask_text", "std"),
        delta_mask_image_std=("delta_mask_image", "std"),
        delta_mask_bio_std=("delta_mask_bio", "std"),
    )
    occlusion_stability = occlusion.groupby("patient_id", as_index=False).agg(
        delta_remove_diffuse_std=("delta_remove_diffuse", "std"),
        delta_remove_negative_std=("delta_remove_negative", "std"),
    )
    feature_stability = features[features["label"] == 1].groupby("patient_id", as_index=False).agg(
        text_anchor_cosine_std=("text_anchor_cosine", "std"),
        image_anchor_cosine_std=("image_anchor_cosine", "std"),
        bio_anchor_cosine_std=("bio_anchor_cosine", "std"),
    )
    stability = stability.merge(occlusion_stability, on="patient_id", how="left").merge(feature_stability, on="patient_id", how="left")
    stability["contribution_std_mean"] = stability[["delta_mask_text_std", "delta_mask_image_std", "delta_mask_bio_std"]].mean(axis=1)
    stability.to_csv(out_dir / "c14b_seedwise_fusion_stability_val.csv", index=False)
    top_stability = stability.sort_values("full_prob_std", ascending=False).head(20)
    stability_report = [
        "# Phase C14-B Seed-Wise Fusion Stability",
        "",
        "The table ranks validation-positive patients by cross-seed prediction and contribution instability.",
        "",
        top_stability.to_markdown(index=False) if not top_stability.empty else "_No stability rows._",
        "",
        "Patients are grouped using the corrected all-seed FN, majority FN, seed-sensitive positive, and all-seed TP definitions.",
        "Seed 42 is not selected or discarded; it is reported alongside seeds 0 and 3407.",
    ]
    (out_dir / "c14b_seedwise_fusion_stability_report.md").write_text("\n".join(stability_report) + "\n", encoding="utf-8")
    write_ranked_cases(features, ablation, occlusion, out_dir)

    missing = pd.DataFrame(input_rows, columns=["path", "status", "notes"])
    missing.to_csv(out_dir / "c14b_inputs_used_and_missing.csv", index=False)
    missing.to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    route, allowed_next_step, route_basis = decide_route(reproduction, features, ablation, occlusion, seeds)
    write_final_report(out_dir, reproduction, group_summary, feature_summary, ablation_summary, occlusion_summary, modality_consistency, occlusion_consistency, stability, missing, route, allowed_next_step, route_basis)
    print(json.dumps({"output_dir": str(out_dir), "route": route, "allowed_next_step": allowed_next_step, "reproduction_gate": "PASS", "all_seed_fn_patients": int((group_counts["cross_seed_group"] == "all_seed_fn").sum()), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
