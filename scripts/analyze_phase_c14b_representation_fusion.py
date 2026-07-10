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
    parser.add_argument("--include-test-reporting-only", action="store_true")
    return parser.parse_args()


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
                feature_rows.append(base)

            condition_outputs: Dict[str, torch.Tensor] = {"full_model": full["prob"]}
            for condition in ABLATION_CONDITIONS[1:]:
                condition_outputs[condition] = forward_with_diagnostics(model, batch, condition)["prob"]
            for index, patient_id_raw in enumerate(batch["patient_id"]):
                patient_id = str(patient_id_raw)
                row = metadata_row(patient_id, manifest_by_patient, group_by_patient, seed)
                for condition in ABLATION_CONDITIONS:
                    row[f"pred_{condition}"] = float(condition_outputs[condition][index].detach().cpu())
                row["delta_mask_text"] = row["pred_mask_text"] - row["pred_full_model"]
                row["delta_mask_image"] = row["pred_mask_image"] - row["pred_full_model"]
                row["delta_mask_bio"] = row["pred_mask_bio"] - row["pred_full_model"]
                row["text_contribution"] = row["pred_full_model"] - row["pred_mask_text"]
                row["image_contribution"] = row["pred_full_model"] - row["pred_mask_image"]
                row["bio_contribution"] = row["pred_full_model"] - row["pred_mask_bio"]
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, Any]] = []
    manifest = build_manifest_frame(Path(args.manifest), input_rows)
    predictions = read_predictions(Path(args.run_dir), "val", input_rows)
    if manifest.empty or predictions.empty:
        raise RuntimeError("C14-B requires a non-empty manifest and validation predictions.")
    positive_groups, group_counts, group_summary = build_cross_seed_groups(predictions, manifest)
    positive_groups.to_csv(out_dir / "c14b_cross_seed_positive_groups_val.csv", index=False)
    group_summary.to_csv(out_dir / "c14b_cross_seed_group_summary_val.csv", index=False)
    c14a = read_c14a(out_dir, input_rows)

    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    manifest_by_patient = {str(row["patient_id"]): row for row in val_rows}
    group_by_patient = dict(zip(group_counts["patient_id"].astype(str), group_counts["cross_seed_group"].astype(str)))
    text_by_patient = {patient_id: report_text(row) for patient_id, row in manifest_by_patient.items()}
    saved_predictions = {(str(row.patient_id), int(row.seed)): float(row.pred_prob) for row in predictions.itertuples()}
    if c14a.empty:
        input_rows.append({"path": "phase_c14a", "status": "not_merged", "notes": "no C14-A fields available"})
    else:
        input_rows.append({"path": "phase_c14a", "status": "available", "notes": "fields retained in prior audit; no inconsistent recomputation"})

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    input_rows.append({"path": "runtime", "status": "loaded", "notes": f"device={device}; analysis-only eval/no_grad"})
    feature_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    occlusion_rows: List[Dict[str, Any]] = []
    reproduction_errors: List[float] = []
    for seed in SEEDS:
        checkpoint_path = Path(args.run_dir) / "checkpoints" / f"seed_{seed}_best.pt"
        if not checkpoint_path.is_file():
            input_rows.append({"path": str(checkpoint_path), "status": "missing", "notes": "required C13 checkpoint"})
            raise FileNotFoundError(checkpoint_path)
        model, config, checkpoint = load_checkpoint(checkpoint_path, device)
        input_rows.append({"path": str(checkpoint_path), "status": "loaded", "notes": f"seed={seed}; best_epoch={checkpoint.get('best_epoch', '')}"})
        project_cfg = config.get("project", {})
        model_cfg = config.get("model", {})
        data_root = str(project_cfg.get("data_root", ""))
        dataset = PatientHTDataset(
            rows=val_rows,
            data_root=data_root,
            split="val",
            max_images=int(model_cfg.get("max_images_per_patient", 28)),
            image_size=int(model_cfg.get("image_size", 224)),
            text_max_length=int(model_cfg.get("text_max_length", args.text_max_length)),
            text_vocab_size=int(model_cfg.get("text_vocab_size", 50000)),
            bio_dim=int(model_cfg.get("bio_dim", 32)),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_patient_batch)
        seed_features, seed_ablation, seed_occlusion, max_error = run_seed(
            seed,
            model,
            loader,
            manifest_by_patient,
            group_by_patient,
            saved_predictions,
            text_by_patient,
            config,
            device,
            input_rows,
        )
        feature_rows.extend(seed_features)
        ablation_rows.extend(seed_ablation)
        occlusion_rows.extend(seed_occlusion)
        reproduction_errors.append(max_error)

    features = pd.DataFrame(feature_rows)
    ablation = pd.DataFrame(ablation_rows)
    occlusion = pd.DataFrame(occlusion_rows)
    features = features.merge(c14a.drop_duplicates("patient_id"), on="patient_id", how="left", suffixes=("", "_c14a")) if not c14a.empty else features
    features.to_csv(out_dir / "c14b_representation_features_val.csv", index=False)
    ablation.to_csv(out_dir / "c14b_modality_ablation_val.csv", index=False)
    occlusion.to_csv(out_dir / "c14b_text_occlusion_val.csv", index=False)

    feature_columns = [
        "text_embedding_norm", "image_embedding_norm", "bio_embedding_norm", "patient_anchor_norm", "fusion_feature_norm",
        "text_anchor_cosine", "image_anchor_cosine", "bio_anchor_cosine", "text_image_cosine", "text_bio_cosine", "image_bio_cosine",
        "e_img", "e_text", "e_bio", "e_synergy", "e_negative", "d_img_txt", "d_img_bio", "d_txt_bio",
    ]
    feature_summary = grouped_summary(features[features["label"] == 1], feature_columns)
    ablation_columns = ["pred_full_model", "pred_text_only_like", "pred_image_only_like", "pred_bio_only_like", "delta_mask_text", "delta_mask_image", "delta_mask_bio", "text_contribution", "image_contribution", "bio_contribution"]
    ablation_summary = grouped_summary(ablation, ablation_columns)
    occlusion_columns = ["pred_full_c13_text", "delta_remove_diffuse", "delta_remove_negative", "delta_prefix_only", "delta_remove_prefix"]
    occlusion_summary = grouped_summary(occlusion, occlusion_columns)
    feature_summary.to_csv(out_dir / "c14b_representation_group_summary_val.csv", index=False)
    ablation_summary.to_csv(out_dir / "c14b_modality_ablation_group_summary_val.csv", index=False)
    occlusion_summary.to_csv(out_dir / "c14b_text_occlusion_group_summary_val.csv", index=False)

    stability_base = ablation[ablation["label"] == 1].copy()
    stability = stability_base.groupby("patient_id", as_index=False).agg(
        cross_seed_group=("cross_seed_group", "first"),
        n_rows=("seed", "nunique"),
        pred_prob_std=("pred_full_model", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        text_contribution_std=("text_contribution", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        image_contribution_std=("image_contribution", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        bio_contribution_std=("bio_contribution", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
    )
    feature_stability = features[features["label"] == 1].groupby("patient_id").agg(
        text_anchor_cosine_std=("text_anchor_cosine", "std"),
        image_anchor_cosine_std=("image_anchor_cosine", "std"),
        bio_anchor_cosine_std=("bio_anchor_cosine", "std"),
    ).reset_index()
    stability = stability.merge(feature_stability, on="patient_id", how="left")
    stability["contribution_std_mean"] = stability[["text_contribution_std", "image_contribution_std", "bio_contribution_std"]].mean(axis=1)
    stability.to_csv(out_dir / "c14b_seedwise_fusion_stability_val.csv", index=False)
    write_ranked_cases(features, ablation, occlusion, out_dir)

    missing = pd.DataFrame(input_rows, columns=["path", "status", "notes"])
    missing.to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    route = write_reports(out_dir, group_summary, feature_summary, ablation_summary, occlusion_summary, stability, features[features["cross_seed_group"] == "all_seed_fn"], max(reproduction_errors) if reproduction_errors else float("nan"), missing)
    print(json.dumps({"output_dir": str(out_dir), "route": route, "all_seed_fn_patients": int((group_counts["cross_seed_group"] == "all_seed_fn").sum()), "max_reproduction_error": max(reproduction_errors) if reproduction_errors else None, "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
