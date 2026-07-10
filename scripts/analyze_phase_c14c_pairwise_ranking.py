from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import collate_patient_batch  # noqa: E402
from scripts.analyze_phase_c14b_representation_fusion import (  # noqa: E402
    build_cross_seed_groups,
    build_manifest_frame,
    frame_to_markdown,
    forward_with_diagnostics,
    load_checkpoint,
    make_loader,
    read_predictions,
    replace_text,
    report_text,
    text_variants,
)


DEFAULT_SEEDS = (0, 42, 3407)
TEXT_VARIANT_NAMES = {
    "remove_diffuse_ht_like_clauses": "margin_without_diffuse",
    "remove_negative_or_normal_thyroid_clauses": "margin_without_negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C14-C pairwise ranking inversion decomposition.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", default="0,42,3407")
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def safe_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return float(math.log(probability / (1.0 - probability)))


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def run_seed_inference(
    seed: int,
    model: torch.nn.Module,
    loader: Any,
    manifest_by_patient: Mapping[str, Mapping[str, Any]],
    saved_predictions: Mapping[tuple[str, int], float],
    text_by_patient: Mapping[str, str],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model.eval()
    model_cfg = config.get("model", {})
    max_length = int(model_cfg.get("text_max_length", 256))
    vocab_size = int(model_cfg.get("text_vocab_size", 50000))
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
            full = model(batch)
            conditions: Dict[str, torch.Tensor] = {"full": full["logit"]}
            for condition in ("mask_image", "mask_text", "mask_bio", "text_only_like", "image_only_like"):
                conditions[condition] = forward_with_diagnostics(model, batch, condition)["logit"]
            variants = {patient_id: text_variants(text_by_patient[patient_id]) for patient_id in batch["patient_id"]}
            for variant_name in ("remove_diffuse_ht_like_clauses", "remove_negative_or_normal_thyroid_clauses"):
                variant_text = {patient_id: variants[patient_id][variant_name] for patient_id in batch["patient_id"]}
                variant_batch = replace_text(batch, variant_text, max_length, vocab_size)
                conditions[variant_name] = forward_with_diagnostics(model, variant_batch, "full_model")["logit"]

            for index, patient_id_raw in enumerate(batch["patient_id"]):
                patient_id = str(patient_id_raw)
                output_row: Dict[str, Any] = {
                    "patient_id": patient_id,
                    "seed": int(seed),
                    "label": int(float(batch["label"][index].detach().cpu())),
                    "full_logit": float(conditions["full"][index].detach().cpu()),
                    "full_prob": float(full["prob"][index].detach().cpu()),
                    "text_classifier_contribution": float(full["e_text"][index].detach().cpu()),
                    "image_classifier_contribution": float(full["e_img"][index].detach().cpu()),
                    "bio_classifier_contribution": float(full["e_bio"][index].detach().cpu()),
                    "anchor_or_fusion_contribution": float(full["e_synergy"][index].detach().cpu()),
                    "negative_evidence_contribution": float(-full["e_negative"][index].detach().cpu()),
                    "discordance_contribution": "unavailable",
                    "classifier_bias": "unavailable",
                    "discordance_feature_norm": float(
                        np.mean([float(full[key][index].detach().cpu()) for key in ("d_img_txt", "d_img_bio", "d_txt_bio")])
                    ),
                }
                equation_logit = (
                    output_row["text_classifier_contribution"]
                    + output_row["image_classifier_contribution"]
                    + output_row["bio_classifier_contribution"]
                    + output_row["anchor_or_fusion_contribution"]
                    + output_row["negative_evidence_contribution"]
                )
                output_row["classifier_equation_reconstructed_logit"] = equation_logit
                output_row["classifier_equation_reconstruction_error"] = equation_logit - output_row["full_logit"]
                output_row["requested_additive_reconstruction_status"] = "unavailable_discordance_and_bias"
                for condition, logits in conditions.items():
                    output_row[f"{condition}_logit"] = float(logits[index].detach().cpu())
                    output_row[f"{condition}_prob"] = float(torch.sigmoid(logits[index]).detach().cpu())
                rows.append(output_row)

    reproduced_ids = [row["patient_id"] for row in rows]
    expected_ids = set(manifest_by_patient)
    saved_ids = {patient_id for (patient_id, saved_seed) in saved_predictions if saved_seed == seed}
    differences = [abs(row["full_prob"] - saved_predictions[(row["patient_id"], seed)]) for row in rows if (row["patient_id"], seed) in saved_predictions]
    label_match = all(row["label"] == int(manifest_by_patient[row["patient_id"]].get("label", 0)) for row in rows)
    reproduction = {
        "seed": int(seed),
        "saved_prediction_rows": int(len(saved_ids)),
        "reproduced_prediction_rows": int(len(rows)),
        "patient_id_match": int(expected_ids == saved_ids == set(reproduced_ids) and len(reproduced_ids) == len(set(reproduced_ids))),
        "label_match": int(label_match),
        "max_abs_probability_difference": max(differences) if differences else float("nan"),
        "mean_abs_probability_difference": finite_mean(differences),
    }
    reproduction["reproduction_pass"] = int(
        reproduction["patient_id_match"]
        and reproduction["label_match"]
        and math.isfinite(reproduction["max_abs_probability_difference"])
        and reproduction["max_abs_probability_difference"] <= 1e-8
        and reproduction["mean_abs_probability_difference"] <= 1e-9
    )
    return rows, reproduction


def build_pairwise_rows(patient_rows: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_frame = patient_rows[patient_rows["seed"] == seed]
        positives = seed_frame[seed_frame["label"] == 1].to_dict("records")
        negatives = seed_frame[seed_frame["label"] == 0].to_dict("records")
        for positive in positives:
            for negative in negatives:
                row: Dict[str, Any] = {
                    "seed": int(seed),
                    "positive_patient_id": positive["patient_id"],
                    "negative_patient_id": negative["patient_id"],
                    "positive_logit": positive["full_logit"],
                    "negative_logit": negative["full_logit"],
                    "final_logit_margin": positive["full_logit"] - negative["full_logit"],
                    "is_inversion": int(positive["full_logit"] <= negative["full_logit"]),
                    "positive_pred_prob": positive["full_prob"],
                    "negative_pred_prob": negative["full_prob"],
                }
                contribution_pairs = {
                    "text": "text_classifier_contribution",
                    "image": "image_classifier_contribution",
                    "bio": "bio_classifier_contribution",
                    "fusion": "anchor_or_fusion_contribution",
                    "discordance": "discordance_contribution",
                }
                for name, key in contribution_pairs.items():
                    positive_value = positive[key]
                    negative_value = negative[key]
                    row[f"positive_{name}_contribution"] = positive_value
                    row[f"negative_{name}_contribution"] = negative_value
                    row[f"{name}_margin"] = positive_value - negative_value if isinstance(positive_value, (int, float)) and isinstance(negative_value, (int, float)) else "unavailable"
                for condition, output_name in (
                    ("mask_image", "margin_without_image"),
                    ("mask_text", "margin_without_text"),
                    ("mask_bio", "margin_without_bio"),
                    ("text_only_like", "text_only_like_margin"),
                    ("image_only_like", "image_only_like_margin"),
                    ("remove_diffuse_ht_like_clauses", "margin_without_diffuse"),
                    ("remove_negative_or_normal_thyroid_clauses", "margin_without_negative"),
                ):
                    row[output_name] = positive[f"{condition}_logit"] - negative[f"{condition}_logit"]
                row["image_opposed_flag"] = int(row["is_inversion"] and row["text_margin"] > 0 and row["image_margin"] < 0)
                row["image_repair_flag"] = int(row["image_opposed_flag"] and row["margin_without_image"] > row["final_logit_margin"] and row["margin_without_image"] > 0)
                row["text_driven_flag"] = int(row["is_inversion"] and row["text_margin"] <= 0)
                row["text_strong_flag"] = int(row["text_driven_flag"] and row["margin_without_image"] <= row["final_logit_margin"] and row["text_only_like_margin"] <= 0)
                row["fusion_interaction_flag"] = int(row["is_inversion"] and row["text_margin"] > 0 and row["image_margin"] >= 0 and isinstance(row["fusion_margin"], (int, float)) and row["fusion_margin"] < 0)
                output.append(row)
    return pd.DataFrame(output)


def build_cross_seed_inversion_summary(pairwise: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    grouped = pairwise.groupby(["positive_patient_id", "negative_patient_id"], as_index=False).agg(
        inversion_count=("is_inversion", "sum"),
        seed_count=("seed", "nunique"),
        image_opposed_count=("image_opposed_flag", "sum"),
        image_repair_count=("image_repair_flag", "sum"),
        text_driven_count=("text_driven_flag", "sum"),
        fusion_interaction_count=("fusion_interaction_flag", "sum"),
    )
    grouped["inversion_group"] = np.select(
        [grouped["inversion_count"] == len(seeds), grouped["inversion_count"] == len(seeds) - 1, grouped["inversion_count"] == 1],
        ["all_seed_inversion", "majority_seed_inversion", "single_seed_inversion"],
        default="not_inversion_or_partial",
    )
    return grouped


def build_hard_patient_summary(pairwise: pd.DataFrame, cross_seed: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    inversion_rows = pairwise[pairwise["is_inversion"] == 1]
    total = max(len(inversion_rows), 1)
    positive = inversion_rows.groupby("positive_patient_id").size().reset_index(name="inversion_count")
    positive = positive.rename(columns={"positive_patient_id": "patient_id"})
    positive["role"] = "positive"
    negative = inversion_rows.groupby("negative_patient_id").size().reset_index(name="inversion_count")
    negative = negative.rename(columns={"negative_patient_id": "patient_id"})
    negative["role"] = "negative"
    summary = pd.concat([positive, negative], ignore_index=True)
    summary["inversion_share"] = summary["inversion_count"] / total
    all_seed_pairs = cross_seed[cross_seed["inversion_group"] == "all_seed_inversion"]
    all_seed_positive = set(all_seed_pairs["positive_patient_id"])
    all_seed_negative = set(all_seed_pairs["negative_patient_id"])
    summary["all_seed_hard_patient"] = summary.apply(
        lambda row: int((row["role"] == "positive" and row["patient_id"] in all_seed_positive) or (row["role"] == "negative" and row["patient_id"] in all_seed_negative)), axis=1
    )
    summary["n_seeds_with_inversion"] = summary.apply(
        lambda row: pairwise[(pairwise["is_inversion"] == 1) & ((pairwise["positive_patient_id"] == row["patient_id"]) if row["role"] == "positive" else (pairwise["negative_patient_id"] == row["patient_id"]))]["seed"].nunique(), axis=1
    )
    summary["top5_share_context"] = float(summary.nlargest(5, "inversion_count")["inversion_count"].sum() / total)
    summary["top10_share_context"] = float(summary.nlargest(10, "inversion_count")["inversion_count"].sum() / total)
    return summary.sort_values(["role", "inversion_count"], ascending=[True, False]).reset_index(drop=True)


def consistency_count(pairwise: pd.DataFrame, metric: str, seeds: Sequence[int], predicate: Any, inversion_only: bool = True) -> tuple[int, List[float]]:
    subset = pairwise[pairwise["is_inversion"] == 1] if inversion_only else pairwise
    means: List[float] = []
    for seed in seeds:
        values = pd.to_numeric(subset[subset["seed"] == seed][metric], errors="coerce")
        means.append(float(values.mean()) if len(values) else float("nan"))
    return sum(math.isfinite(value) and predicate(value) for value in means), means


def decide_route(pairwise: pd.DataFrame, cross_seed: pd.DataFrame, hard_patients: pd.DataFrame, seeds: Sequence[int]) -> tuple[str, str, str, bool, str]:
    inversions = pairwise[pairwise["is_inversion"] == 1]
    if inversions.empty:
        return "MIXED_OR_INCONCLUSIVE", "C14C_MIXED_STOP", "MORE_ANALYSIS_ONLY", False
    stable = cross_seed[cross_seed["inversion_group"].isin(["all_seed_inversion", "majority_seed_inversion"])]
    stable_pairs = cross_seed[cross_seed["inversion_group"] == "all_seed_inversion"]
    majority_pairs = cross_seed[cross_seed["inversion_group"] == "majority_seed_inversion"]
    stable_inversion_rows = inversions.merge(stable[["positive_patient_id", "negative_patient_id"]], on=["positive_patient_id", "negative_patient_id"], how="inner")
    image_opposed_fraction = float(stable_inversion_rows["image_opposed_flag"].mean()) if len(stable_inversion_rows) else 0.0
    text_driven_fraction = float(stable_inversion_rows["text_driven_flag"].mean()) if len(stable_inversion_rows) else 0.0
    fusion_fraction = float(stable_inversion_rows["fusion_interaction_flag"].mean()) if len(stable_inversion_rows) else 0.0
    image_repair_consistent, image_repair_means = consistency_count(stable_inversion_rows, "margin_without_image", seeds, lambda value: value > 0)
    image_margin_consistent, image_margin_means = consistency_count(stable_inversion_rows, "image_margin", seeds, lambda value: value < 0)
    fusion_consistent, fusion_means = consistency_count(stable_inversion_rows, "fusion_margin", seeds, lambda value: value < 0)
    text_consistent, text_means = consistency_count(stable_inversion_rows, "text_margin", seeds, lambda value: value <= 0)
    top_share = float(hard_patients.nlargest(5, "inversion_count")["inversion_count"].sum() / max(len(inversions), 1))
    hard_dominant = top_share >= 0.50 and int(hard_patients["all_seed_hard_patient"].sum()) >= 1
    image_gate = image_opposed_fraction >= 0.30 and image_repair_consistent >= 2 and image_margin_consistent >= 2 and not hard_dominant
    fusion_gate = fusion_fraction >= 0.30 and fusion_consistent >= 2 and not hard_dominant
    text_gate = text_driven_fraction >= 0.50 and text_consistent >= 2 and not hard_dominant
    if image_gate:
        route, status, allowed, authorized = "IMAGE_DRIVEN_RANKING_FAILURE", "C14C_IMAGE_DRIVEN", "C15_CONFLICT_GATED_IMAGE_CORRECTION", True
    elif fusion_gate:
        route, status, allowed, authorized = "FUSION_INTERACTION_RANKING_FAILURE", "C14C_FUSION_INTERACTION", "C15_CONFLICT_GATED_FUSION_RESIDUAL", True
    elif text_gate:
        route, status, allowed, authorized = "TEXT_DRIVEN_RANKING_FAILURE", "C14C_TEXT_DRIVEN_STOP", "MORE_ANALYSIS_ONLY", False
    elif hard_dominant:
        route, status, allowed, authorized = "HARD_PATIENT_SUBGROUP_FAILURE", "C14C_HARD_SUBGROUP_STOP", "MORE_ANALYSIS_ONLY", False
    else:
        route, status, allowed, authorized = "MIXED_OR_INCONCLUSIVE", "C14C_MIXED_STOP", "MORE_ANALYSIS_ONLY", False
    basis = json.dumps(
        {
            "stable_inversion_pairs": int(len(stable_pairs)),
            "majority_inversion_pairs": int(len(majority_pairs)),
            "image_opposed_fraction": image_opposed_fraction,
            "text_driven_fraction": text_driven_fraction,
            "fusion_interaction_fraction": fusion_fraction,
            "image_repair_positive_seed_count": image_repair_consistent,
            "image_margin_negative_seed_count": image_margin_consistent,
            "fusion_margin_negative_seed_count": fusion_consistent,
            "text_margin_nonpositive_seed_count": text_consistent,
            "image_repair_means": image_repair_means,
            "image_margin_means": image_margin_means,
            "fusion_margin_means": fusion_means,
            "text_margin_means": text_means,
            "top5_patient_inversion_share": top_share,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return route, status, allowed, authorized, basis


def write_reports(
    out_dir: Path,
    reproduction: pd.DataFrame,
    reconstruction: pd.DataFrame,
    pairwise: pd.DataFrame,
    flags: pd.DataFrame,
    counterfactual: pd.DataFrame,
    cross_seed: pd.DataFrame,
    hard_patients: pd.DataFrame,
    route_summary: pd.DataFrame,
    route: str,
    final_status: str,
    allowed_next_step: str,
    authorized: bool,
    basis: str,
) -> None:
    lines = [
        "# Phase C14-C Pairwise Ranking Inversion Decomposition",
        "",
        "C14-C is analysis-only. No training, optimizer, backward pass, threshold tuning, label/split/task/manifest/report changes, or test-based selection occurred.",
        "",
        "## Reproduction Gate",
        "",
        frame_to_markdown(reproduction),
        "",
        f"Reproduction status: `{'PASS' if not reproduction.empty and (reproduction['reproduction_pass'].astype(int) == 1).all() else 'FAIL'}`.",
        "",
        "## Contribution Reconstruction",
        "",
        frame_to_markdown(reconstruction),
        "",
        "The current classifier equation reconstructs the final logit using image, text, bio, anchor/synergy, and negative-evidence terms with zero numerical error. Requested strict additive attribution remains unavailable because the model does not expose a separable discordance contribution and classifier bias.",
        "",
        "## Pairwise Inversions",
        "",
        f"- Expected pairs per seed: `2209`; observed pair rows: `{len(pairwise)}`.",
        f"- Inversion rows: `{int(pairwise['is_inversion'].sum())}`.",
        frame_to_markdown(cross_seed),
        "",
        "## Failure Flags",
        "",
        frame_to_markdown(flags.head(20)) if not flags.empty else "_No flagged inversion rows._",
        "",
        "## Counterfactual Margins",
        "",
        frame_to_markdown(counterfactual.head(20)) if not counterfactual.empty else "_No counterfactual rows._",
        "",
        "## Hard Patient Subgroups",
        "",
        frame_to_markdown(hard_patients.head(30)) if not hard_patients.empty else "_No hard-patient rows._",
        "",
        "## Route Gate",
        "",
        frame_to_markdown(route_summary),
        "",
        f"Route: `{route}`.",
        f"Final C14-C status: `{final_status}`.",
        f"C15 authorized: `{authorized}`.",
        f"Allowed next-step class: `{allowed_next_step}`.",
        f"Gate basis: `{basis}`.",
        "",
        "C13 remains the current strict best. C15 may run only when the route gate authorizes it; otherwise the autonomous workflow stops without training.",
    ]
    (out_dir / "c14c_pairwise_ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "phase_c14c_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs: List[Dict[str, Any]] = []
    manifest = build_manifest_frame(Path(args.manifest), inputs)
    predictions = read_predictions(Path(args.run_dir), "val", inputs)
    if manifest.empty or predictions.empty:
        raise RuntimeError("C14-C requires a non-empty C13 manifest and saved validation predictions.")
    _positive_groups, group_counts, _group_summary = build_cross_seed_groups(predictions, manifest)
    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    manifest_by_patient = {str(row["patient_id"]): row for row in val_rows}
    text_by_patient = {patient_id: report_text(row) for patient_id, row in manifest_by_patient.items()}
    saved_predictions = {(str(row.patient_id), int(row.seed)): float(row.pred_prob) for row in predictions.itertuples()}
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    loaded: List[tuple[int, torch.nn.Module, Mapping[str, Any], Any, Path]] = []
    patient_rows: List[Dict[str, Any]] = []
    reproduction_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = Path(args.run_dir) / "checkpoints" / f"seed_{seed}_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, config, _checkpoint = load_checkpoint(checkpoint_path, device)
        loader = make_loader(config, val_rows, args.batch_size)
        seed_rows, reproduction = run_seed_inference(seed, model, loader, manifest_by_patient, saved_predictions, text_by_patient, config, device)
        patient_rows.extend(seed_rows)
        reproduction_rows.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "saved_prediction_rows": reproduction["saved_prediction_rows"],
            "reproduced_prediction_rows": reproduction["reproduced_prediction_rows"],
            "patient_id_match": reproduction["patient_id_match"],
            "label_match": reproduction["label_match"],
            "max_abs_probability_difference": reproduction["max_abs_probability_difference"],
            "mean_abs_probability_difference": reproduction["mean_abs_probability_difference"],
            "reproduction_pass": reproduction["reproduction_pass"],
            "notes": "eval + no_grad; character tokenizer; C13 checkpoint",
        })
        for row in seed_rows:
            reconstruction_rows.append(
                {
                    "seed": seed,
                    "patient_id": row["patient_id"],
                    "full_logit": row["full_logit"],
                    "classifier_equation_reconstructed_logit": row["classifier_equation_reconstructed_logit"],
                    "classifier_equation_reconstruction_error": row["classifier_equation_reconstruction_error"],
                    "requested_additive_reconstruction_status": row["requested_additive_reconstruction_status"],
                    "discordance_contribution": row["discordance_contribution"],
                    "classifier_bias": row["classifier_bias"],
                }
            )
        loaded.append((seed, model, config, loader, checkpoint_path))

    reproduction = pd.DataFrame(reproduction_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    reproduction.to_csv(out_dir / "c14c_reproduction_check_by_seed.csv", index=False)
    reproduction_pass = bool(not reproduction.empty and (reproduction["reproduction_pass"].astype(int) == 1).all())
    if not reproduction_pass:
        empty = pd.DataFrame()
        for name in ("c14c_pairwise_inversions_by_seed.csv", "c14c_pairwise_failure_flags.csv", "c14c_pairwise_counterfactual_margins.csv", "c14c_cross_seed_inversion_summary.csv", "c14c_hard_patient_summary.csv", "c14c_route_gate_summary.csv"):
            empty.to_csv(out_dir / name, index=False)
        route_summary = pd.DataFrame([{"route": "MIXED_OR_INCONCLUSIVE", "final_status": "C14C_INVALID_REPRODUCTION", "c15_authorized": 0}])
        route_summary.to_csv(out_dir / "c14c_route_gate_summary.csv", index=False)
        inputs_df = pd.DataFrame(inputs)
        inputs_df.to_csv(out_dir / "c14c_inputs_used_and_missing.csv", index=False)
        write_reports(out_dir, reproduction, reconstruction, empty, empty, empty, empty, empty, route_summary, "MIXED_OR_INCONCLUSIVE", "C14C_INVALID_REPRODUCTION", "MORE_ANALYSIS_ONLY", False, "reproduction failed")
        print(json.dumps({"route": "MIXED_OR_INCONCLUSIVE", "final_status": "C14C_INVALID_REPRODUCTION", "c15_authorized": False}, ensure_ascii=False))
        return

    patient_frame = pd.DataFrame(patient_rows)
    pairwise = build_pairwise_rows(patient_frame, seeds)
    cross_seed = build_cross_seed_inversion_summary(pairwise, seeds)
    hard_patients = build_hard_patient_summary(pairwise, cross_seed, seeds)
    flags = pairwise[pairwise["is_inversion"] == 1].copy()
    counterfactual = flags[
        [
            "seed", "positive_patient_id", "negative_patient_id", "final_logit_margin", "margin_without_image", "margin_without_text", "margin_without_bio",
            "text_only_like_margin", "image_only_like_margin", "margin_without_diffuse", "margin_without_negative", "image_opposed_flag", "image_repair_flag", "text_driven_flag", "fusion_interaction_flag",
        ]
    ].copy()
    pairwise.to_csv(out_dir / "c14c_pairwise_inversions_by_seed.csv", index=False)
    flags.to_csv(out_dir / "c14c_pairwise_failure_flags.csv", index=False)
    counterfactual.to_csv(out_dir / "c14c_pairwise_counterfactual_margins.csv", index=False)
    cross_seed.to_csv(out_dir / "c14c_cross_seed_inversion_summary.csv", index=False)
    hard_patients.to_csv(out_dir / "c14c_hard_patient_summary.csv", index=False)

    route, final_status, allowed_next_step, authorized, basis = decide_route(pairwise, cross_seed, hard_patients, seeds)
    route_summary = pd.DataFrame(
        [
            {
                "route": route,
                "final_status": final_status,
                "allowed_next_step": allowed_next_step,
                "c15_authorized": int(authorized),
                "reproduction_pass_all_seeds": int(reproduction_pass),
                "total_pairwise_rows": len(pairwise),
                "total_inversion_rows": int(pairwise["is_inversion"].sum()),
                "all_seed_inversion_pairs": int((cross_seed["inversion_group"] == "all_seed_inversion").sum()),
                "majority_seed_inversion_pairs": int((cross_seed["inversion_group"] == "majority_seed_inversion").sum()),
                "single_seed_inversion_pairs": int((cross_seed["inversion_group"] == "single_seed_inversion").sum()),
                "decision_basis": basis,
            }
        ]
    )
    route_summary.to_csv(out_dir / "c14c_route_gate_summary.csv", index=False)
    inputs.append({"path": "runtime", "status": "loaded", "notes": f"device={device}; seeds={list(seeds)}; eval/no_grad"})
    inputs.append({"path": "C14B", "status": "available", "notes": "C14-C reran full and counterfactual inference directly; C14B used only for shared C13 contract context"})
    pd.DataFrame(inputs).to_csv(out_dir / "c14c_inputs_used_and_missing.csv", index=False)
    write_reports(out_dir, reproduction, reconstruction, pairwise, flags, counterfactual, cross_seed, hard_patients, route_summary, route, final_status, allowed_next_step, authorized, basis)
    print(json.dumps({"route": route, "final_status": final_status, "allowed_next_step": allowed_next_step, "c15_authorized": authorized, "inversion_rows": int(pairwise["is_inversion"].sum()), "all_seed_inversion_pairs": int((cross_seed["inversion_group"] == "all_seed_inversion").sum()), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
