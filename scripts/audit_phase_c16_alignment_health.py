from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.models import DMEAHTModel  # noqa: E402
from train import DSSA_LOSS_WEIGHTS, dssa_loss_terms, pairwise_ranking_loss, set_seed  # noqa: E402


PATIENT_DIAGNOSTIC_COLUMNS = (
    "patient_id",
    "seed",
    "split",
    "label",
    "pred_prob",
    "logit",
    "prototype_similarity_non_ht",
    "prototype_similarity_ht",
    "disease_margin",
    "shared_attention_img",
    "shared_attention_txt",
    "shared_attention_bio",
    "specific_gate_img",
    "specific_gate_txt",
    "specific_gate_bio",
    "shared_img_norm",
    "shared_txt_norm",
    "shared_bio_norm",
    "specific_img_norm",
    "specific_txt_norm",
    "specific_bio_norm",
    "specific_residual_norm",
    "patient_shared_norm",
    "soft_disease_anchor_norm",
    "specific_residual_shared_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit C16 DSSA synthetic and saved-run alignment health.")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--c13-config", default="configs/dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml")
    parser.add_argument("--c16-config", default="configs/dmea_ht_v2_c16_dssa_smoke.yaml")
    parser.add_argument("--pilot-config", default="configs/dmea_ht_v2_c16_dssa_seed0_pilot.yaml")
    parser.add_argument("--stress-config", default="configs/dmea_ht_v2_c16_dssa_stress_seeds.yaml")
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="analysis_reports/phase_c16")
    parser.add_argument("--require-health-pass", action="store_true")
    return parser.parse_args()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.to_dict(orient="records"):
        values = []
        for column in frame.columns:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                text = "NA"
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def synthetic_batch(batch_size: int = 4, bio_dim: int = 7) -> Dict[str, Any]:
    images = torch.randn(batch_size, 2, 3, 32, 32)
    image_mask = torch.ones(batch_size, 2)
    image_mask[1] = 0
    input_ids = torch.randint(2, 200, (batch_size, 12))
    attention_mask = torch.ones(batch_size, 12, dtype=torch.long)
    attention_mask[3] = 0
    bio_values = torch.randn(batch_size, bio_dim)
    bio_missing_mask = torch.zeros(batch_size, bio_dim)
    bio_missing_mask[2] = 1
    return {
        "patient_id": [f"synthetic_{index}" for index in range(batch_size)],
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "images": images,
        "image_mask": image_mask,
        "report_input_ids": input_ids,
        "report_attention_mask": attention_mask,
        "bio_values": bio_values,
        "bio_missing_mask": bio_missing_mask,
        "bio_abnormal_flags": torch.zeros(batch_size, bio_dim),
        "sample_weight": torch.ones(batch_size),
    }


def all_floating_outputs_finite(outputs: Dict[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(value).all().item())
        for value in outputs.values()
        if torch.is_tensor(value) and torch.is_floating_point(value)
    )


def gradient_present(model: torch.nn.Module, prefix: str) -> bool:
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
    return bool(parameters) and any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in parameters
    )


def run_synthetic_smoke(
    c13_config_path: Path,
    c16_config_path: Path,
    pilot_config_path: Path,
    stress_config_path: Path,
    output_dir: Path,
) -> bool:
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    c13_config = load_config(c13_config_path)
    c13_explicit_disabled = copy.deepcopy(c13_config)
    c13_explicit_disabled.setdefault("model", {})["use_dssa"] = False
    batch = synthetic_batch(bio_dim=int(c13_config["model"].get("bio_dim", 7)))

    set_seed(1701)
    legacy_default = DMEAHTModel(c13_config).eval()
    set_seed(1701)
    legacy_disabled = DMEAHTModel(c13_explicit_disabled).eval()
    with torch.no_grad():
        legacy_default_output = legacy_default(batch)
        legacy_disabled_output = legacy_disabled(batch)
    check("legacy_state_dict_keys", legacy_default.state_dict().keys() == legacy_disabled.state_dict().keys(), len(legacy_default.state_dict()))
    max_legacy_difference = float((legacy_default_output["logit"] - legacy_disabled_output["logit"]).abs().max())
    check("legacy_forward_equivalence", max_legacy_difference == 0.0, max_legacy_difference)

    c16_config = load_config(c16_config_path)
    pilot_config = load_config(pilot_config_path)
    stress_config = load_config(stress_config_path)
    c16_configs = (c16_config, pilot_config, stress_config)
    check(
        "frozen_data_contract",
        all(
            config["project"]["data_root"] == c13_config["project"]["data_root"]
            and config["project"]["manifest"] == c13_config["project"]["manifest"]
            for config in c16_configs
        ),
        [config["project"]["manifest"] for config in c16_configs],
    )
    frozen_model_keys = (
        "variant",
        "hidden_dim",
        "text_vocab_size",
        "text_max_length",
        "bio_dim",
        "max_images_per_patient",
        "image_size",
        "dropout",
        "use_patient_anchor",
        "use_evidence_role",
        "use_discordance",
    )
    model_differences = {
        key: [config["model"].get(key) for config in c16_configs]
        for key in frozen_model_keys
        if any(config["model"].get(key) != c13_config["model"].get(key) for config in c16_configs)
    }
    check("frozen_encoder_and_input_contract", not model_differences, model_differences)
    frozen_training_keys = ("batch_size", "lr", "weight_decay", "epochs", "patience", "primary_metric", "num_workers")
    training_differences = {
        key: [config["training"].get(key) for config in (pilot_config, stress_config)]
        for key in frozen_training_keys
        if any(config["training"].get(key) != c13_config["training"].get(key) for config in (pilot_config, stress_config))
    }
    check("frozen_pilot_optimizer_contract", not training_differences, training_differences)
    check(
        "formal_seed_contract",
        pilot_config["training"].get("seeds") == [0]
        and stress_config["training"].get("seeds") == [42, 3407],
        [pilot_config["training"].get("seeds"), stress_config["training"].get("seeds")],
    )
    expected_weights = {
        "lambda_proto": 0.05,
        "lambda_shared": 0.02,
        "lambda_orth": 0.01,
        "lambda_var": 0.005,
        "lambda_sep": 0.01,
        "lambda_rank": 0.02,
    }
    weight_differences = {
        key: [config["loss"].get(key) for config in c16_configs]
        for key, expected in expected_weights.items()
        if any(float(config["loss"].get(key, float("nan"))) != expected for config in c16_configs)
    }
    check("fixed_dssa_loss_weights", not weight_differences, weight_differences)
    forbidden_loss_keys = (
        "text_morphology_weight",
        "image_morphology_weight",
        "text_negative_weight",
        "bio_evidence_weight",
        "discordance_label_weight",
        "counterfactual_weight",
        "matched_supcon_weight",
    )
    enabled_forbidden_losses = {
        key: [config["loss"].get(key) for config in c16_configs]
        for key in forbidden_loss_keys
        if any(float(config["loss"].get(key, 0.0) or 0.0) != 0.0 for config in c16_configs)
    }
    check("forbidden_legacy_losses_disabled", not enabled_forbidden_losses, enabled_forbidden_losses)
    residual_scales = [float(config["model"].get("dssa_residual_scale", 1.0)) for config in c16_configs]
    check("bounded_specific_residual_scale", all(0.0 <= value <= 0.15 for value in residual_scales), residual_scales)
    check(
        "validation_auc_checkpoint_contract",
        all(config["training"].get("primary_metric") == "val_AUC" for config in c16_configs),
        [config["training"].get("primary_metric") for config in c16_configs],
    )
    set_seed(1701)
    model = DMEAHTModel(c16_config).train()
    outputs = model(batch)
    check("c16_output_shape", tuple(outputs["logit"].shape) == (4,), tuple(outputs["logit"].shape))
    check("all_outputs_finite", all_floating_outputs_finite(outputs), "floating outputs")

    available = outputs["dssa_available_mask"].bool()
    attention = torch.stack([outputs[f"shared_attention_{name}"] for name in ("img", "txt", "bio")], dim=1)
    expected_attention_sum = available.any(dim=1).to(attention.dtype)
    max_attention_error = float((attention.sum(dim=1) - expected_attention_sum).abs().max().detach())
    check("shared_attention_masked_sum", max_attention_error <= 1e-6, max_attention_error)
    check("missing_modalities_masked", bool((attention[~available] == 0).all().item()), attention.detach().tolist())

    mixed_rank = pairwise_ranking_loss(outputs["logit"], batch["label"])
    all_positive_rank = pairwise_ranking_loss(outputs["logit"], torch.ones_like(batch["label"]))
    all_negative_rank = pairwise_ranking_loss(outputs["logit"], torch.zeros_like(batch["label"]))
    check("ranking_mixed_finite", bool(torch.isfinite(mixed_rank).item()), float(mixed_rank.detach()))
    check("ranking_all_positive_zero", float(all_positive_rank.detach()) == 0.0, float(all_positive_rank.detach()))
    check("ranking_all_negative_zero", float(all_negative_rank.detach()) == 0.0, float(all_negative_rank.detach()))

    terms = dssa_loss_terms(outputs, batch["label"], include_ranking=True)
    weighted = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"])
    loss_cfg = c16_config["loss"]
    for term_name, term in terms.items():
        weighted = weighted + float(loss_cfg[f"lambda_{DSSA_LOSS_WEIGHTS[term_name]}"]) * term
    check("all_losses_finite", bool(torch.isfinite(weighted).item()) and all(bool(torch.isfinite(term).item()) for term in terms.values()), {key: float(value.detach()) for key, value in terms.items()})
    weighted.backward()
    for prefix in (
        "dssa.shared_projectors",
        "dssa.specific_projectors",
        "dssa.prototypes",
        "dssa.shared_score",
        "dssa.specific_gates",
        "dssa.specific_residual_projectors",
        "dssa.classifier",
    ):
        check(f"gradient_{prefix.replace('.', '_')}", gradient_present(model, prefix), prefix)

    prototype_cosine = float(outputs["prototype_cosine"].detach())
    residual_ratio = float(outputs["specific_residual_shared_ratio"].detach().max())
    check("prototype_not_collapsed", prototype_cosine < 0.95, prototype_cosine)
    check("specific_residual_controlled", residual_ratio < 1.0, residual_ratio)

    forbidden = (
        "n_images",
        "n_visits",
        "selected_n_visits",
        "used_images",
        "image_padding_count",
        "report_length",
        "source_folder",
    )
    alignment_source = (Path(__file__).resolve().parents[1] / "dmea_ht" / "alignment.py").read_text(encoding="utf-8")
    leaked = [field for field in forbidden if field in alignment_source]
    check("shortcut_fields_absent", not leaked, leaked)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "checks": checks,
    }
    (output_dir / "c16_synthetic_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(output_dir / "c16_synthetic_smoke_checks.csv", index=False)
    print(json.dumps({"synthetic_smoke": result["status"], "checks": len(checks), "output_dir": str(output_dir)}))
    return result["status"] == "PASS"


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected name=path, got: {value}")
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def read_run_frames(run_specs: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    for model_id, run_dir in [parse_named_path(value) for value in run_specs]:
        metric_path = run_dir / "reports" / "metrics_by_seed.csv"
        if metric_path.is_file():
            metrics = pd.read_csv(metric_path)
            metrics.insert(0, "model_id", model_id)
            metric_frames.append(metrics)
        for split in ("train", "val"):
            for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
                frame = pd.read_csv(path)
                frame["patient_id"] = frame["patient_id"].astype(str)
                frame["split"] = split
                frame["model_id"] = model_id
                prediction_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def finite_mean(values: Iterable[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def prototype_health(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_id, seed, split), group in predictions.groupby(["model_id", "seed", "split"]):
        label = pd.to_numeric(group["label"], errors="coerce").astype(int)
        margin = pd.to_numeric(group["disease_margin"], errors="coerce")
        metric_row = metrics[(metrics["model_id"] == model_id) & (metrics["seed"] == seed) & (metrics["split"] == split)]
        cosine = float(metric_row.iloc[0].get("prototype_cosine", float("nan"))) if not metric_row.empty else float("nan")
        rows.append(
            {
                "model_id": model_id,
                "seed": int(seed),
                "split": split,
                "n_patients": int(len(group)),
                "prototype_cosine": cosine,
                "prototype_distance": 1.0 - cosine,
                "non_ht_similarity_label0_mean": finite_mean(group.loc[label == 0, "prototype_similarity_non_ht"]),
                "ht_similarity_label1_mean": finite_mean(group.loc[label == 1, "prototype_similarity_ht"]),
                "disease_margin_label0_mean": finite_mean(margin[label == 0]),
                "disease_margin_label0_std": float(margin[label == 0].std(ddof=1)),
                "disease_margin_label1_mean": finite_mean(margin[label == 1]),
                "disease_margin_label1_std": float(margin[label == 1].std(ddof=1)),
                "prototype_assignment_accuracy": float(((margin >= 0).astype(int) == label).mean()),
                "prototype_collapse_flag": int(math.isfinite(cosine) and cosine >= 0.95),
            }
        )
    return pd.DataFrame(rows)


def shared_health(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = ((pd.to_numeric(frame["pred_prob"]) >= 0.5).astype(int) == pd.to_numeric(frame["label"]).astype(int)).astype(int)
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(["model_id", "seed", "split", "label", "correct"]):
        model_id, seed, split, label, correct = keys
        row: Dict[str, Any] = {
            "model_id": model_id,
            "seed": int(seed),
            "split": split,
            "label": int(label),
            "correct": int(correct),
            "n_patients": int(len(group)),
        }
        for left, right in (("img", "txt"), ("img", "bio"), ("txt", "bio")):
            available = pd.to_numeric(group[f"shared_pair_available_{left}_{right}"], errors="coerce") > 0.5
            row[f"mean_shared_cosine_{left}_{right}"] = finite_mean(group.loc[available, f"shared_cosine_{left}_{right}"])
        for modality in ("img", "txt", "bio"):
            available = pd.to_numeric(group[f"modality_available_{modality}"], errors="coerce") > 0.5
            attention = pd.to_numeric(group.loc[available, f"shared_attention_{modality}"], errors="coerce").dropna()
            row[f"shared_attention_{modality}_mean"] = float(attention.mean()) if not attention.empty else float("nan")
            row[f"shared_attention_{modality}_std"] = float(attention.std(ddof=1)) if len(attention) > 1 else 0.0
            row[f"shared_attention_{modality}_p05"] = float(attention.quantile(0.05)) if not attention.empty else float("nan")
            row[f"shared_attention_{modality}_p95"] = float(attention.quantile(0.95)) if not attention.empty else float("nan")
        metric_row = metrics[(metrics["model_id"] == model_id) & (metrics["seed"] == seed) & (metrics["split"] == split)]
        collapse_inputs = []
        if not metric_row.empty:
            for modality in ("img", "txt", "bio"):
                row[f"shared_{modality}_feature_std_mean"] = metric_row.iloc[0].get(f"shared_{modality}_feature_std_mean", float("nan"))
                row[f"shared_{modality}_offdiag_cosine_mean"] = metric_row.iloc[0].get(f"shared_{modality}_offdiag_cosine_mean", float("nan"))
                collapse_inputs.append(float(row[f"shared_{modality}_offdiag_cosine_mean"]))
        row["shared_sample_collapse_flag"] = int(bool(collapse_inputs) and max(collapse_inputs) >= 0.98)
        rows.append(row)
    return pd.DataFrame(rows)


def specific_health(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_id, seed, split), group in predictions.groupby(["model_id", "seed", "split"]):
        metric_row = metrics[(metrics["model_id"] == model_id) & (metrics["seed"] == seed) & (metrics["split"] == split)]
        row: Dict[str, Any] = {
            "model_id": model_id,
            "seed": int(seed),
            "split": split,
            "n_patients": int(len(group)),
            "specific_residual_shared_ratio_mean": finite_mean(group["specific_residual_shared_ratio"]),
        }
        duplicate_flags = []
        collapse_flags = []
        gate_flags = []
        for modality in ("img", "txt", "bio"):
            available = pd.to_numeric(group[f"modality_available_{modality}"], errors="coerce") > 0.5
            cosine = pd.to_numeric(group.loc[available, f"shared_specific_cosine_{modality}"], errors="coerce")
            gate = pd.to_numeric(group.loc[available, f"specific_gate_{modality}"], errors="coerce")
            row[f"mean_abs_shared_specific_cosine_{modality}"] = float(cosine.abs().mean()) if not cosine.empty else float("nan")
            row[f"mean_specific_gate_{modality}"] = float(gate.mean()) if not gate.empty else float("nan")
            row[f"specific_gate_{modality}_std"] = float(gate.std(ddof=1)) if len(gate) > 1 else 0.0
            row[f"specific_gate_{modality}_p05"] = float(gate.quantile(0.05)) if not gate.empty else float("nan")
            row[f"specific_gate_{modality}_p95"] = float(gate.quantile(0.95)) if not gate.empty else float("nan")
            row[f"specific_gate_{modality}_saturation_fraction"] = (
                float(((gate <= 0.01) | (gate >= 0.99)).mean()) if not gate.empty else float("nan")
            )
            feature_std = float(metric_row.iloc[0].get(f"specific_{modality}_feature_std_mean", float("nan"))) if not metric_row.empty else float("nan")
            row[f"specific_{modality}_feature_std_mean"] = feature_std
            duplicate_flags.append(math.isfinite(row[f"mean_abs_shared_specific_cosine_{modality}"]) and row[f"mean_abs_shared_specific_cosine_{modality}"] >= 0.95)
            collapse_flags.append(math.isfinite(feature_std) and feature_std <= 1e-3)
            gate_saturation_fraction = float(((gate <= 0.01) | (gate >= 0.99)).mean()) if not gate.empty else 1.0
            gate_flags.append(
                not gate.empty
                and (
                    float(gate.mean()) <= 0.01
                    or float(gate.mean()) >= 0.99
                    or gate_saturation_fraction >= 0.50
                )
            )
        row["specific_duplicates_shared_flag"] = int(any(duplicate_flags))
        row["specific_collapse_flag"] = int(any(collapse_flags))
        row["specific_dominates_shared_flag"] = int(row["specific_residual_shared_ratio_mean"] >= 1.0)
        row["global_gate_saturation_flag"] = int(any(gate_flags))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    validation = predictions[predictions["split"] == "val"]
    for (model_id, seed), group in validation.groupby(["model_id", "seed"]):
        positives = group[pd.to_numeric(group["label"]).astype(int) == 1].to_dict("records")
        negatives = group[pd.to_numeric(group["label"]).astype(int) == 0].to_dict("records")
        for positive in positives:
            for negative in negatives:
                margin = float(positive["logit"]) - float(negative["logit"])
                rows.append(
                    {
                        "model_id": model_id,
                        "seed": int(seed),
                        "positive_patient_id": positive["patient_id"],
                        "negative_patient_id": negative["patient_id"],
                        "positive_logit": float(positive["logit"]),
                        "negative_logit": float(negative["logit"]),
                        "pair_margin": margin,
                        "is_inversion": int(margin <= 0.0),
                    }
                )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = pairwise.groupby(["model_id", "seed"], as_index=False).agg(
        pair_count=("is_inversion", "size"),
        inversion_count=("is_inversion", "sum"),
        pair_margin_mean=("pair_margin", "mean"),
        pair_margin_std=("pair_margin", "std"),
    )
    summary["inversion_rate"] = summary["inversion_count"] / summary["pair_count"].clip(lower=1)
    return pairwise, summary


def alignment_health_gate(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    prototype: pd.DataFrame,
    shared: pd.DataFrame,
    specific: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    validation = predictions[predictions["split"] == "val"]
    for (model_id, seed), group in validation.groupby(["model_id", "seed"]):
        metric_row = metrics[
            (metrics["model_id"] == model_id) & (metrics["seed"] == seed) & (metrics["split"] == "val")
        ]
        proto_row = prototype[
            (prototype["model_id"] == model_id) & (prototype["seed"] == seed) & (prototype["split"] == "val")
        ]
        shared_rows = shared[
            (shared["model_id"] == model_id) & (shared["seed"] == seed) & (shared["split"] == "val")
        ]
        specific_row = specific[
            (specific["model_id"] == model_id) & (specific["seed"] == seed) & (specific["split"] == "val")
        ]
        probabilities = pd.to_numeric(group["pred_prob"], errors="coerce")
        residual_ratio = pd.to_numeric(group["specific_residual_shared_ratio"], errors="coerce")
        checks = {
            "finite_predictions": bool(np.isfinite(probabilities).all()),
            "nonconstant_predictions": bool(probabilities.std(ddof=1) > 1e-6),
            "prototype_not_collapsed": bool(not proto_row.empty and int(proto_row["prototype_collapse_flag"].max()) == 0),
            "shared_not_collapsed": bool(not shared_rows.empty and int(shared_rows["shared_sample_collapse_flag"].max()) == 0),
            "specific_not_collapsed": bool(not specific_row.empty and int(specific_row["specific_collapse_flag"].max()) == 0),
            "specific_not_duplicate": bool(not specific_row.empty and int(specific_row["specific_duplicates_shared_flag"].max()) == 0),
            "specific_not_dominant": bool(not residual_ratio.empty and float(residual_ratio.max()) < 1.0),
            "attention_not_collapsed": bool(not metric_row.empty and int(metric_row.iloc[0].get("global_attention_collapse_flag", 1)) == 0),
            "gates_not_saturated": bool(not specific_row.empty and int(specific_row["global_gate_saturation_flag"].max()) == 0),
        }
        row: Dict[str, Any] = {
            "model_id": model_id,
            "seed": int(seed),
            **{name: int(passed) for name, passed in checks.items()},
            "prediction_std": float(probabilities.std(ddof=1)),
            "max_specific_residual_shared_ratio": float(residual_ratio.max()),
            "health_pass": int(all(checks.values())),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_health_report(title: str, frame: pd.DataFrame, path: Path, notes: Sequence[str]) -> None:
    lines = [f"# {title}", "", *notes, "", frame_to_markdown(frame)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_saved_audit(run_specs: Sequence[str], output_dir: Path) -> bool:
    metrics, predictions = read_run_frames(run_specs)
    if predictions.empty:
        raise FileNotFoundError("No C16 train/validation prediction files were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype = prototype_health(metrics, predictions)
    shared = shared_health(metrics, predictions)
    specific = specific_health(metrics, predictions)
    pairwise, inversion = pairwise_rows(predictions)
    health_gate = alignment_health_gate(metrics, predictions, prototype, shared, specific)

    prototype.to_csv(output_dir / "c16_prototype_health_by_seed.csv", index=False)
    shared.to_csv(output_dir / "c16_shared_alignment_by_seed.csv", index=False)
    specific.to_csv(output_dir / "c16_specific_health_by_seed.csv", index=False)
    pairwise.to_csv(output_dir / "c16_pairwise_ranking_val.csv", index=False)
    inversion.to_csv(output_dir / "c16_pairwise_inversion_summary.csv", index=False)
    health_gate.to_csv(output_dir / "c16_alignment_health_gate.csv", index=False)
    validation = predictions[predictions["split"] == "val"].copy()
    keep = [column for column in PATIENT_DIAGNOSTIC_COLUMNS if column in validation.columns]
    validation[keep].to_csv(output_dir / "c16_patient_diagnostics_val.csv", index=False)

    write_health_report(
        "C16 Prototype Health",
        prototype,
        output_dir / "c16_prototype_health_report.md",
        ["Prototype metrics are audit-only and never select checkpoints."],
    )
    write_health_report(
        "C16 Shared Alignment Health",
        shared,
        output_dir / "c16_shared_alignment_report.md",
        ["Shared cosine is stratified by split, label, and prediction correctness."],
    )
    write_health_report(
        "C16 Specific Branch Health",
        specific,
        output_dir / "c16_specific_health_report.md",
        ["Specific features must remain non-collapsed, non-duplicative, and norm-controlled."],
    )
    health_pass = bool(not health_gate.empty and (health_gate["health_pass"] == 1).all())
    (output_dir / "c16_alignment_health_gate.json").write_text(
        json.dumps({"health_pass": health_pass, "rows": health_gate.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"prediction_rows": len(predictions), "pairwise_rows": len(pairwise), "health_pass": health_pass, "output_dir": str(output_dir)}))
    return health_pass


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    passed = True
    if args.synthetic_smoke:
        passed = run_synthetic_smoke(
            Path(args.c13_config),
            Path(args.c16_config),
            Path(args.pilot_config),
            Path(args.stress_config),
            output_dir,
        )
    health_pass = True
    if args.run_dirs:
        health_pass = run_saved_audit(args.run_dirs, output_dir)
    if not args.synthetic_smoke and not args.run_dirs:
        raise SystemExit("Specify --synthetic-smoke and/or --run-dirs")
    if not passed or (args.require_health_pass and not health_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
