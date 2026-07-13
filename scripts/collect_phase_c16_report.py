from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_prediction_shortcut_residual import (  # noqa: E402
    DEFAULT_SHORTCUT_FIELDS,
    audit_frame,
    merge_shortcuts,
    read_manifest_frame,
)


C13_AUC_MEAN = 0.8664554096876415
C13_AUPRC_MEAN = 0.8570449989296317
FORMAL_SEEDS = (0, 42, 3407)


def collect_environment() -> Dict[str, Any]:
    def git_value(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"

    environment: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_commit": git_value("rev-parse", "HEAD"),
        "repo_commit_short": git_value("rev-parse", "--short", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "tracked_worktree_status": git_value("status", "--porcelain", "--untracked-files=no"),
    }
    try:
        import torch

        environment.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    except Exception as exc:
        environment["torch_environment_error"] = str(exc)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect C16 DSSA pilot or formal gate report.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--c13-run-dir", required=True)
    parser.add_argument("--c16-runs", nargs="+", required=True, help="name=run_dir entries for the selected C16 route")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c16")
    parser.add_argument("--synthetic-smoke", default="analysis_reports/phase_c16/c16_synthetic_smoke.json")
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected name=path, got: {value}")
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path.strip())


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


def load_run_data(run_specs: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    metric_frames: List[pd.DataFrame] = []
    epoch_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    configs: Dict[str, Dict[str, Any]] = {}
    for model_id, run_dir in [parse_named_path(value) for value in run_specs]:
        metrics_path = run_dir / "reports" / "metrics_by_seed.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "model_id", model_id)
        metric_frames.append(metrics)
        epochs_path = run_dir / "reports" / "metrics_by_epoch.csv"
        if epochs_path.is_file():
            epochs = pd.read_csv(epochs_path)
            epochs.insert(0, "model_id", model_id)
            epoch_frames.append(epochs)
        config_path = run_dir / "reports" / "run_config.json"
        configs[model_id] = json.loads(config_path.read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
                frame = pd.read_csv(path)
                frame["patient_id"] = frame["patient_id"].astype(str)
                frame["split"] = split
                frame["model_id"] = model_id
                prediction_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    epochs = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    duplicated = metrics[metrics["split"] == "val"].duplicated(["seed"], keep=False)
    if bool(duplicated.any()):
        duplicates = metrics.loc[metrics["split"] == "val"].loc[duplicated, ["model_id", "seed"]].to_dict("records")
        raise ValueError(f"selected C16 route contains duplicate validation seeds: {duplicates}")
    return metrics, epochs, predictions, configs


def load_c13(c13_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(c13_run_dir / "reports" / "metrics_by_seed.csv")
    predictions = []
    for split in ("val", "test"):
        for path in sorted((c13_run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
            frame = pd.read_csv(path)
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["split"] = split
            predictions.append(frame)
    return metrics, pd.concat(predictions, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = (
        "AUC",
        "AUPRC",
        "ACC",
        "F1",
        "Sensitivity",
        "Specificity",
        "Balanced_ACC",
        "pos_neg_gap",
        "pairwise_inversion_count",
        "prototype_cosine",
        "prototype_assignment_accuracy",
        "specific_residual_shared_ratio",
    )
    rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        group = metrics[metrics["split"] == split]
        if group.empty:
            continue
        row: Dict[str, Any] = {"split": split, "seed_count": int(group["seed"].nunique())}
        for key in keys:
            if key not in group.columns:
                continue
            values = pd.to_numeric(group[key], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                row[f"{key}_mean"] = float(values.mean())
                row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, group in predictions[predictions["split"] == "val"].groupby("seed"):
        labels = pd.to_numeric(group["label"], errors="coerce").astype(int)
        positives = pd.to_numeric(group.loc[labels == 1, "logit"], errors="coerce").to_numpy(dtype=float)
        negatives = pd.to_numeric(group.loc[labels == 0, "logit"], errors="coerce").to_numpy(dtype=float)
        margins = positives[:, None] - negatives[None, :]
        rows.append(
            {
                "seed": int(seed),
                "pair_count": int(margins.size),
                "inversion_count": int((margins <= 0.0).sum()),
                "inversion_rate": float((margins <= 0.0).mean()),
                "pair_margin_mean": float(margins.mean()),
            }
        )
    return pd.DataFrame(rows)


def positive_preservation(c16_predictions: pd.DataFrame, c13_predictions: pd.DataFrame) -> pd.DataFrame:
    c16 = c16_predictions[c16_predictions["split"] == "val"].copy()
    c13 = c13_predictions[c13_predictions["split"] == "val"].copy()
    keep = ["patient_id", "seed", "label", "pred_prob", "logit"]
    merged = c16[keep].merge(c13[keep], on=["patient_id", "seed", "label"], suffixes=("_c16", "_c13"), how="inner")
    merged["positive_probability_delta"] = merged["pred_prob_c16"] - merged["pred_prob_c13"]
    merged["absolute_error_delta"] = (
        (merged["pred_prob_c16"] - merged["label"]).abs() - (merged["pred_prob_c13"] - merged["label"]).abs()
    )
    merged["c13_prediction"] = (merged["pred_prob_c13"] >= 0.5).astype(int)
    merged["c16_prediction"] = (merged["pred_prob_c16"] >= 0.5).astype(int)
    merged["c13_error_type"] = np.select(
        [(merged["label"] == 1) & (merged["c13_prediction"] == 0), (merged["label"] == 0) & (merged["c13_prediction"] == 1)],
        ["FN", "FP"],
        default="correct",
    )
    merged["c16_error_type"] = np.select(
        [(merged["label"] == 1) & (merged["c16_prediction"] == 0), (merged["label"] == 0) & (merged["c16_prediction"] == 1)],
        ["FN", "FP"],
        default="correct",
    )
    return merged


def positive_preservation_summary(preservation: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, group in preservation.groupby("seed"):
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        rows.append(
            {
                "scope": "seed",
                "seed": int(seed),
                "positive_probability_delta_mean": float(positives["positive_probability_delta"].mean()),
                "negative_probability_delta_mean": float(negatives["positive_probability_delta"].mean()),
                "positive_absolute_error_delta_mean": float(positives["absolute_error_delta"].mean()),
                "negative_absolute_error_delta_mean": float(negatives["absolute_error_delta"].mean()),
                "c13_fn": int((positives["c13_error_type"] == "FN").sum()),
                "c16_fn": int((positives["c16_error_type"] == "FN").sum()),
                "c13_tp": int((positives["c13_error_type"] == "correct").sum()),
                "c16_tp": int((positives["c16_error_type"] == "correct").sum()),
                "c13_fp": int((negatives["c13_error_type"] == "FP").sum()),
                "c16_fp": int((negatives["c16_error_type"] == "FP").sum()),
            }
        )
    seed_count = int(preservation["seed"].nunique())
    if seed_count > 1:
        positives = preservation[preservation["label"] == 1]
        negatives = preservation[preservation["label"] == 0]
        positive_patient = positives.groupby("patient_id").agg(
            seed_count=("seed", "nunique"),
            c13_fn_count=("c13_error_type", lambda values: int((values == "FN").sum())),
            c16_fn_count=("c16_error_type", lambda values: int((values == "FN").sum())),
            c13_tp_count=("c13_error_type", lambda values: int((values == "correct").sum())),
            c16_tp_count=("c16_error_type", lambda values: int((values == "correct").sum())),
        )
        negative_patient = negatives.groupby("patient_id").agg(
            seed_count=("seed", "nunique"),
            c13_fp_count=("c13_error_type", lambda values: int((values == "FP").sum())),
            c16_fp_count=("c16_error_type", lambda values: int((values == "FP").sum())),
        )
        rows.append(
            {
                "scope": "all_seed_patient",
                "seed": "all",
                "formal_seed_count": seed_count,
                "c13_all_seed_fn_patients": int(((positive_patient["seed_count"] == seed_count) & (positive_patient["c13_fn_count"] == seed_count)).sum()),
                "c16_all_seed_fn_patients": int(((positive_patient["seed_count"] == seed_count) & (positive_patient["c16_fn_count"] == seed_count)).sum()),
                "c13_all_seed_tp_patients": int(((positive_patient["seed_count"] == seed_count) & (positive_patient["c13_tp_count"] == seed_count)).sum()),
                "c16_all_seed_tp_patients": int(((positive_patient["seed_count"] == seed_count) & (positive_patient["c16_tp_count"] == seed_count)).sum()),
                "c13_all_seed_fp_patients": int(((negative_patient["seed_count"] == seed_count) & (negative_patient["c13_fp_count"] == seed_count)).sum()),
                "c16_all_seed_fp_patients": int(((negative_patient["seed_count"] == seed_count) & (negative_patient["c16_fp_count"] == seed_count)).sum()),
            }
        )
    return pd.DataFrame(rows)


def model_comparison(
    c16_metrics: pd.DataFrame,
    c13_metrics: pd.DataFrame,
    c16_pairwise: pd.DataFrame,
    c13_pairwise: pd.DataFrame,
) -> pd.DataFrame:
    metric_keys = ("AUC", "AUPRC", "Sensitivity", "Specificity", "Balanced_ACC", "pos_neg_gap")
    c16 = c16_metrics[c16_metrics["split"] == "val"][["seed", *metric_keys]].copy()
    c13 = c13_metrics[c13_metrics["split"] == "val"][["seed", *metric_keys]].copy()
    comparison = c16.merge(c13, on="seed", suffixes=("_c16", "_c13"), how="inner")
    for key in metric_keys:
        comparison[f"{key}_delta"] = pd.to_numeric(comparison[f"{key}_c16"]) - pd.to_numeric(comparison[f"{key}_c13"])
    comparison = comparison.merge(
        c16_pairwise[["seed", "inversion_count"]].rename(columns={"inversion_count": "inversion_count_c16"}),
        on="seed",
        how="left",
    ).merge(
        c13_pairwise[["seed", "inversion_count"]].rename(columns={"inversion_count": "inversion_count_c13"}),
        on="seed",
        how="left",
    )
    comparison["inversion_count_delta"] = comparison["inversion_count_c16"] - comparison["inversion_count_c13"]
    return comparison


def shortcut_audit(manifest_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    manifest = read_manifest_frame(manifest_path, DEFAULT_SHORTCUT_FIELDS)
    rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        split_frame = predictions[predictions["split"] == split].copy()
        if split_frame.empty:
            continue
        split_frame["pred_prob_for_audit"] = pd.to_numeric(split_frame["pred_prob"], errors="coerce")
        merged = merge_shortcuts(split_frame, manifest, DEFAULT_SHORTCUT_FIELDS)
        for seed, seed_frame in merged.groupby("seed"):
            rows.append(audit_frame("C16_DSSA", split, str(int(seed)), seed_frame, DEFAULT_SHORTCUT_FIELDS))
        rows.append(audit_frame("C16_DSSA", split, "pooled", merged, DEFAULT_SHORTCUT_FIELDS))
    return pd.DataFrame(rows)


def load_health(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prototype = pd.read_csv(output_dir / "c16_prototype_health_by_seed.csv")
    shared = pd.read_csv(output_dir / "c16_shared_alignment_by_seed.csv")
    specific = pd.read_csv(output_dir / "c16_specific_health_by_seed.csv")
    return prototype, shared, specific


def health_gate(
    seed: int,
    metrics: pd.DataFrame,
    epochs: pd.DataFrame,
    prototype: pd.DataFrame,
    shared: pd.DataFrame,
    specific: pd.DataFrame,
) -> tuple[bool, Dict[str, Any]]:
    val_metrics = metrics[(metrics["seed"] == seed) & (metrics["split"] == "val")]
    proto = prototype[(prototype["seed"] == seed) & (prototype["split"] == "val")]
    shared_seed = shared[(shared["seed"] == seed) & (shared["split"] == "val")]
    specific_seed = specific[(specific["seed"] == seed) & (specific["split"] == "val")]
    seed_epochs = epochs[epochs["seed"] == seed].sort_values("epoch")
    selected_epochs = seed_epochs[pd.to_numeric(seed_epochs.get("selected_by_val_auc", 0), errors="coerce") == 1]
    first_shared_loss = float(seed_epochs.iloc[0].get("train_shared_consistency_loss", float("nan"))) if not seed_epochs.empty else float("nan")
    selected_shared_loss = float(selected_epochs.iloc[0].get("train_shared_consistency_loss", float("nan"))) if not selected_epochs.empty else float("nan")
    margin_label0 = float(proto.iloc[0].get("disease_margin_label0_mean", float("nan"))) if not proto.empty else float("nan")
    margin_label1 = float(proto.iloc[0].get("disease_margin_label1_mean", float("nan"))) if not proto.empty else float("nan")
    prototype_accuracy = float(proto.iloc[0].get("prototype_assignment_accuracy", float("nan"))) if not proto.empty else float("nan")
    evidence = {
        "prototype_collapse": int(proto["prototype_collapse_flag"].max()) if not proto.empty else 1,
        "shared_sample_collapse": int(shared_seed["shared_sample_collapse_flag"].max()) if not shared_seed.empty else 1,
        "specific_collapse": int(specific_seed["specific_collapse_flag"].max()) if not specific_seed.empty else 1,
        "specific_duplicates_shared": int(specific_seed["specific_duplicates_shared_flag"].max()) if not specific_seed.empty else 1,
        "specific_dominates_shared": int(specific_seed["specific_dominates_shared_flag"].max()) if not specific_seed.empty else 1,
        "global_gate_saturation": int(specific_seed["global_gate_saturation_flag"].max()) if not specific_seed.empty else 1,
        "global_attention_collapse": int(val_metrics.iloc[0].get("global_attention_collapse_flag", 1)) if not val_metrics.empty else 1,
        "disease_margin_direction_fail": int(
            not (math.isfinite(margin_label0) and math.isfinite(margin_label1) and margin_label1 > margin_label0)
        ),
        "prototype_assignment_fail": int(not (math.isfinite(prototype_accuracy) and prototype_accuracy > 0.50)),
        "shared_alignment_improvement_fail": int(
            not (
                math.isfinite(first_shared_loss)
                and math.isfinite(selected_shared_loss)
                and selected_shared_loss < first_shared_loss
            )
        ),
    }
    evidence.update(
        {
            "disease_margin_label0_mean": margin_label0,
            "disease_margin_label1_mean": margin_label1,
            "prototype_assignment_accuracy": prototype_accuracy,
            "first_train_shared_consistency_loss": first_shared_loss,
            "selected_train_shared_consistency_loss": selected_shared_loss,
        }
    )
    flag_keys = [key for key in evidence if key.endswith("collapse") or key.endswith("shared") or key.endswith("dominant") or key.endswith("saturation") or key.endswith("fail")]
    return all(int(evidence[key]) == 0 for key in flag_keys), evidence


def shortcut_gate(seed: int, shortcut: pd.DataFrame) -> tuple[bool, Dict[str, Any]]:
    row = shortcut[(shortcut["split"] == "val") & (shortcut["seed"].astype(str) == str(seed))]
    if row.empty:
        return False, {"reason": "missing shortcut row"}
    values = row.iloc[0]
    evidence = {
        "max_abs_spearman": float(values["max_abs_spearman"]),
        "linear_r2_prob_from_shortcuts": float(values["linear_r2_prob_from_shortcuts"]),
        "shortcut_only_label_auc_audit_only": float(values["shortcut_only_label_auc_audit_only"]),
    }
    passed = (
        evidence["max_abs_spearman"] < 0.20
        and evidence["linear_r2_prob_from_shortcuts"] < 0.10
        and evidence["shortcut_only_label_auc_audit_only"] < 0.65
    )
    return passed, evidence


def pilot_gate(
    c16_metrics: pd.DataFrame,
    c16_epochs: pd.DataFrame,
    c13_metrics: pd.DataFrame,
    c16_pairwise: pd.DataFrame,
    c13_pairwise: pd.DataFrame,
    prototype: pd.DataFrame,
    shared: pd.DataFrame,
    specific: pd.DataFrame,
    shortcut: pd.DataFrame,
) -> pd.DataFrame:
    c16 = c16_metrics[(c16_metrics["seed"] == 0) & (c16_metrics["split"] == "val")].iloc[0]
    c13 = c13_metrics[(c13_metrics["seed"] == 0) & (c13_metrics["split"] == "val")].iloc[0]
    c16_inversions = int(c16_pairwise[c16_pairwise["seed"] == 0].iloc[0]["inversion_count"])
    c13_inversions = int(c13_pairwise[c13_pairwise["seed"] == 0].iloc[0]["inversion_count"])
    health_pass, health_evidence = health_gate(0, c16_metrics, c16_epochs, prototype, shared, specific)
    shortcut_pass, shortcut_evidence = shortcut_gate(0, shortcut)
    checks = [
        ("val_auc_not_below_c13", float(c16["AUC"]) >= float(c13["AUC"]), f"{c16['AUC']} vs {c13['AUC']}"),
        ("preferred_auc_gain", float(c16["AUC"]) - float(c13["AUC"]) >= 0.005, float(c16["AUC"]) - float(c13["AUC"])),
        ("auprc_preserved", float(c16["AUPRC"]) >= float(c13["AUPRC"]) - 0.005, f"{c16['AUPRC']} vs {c13['AUPRC']}"),
        ("positive_negative_gap_preserved", float(c16["pos_neg_gap"]) >= float(c13["pos_neg_gap"]) - 0.02, f"{c16['pos_neg_gap']} vs {c13['pos_neg_gap']}"),
        ("sensitivity_floor", float(c16["Sensitivity"]) >= 0.55, c16["Sensitivity"]),
        ("specificity_floor", float(c16["Specificity"]) >= 0.75, c16["Specificity"]),
        ("inversions_not_increased", c16_inversions <= c13_inversions, f"{c16_inversions} vs {c13_inversions}"),
        ("alignment_health", health_pass, json.dumps(health_evidence, sort_keys=True)),
        ("shortcut_safety", shortcut_pass, json.dumps(shortcut_evidence, sort_keys=True)),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence} for name, passed, evidence in checks]
    )


def formal_gate(
    metrics: pd.DataFrame,
    epochs: pd.DataFrame,
    c13_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    c13_pairwise: pd.DataFrame,
    prototype: pd.DataFrame,
    shared: pd.DataFrame,
    specific: pd.DataFrame,
    shortcut: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    validation = metrics[metrics["split"] == "val"].copy().sort_values("seed")
    c13_validation = c13_metrics[c13_metrics["split"] == "val"].copy().sort_values("seed")
    auc = pd.to_numeric(validation["AUC"]).to_numpy(dtype=float)
    auprc = pd.to_numeric(validation["AUPRC"]).to_numpy(dtype=float)
    gap = pd.to_numeric(validation["pos_neg_gap"]).to_numpy(dtype=float)
    c13_gap = pd.to_numeric(c13_validation["pos_neg_gap"]).to_numpy(dtype=float)
    inversion_compare = pairwise[["seed", "inversion_count"]].merge(
        c13_pairwise[["seed", "inversion_count"]], on="seed", suffixes=("_c16", "_c13")
    )
    inversion_improved = int((inversion_compare["inversion_count_c16"] < inversion_compare["inversion_count_c13"]).sum())
    health_rows = [health_gate(seed, metrics, epochs, prototype, shared, specific)[0] for seed in FORMAL_SEEDS]
    shortcut_rows = [shortcut_gate(seed, shortcut)[0] for seed in FORMAL_SEEDS]
    checks = [
        ("formal_seeds_complete", set(validation["seed"].astype(int)) == set(FORMAL_SEEDS), validation["seed"].tolist()),
        ("mean_auc_above_c13", float(auc.mean()) > C13_AUC_MEAN, float(auc.mean())),
        ("meaningful_auc_gain", float(auc.mean()) - C13_AUC_MEAN >= 0.01, float(auc.mean()) - C13_AUC_MEAN),
        ("minimum_seed_auc", float(auc.min()) >= 0.85, float(auc.min())),
        ("auc_std", float(auc.std(ddof=1)) <= 0.02, float(auc.std(ddof=1))),
        ("mean_auprc", float(auprc.mean()) >= C13_AUPRC_MEAN, float(auprc.mean())),
        ("inversions_improved_two_seeds", inversion_improved >= 2, inversion_compare.to_dict("records")),
        ("positive_negative_gap", float(gap.mean()) >= float(c13_gap.mean()) - 0.02, f"{gap.mean()} vs {c13_gap.mean()}"),
        ("sensitivity_no_collapse", float(pd.to_numeric(validation["Sensitivity"]).min()) >= 0.50, pd.to_numeric(validation["Sensitivity"]).tolist()),
        ("specificity_no_collapse", float(pd.to_numeric(validation["Specificity"]).min()) >= 0.70, pd.to_numeric(validation["Specificity"]).tolist()),
        ("alignment_health_all_seeds", all(health_rows), health_rows),
        ("shortcut_safety_all_seeds", all(shortcut_rows), shortcut_rows),
    ]
    gate = pd.DataFrame(
        [{"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence} for name, passed, evidence in checks]
    )
    health_failed = not all(health_rows)
    if bool((gate["status"] == "PASS").all()):
        decision = "PROMOTE_C16_DSSA"
    elif health_failed:
        decision = "C16_DSSA_ALIGNMENT_HEALTH_FAIL"
    elif float(auc.mean()) > C13_AUC_MEAN:
        decision = "C16_DSSA_PARTIAL_IMPROVEMENT_NOT_STABLE"
    else:
        decision = "C16_DSSA_FORMAL_FAIL_KEEP_C13"
    return gate, decision


def write_reports(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    preservation_summary: pd.DataFrame,
    prototype: pd.DataFrame,
    shared: pd.DataFrame,
    specific: pd.DataFrame,
    shortcut: pd.DataFrame,
    pilot: pd.DataFrame,
    formal: pd.DataFrame,
    state: str,
    decision: str | None,
    environment: Dict[str, Any],
    synthetic: Dict[str, Any],
) -> None:
    validation = metrics[metrics["split"] == "val"].sort_values("seed")
    testing = metrics[metrics["split"] == "test"].sort_values("seed")
    validation_summary = summary[summary["split"] == "val"]
    mean_validation_auc = float(validation_summary.iloc[0]["AUC_mean"]) if not validation_summary.empty else float("nan")
    formal_complete = set(validation["seed"].astype(int)) == set(FORMAL_SEEDS)
    current_strict_best = "C16 DSSA" if decision == "PROMOTE_C16_DSSA" else "C13 temporal-focus DMEA-HT"
    lines = [
        "# C16 DSSA Model Comparison",
        "",
        "C13 remains the frozen fallback unless the full three-seed promotion gate passes.",
        "",
        "## Seed-Wise C13 Comparison",
        "",
        frame_to_markdown(comparison),
        "",
        "## C16 Aggregate",
        "",
        frame_to_markdown(summary),
        "",
        "## Pilot Gate",
        "",
        frame_to_markdown(pilot),
    ]
    if not formal.empty:
        lines.extend(["", "## Formal Gate", "", frame_to_markdown(formal)])
    (output_dir / "c16_model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stability = validation[[column for column in ("seed", "AUC", "AUPRC", "Sensitivity", "Specificity", "prototype_cosine", "mean_shared_attention_img", "mean_shared_attention_txt", "mean_shared_attention_bio", "mean_specific_gate_img", "mean_specific_gate_txt", "mean_specific_gate_bio") if column in validation.columns]]
    (output_dir / "c16_seed_stability_report.md").write_text(
        "# C16 Seed Stability\n\n" + frame_to_markdown(stability) + "\n\n" + frame_to_markdown(summary) + "\n",
        encoding="utf-8",
    )

    final_lines = [
        "# Phase C16 Final Report",
        "",
        f"Status: `{state}`.",
        f"Final decision label: `{decision}`." if decision else "Final decision label: not assigned while the serial gate is incomplete.",
        "",
        f"- Current strict best: `{current_strict_best}`.",
        f"- Stress seeds completed: `{formal_complete}`.",
        "- Checkpoints are selected by validation AUC only; test is reporting-only.",
        "- Labels, patient split, C13 temporal-focus manifest, and report construction remain unchanged.",
        "- Shortcut fields are audit-only and are not model or loss inputs.",
        f"- Mean validation AUC: `{mean_validation_auc}`.",
        f"- Mean validation AUC reached 0.90: `{bool(math.isfinite(mean_validation_auc) and mean_validation_auc >= 0.90)}`.",
        "",
        "## Commit, Worktree, And Server Environment",
        "",
        frame_to_markdown(pd.DataFrame([{"field": key, "value": value} for key, value in environment.items()])),
        "",
        "## Legacy Compatibility And Synthetic Gate",
        "",
        f"Synthetic/config contract status: `{synthetic.get('status', 'missing')}`.",
        "",
        frame_to_markdown(pd.DataFrame(synthetic.get("checks", []))),
        "",
        "## Seed-0 Pilot Gate",
        "",
        frame_to_markdown(pilot),
        "",
        "## Seed-Wise Validation Metrics",
        "",
        frame_to_markdown(validation),
        "",
        "## Mean And Standard Deviation",
        "",
        frame_to_markdown(summary),
        "",
        "## Test Metrics, Reporting Only",
        "",
        "These metrics were computed only after validation-AUC checkpoint selection and do not affect any gate.",
        "",
        frame_to_markdown(testing),
        "",
        "## C13 Performance And Inversion Comparison",
        "",
        frame_to_markdown(comparison),
        "",
        "## Prototype Health",
        "",
        frame_to_markdown(prototype[prototype["split"] == "val"]),
        "",
        "## Shared Alignment Health",
        "",
        frame_to_markdown(shared[shared["split"] == "val"]),
        "",
        "## Specific Branch Health",
        "",
        frame_to_markdown(specific[specific["split"] == "val"]),
        "",
        "## Positive Preservation And FN/TP Changes",
        "",
        frame_to_markdown(preservation_summary),
        "",
        "## Shortcut Safety",
        "",
        frame_to_markdown(shortcut[shortcut["split"] == "val"]),
        "",
        "## Active Gate",
        "",
        frame_to_markdown(formal if not formal.empty else pilot),
    ]
    (output_dir / "phase_c16_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = Path(args.synthetic_smoke)
    synthetic = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
    if synthetic.get("status") != "PASS":
        raise RuntimeError("C16 synthetic smoke must pass before report collection")

    metrics, epochs, predictions, configs = load_run_data(args.c16_runs)
    c13_metrics, c13_predictions = load_c13(Path(args.c13_run_dir))
    prototype, shared, specific = load_health(output_dir)
    pairwise = build_pairwise_summary(predictions)
    c13_pairwise = build_pairwise_summary(c13_predictions)
    preservation = positive_preservation(predictions, c13_predictions)
    preservation_summary = positive_preservation_summary(preservation)
    comparison = model_comparison(metrics, c13_metrics, pairwise, c13_pairwise)
    shortcut = shortcut_audit(Path(args.manifest), predictions)
    summary = summarize(metrics)
    environment = collect_environment()

    metrics.to_csv(output_dir / "c16_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "c16_metrics_summary.csv", index=False)
    epochs.to_csv(output_dir / "c16_metrics_by_epoch.csv", index=False)
    preservation.to_csv(output_dir / "c16_positive_preservation_audit.csv", index=False)
    preservation_summary.to_csv(output_dir / "c16_positive_preservation_summary.csv", index=False)
    comparison.to_csv(output_dir / "c16_model_comparison_by_seed.csv", index=False)
    shortcut.to_csv(output_dir / "c16_shortcut_residual_audit.csv", index=False)
    (output_dir / "c16_server_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    pilot = pilot_gate(metrics, epochs, c13_metrics, pairwise, c13_pairwise, prototype, shared, specific, shortcut)
    pilot.to_csv(output_dir / "c16_pilot_gate.csv", index=False)
    pilot_pass = bool((pilot["status"] == "PASS").all())
    validation_seeds = set(metrics[metrics["split"] == "val"]["seed"].astype(int))
    formal = pd.DataFrame()
    decision: str | None = None
    if validation_seeds == set(FORMAL_SEEDS):
        formal, decision = formal_gate(metrics, epochs, c13_metrics, pairwise, c13_pairwise, prototype, shared, specific, shortcut)
        state = "FORMAL_COMPLETE"
    elif validation_seeds == {0} and pilot_pass:
        state = "PILOT_PASS_STRESS_AUTHORIZED"
    elif validation_seeds == {0}:
        seed0_model_id = str(metrics[(metrics["seed"] == 0) & (metrics["split"] == "val")].iloc[0]["model_id"])
        rank_weight = float(configs[seed0_model_id].get("loss", {}).get("lambda_rank", 0.0))
        if rank_weight > 0.0:
            state = "PILOT_FAIL_ONLY_RANK0_FALLBACK_AUTHORIZED"
        else:
            state = "PILOT_AND_FALLBACK_FAIL"
            decision = "C16_DSSA_PILOT_FAIL_KEEP_C13"
    else:
        state = "C16_TRAINING_INVALID"
        decision = "C16_TRAINING_INVALID"

    if not formal.empty:
        formal.to_csv(output_dir / "c16_formal_gate.csv", index=False)
    write_reports(
        output_dir,
        metrics,
        summary,
        comparison,
        preservation_summary,
        prototype,
        shared,
        specific,
        shortcut,
        pilot,
        formal,
        state,
        decision,
        environment,
        synthetic,
    )
    result = {
        "state": state,
        "decision": decision,
        "validation_seeds": sorted(validation_seeds),
        "pilot_pass": pilot_pass,
        "mean_validation_auc": float(summary[summary["split"] == "val"].iloc[0]["AUC_mean"]),
        "auc_0_90_reached": bool(float(summary[summary["split"] == "val"].iloc[0]["AUC_mean"]) >= 0.90),
    }
    (output_dir / "c16_gate_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
