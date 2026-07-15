#!/usr/bin/env python3
"""Audit common Validation failures after three healthy post-goal hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import read_manifest  # noqa: E402
from dmea_ht.visit_data import sha256_file  # noqa: E402


SEEDS = (0, 42, 3407)
EXPECTED_VAL_COUNT = 94
EXPECTED_VAL_POSITIVES = 47
EXPECTED_MANIFEST_SHA256 = "cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b"
MODEL_ORDER = ("C17", "C27", "C38", "C39", "C40")
EVIDENCE_FIELDS = (
    "selected_n_visits",
    "n_visits",
    "raw_n_visits",
    "n_images",
    "used_images",
    "raw_n_images",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "image_padding_count",
    "padding_count",
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
    "txt_morphology_label",
    "txt_negative_label",
    "txt_uncertain_label",
    "txt_diag_hint_label",
    "image_morphology_weak_label",
    "bio_immune_abnormal_label",
    "bio_function_abnormal_label",
    "bio_missing_label",
    "discordance_state_label",
)
STRATA_FIELDS = (
    "visit_count_bin",
    "report_coverage_bin",
    "dated_bio_bin",
    "bio_missing_bin",
    "evidence_conflict",
    "weak_evidence_group",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def as_int(value: Any, default: int = 0) -> int:
    number = as_float(value, float(default))
    return int(number) if np.isfinite(number) else int(default)


def safe_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    values = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(np.unique(values)) < 2:
        return float("nan")
    return float(roc_auc_score(values, scores))


def read_prediction(path: Path, model: str, seed: int) -> pd.DataFrame:
    if not path.name.startswith(f"val_predictions_seed_{seed}."):
        raise RuntimeError(f"{model} seed {seed} is not a Validation prediction filename: {path}")
    if any(part.lower() == "test" for part in path.parts):
        raise RuntimeError(f"Test path is forbidden in the data-limit audit: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing {model} seed {seed} Validation prediction: {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{model} seed {seed} prediction is missing columns: {missing}")
    probability_column = next(
        (name for name in ("final_prob", "prob", "prediction", "y_prob") if name in frame.columns),
        None,
    )
    if probability_column is None:
        raise RuntimeError(f"{model} seed {seed} has no probability column: {list(frame.columns)}")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["probability"] = pd.to_numeric(frame[probability_column], errors="raise").astype(float)
    if frame["patient_id"].duplicated().any() or not np.isfinite(frame["probability"]).all():
        raise RuntimeError(f"{model} seed {seed} contains duplicate patients or non-finite probabilities")
    if "split" in frame and set(frame["split"].astype(str).str.lower()) != {"val"}:
        raise RuntimeError(f"{model} seed {seed} prediction is not Validation-only")
    return frame[["patient_id", "label", "probability"]].sort_values("patient_id").reset_index(drop=True)


def row_float(row: Mapping[str, Any], field: str, default: float = np.nan) -> float:
    value = row.get(field, default)
    if isinstance(value, (list, tuple, dict)):
        return default
    return as_float(value, default)


def manifest_features(row: Mapping[str, Any]) -> Dict[str, Any]:
    visits = list(row.get("visits") or [])
    report_present = [bool(str(visit.get("report_text", "") or "").strip()) for visit in visits]
    dated_bio = [visit.get("dated_bio_row_id") is not None for visit in visits]
    image_count = sum(len(visit.get("image_paths") or []) for visit in visits)
    feature: Dict[str, Any] = {
        "patient_id": str(row["patient_id"]),
        "split": str(row.get("split", "")).lower(),
        "label": as_int(row.get("label")),
        "selected_n_visits": as_int(row.get("selected_n_visits"), len(visits)),
        "n_visits": as_int(row.get("n_visits"), len(visits)),
        "raw_n_visits": as_int(row.get("raw_n_visits"), len(visits)),
        "n_images": as_int(row.get("n_images"), image_count),
        "used_images": as_int(row.get("used_images"), image_count),
        "raw_n_images": as_int(row.get("raw_n_images"), image_count),
        "has_bio": as_int(row.get("has_bio"), int(any(dated_bio))),
        "bio_missing_count": as_int(row.get("bio_missing_count"), 0),
        "report_length": as_int(row.get("report_length"), sum(len(str(visit.get("report_text", "") or "")) for visit in visits)),
        "image_padding_count": as_int(row.get("image_padding_count"), 0),
        "padding_count": as_int(row.get("padding_count"), 0),
        "reconstructable_visit_count": as_int(row.get("reconstructable_visit_count"), sum(report_present)),
        "visit_report_coverage": row_float(row, "visit_report_coverage", sum(report_present) / max(len(visits), 1)),
        "dated_bio_visit_count": as_int(row.get("dated_bio_visit_count"), sum(dated_bio)),
    }
    for field in EVIDENCE_FIELDS:
        if field not in feature:
            feature[field] = as_int(row.get(field), -1)

    txt_negative = feature["txt_negative_label"]
    txt_morphology = feature["txt_morphology_label"]
    image_morphology = feature["image_morphology_weak_label"]
    conflict = (
        (feature["label"] == 1 and (txt_negative == 1 or image_morphology == 0))
        or (feature["label"] == 0 and (txt_morphology == 1 or image_morphology == 1))
    )
    feature["evidence_conflict"] = "conflict" if conflict else "non_conflict"
    feature["weak_evidence_group"] = f"morph{txt_morphology}_neg{txt_negative}"
    selected_visits = feature["selected_n_visits"]
    feature["visit_count_bin"] = "1" if selected_visits <= 1 else "2" if selected_visits == 2 else "3+"
    coverage = row_float(feature, "visit_report_coverage", 0.0)
    feature["report_coverage_bin"] = "0" if coverage <= 0 else "partial" if coverage < 1 else "full"
    feature["dated_bio_bin"] = "0" if feature["dated_bio_visit_count"] <= 0 else "1+"
    missing_bio = feature["bio_missing_count"]
    feature["bio_missing_bin"] = "all_missing" if missing_bio >= 7 else "partial_missing" if missing_bio > 0 else "complete"
    return feature


def inversion_burdens(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Tuple[int, int]]:
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    inversions = probabilities[positive, None] < probabilities[negative][None, :]
    positive_counts = inversions.sum(axis=1).astype(int)
    negative_counts = inversions.sum(axis=0).astype(int)
    output: Dict[str, Tuple[int, int]] = {}
    for index, count in zip(positive, positive_counts):
        output[str(index)] = (int(count), len(negative))
    for index, count in zip(negative, negative_counts):
        output[str(index)] = (int(count), len(positive))
    return output


def load_decisions(config: Mapping[str, Any], candidates: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for model in candidates:
        path = resolve_path(config["project"]["models"][model]["decision_json"])
        if any(part.lower() == "test" for part in path.parts):
            failures.append(f"{model}: decision path contains a Test component")
            continue
        if not path.exists():
            failures.append(f"{model}: missing decision JSON {path}")
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        decisions[model] = decision
        if not bool(decision.get("training_health_pass")):
            failures.append(f"{model}: training_health_pass is false")
        if not bool(decision.get("shortcut_safety_pass")):
            failures.append(f"{model}: shortcut_safety_pass is false")
        if bool(decision.get("goal_reached")):
            failures.append(f"{model}: decision claims goal reached unexpectedly")
        if as_float(decision.get("validation_mean_AUC"), 1.0) >= 0.89:
            failures.append(f"{model}: Validation mean is not below 0.89")
        for key, expected in (("validation_decision_frozen_before_test", True), ("test_used_for_decision", False), ("ensemble_used", False), ("threshold_tuned", False)):
            if bool(decision.get(key)) is not expected:
                failures.append(f"{model}: {key} failed isolation check")
    return decisions, failures


def load_predictions(config: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], List[str], np.ndarray]:
    val_rows = sorted((row for row in manifest_rows if str(row.get("split", "")).lower() == "val"), key=lambda row: str(row["patient_id"]))
    ids = [str(row["patient_id"]) for row in val_rows]
    labels = np.asarray([as_int(row.get("label")) for row in val_rows], dtype=int)
    if len(ids) != EXPECTED_VAL_COUNT or int((labels == 1).sum()) != EXPECTED_VAL_POSITIVES:
        raise RuntimeError(f"Canonical Validation split mismatch: n={len(ids)} positives={(labels == 1).sum()}")
    predictions: Dict[str, Dict[int, pd.DataFrame]] = OrderedDict()
    for model in MODEL_ORDER:
        run_dir = resolve_path(config["project"]["models"][model]["run_dir"])
        predictions[model] = {}
        for seed in SEEDS:
            frame = read_prediction(run_dir / "predictions" / f"val_predictions_seed_{seed}.csv", model, seed)
            if frame["patient_id"].tolist() != ids:
                raise RuntimeError(f"{model} seed {seed} patient alignment does not match the canonical Validation split")
            if not np.array_equal(frame["label"].to_numpy(dtype=int), labels):
                raise RuntimeError(f"{model} seed {seed} label alignment does not match the canonical Validation split")
            predictions[model][seed] = frame
    return predictions, ids, labels


def aggregate_patient_errors(
    predictions: Mapping[str, Mapping[int, pd.DataFrame]],
    ids: Sequence[str],
    labels: np.ndarray,
    feature_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    total_runs = len(MODEL_ORDER) * len(SEEDS)
    records: List[Dict[str, Any]] = []
    for index, patient_id in enumerate(ids):
        record = dict(feature_by_id[patient_id])
        record.update({"error_count": 0, "fn_count": 0, "fp_count": 0, "rank_bad_comparisons": 0, "rank_total_comparisons": 0, "rank_bad_run_count": 0})
        for model in MODEL_ORDER:
            record[f"{model}_error_count"] = 0
            record[f"{model}_fn_count"] = 0
            record[f"{model}_fp_count"] = 0
        for model in MODEL_ORDER:
            for seed in SEEDS:
                frame = predictions[model][seed]
                probabilities = frame["probability"].to_numpy(dtype=float)
                predicted = probabilities >= 0.5
                burdens = inversion_burdens(labels, probabilities)
                error = bool(predicted[index] != labels[index])
                fn = bool(labels[index] == 1 and not predicted[index])
                fp = bool(labels[index] == 0 and predicted[index])
                record["error_count"] += int(error)
                record["fn_count"] += int(fn)
                record["fp_count"] += int(fp)
                record[f"{model}_error_count"] += int(error)
                record[f"{model}_fn_count"] += int(fn)
                record[f"{model}_fp_count"] += int(fp)
                bad, total = burdens[str(index)]
                record["rank_bad_comparisons"] += bad
                record["rank_total_comparisons"] += total
                record["rank_bad_run_count"] += int(bad > 0)
        record["error_rate"] = record["error_count"] / total_runs
        record["fn_rate"] = record["fn_count"] / total_runs
        record["fp_rate"] = record["fp_count"] / total_runs
        record["rank_error_rate"] = record["rank_bad_comparisons"] / max(record["rank_total_comparisons"], 1)
        record["hard_common_error"] = record["error_rate"] >= 0.5
        record["hard_common_fn"] = bool(labels[index] == 1 and record["fn_rate"] >= 0.5)
        record["hard_common_fp"] = bool(labels[index] == 0 and record["fp_rate"] >= 0.5)
        record["hard_common_rank"] = record["rank_error_rate"] >= 0.5
        records.append(record)
    return pd.DataFrame(records)


def error_concentration(patient_errors: pd.DataFrame) -> Dict[str, float]:
    def share(frame: pd.DataFrame, count_field: str) -> float:
        total = float(frame[count_field].sum())
        if frame.empty or total <= 0:
            return 0.0
        take = max(1, int(np.ceil(len(frame) * 0.25)))
        return float(frame[count_field].nlargest(take).sum() / total)

    positive = patient_errors[patient_errors["label"] == 1]
    negative = patient_errors[patient_errors["label"] == 0]
    return {
        "all_error_count": float(patient_errors["error_count"].sum()),
        "top_quartile_all_error_share": share(patient_errors, "error_count"),
        "top_quartile_positive_fn_share": share(positive, "fn_count"),
        "top_quartile_negative_fp_share": share(negative, "fp_count"),
    }


def evidence_strata(predictions: Mapping[str, Mapping[int, pd.DataFrame]], patient_errors: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for field in STRATA_FIELDS:
        for value in sorted(patient_errors[field].astype(str).unique()):
            mask = patient_errors[field].astype(str).eq(value).to_numpy()
            indices = np.where(mask)[0]
            if len(indices) < 4:
                continue
            for model in MODEL_ORDER:
                for seed in SEEDS:
                    frame = predictions[model][seed]
                    labels = frame["label"].to_numpy(dtype=int)[indices]
                    probabilities = frame["probability"].to_numpy(dtype=float)[indices]
                    predicted = probabilities >= 0.5
                    positives = labels == 1
                    negatives = labels == 0
                    rows.append(
                        {
                            "stratum_field": field,
                            "stratum": value,
                            "model": model,
                            "seed": seed,
                            "n": int(len(indices)),
                            "positive_n": int(positives.sum()),
                            "negative_n": int(negatives.sum()),
                            "AUC": safe_auc(labels, probabilities),
                            "error_rate": float((predicted != labels).mean()),
                            "fn_rate": float((~predicted[positives]).mean()) if positives.any() else np.nan,
                            "fp_rate": float(predicted[negatives].mean()) if negatives.any() else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def evidence_audit(patient_errors: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hard_positive = patient_errors[patient_errors["hard_common_fn"]]
    easy_positive = patient_errors[(patient_errors["label"] == 1) & ~patient_errors["hard_common_fn"]]
    for field in ("evidence_conflict", "visit_count_bin", "report_coverage_bin", "dated_bio_bin", "bio_missing_bin", "weak_evidence_group"):
        for value in sorted(patient_errors[field].astype(str).unique()):
            group = patient_errors[patient_errors[field].astype(str) == value]
            positives = group[group["label"] == 1]
            hard = positives[positives["hard_common_fn"]]
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "n": int(len(group)),
                    "positive_n": int(len(positives)),
                    "hard_positive_n": int(len(hard)),
                    "hard_positive_rate": float(len(hard) / len(positives)) if len(positives) else np.nan,
                    "all_error_rate": float(group["error_rate"].mean()) if len(group) else np.nan,
                }
            )
    overall = float(len(hard_positive) / max(int((patient_errors["label"] == 1).sum()), 1))
    enriched = [
        row
        for row in rows
        if int(row["positive_n"]) >= 8
        and np.isfinite(row["hard_positive_rate"])
        and row["hard_positive_rate"] >= overall + 0.15
    ]
    summary = {
        "hard_positive_count": int(len(hard_positive)),
        "positive_count": int((patient_errors["label"] == 1).sum()),
        "hard_positive_rate": overall,
        "hard_positive_conflict_rate": float(hard_positive["evidence_conflict"].eq("conflict").mean()) if len(hard_positive) else 0.0,
        "easy_positive_conflict_rate": float(easy_positive["evidence_conflict"].eq("conflict").mean()) if len(easy_positive) else 0.0,
        "enriched_evidence_strata_count": len(enriched),
        "enriched_evidence_strata": enriched,
    }
    return pd.DataFrame(rows), summary


def write_report(
    output_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    decisions: Mapping[str, Mapping[str, Any]],
    decision_failures: Sequence[str],
    patient_errors: pd.DataFrame,
    concentration: Mapping[str, Any],
    strata: pd.DataFrame,
    evidence: pd.DataFrame,
    evidence_summary: Mapping[str, Any],
    auc_rows: Sequence[Mapping[str, Any]],
    structural_pass: bool,
) -> Dict[str, Any]:
    candidate_means = {model: as_float(decisions[model].get("validation_mean_AUC")) for model in decisions}
    common_hard_patient_rate = float(patient_errors["hard_common_error"].mean())
    common_hard_positive_rate = float(patient_errors.loc[patient_errors["label"] == 1, "hard_common_fn"].mean())
    common_hard_negative_rate = float(patient_errors.loc[patient_errors["label"] == 0, "hard_common_fp"].mean())
    all_candidates_healthy_safe = not decision_failures and len(decisions) == 3
    data_limit_supported = bool(
        structural_pass
        and all_candidates_healthy_safe
        and max(candidate_means.values()) < 0.89
        and common_hard_positive_rate >= 0.25
        and common_hard_patient_rate >= 0.20
        and float(concentration["top_quartile_all_error_share"]) >= 0.35
        and int(evidence_summary["enriched_evidence_strata_count"]) >= 1
    )
    conclusion = "DEMA_HT_AUC_090_DATA_LIMIT_SUSPECTED" if data_limit_supported else "DATA_LIMIT_AUDIT_INCONCLUSIVE"
    summary = {
        "phase": "GOAL_DATA_LIMIT_AUDIT",
        "conclusion": conclusion,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "validation_only": True,
        "test_used": False,
        "structural_pass": structural_pass,
        "decision_failures": list(decision_failures),
        "candidate_validation_mean_AUC": candidate_means,
        "common_hard_patient_rate": common_hard_patient_rate,
        "common_hard_positive_fn_rate": common_hard_positive_rate,
        "common_hard_negative_fp_rate": common_hard_negative_rate,
        "error_concentration": dict(concentration),
        "evidence_summary": dict(evidence_summary),
        "criteria": {
            "three_healthy_safe_candidates": all_candidates_healthy_safe,
            "all_candidate_means_below_0.89": bool(candidate_means) and max(candidate_means.values()) < 0.89,
            "common_hard_positive_fn_rate_at_least_0.25": common_hard_positive_rate >= 0.25,
            "common_hard_patient_rate_at_least_0.20": common_hard_patient_rate >= 0.20,
            "top_quartile_error_share_at_least_0.35": float(concentration["top_quartile_all_error_share"]) >= 0.35,
            "enriched_evidence_stratum_present": int(evidence_summary["enriched_evidence_strata_count"]) >= 1,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_errors.to_csv(output_dir / "patient_error_concentration.csv", index=False)
    strata.to_csv(output_dir / "validation_evidence_strata.csv", index=False)
    evidence.to_csv(output_dir / "evidence_audit_summary.csv", index=False)
    pd.DataFrame(list(auc_rows)).to_csv(output_dir / "validation_auc_by_model_seed.csv", index=False)
    (output_dir / "data_limit_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Goal Data-Limit Audit",
        "",
        f"- Conclusion: `{conclusion}`.",
        f"- Manifest SHA-256: `{manifest_sha256}`; Validation-only: `{summary['validation_only']}`; Test used: `{summary['test_used']}`.",
        f"- Candidate Validation means: `{candidate_means}`; all three health/shortcut/isolation checks passed: `{all_candidates_healthy_safe and structural_pass}`.",
        f"- Common hard patient rate: `{common_hard_patient_rate:.6f}`; common positive false-negative rate: `{common_hard_positive_rate:.6f}`; common negative false-positive rate: `{common_hard_negative_rate:.6f}`.",
        f"- Top-quartile error share: `{float(concentration['top_quartile_all_error_share']):.6f}`; enriched evidence strata: `{int(evidence_summary['enriched_evidence_strata_count'])}`.",
        "- All evidence fields are audit-only and were not model inputs.",
        "- No Test prediction, Test label, Test metric, or Test-derived artifact was opened by this audit.",
    ]
    (output_dir / "goal_data_limit_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_goal_data_limit_audit.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if str(config.get("phase", "")).lower() != "goal_data_limit_audit":
        raise RuntimeError("This script requires the goal_data_limit_audit config")
    manifest_path = resolve_path(config["project"]["manifest"])
    manifest_sha256 = sha256_file(manifest_path)
    expected_sha256 = str(config["project"].get("expected_manifest_sha256", EXPECTED_MANIFEST_SHA256))
    if manifest_sha256 != expected_sha256:
        raise RuntimeError(f"Manifest SHA-256 mismatch: expected {expected_sha256}, got {manifest_sha256}")
    manifest_rows = read_manifest(manifest_path)
    if len({str(row["patient_id"]) for row in manifest_rows}) != len(manifest_rows):
        raise RuntimeError("Manifest has duplicate patient IDs")
    if any(str(row.get("split", "")).lower() not in {"train", "val", "test"} for row in manifest_rows):
        raise RuntimeError("Manifest contains an unknown split")
    candidates = tuple(config["project"].get("candidate_models", ("C38", "C39", "C40")))
    if tuple(config["project"].get("seeds", SEEDS)) != SEEDS or candidates != ("C38", "C39", "C40"):
        raise RuntimeError("The goal audit requires candidates C38/C39/C40 and seeds [0, 42, 3407]")
    decisions, decision_failures = load_decisions(config, candidates)
    predictions, ids, labels = load_predictions(config, manifest_rows)
    feature_by_id = {
        str(row["patient_id"]): manifest_features(row)
        for row in manifest_rows
        if str(row.get("split", "")).lower() == "val"
    }
    patient_errors = aggregate_patient_errors(predictions, ids, labels, feature_by_id)
    concentration = error_concentration(patient_errors)
    strata = evidence_strata(predictions, patient_errors)
    evidence, evidence_summary = evidence_audit(patient_errors)
    auc_rows: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        for seed in SEEDS:
            frame = predictions[model][seed]
            auc_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "n": len(frame),
                    "positive_n": int((frame["label"] == 1).sum()),
                    "Validation_AUC": safe_auc(frame["label"], frame["probability"]),
                }
            )
    structural_pass = manifest_sha256 == expected_sha256 and not decision_failures
    summary = write_report(
        resolve_path(config["project"]["output_dir"]),
        manifest_path,
        manifest_sha256,
        decisions,
        decision_failures,
        patient_errors,
        concentration,
        strata,
        evidence,
        evidence_summary,
        auc_rows,
        structural_pass,
    )
    print(json.dumps({"status": "GOAL_DATA_LIMIT_AUDIT_COMPLETE", **summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
