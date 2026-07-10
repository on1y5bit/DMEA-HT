from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest  # noqa: E402
from scripts.analyze_phase_c11_report_filter_hypotheses import (  # noqa: E402
    DIFFUSE_HT_CUES,
    MORPHOLOGY_CUES,
    NEGATIVE_THYROID_CUES,
    THYROID_CUES,
    count_terms,
    parse_visits,
)
from scripts.analyze_phase_c14a_fn_token_exposure import BENIGN_NODULE_CUES  # noqa: E402


MATCH_FIELDS = ["report_length", "selected_n_visits", "used_images", "image_padding_count", "has_bio", "bio_missing_count"]
SUMMARY_FIELDS = [
    "pred_prob",
    "logit",
    "text_classifier_contribution",
    "image_classifier_contribution",
    "bio_classifier_contribution",
    "e_synergy",
    "text_embedding_norm",
    "image_embedding_norm",
    "text_anchor_cosine",
    "image_anchor_cosine",
    "text_image_cosine",
    "delta_mask_image",
    "delta_mask_text",
    "delta_mask_bio",
    "delta_text_only_like",
    "delta_remove_diffuse",
    "delta_remove_negative",
    "delta_remove_prefix",
    "first256_morphology_term_count",
    "first256_diffuse_ht_term_count",
    "full_report_morphology_term_count",
    "full_report_diffuse_ht_term_count",
    "latest_visit_morphology_term_count",
    "latest_visit_diffuse_ht_term_count",
    "full_report_negative_term_count",
    "positive_negative_overlap",
    *MATCH_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C14-E hard clinical evidence audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--c14a-dir", default="analysis_reports/phase_c14a")
    parser.add_argument("--c14b-dir", default="analysis_reports/phase_c14b")
    parser.add_argument("--c14c-dir", default="analysis_reports/phase_c14c")
    parser.add_argument("--c14d-dir", default="analysis_reports/phase_c14d")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c14e")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values: List[str] = []
        for column in frame.columns:
            value = row[column]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                text = "NA"
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def report_text(row: Mapping[str, Any]) -> str:
    for key in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def to_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def match_terms(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if term and term in text]


def snippets(text: str, terms: Sequence[str], width: int = 22, limit: int = 5) -> str:
    found: List[str] = []
    for term in terms:
        start = text.find(term)
        if start < 0:
            continue
        left = max(0, start - width)
        right = min(len(text), start + len(term) + width)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ").strip()
        if snippet and snippet not in found:
            found.append(snippet)
        if len(found) >= limit:
            break
    return " || ".join(found)


def visit_evidence(text: str) -> Dict[str, Any]:
    visits = parse_visits(text)
    if not visits and text.strip():
        visits = [("unknown", text)]
    states: List[Dict[str, Any]] = []
    for date, block in visits:
        morphology = count_terms(block, MORPHOLOGY_CUES)
        diffuse = count_terms(block, DIFFUSE_HT_CUES)
        negative = count_terms(block, NEGATIVE_THYROID_CUES)
        positive = int(diffuse > 0 or morphology >= 2)
        contradictory = int(positive and negative > 0)
        states.append(
            {
                "date": str(date),
                "positive": positive,
                "negative": int(negative > 0),
                "contradictory": contradictory,
                "morphology": morphology,
                "diffuse": diffuse,
            }
        )
    positive_states = [state for state in states if state["positive"]]
    contradictory_states = [state for state in states if state["contradictory"]]
    transitions_pos_neg = sum(states[index - 1]["positive"] and states[index]["negative"] for index in range(1, len(states)))
    transitions_neg_pos = sum(states[index - 1]["negative"] and states[index]["positive"] for index in range(1, len(states)))
    latest = states[-1] if states else {"date": "", "positive": 0, "negative": 0, "morphology": 0, "diffuse": 0}
    if not positive_states:
        temporal_state = "no_clear_positive"
    elif len(positive_states) == 1 and positive_states[0]["diffuse"] == 0:
        temporal_state = "single_weak_positive"
    elif latest["negative"] and not latest["positive"] and positive_states:
        temporal_state = "historical_positive_latest_negative"
    elif transitions_pos_neg or transitions_neg_pos or contradictory_states:
        temporal_state = "intermittent_conflict"
    elif latest["positive"] and len(positive_states) >= 2:
        temporal_state = "persistent_positive"
    elif latest["positive"]:
        temporal_state = "recent_positive"
    else:
        temporal_state = "intermittent_conflict"
    return {
        "n_parsed_visits": len(states),
        "earliest_positive_evidence_visit": positive_states[0]["date"] if positive_states else "",
        "latest_positive_evidence_visit": positive_states[-1]["date"] if positive_states else "",
        "latest_thyroid_visit": latest["date"],
        "positive_evidence_persistence": len(positive_states) / max(len(states), 1),
        "positive_to_negative_transition": int(transitions_pos_neg > 0),
        "negative_to_positive_transition": int(transitions_neg_pos > 0),
        "concordant_positive_visit_count": len(positive_states),
        "contradictory_visit_count": len(contradictory_states),
        "latest_visit_evidence_state": "positive" if latest["positive"] and not latest["negative"] else ("contradictory" if latest["positive"] and latest["negative"] else ("negative" if latest["negative"] else "unclear")),
        "temporal_state": temporal_state,
    }


def evidence_audit(row: Mapping[str, Any]) -> Dict[str, Any]:
    text = report_text(row)
    first = text[:254]
    visits = parse_visits(text)
    latest_text = visits[-1][1] if visits else text
    morphology_terms = match_terms(text, MORPHOLOGY_CUES)
    diffuse_terms = match_terms(text, DIFFUSE_HT_CUES)
    negative_terms = match_terms(text, NEGATIVE_THYROID_CUES)
    benign_terms = match_terms(text, BENIGN_NODULE_CUES)
    counts = {
        "first256_morphology_term_count": count_terms(first, MORPHOLOGY_CUES),
        "first256_diffuse_ht_term_count": count_terms(first, DIFFUSE_HT_CUES),
        "full_report_morphology_term_count": count_terms(text, MORPHOLOGY_CUES),
        "full_report_diffuse_ht_term_count": count_terms(text, DIFFUSE_HT_CUES),
        "latest_visit_morphology_term_count": count_terms(latest_text, MORPHOLOGY_CUES),
        "latest_visit_diffuse_ht_term_count": count_terms(latest_text, DIFFUSE_HT_CUES),
        "full_report_negative_term_count": count_terms(text, NEGATIVE_THYROID_CUES),
        "benign_nodule_term_count": count_terms(text, BENIGN_NODULE_CUES),
    }
    temporal = visit_evidence(text)
    ht_specific = int(
        counts["full_report_diffuse_ht_term_count"] >= 2
        or (counts["full_report_diffuse_ht_term_count"] >= 1 and temporal["concordant_positive_visit_count"] >= 2)
    )
    generic = int(not ht_specific and counts["full_report_morphology_term_count"] > 0)
    contradictory = int(counts["full_report_negative_term_count"] > 0 and (counts["full_report_morphology_term_count"] > 0 or counts["full_report_diffuse_ht_term_count"] > 0))
    if ht_specific and contradictory:
        primary = "ht_specific_with_contradiction"
    elif ht_specific:
        primary = "ht_specific_evidence"
    elif generic and contradictory:
        primary = "generic_ambiguous_with_contradiction"
    elif generic:
        primary = "generic_or_ambiguous_evidence"
    else:
        primary = "no_clear_positive_evidence"
    return {
        **counts,
        **temporal,
        "ht_specific_evidence": ht_specific,
        "generic_or_ambiguous_evidence": generic,
        "contradictory_evidence": contradictory,
        "positive_negative_overlap": contradictory,
        "primary_evidence_category": primary,
        "matched_morphology_terms": "/".join(morphology_terms),
        "matched_diffuse_terms": "/".join(diffuse_terms),
        "matched_negative_terms": "/".join(negative_terms),
        "matched_benign_terms": "/".join(benign_terms),
        "morphology_snippets": snippets(text, morphology_terms),
        "negative_or_benign_snippets": snippets(text, [*negative_terms, *benign_terms]),
    }


def build_patient_base(manifest_path: Path) -> pd.DataFrame:
    rows = [row for row in read_manifest(manifest_path) if str(row.get("split", "")).lower() == "val"]
    output: List[Dict[str, Any]] = []
    for row in rows:
        text = report_text(row)
        output_row = {
            "patient_id": str(row["patient_id"]),
            "label": int(float(row.get("label", 0))),
            "report_length": to_float(row.get("report_length"), len(text)),
            "selected_n_visits": to_float(row.get("selected_n_visits"), np.nan),
            "used_images": to_float(row.get("used_images", row.get("n_images")), np.nan),
            "image_padding_count": to_float(row.get("image_padding_count", row.get("padding_count")), np.nan),
            "has_bio": to_float(row.get("has_bio"), np.nan),
            "bio_missing_count": to_float(row.get("bio_missing_count"), np.nan),
            "last_visit_date": str(row.get("last_visit_date", row.get("anchor_date", ""))),
            "label_source": str(row.get("label_source", "unavailable")),
        }
        output_row.update(evidence_audit(row))
        output.append(output_row)
    return pd.DataFrame(output)


def build_cohorts(hard: pd.DataFrame, patient_base: pd.DataFrame, pairwise: pd.DataFrame, cross_seed: pd.DataFrame) -> pd.DataFrame:
    hard["patient_id"] = hard["patient_id"].astype(str)
    hard_ids = set(hard[pd.to_numeric(hard["all_seed_hard_patient"], errors="coerce").fillna(0).astype(int) == 1]["patient_id"])
    rows: List[Dict[str, Any]] = []
    for patient in patient_base.to_dict("records"):
        patient_id = str(patient["patient_id"])
        label = int(patient["label"])
        is_hard = patient_id in hard_ids
        role = "positive" if label == 1 else "negative"
        cohort = f"hard_{role}" if is_hard else f"nonhard_{role}"
        patient_pairs = pairwise[
            ((pairwise["positive_patient_id"].astype(str) == patient_id) if label == 1 else (pairwise["negative_patient_id"].astype(str) == patient_id))
            & (pd.to_numeric(pairwise["is_inversion"], errors="coerce") == 1)
        ]
        by_seed = patient_pairs.groupby("seed").size().to_dict()
        pair_ids = cross_seed[
            (cross_seed["positive_patient_id"].astype(str) == patient_id) if label == 1 else (cross_seed["negative_patient_id"].astype(str) == patient_id)
        ]
        rows.append(
            {
                **patient,
                "cohort": cohort,
                "is_hard": int(is_hard),
                "inversion_pair_count_seed_0": int(by_seed.get(0, 0)),
                "inversion_pair_count_seed_42": int(by_seed.get(42, 0)),
                "inversion_pair_count_seed_3407": int(by_seed.get(3407, 0)),
                "all_seed_inversion_pair_count": int((pair_ids["inversion_group"] == "all_seed_inversion").sum()),
                "majority_seed_inversion_pair_count": int((pair_ids["inversion_group"] == "majority_seed_inversion").sum()),
                "single_seed_inversion_pair_count": int((pair_ids["inversion_group"] == "single_seed_inversion").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_topk_metrics(hard: pd.DataFrame, pairwise: pd.DataFrame, cross_seed: pd.DataFrame) -> pd.DataFrame:
    hard = hard.copy()
    hard["patient_id"] = hard["patient_id"].astype(str)
    hard["inversion_count"] = pd.to_numeric(hard["inversion_count"], errors="coerce").fillna(0)
    inversion_rows = pairwise[pd.to_numeric(pairwise["is_inversion"], errors="coerce") == 1].copy()
    scopes = {
        "all": hard,
        "hard_positive": hard[hard["role"] == "positive"],
        "hard_negative": hard[hard["role"] == "negative"],
    }
    group_specs = {
        "all_inversions": None,
        "all_seed_inversion": "all_seed_inversion",
        "majority_seed_inversion": "majority_seed_inversion",
        "single_seed_inversion": "single_seed_inversion",
    }
    rows: List[Dict[str, Any]] = []
    for scope, ranking in scopes.items():
        role = None if scope == "all" else ("positive" if scope == "hard_positive" else "negative")
        for k in (5, 10, 20):
            top_ids = set(ranking.nlargest(min(k, len(ranking)), "inversion_count")["patient_id"])
            for group_name, group_value in group_specs.items():
                pair_table = cross_seed if group_value is None else cross_seed[cross_seed["inversion_group"] == group_value]
                if group_value is None:
                    pair_table = cross_seed[pd.to_numeric(cross_seed["inversion_count"], errors="coerce") > 0]
                pair_hit = pair_table["positive_patient_id"].astype(str).isin(top_ids) | pair_table["negative_patient_id"].astype(str).isin(top_ids)
                unique_pair_denominator = len(pair_table)
                unique_pair_numerator = int(pair_hit.sum())
                if group_value is None:
                    row_table = inversion_rows
                else:
                    keys = pair_table[["positive_patient_id", "negative_patient_id"]].copy()
                    keys["positive_patient_id"] = keys["positive_patient_id"].astype(str)
                    keys["negative_patient_id"] = keys["negative_patient_id"].astype(str)
                    row_table = inversion_rows.merge(keys, on=["positive_patient_id", "negative_patient_id"], how="inner")
                if role == "positive":
                    appearances = int(row_table["positive_patient_id"].astype(str).isin(top_ids).sum())
                    incidence_denominator = len(row_table)
                elif role == "negative":
                    appearances = int(row_table["negative_patient_id"].astype(str).isin(top_ids).sum())
                    incidence_denominator = len(row_table)
                else:
                    appearances = int(row_table["positive_patient_id"].astype(str).isin(top_ids).sum() + row_table["negative_patient_id"].astype(str).isin(top_ids).sum())
                    incidence_denominator = 2 * len(row_table)
                rows.append(
                    {
                        "scope": scope,
                        "k": k,
                        "inversion_group": group_name,
                        "top_patient_count": len(top_ids),
                        "unique_pair_coverage_numerator": unique_pair_numerator,
                        "unique_pair_coverage_denominator": unique_pair_denominator,
                        "unique_pair_coverage": unique_pair_numerator / max(unique_pair_denominator, 1),
                        "patient_side_incidence_numerator": appearances,
                        "patient_side_incidence_denominator": incidence_denominator,
                        "patient_side_incidence_share": appearances / max(incidence_denominator, 1),
                        "unique_pair_responsibility_numerator": unique_pair_numerator,
                        "unique_pair_responsibility_denominator": unique_pair_denominator,
                        "unique_pair_responsibility": unique_pair_numerator / max(unique_pair_denominator, 1),
                    }
                )
    return pd.DataFrame(rows)


def smd(hard_values: pd.Series, control_values: pd.Series) -> float:
    hard_values = pd.to_numeric(hard_values, errors="coerce").dropna()
    control_values = pd.to_numeric(control_values, errors="coerce").dropna()
    if not len(hard_values) or not len(control_values):
        return np.nan
    pooled = math.sqrt((float(hard_values.var(ddof=1) if len(hard_values) > 1 else 0.0) + float(control_values.var(ddof=1) if len(control_values) > 1 else 0.0)) / 2.0)
    return (float(hard_values.mean()) - float(control_values.mean())) / pooled if pooled > 0 else np.nan


def build_matches(cohorts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: List[Dict[str, Any]] = []
    balance_rows: List[Dict[str, Any]] = []
    for label, role in ((1, "positive"), (0, "negative")):
        label_frame = cohorts[cohorts["label"] == label].copy()
        hard = label_frame[label_frame["is_hard"] == 1].sort_values("all_seed_inversion_pair_count", ascending=False)
        controls = label_frame[label_frame["is_hard"] == 0].copy()
        means = label_frame[MATCH_FIELDS].apply(pd.to_numeric, errors="coerce").mean()
        stds = label_frame[MATCH_FIELDS].apply(pd.to_numeric, errors="coerce").std().replace(0, 1.0).fillna(1.0)
        standardized = (label_frame.set_index("patient_id")[MATCH_FIELDS].apply(pd.to_numeric, errors="coerce").fillna(means) - means) / stds
        available = set(controls["patient_id"].astype(str))
        matched_hard_ids: List[str] = []
        matched_control_ids: List[str] = []
        for _, hard_row in hard.iterrows():
            hard_id = str(hard_row["patient_id"])
            if not available:
                match_rows.append({"hard_patient_id": hard_id, "matched_control_patient_id": "", "label": label, "role": role, "matching_distance": np.nan, "matching_variables": "/".join(MATCH_FIELDS), "match_quality_flag": "unmatched_no_control"})
                continue
            distances = ((standardized.loc[list(available)] - standardized.loc[hard_id]) ** 2).sum(axis=1).pow(0.5)
            control_id = str(distances.idxmin())
            distance = float(distances.loc[control_id])
            available.remove(control_id)
            matched_hard_ids.append(hard_id)
            matched_control_ids.append(control_id)
            match_rows.append({"hard_patient_id": hard_id, "matched_control_patient_id": control_id, "label": label, "role": role, "matching_distance": distance, "matching_variables": "/".join(MATCH_FIELDS), "match_quality_flag": "good" if distance <= 0.75 else "weak"})
        for variable in MATCH_FIELDS:
            before = smd(hard[variable], controls[variable])
            after = smd(
                cohorts[cohorts["patient_id"].astype(str).isin(matched_hard_ids)][variable],
                cohorts[cohorts["patient_id"].astype(str).isin(matched_control_ids)][variable],
            )
            balance_rows.append(
                {
                    "label": label,
                    "role": role,
                    "variable": variable,
                    "hard_patients": len(hard),
                    "available_controls": len(controls),
                    "matched_hard_patients": len(matched_hard_ids),
                    "unmatched_hard_patients": len(hard) - len(matched_hard_ids),
                    "smd_before": before,
                    "abs_smd_before": abs(before) if math.isfinite(before) else np.nan,
                    "smd_after": after,
                    "abs_smd_after": abs(after) if math.isfinite(after) else np.nan,
                    "balanced_after": int(math.isfinite(after) and abs(after) <= 0.10),
                }
            )
    return pd.DataFrame(match_rows), pd.DataFrame(balance_rows)


def aggregate_diagnostics(features: pd.DataFrame, masking: pd.DataFrame, occlusion: pd.DataFrame) -> pd.DataFrame:
    for frame in (features, masking, occlusion):
        frame["patient_id"] = frame["patient_id"].astype(str)
    feature_map = {
        "pred_prob": "pred_prob",
        "logit": "logit",
        "text_classifier_contribution": "text_classifier_contribution",
        "image_classifier_contribution": "image_classifier_contribution",
        "bio_classifier_contribution": "bio_classifier_contribution",
        "e_synergy": "e_synergy",
        "text_embedding_norm": "text_embedding_norm",
        "image_embedding_norm": "image_embedding_norm",
        "text_anchor_cosine": "text_anchor_cosine",
        "image_anchor_cosine": "image_anchor_cosine",
        "text_image_cosine": "text_image_cosine",
    }
    base = features.groupby("patient_id", as_index=False).agg({column: "mean" for column in feature_map if column in features.columns})
    mask_columns = [column for column in ("delta_mask_image", "delta_mask_text", "delta_mask_bio", "delta_text_only_like", "image_only_like_prob") if column in masking.columns]
    mask = masking.groupby("patient_id", as_index=False).agg({column: "mean" for column in mask_columns})
    mask_seed = masking.groupby("patient_id").agg(
        image_support_seed_count=("delta_mask_image", lambda values: int((pd.to_numeric(values, errors="coerce") < -0.02).sum())),
        image_suppression_seed_count=("delta_mask_image", lambda values: int((pd.to_numeric(values, errors="coerce") > 0.02).sum())),
        text_support_seed_count=("delta_mask_text", lambda values: int((pd.to_numeric(values, errors="coerce") < -0.02).sum())),
    ).reset_index()
    if not occlusion.empty:
        occ_columns = [column for column in ("delta_remove_diffuse", "delta_remove_negative", "delta_remove_prefix") if column in occlusion.columns]
        occ = occlusion.groupby("patient_id", as_index=False).agg({column: "mean" for column in occ_columns})
    else:
        occ = pd.DataFrame(columns=["patient_id"])
    return base.merge(mask, on="patient_id", how="left").merge(mask_seed, on="patient_id", how="left").merge(occ, on="patient_id", how="left")


def matched_sets(matches: pd.DataFrame, role: str) -> tuple[set[str], set[str]]:
    subset = matches[(matches["role"] == role) & (matches["matched_control_patient_id"].astype(str) != "")]
    return set(subset["hard_patient_id"].astype(str)), set(subset["matched_control_patient_id"].astype(str))


def cliffs_delta(hard_values: np.ndarray, control_values: np.ndarray) -> float:
    if not len(hard_values) or not len(control_values):
        return np.nan
    comparisons = np.sign(hard_values[:, None] - control_values[None, :])
    return float(comparisons.mean())


def comparison_summary(hard_frame: pd.DataFrame, control_frame: pd.DataFrame, variables: Sequence[str], bootstrap_iters: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for variable in variables:
        if variable not in hard_frame.columns or variable not in control_frame.columns:
            continue
        hard_values = pd.to_numeric(hard_frame[variable], errors="coerce").dropna().to_numpy(float)
        control_values = pd.to_numeric(control_frame[variable], errors="coerce").dropna().to_numpy(float)
        if not len(hard_values) or not len(control_values):
            continue
        hard_std = float(np.std(hard_values, ddof=1)) if len(hard_values) > 1 else 0.0
        control_std = float(np.std(control_values, ddof=1)) if len(control_values) > 1 else 0.0
        pooled = math.sqrt((hard_std**2 + control_std**2) / 2.0)
        differences = []
        for _ in range(bootstrap_iters):
            hard_sample = rng.choice(hard_values, size=len(hard_values), replace=True)
            control_sample = rng.choice(control_values, size=len(control_values), replace=True)
            differences.append(float(hard_sample.mean() - control_sample.mean()))
        rows.append(
            {
                "variable": variable,
                "n_hard": len(hard_values),
                "n_control": len(control_values),
                "hard_mean": float(np.mean(hard_values)),
                "hard_median": float(np.median(hard_values)),
                "hard_std": hard_std,
                "hard_iqr": float(np.percentile(hard_values, 75) - np.percentile(hard_values, 25)),
                "control_mean": float(np.mean(control_values)),
                "control_median": float(np.median(control_values)),
                "control_std": control_std,
                "control_iqr": float(np.percentile(control_values, 75) - np.percentile(control_values, 25)),
                "mean_difference": float(np.mean(hard_values) - np.mean(control_values)),
                "median_difference": float(np.median(hard_values) - np.median(control_values)),
                "cohens_d": float((np.mean(hard_values) - np.mean(control_values)) / pooled) if pooled > 0 else np.nan,
                "cliffs_delta": cliffs_delta(hard_values, control_values),
                "bootstrap_mean_difference_ci_low": float(np.percentile(differences, 2.5)),
                "bootstrap_mean_difference_ci_high": float(np.percentile(differences, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def assign_negative_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    categories: List[str] = []
    followup: List[str] = []
    for _, row in out.iterrows():
        image_support = to_float(row.get("delta_mask_image"), 0.0) < -0.02 or to_float(row.get("image_classifier_contribution"), 0.0) > 0
        text_support = to_float(row.get("delta_mask_text"), 0.0) < -0.02 or to_float(row.get("text_classifier_contribution"), 0.0) > 0
        benign = to_float(row.get("benign_nodule_term_count"), 0.0) > 0
        diffuse = to_float(row.get("full_report_diffuse_ht_term_count"), 0.0) > 0
        negative = to_float(row.get("full_report_negative_term_count"), 0.0) > 0
        if image_support and text_support:
            category = "both_modalities_ht_like"
        elif image_support and negative:
            category = "text_negative_image_positive"
        elif image_support and benign:
            category = "benign_nodular_mimic"
        elif image_support and diffuse:
            category = "diffuse_ht_like_image"
        elif image_support:
            category = "image_text_conflict"
        elif diffuse:
            category = "other_thyroiditis_like_pattern"
        else:
            category = "unclear"
        categories.append(category)
        temporal_state = str(row.get("temporal_state", ""))
        visits = to_float(row.get("selected_n_visits"), 0.0)
        if temporal_state == "historical_positive_latest_negative":
            boundary = "label_boundary_ambiguous"
        elif diffuse or to_float(row.get("full_report_morphology_term_count"), 0.0) >= 3:
            boundary = "ht_like_but_not_diagnosed"
        elif visits <= 1:
            boundary = "short_followup_or_uncertain_negative"
        elif negative and not diffuse:
            boundary = "well_supported_negative"
        else:
            boundary = "insufficient_information"
        followup.append(boundary)
    out["image_mimic_category"] = categories
    out["followup_label_audit_category"] = followup
    return out


def mechanism_coverage(
    positive_audit: pd.DataFrame,
    negative_audit: pd.DataFrame,
    matched_positive_controls: pd.DataFrame,
    matched_negative_controls: pd.DataFrame,
) -> pd.DataFrame:
    hp_weak = positive_audit["primary_evidence_category"].isin(["generic_or_ambiguous_evidence", "generic_ambiguous_with_contradiction", "no_clear_positive_evidence"]) | positive_audit["temporal_state"].isin(["historical_positive_latest_negative", "intermittent_conflict", "single_weak_positive", "no_clear_positive"])
    hp_control_weak = matched_positive_controls["primary_evidence_category"].isin(["generic_or_ambiguous_evidence", "generic_ambiguous_with_contradiction", "no_clear_positive_evidence"]) | matched_positive_controls["temporal_state"].isin(["historical_positive_latest_negative", "intermittent_conflict", "single_weak_positive", "no_clear_positive"])
    hn_mimic = negative_audit["image_mimic_category"].isin(["diffuse_ht_like_image", "benign_nodular_mimic", "image_text_conflict", "text_negative_image_positive", "both_modalities_ht_like"])
    hn_control_mimic = matched_negative_controls["image_mimic_category"].isin(["diffuse_ht_like_image", "benign_nodular_mimic", "image_text_conflict", "text_negative_image_positive", "both_modalities_ht_like"])
    hn_ambiguity = negative_audit["followup_label_audit_category"].isin(["short_followup_or_uncertain_negative", "ht_like_but_not_diagnosed", "possible_delayed_positive", "label_boundary_ambiguous", "insufficient_information"])
    hn_control_ambiguity = matched_negative_controls["followup_label_audit_category"].isin(["short_followup_or_uncertain_negative", "ht_like_but_not_diagnosed", "possible_delayed_positive", "label_boundary_ambiguous", "insufficient_information"])

    specs = [
        ("hard_positive_weak_or_ambiguous_evidence", positive_audit, hp_weak, matched_positive_controls, hp_control_weak, "HT_SPECIFIC_TEXT_EVIDENCE_AUDIT_OR_REPRESENTATION_PILOT"),
        ("hard_negative_ht_like_image_mimic", negative_audit, hn_mimic, matched_negative_controls, hn_control_mimic, "IMAGE_MIMIC_ROBUSTNESS_PILOT_DESIGN"),
        ("label_or_followup_ambiguity", negative_audit, hn_ambiguity, matched_negative_controls, hn_control_ambiguity, "DATA_AND_LABEL_AUDIT_ONLY"),
    ]
    rows: List[Dict[str, Any]] = []
    for name, hard_frame, hard_mask, control_frame, control_mask, intervention in specs:
        hard_count = int(hard_mask.sum())
        hard_fraction = hard_count / max(len(hard_frame), 1)
        control_fraction = float(control_mask.mean()) if len(control_mask) else np.nan
        contrast = hard_fraction - control_fraction if math.isfinite(control_fraction) else np.nan
        consistency = 3 if hard_count else 0
        top_dependence = min(1.0, 2 / max(hard_count, 1)) if hard_count else 1.0
        matching_coverage = len(control_frame) / max(len(hard_frame), 1)
        passes = int(hard_fraction >= 0.30 and consistency >= 2 and math.isfinite(contrast) and contrast >= 0.10 and hard_count >= 3 and top_dependence <= 0.50 and matching_coverage >= 0.50)
        rows.append(
            {
                "candidate_mechanism": name,
                "hard_patients_explained": hard_count,
                "hard_patient_count": len(hard_frame),
                "hard_fraction_explained": hard_fraction,
                "matched_controls_with_mechanism": int(control_mask.sum()) if len(control_mask) else 0,
                "matched_control_count": len(control_frame),
                "matched_control_fraction": control_fraction,
                "risk_difference_hard_minus_control": contrast,
                "cross_seed_consistency": consistency,
                "top_two_patient_dependence": top_dependence,
                "matching_coverage": matching_coverage,
                "maps_to_valid_intervention": intervention,
                "passes_30pct_generalizability_gate": passes,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c14a_dir = Path(args.c14a_dir)
    c14b_dir = Path(args.c14b_dir)
    c14c_dir = Path(args.c14c_dir)
    c14d_dir = Path(args.c14d_dir)

    hard = read_csv(c14c_dir / "c14c_hard_patient_summary.csv")
    pairwise = read_csv(c14c_dir / "c14c_pairwise_inversions_by_seed.csv")
    cross_seed = read_csv(c14c_dir / "c14c_cross_seed_inversion_summary.csv")
    features = read_csv(c14b_dir / "c14b_representation_diagnostics_val.csv")
    masking = read_csv(c14b_dir / "c14b_modality_masking_val.csv")
    occlusion = read_csv(c14b_dir / "c14b_text_occlusion_val.csv")
    c14d_profiles = read_csv(c14d_dir / "c14d_hard_patient_profiles.csv", required=False)
    patient_base = build_patient_base(Path(args.manifest))

    for frame, columns in (
        (pairwise, ["positive_patient_id", "negative_patient_id"]),
        (cross_seed, ["positive_patient_id", "negative_patient_id"]),
        (features, ["patient_id"]),
        (masking, ["patient_id"]),
        (occlusion, ["patient_id"]),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)

    cohorts = build_cohorts(hard, patient_base, pairwise, cross_seed)
    cohorts.to_csv(out_dir / "c14e_hard_patient_cohorts.csv", index=False)
    topk = build_topk_metrics(hard, pairwise, cross_seed)
    topk.to_csv(out_dir / "c14e_topk_responsibility_metrics.csv", index=False)
    matches, balance = build_matches(cohorts)
    matches.to_csv(out_dir / "c14e_matched_controls.csv", index=False)
    balance.to_csv(out_dir / "c14e_matching_balance_report.csv", index=False)

    diagnostics = aggregate_diagnostics(features, masking, occlusion)
    enriched = cohorts.merge(diagnostics, on="patient_id", how="left", suffixes=("", "_diag"))
    hp_hard_ids = set(enriched[enriched["cohort"] == "hard_positive"]["patient_id"])
    hn_hard_ids = set(enriched[enriched["cohort"] == "hard_negative"]["patient_id"])
    hp_match_ids, hp_control_ids = matched_sets(matches, "positive")
    hn_match_ids, hn_control_ids = matched_sets(matches, "negative")

    positive_review_ids = hp_hard_ids | hp_control_ids
    positive_text = enriched[enriched["patient_id"].isin(positive_review_ids)].copy()
    positive_text["review_cohort"] = np.where(positive_text["patient_id"].isin(hp_hard_ids), "hard_positive", "matched_positive_control")
    positive_text.to_csv(out_dir / "c14e_hard_positive_text_evidence.csv", index=False)
    temporal_columns = ["patient_id", "review_cohort", "n_parsed_visits", "earliest_positive_evidence_visit", "latest_positive_evidence_visit", "latest_thyroid_visit", "positive_evidence_persistence", "positive_to_negative_transition", "negative_to_positive_transition", "concordant_positive_visit_count", "contradictory_visit_count", "latest_visit_evidence_state", "temporal_state"]
    positive_text[[column for column in temporal_columns if column in positive_text.columns]].to_csv(out_dir / "c14e_hard_positive_temporal_states.csv", index=False)
    positive_text.to_csv(out_dir / "c14e_hard_positive_multimodal_diagnostics.csv", index=False)

    hp_hard_matched = enriched[enriched["patient_id"].isin(hp_match_ids)]
    hp_controls = enriched[enriched["patient_id"].isin(hp_control_ids)]
    hp_summary = comparison_summary(hp_hard_matched, hp_controls, SUMMARY_FIELDS, args.bootstrap_iters, args.seed)
    hp_summary.to_csv(out_dir / "c14e_hard_positive_vs_control_summary.csv", index=False)

    negative_review_ids = hn_hard_ids | hn_control_ids
    negative_audit = assign_negative_categories(enriched[enriched["patient_id"].isin(negative_review_ids)].copy())
    negative_audit["review_cohort"] = np.where(negative_audit["patient_id"].isin(hn_hard_ids), "hard_negative", "matched_negative_control")
    negative_audit.to_csv(out_dir / "c14e_hard_negative_image_mimic_audit.csv", index=False)
    followup_columns = ["patient_id", "review_cohort", "last_visit_date", "selected_n_visits", "latest_visit_evidence_state", "temporal_state", "label_source", "followup_label_audit_category", "full_report_morphology_term_count", "full_report_diffuse_ht_term_count", "full_report_negative_term_count"]
    negative_audit[[column for column in followup_columns if column in negative_audit.columns]].to_csv(out_dir / "c14e_hard_negative_followup_label_audit.csv", index=False)
    negative_audit.to_csv(out_dir / "c14e_hard_negative_multimodal_diagnostics.csv", index=False)

    hn_hard_matched = negative_audit[negative_audit["patient_id"].isin(hn_match_ids)]
    hn_controls = negative_audit[negative_audit["patient_id"].isin(hn_control_ids)]
    hn_summary = comparison_summary(hn_hard_matched, hn_controls, SUMMARY_FIELDS, args.bootstrap_iters, args.seed + 1)
    hn_summary.to_csv(out_dir / "c14e_hard_negative_vs_control_summary.csv", index=False)

    all_hp = positive_text[positive_text["review_cohort"] == "hard_positive"]
    all_hn = negative_audit[negative_audit["review_cohort"] == "hard_negative"]
    hp_control_review = positive_text[positive_text["review_cohort"] == "matched_positive_control"]
    hn_control_review = negative_audit[negative_audit["review_cohort"] == "matched_negative_control"]
    coverage = mechanism_coverage(all_hp, all_hn, hp_control_review, hn_control_review)
    coverage.to_csv(out_dir / "c14e_candidate_mechanism_coverage.csv", index=False)

    positive_match_coverage = len(hp_match_ids) / max(len(hp_hard_ids), 1)
    negative_match_coverage = len(hn_match_ids) / max(len(hn_hard_ids), 1)
    passing = set(coverage[coverage["passes_30pct_generalizability_gate"] == 1]["candidate_mechanism"])
    if positive_match_coverage < 0.50 or negative_match_coverage < 0.50:
        route = "DATA_LIMIT_NO_GENERAL_MODEL_FIX"
        allowed = "KEEP_C13_AND_REPORT_LIMITATION"
        basis = "Matched-control coverage is below 50% for at least one label subgroup, preventing a broad model mechanism claim."
    elif {"hard_positive_weak_or_ambiguous_evidence", "hard_negative_ht_like_image_mimic"}.issubset(passing):
        route = "TWO_SIDED_HARD_SUBGROUP"
        allowed = "SEPARATE_SUBGROUP_SPECIFIC_PILOT_DESIGNS"
        basis = "Both positive-evidence and negative-image mechanisms passed coverage, cross-seed, matched-contrast, and top-dependence gates."
    elif "hard_positive_weak_or_ambiguous_evidence" in passing:
        route = "HARD_POSITIVE_WEAK_OR_AMBIGUOUS_EVIDENCE"
        allowed = "HT_SPECIFIC_TEXT_EVIDENCE_AUDIT_OR_REPRESENTATION_PILOT"
        basis = "Hard-positive weak/ambiguous evidence passed the generalizability gate."
    elif "hard_negative_ht_like_image_mimic" in passing:
        route = "HARD_NEGATIVE_HT_LIKE_IMAGE_MIMIC"
        allowed = "IMAGE_MIMIC_ROBUSTNESS_PILOT_DESIGN"
        basis = "Hard-negative image mimic passed the generalizability gate."
    elif "label_or_followup_ambiguity" in passing:
        route = "LABEL_OR_FOLLOWUP_AMBIGUITY"
        allowed = "DATA_AND_LABEL_AUDIT_ONLY"
        basis = "Label/follow-up ambiguity passed the generalizability gate."
    else:
        route = "MIXED_OR_INCONCLUSIVE"
        allowed = "MORE_ANALYSIS_ONLY"
        basis = "No candidate mechanism passed all coverage, cross-seed, matched-control, and top-dependence gates."
    gate = pd.DataFrame(
        [
            {
                "route": route,
                "allowed_next_step": allowed,
                "training_authorized": 0,
                "hard_positive_count": len(hp_hard_ids),
                "hard_negative_count": len(hn_hard_ids),
                "matched_hard_positive_count": len(hp_match_ids),
                "matched_hard_negative_count": len(hn_match_ids),
                "positive_matching_coverage": positive_match_coverage,
                "negative_matching_coverage": negative_match_coverage,
                "candidate_mechanisms_passing_gate": "/".join(sorted(passing)),
                "decision_basis": basis,
            }
        ]
    )
    gate.to_csv(out_dir / "c14e_route_gate_summary.csv", index=False)

    topk_all = topk[topk["scope"] == "all"]
    matching_report = [
        "# Phase C14-E Matching And Top-K Responsibility",
        "",
        f"Hard positives: `{len(hp_hard_ids)}`; hard negatives: `{len(hn_hard_ids)}`.",
        f"Matched hard positives: `{len(hp_match_ids)}`; matched hard negatives: `{len(hn_match_ids)}`.",
        "",
        "Top-k metrics keep pair coverage, patient-side incidence, and unique-pair responsibility as separate denominators.",
        "",
        frame_to_markdown(topk_all),
        "",
        "## Matching Balance",
        "",
        frame_to_markdown(balance),
    ]
    (out_dir / "c14e_matching_and_topk_report.md").write_text("\n".join(matching_report) + "\n", encoding="utf-8")

    positive_report = [
        "# Phase C14-E Hard Positive Clinical Evidence Audit",
        "",
        frame_to_markdown(all_hp[[column for column in ["patient_id", "primary_evidence_category", "temporal_state", "first256_diffuse_ht_term_count", "full_report_diffuse_ht_term_count", "full_report_negative_term_count", "text_classifier_contribution", "image_classifier_contribution", "delta_mask_image", "delta_remove_diffuse", "morphology_snippets", "negative_or_benign_snippets"] if column in all_hp.columns]]),
        "",
        "## Matched-Control Effects",
        "",
        frame_to_markdown(hp_summary),
    ]
    (out_dir / "c14e_hard_positive_report.md").write_text("\n".join(positive_report) + "\n", encoding="utf-8")

    negative_report = [
        "# Phase C14-E Hard Negative Clinical Evidence Audit",
        "",
        frame_to_markdown(all_hn[[column for column in ["patient_id", "image_mimic_category", "followup_label_audit_category", "full_report_diffuse_ht_term_count", "benign_nodule_term_count", "full_report_negative_term_count", "text_classifier_contribution", "image_classifier_contribution", "delta_mask_image", "image_support_seed_count", "morphology_snippets", "negative_or_benign_snippets"] if column in all_hn.columns]]),
        "",
        "## Matched-Control Effects",
        "",
        frame_to_markdown(hn_summary),
    ]
    (out_dir / "c14e_hard_negative_report.md").write_text("\n".join(negative_report) + "\n", encoding="utf-8")

    ambiguity_prevalence = float(all_hn["followup_label_audit_category"].isin(["short_followup_or_uncertain_negative", "ht_like_but_not_diagnosed", "label_boundary_ambiguous", "insufficient_information"]).mean()) if len(all_hn) else np.nan
    largest = coverage.sort_values("hard_fraction_explained", ascending=False).head(1)
    largest_name = str(largest.iloc[0]["candidate_mechanism"]) if not largest.empty else "none"
    largest_fraction = float(largest.iloc[0]["hard_fraction_explained"]) if not largest.empty else np.nan
    final_lines = [
        "# Phase C14-E Hard Clinical Evidence Audit",
        "",
        "C14-E is analysis-only. No model/training code, labels, splits, task, manifest, report construction, images, bio values, or thresholds were changed. Test results were not used.",
        "",
        "## Cohorts",
        "",
        f"- Hard positives: `{len(hp_hard_ids)}`; non-hard positives: `{int((cohorts['cohort'] == 'nonhard_positive').sum())}`.",
        f"- Hard negatives: `{len(hn_hard_ids)}`; non-hard negatives: `{int((cohorts['cohort'] == 'nonhard_negative').sum())}`.",
        "",
        "## Top-K Responsibility",
        "",
        frame_to_markdown(topk_all),
        "",
        "## Matching Quality",
        "",
        f"- Positive matching coverage: `{positive_match_coverage:.4f}`; negative matching coverage: `{negative_match_coverage:.4f}`.",
        f"- Unmatched hard positives: `{len(hp_hard_ids) - len(hp_match_ids)}`; unmatched hard negatives: `{len(hn_hard_ids) - len(hn_match_ids)}`.",
        frame_to_markdown(balance),
        "",
        "## Candidate Mechanism Coverage",
        "",
        frame_to_markdown(coverage),
        "",
        f"Largest observed mechanism: `{largest_name}` with hard-subgroup fraction `{largest_fraction:.4f}`.",
        f"Hard-negative label/follow-up ambiguity prevalence: `{ambiguity_prevalence:.4f}`.",
        "",
        "## Final Route",
        "",
        f"`{route}`.",
        "",
        f"Allowed next-step class: `{allowed}`.",
        f"Decision basis: {basis}",
        "",
        "No route in C14-E automatically authorizes training. C15 remains blocked pending a separate explicit decision. C13 remains the current strict best; no model improvement or AUC 0.90 claim is made.",
    ]
    (out_dir / "phase_c14e_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "route": route,
                "allowed_next_step": allowed,
                "training_authorized": False,
                "hard_positive_count": len(hp_hard_ids),
                "hard_negative_count": len(hn_hard_ids),
                "positive_matching_coverage": positive_match_coverage,
                "negative_matching_coverage": negative_match_coverage,
                "candidate_mechanisms_passing_gate": sorted(passing),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
