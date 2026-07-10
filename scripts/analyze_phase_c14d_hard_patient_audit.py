from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_C14C_DIR = "analysis_reports/phase_c14c"
DEFAULT_C14B_DIR = "analysis_reports/phase_c14b"
DEFAULT_OUTPUT_DIR = "analysis_reports/phase_c14d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C14-D hard-patient subgroup audit.")
    parser.add_argument("--c14c-dir", default=DEFAULT_C14C_DIR)
    parser.add_argument("--c14b-dir", default=DEFAULT_C14B_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
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


def numeric(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def main() -> None:
    args = parse_args()
    c14c_dir = Path(args.c14c_dir)
    c14b_dir = Path(args.c14b_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = read_csv(c14c_dir / "c14c_hard_patient_summary.csv")
    pairwise = read_csv(c14c_dir / "c14c_pairwise_inversions_by_seed.csv")
    route = read_csv(c14c_dir / "c14c_route_gate_summary.csv")
    features = read_csv(c14b_dir / "c14b_representation_diagnostics_val.csv")
    hard["patient_id"] = hard["patient_id"].astype(str)
    pairwise["positive_patient_id"] = pairwise["positive_patient_id"].astype(str)
    pairwise["negative_patient_id"] = pairwise["negative_patient_id"].astype(str)
    features["patient_id"] = features["patient_id"].astype(str)

    hard = numeric(hard, ["inversion_count", "inversion_share", "n_seeds_with_inversion", "all_seed_hard_patient"])
    pairwise = numeric(pairwise, ["seed", "is_inversion", "image_opposed_flag", "image_repair_flag", "text_driven_flag", "fusion_interaction_flag", "final_logit_margin", "margin_without_image", "margin_without_text", "margin_without_bio", "text_only_like_margin", "margin_without_diffuse", "margin_without_negative"])
    features = numeric(
        features,
        [
            "seed", "label", "pred_prob", "logit", "e_text", "e_img", "e_bio", "e_synergy", "e_negative", "text_embedding_norm", "image_embedding_norm", "bio_embedding_norm",
            "text_classifier_contribution", "image_classifier_contribution", "bio_classifier_contribution", "report_length", "selected_n_visits", "used_images", "has_bio", "bio_missing_count",
            "full_report_morphology_term_count", "full_report_diffuse_ht_term_count", "first256_morphology_term_count", "first256_diffuse_ht_term_count", "full_report_negative_term_count", "evidence_exposed_in_first256", "positive_negative_overlap",
        ],
    )

    inversion_rows = pairwise[pairwise["is_inversion"] == 1].copy()
    hard_keys = set(hard[hard["all_seed_hard_patient"] == 1]["patient_id"])
    top_keys = set(hard.nlargest(args.top_k, "inversion_count")["patient_id"])

    profile_rows: List[Dict[str, Any]] = []
    for patient_id in sorted(hard_keys):
        hard_row = hard[hard["patient_id"] == patient_id].iloc[0]
        patient_features = features[features["patient_id"] == patient_id]
        patient_inversions = inversion_rows[(inversion_rows["positive_patient_id"] == patient_id) | (inversion_rows["negative_patient_id"] == patient_id)]
        role = str(hard_row["role"])
        row: Dict[str, Any] = {
            "patient_id": patient_id,
            "role": role,
            "is_top_k_hard": int(patient_id in top_keys),
            "inversion_count": hard_row["inversion_count"],
            "inversion_share": hard_row["inversion_share"],
            "n_seeds_with_inversion": hard_row["n_seeds_with_inversion"],
            "all_seed_hard_patient": hard_row["all_seed_hard_patient"],
            "inversion_image_opposed_rate": float(patient_inversions["image_opposed_flag"].mean()) if len(patient_inversions) else np.nan,
            "inversion_image_repair_rate": float(patient_inversions["image_repair_flag"].mean()) if len(patient_inversions) else np.nan,
            "inversion_text_driven_rate": float(patient_inversions["text_driven_flag"].mean()) if len(patient_inversions) else np.nan,
            "inversion_fusion_interaction_rate": float(patient_inversions["fusion_interaction_flag"].mean()) if len(patient_inversions) else np.nan,
            "mean_final_margin": float(patient_inversions["final_logit_margin"].mean()) if len(patient_inversions) else np.nan,
            "mean_margin_without_image": float(patient_inversions["margin_without_image"].mean()) if len(patient_inversions) else np.nan,
            "mean_margin_without_text": float(patient_inversions["margin_without_text"].mean()) if len(patient_inversions) else np.nan,
            "mean_margin_without_bio": float(patient_inversions["margin_without_bio"].mean()) if len(patient_inversions) else np.nan,
        }
        for column in (
            "label", "pred_prob", "logit", "text_classifier_contribution", "image_classifier_contribution", "bio_classifier_contribution", "text_embedding_norm", "image_embedding_norm", "bio_embedding_norm",
            "report_length", "selected_n_visits", "used_images", "has_bio", "bio_missing_count", "full_report_morphology_term_count", "full_report_diffuse_ht_term_count", "first256_morphology_term_count", "first256_diffuse_ht_term_count", "full_report_negative_term_count", "evidence_exposed_in_first256", "positive_negative_overlap",
        ):
            if column in patient_features.columns:
                row[f"mean_{column}"] = float(patient_features[column].mean())
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows).sort_values("inversion_count", ascending=False)
    profiles.to_csv(out_dir / "c14d_hard_patient_profiles.csv", index=False)

    role_profiles = profiles[["patient_id", "role", "is_top_k_hard", "inversion_count", "inversion_share", "inversion_image_opposed_rate", "inversion_image_repair_rate", "inversion_text_driven_rate", "inversion_fusion_interaction_rate"]].copy()
    role_profiles.to_csv(out_dir / "c14d_hard_patient_mechanism_summary.csv", index=False)

    feature_cols = [
        "pred_prob", "logit", "text_classifier_contribution", "image_classifier_contribution", "bio_classifier_contribution", "text_embedding_norm", "image_embedding_norm", "bio_embedding_norm",
        "report_length", "selected_n_visits", "used_images", "has_bio", "bio_missing_count", "full_report_morphology_term_count", "full_report_diffuse_ht_term_count", "first256_morphology_term_count", "first256_diffuse_ht_term_count", "full_report_negative_term_count", "evidence_exposed_in_first256", "positive_negative_overlap",
    ]
    cohort_rows: List[Dict[str, Any]] = []
    for role, label in (("positive", 1), ("negative", 0)):
        role_patients = set(features[features["label"] == label]["patient_id"])
        hard_role = set(profiles[profiles["role"] == role]["patient_id"])
        for cohort, patients in (("hard_all_seed", hard_role), ("nonhard_validation", role_patients - hard_role)):
            subset = features[features["patient_id"].isin(patients)]
            row: Dict[str, Any] = {"role": role, "cohort": cohort, "n_patients": len(patients), "n_rows": len(subset)}
            for column in feature_cols:
                if column in subset.columns:
                    row[f"mean_{column}"] = float(subset[column].mean()) if len(subset) else np.nan
                    row[f"std_{column}"] = float(subset[column].std()) if len(subset) > 1 else np.nan
            cohort_rows.append(row)
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(out_dir / "c14d_hard_vs_nonhard_summary.csv", index=False)

    top_profile = profiles.head(args.top_k)
    patient_inversion_incidence_total = max(2 * len(inversion_rows), 1)
    top_k_inversion_share = float(top_profile["inversion_count"].sum() / patient_inversion_incidence_total)
    mechanism_means = profiles.groupby("role", as_index=False).agg(
        n_patients=("patient_id", "nunique"),
        total_inversion_count=("inversion_count", "sum"),
        mean_inversion_share=("inversion_share", "mean"),
        mean_image_opposed_rate=("inversion_image_opposed_rate", "mean"),
        mean_image_repair_rate=("inversion_image_repair_rate", "mean"),
        mean_text_driven_rate=("inversion_text_driven_rate", "mean"),
        mean_fusion_interaction_rate=("inversion_fusion_interaction_rate", "mean"),
    )
    mechanism_means.to_csv(out_dir / "c14d_role_mechanism_summary.csv", index=False)

    route_value = str(route.iloc[0].get("route", "HARD_PATIENT_SUBGROUP_FAILURE")) if not route.empty else "HARD_PATIENT_SUBGROUP_FAILURE"
    next_step = "MORE_ANALYSIS_ONLY"
    gate = pd.DataFrame(
        [
            {
                "input_route": route_value,
                "c14d_route": "HARD_PATIENT_SUBGROUP_AUDIT_CONFIRMED",
                "next_step": next_step,
                "c15_authorized": 0,
                "training_started": 0,
                "top_k": args.top_k,
                "top_k_inversion_share": top_k_inversion_share,
                "all_seed_hard_patients": len(hard_keys),
                "notes": "Audit-only subgroup/profile comparison; no shortcut field entered a predictor or gate.",
            }
        ]
    )
    gate.to_csv(out_dir / "c14d_next_step_gate.csv", index=False)

    report_lines = [
        "# Phase C14-D Hard-Patient Subgroup Audit",
        "",
        "C14-D is analysis-only and follows the C14-C `HARD_PATIENT_SUBGROUP_FAILURE` stop. No training, threshold tuning, label/split/task/manifest changes, or classifier input changes were made.",
        "",
        "## Cohort Definition",
        "",
        f"- All-seed hard patients: `{len(hard_keys)}` patients with inversion rows in all three seeds.",
        f"- Top-{args.top_k} hard patient inversion incidence share: `{top_k_inversion_share:.4f}` (patient-side incidence denominator is `2 x inversion rows`).",
        "- C14-A exposure and C14-B representation fields are audit metadata only; they were not used as model inputs or learned gate inputs.",
        "",
        "## Top Hard Patients",
        "",
        frame_to_markdown(top_profile.head(args.top_k)),
        "",
        "## Hard Versus Non-Hard Validation Cohorts",
        "",
        frame_to_markdown(cohorts),
        "",
        "## Mechanism Profile",
        "",
        frame_to_markdown(mechanism_means),
        "",
        "The purpose of this comparison is to determine whether the inversion concentration is a small patient-specific subgroup pattern rather than evidence for a general image or fusion correction route.",
        "",
        "## Next-Step Gate",
        "",
        "`HARD_PATIENT_SUBGROUP_AUDIT_CONFIRMED`.",
        "",
        "C15 remains unauthorized. The next valid step is manual/clinical subgroup audit of the highest-impact positive and negative patients, followed by a new evidence-gated decision if needed.",
        "",
        "C13 remains the current strict best. No model improvement or AUC 0.90 claim is made.",
    ]
    (out_dir / "c14d_hard_patient_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "phase_c14d_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "c14d_route": "HARD_PATIENT_SUBGROUP_AUDIT_CONFIRMED", "c15_authorized": False, "all_seed_hard_patients": len(hard_keys), "top_k_inversion_share": top_k_inversion_share}, ensure_ascii=False))


if __name__ == "__main__":
    main()
