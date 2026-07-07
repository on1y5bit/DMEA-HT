from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


STRICT_MVP_VAL_AUC = 0.7581
STRICT_MVP_VAL_AUC_STD = 0.0171
STRICT_MVP_TEST_AUC = 0.7729
STRICT_MVP_TEST_AUC_STD = 0.0363
C1_INITIAL_VAL_AUC = 0.7782
C1_INITIAL_VAL_AUC_STD = 0.0350
C1_EXTENDED_VAL_AUC = 0.7718
C1_EXTENDED_VAL_AUC_STD = 0.0278
C1_EXTENDED_VAL_MEDIAN = 0.7868
C1_EXTENDED_VAL_MIN = 0.7379
C1_EXTENDED_VAL_MAX = 0.8040
C1_BADSEED_VAL_AUC = 0.7430
C1_GOODSEED_VAL_AUC = 0.7933
C1_BADSEED_GAP = 0.1430
C1_GOODSEED_GAP = 0.2168
C1_BADSEED_POS_DELTA = 0.2083
C1_BADSEED_NEG_DELTA = -0.1716
C1_BADSEED_RESIDUAL = 0.2291
C1_GOODSEED_RESIDUAL = 0.2744


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values: List[str] = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value))
            elif pd.isna(value):
                values.append("NA")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv(path: Path, required: bool, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    if not path.exists():
        input_rows.append(
            {
                "path": str(path),
                "status": "missing_required" if required else "missing_optional",
                "notes": "Report input was not found; constants and available files were used where possible.",
            }
        )
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
        return pd.DataFrame()
    input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows"})
    return frame


def read_text(path: Path, required: bool, input_rows: List[Dict[str, str]]) -> str:
    if not path.exists():
        input_rows.append(
            {
                "path": str(path),
                "status": "missing_required" if required else "missing_optional",
                "notes": "Report input was not found.",
            }
        )
        return ""
    text = path.read_text(encoding="utf-8")
    input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(text)} chars"})
    return text


def first_float(frame: pd.DataFrame, row_filter: pd.Series, columns: List[str]) -> float | None:
    if frame.empty:
        return None
    rows = frame[row_filter]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return float(row[column])
    return None


def find_by_name(frame: pd.DataFrame, name_columns: List[str], candidates: List[str]) -> pd.Series:
    if frame.empty:
        return pd.Series([False] * len(frame), index=frame.index)
    mask = pd.Series([False] * len(frame), index=frame.index)
    lowered = [candidate.lower() for candidate in candidates]
    for column in name_columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.lower()
        for candidate in lowered:
            mask = mask | values.eq(candidate) | values.str.contains(candidate, regex=False)
    return mask


def summarize_c4(c4: pd.DataFrame) -> Dict[str, Any]:
    if c4.empty or "val_auc" not in c4.columns:
        return {
            "auc_mean": C1_EXTENDED_VAL_AUC,
            "auc_std": C1_EXTENDED_VAL_AUC_STD,
            "auc_min": C1_EXTENDED_VAL_MIN,
            "auc_max": C1_EXTENDED_VAL_MAX,
            "status": "STABILITY_FAIL",
        }
    auc = pd.to_numeric(c4["val_auc"], errors="coerce").dropna()
    return {
        "auc_mean": float(auc.mean()) if not auc.empty else C1_EXTENDED_VAL_AUC,
        "auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else C1_EXTENDED_VAL_AUC_STD,
        "auc_min": float(auc.min()) if not auc.empty else C1_EXTENDED_VAL_MIN,
        "auc_max": float(auc.max()) if not auc.empty else C1_EXTENDED_VAL_MAX,
        "status": "STABILITY_FAIL",
    }


def summarize_c6(c6: pd.DataFrame, candidate: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if c6.empty or "candidate_name" not in c6.columns:
        return fallback
    rows = c6[c6["candidate_name"].astype(str).str.lower() == candidate.lower()]
    if rows.empty:
        return fallback
    row = rows.iloc[0].to_dict()
    return {
        "val_auc_mean": row.get("val_auc_mean", fallback["val_auc_mean"]),
        "val_auc_std": row.get("val_auc_std", fallback["val_auc_std"]),
        "val_auprc_mean": row.get("val_auprc_mean", fallback.get("val_auprc_mean")),
        "sensitivity": row.get("val_sensitivity_mean", fallback.get("sensitivity")),
        "specificity": row.get("val_specificity_mean", fallback.get("specificity")),
        "gap": row.get("val_pos_neg_gap_mean", fallback.get("gap")),
        "residual": row.get("max_abs_prediction_shortcut_residual_spearman", fallback.get("residual")),
        "decision": row.get("stabilization_decision", fallback["decision"]),
        "reason": row.get("failure_reason", fallback["reason"]),
    }


def aggregate_delta(delta: pd.DataFrame, seed_group: str, label: int) -> float | None:
    if delta.empty or "abs_error_delta" not in delta.columns:
        return None
    frame = delta.copy()
    if "seed_group" in frame.columns:
        frame = frame[frame["seed_group"].astype(str) == seed_group]
    if "label" in frame.columns:
        frame = frame[frame["label"].astype(int) == int(label)]
    values = pd.to_numeric(frame["abs_error_delta"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def c6_positive_delta(c6_positive: pd.DataFrame, candidate: str, label: int) -> float | None:
    if c6_positive.empty or "candidate_name" not in c6_positive.columns:
        return None
    rows = c6_positive[
        (c6_positive["candidate_name"].astype(str) == candidate)
        & (c6_positive["label"].astype(int) == int(label))
    ]
    if rows.empty or "mean_abs_error_delta_vs_mvp" not in rows.columns:
        return None
    values = pd.to_numeric(rows["mean_abs_error_delta_vs_mvp"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def build_main_path_summary(c3: pd.DataFrame, c4: pd.DataFrame, c6: pd.DataFrame) -> pd.DataFrame:
    c4_summary = summarize_c4(c4)
    c6_defaults = {
        "delay_w001_start5": {
            "val_auc_mean": 0.7450,
            "val_auc_std": 0.0070,
            "val_auprc_mean": 0.7254,
            "sensitivity": 0.6028,
            "specificity": 0.7234,
            "gap": 0.1548,
            "residual": 0.2257,
            "decision": "STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS",
            "reason": "mean validation AUC is not close to strict MVP reference",
        },
        "w001": {
            "val_auc_mean": 0.7438,
            "val_auc_std": 0.0090,
            "val_auprc_mean": 0.7282,
            "sensitivity": 0.5816,
            "specificity": 0.7518,
            "gap": 0.1576,
            "residual": 0.2225,
            "decision": "STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS",
            "reason": "mean validation AUC is not close to strict MVP reference",
        },
        "w0005": {
            "val_auc_mean": 0.7421,
            "val_auc_std": 0.0135,
            "val_auprc_mean": 0.7288,
            "sensitivity": 0.6738,
            "specificity": 0.6525,
            "gap": 0.1531,
            "residual": 0.1983,
            "decision": "STABILIZATION_FAIL",
            "reason": "mean validation AUC does not beat original bad-seed C1; mean validation AUC is not close to strict MVP reference",
        },
    }
    c6_delay = summarize_c6(c6, "delay_w001_start5", c6_defaults["delay_w001_start5"])
    c6_w001 = summarize_c6(c6, "w001", c6_defaults["w001"])
    c6_w0005 = summarize_c6(c6, "w0005", c6_defaults["w0005"])

    def c3_auc(names: List[str], default: float | None) -> float | None:
        mask = find_by_name(c3, ["model_id", "candidate_name"], names)
        return first_float(c3, mask, ["val_auc_mean", "validation_auc_mean", "AUC_mean"]) or default

    def c3_std(names: List[str], default: float | None) -> float | None:
        mask = find_by_name(c3, ["model_id", "candidate_name"], names)
        return first_float(c3, mask, ["val_auc_std", "validation_auc_std", "AUC_std"]) or default

    rows = [
        {
            "candidate_name": "Strict MVP",
            "phase": "MVP",
            "status": "current_main_path / stable_reference",
            "validation_auc_mean": c3_auc(["strict_mvp"], STRICT_MVP_VAL_AUC),
            "validation_auc_std": c3_std(["strict_mvp"], STRICT_MVP_VAL_AUC_STD),
            "stability_status": "stable_reference",
            "positive_preservation_status": "reference",
            "shortcut_residual_status": "structural matched reference",
            "promotion_decision": "current_main_path",
            "reason": "Strict structural matched DMEA-MVP remains the stable validation reference; test AUC is reporting-only.",
        },
        {
            "candidate_name": "C1 text morphology only",
            "phase": "C1/C4/C5",
            "status": "ablation_only_unstable",
            "validation_auc_mean": c4_summary["auc_mean"],
            "validation_auc_std": c4_summary["auc_std"],
            "stability_status": "STABILITY_FAIL",
            "positive_preservation_status": "fails_positive_preservation",
            "shortcut_residual_status": "not_primary_failure_signal",
            "promotion_decision": "not_promoted",
            "reason": "Extended seeds fell below strict MVP on multiple bad seeds, and C5 showed positive-label harm.",
        },
        {
            "candidate_name": "C1 text + image evidence",
            "phase": "C1/C3",
            "status": "failed_ablation",
            "validation_auc_mean": c3_auc(["c1_text_image_evidence", "text_image"], None),
            "validation_auc_std": c3_std(["c1_text_image_evidence", "text_image"], None),
            "stability_status": "not_promoted_in_c3",
            "positive_preservation_status": "not_established",
            "shortcut_residual_status": "not_promoted",
            "promotion_decision": "not_promoted",
            "reason": "Text plus image evidence did not become the stable main candidate and image morphology BCE remains disabled for future use.",
        },
        {
            "candidate_name": "C2 text anchor w=0.05",
            "phase": "C2/C3",
            "status": "failed_ablation",
            "validation_auc_mean": c3_auc(["c2_text_anchor_w0.05", "w0.05"], None),
            "validation_auc_std": c3_std(["c2_text_anchor_w0.05", "w0.05"], None),
            "stability_status": "not_promoted_in_c3",
            "positive_preservation_status": "not_established",
            "shortcut_residual_status": "not_promoted",
            "promotion_decision": "not_promoted",
            "reason": "Text anchor did not rescue the evidence branch or supersede the strict MVP reference.",
        },
        {
            "candidate_name": "C6 delay_w001_start5",
            "phase": "C6",
            "status": "partial_no_formal",
            "validation_auc_mean": c6_delay["val_auc_mean"],
            "validation_auc_std": c6_delay["val_auc_std"],
            "stability_status": c6_delay["decision"],
            "positive_preservation_status": "fails_positive_preservation",
            "shortcut_residual_status": "not_alarming",
            "promotion_decision": "not_promoted",
            "reason": c6_delay["reason"],
        },
        {
            "candidate_name": "C6 w001",
            "phase": "C6",
            "status": "partial_no_formal",
            "validation_auc_mean": c6_w001["val_auc_mean"],
            "validation_auc_std": c6_w001["val_auc_std"],
            "stability_status": c6_w001["decision"],
            "positive_preservation_status": "fails_positive_preservation",
            "shortcut_residual_status": "not_alarming",
            "promotion_decision": "not_promoted",
            "reason": c6_w001["reason"],
        },
        {
            "candidate_name": "C6 w0005",
            "phase": "C6",
            "status": "failed_no_formal",
            "validation_auc_mean": c6_w0005["val_auc_mean"],
            "validation_auc_std": c6_w0005["val_auc_std"],
            "stability_status": c6_w0005["decision"],
            "positive_preservation_status": "less_positive_harm_but_auc_gate_failed",
            "shortcut_residual_status": "not_alarming",
            "promotion_decision": "not_promoted",
            "reason": c6_w0005["reason"],
        },
    ]
    return pd.DataFrame(rows)


def build_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "C1",
                "candidate_or_action": "text morphology BCE",
                "initial_observation": "Appeared to improve validation AUC.",
                "validation_result": "Initial validation AUC 0.7782 +/- 0.0350.",
                "stability_result": "Not yet established.",
                "failure_signal": "Single-stage result later failed extended-seed checks.",
                "decision": "Require stability checking.",
            },
            {
                "phase": "C2",
                "candidate_or_action": "text evidence anchor",
                "initial_observation": "Attempted to rescue evidence alignment without direct stronger BCE.",
                "validation_result": "Did not beat or replace C1/strict MVP as a stable path.",
                "stability_result": "Not promoted.",
                "failure_signal": "No reliable improvement over strict MVP reference.",
                "decision": "Failed ablation.",
            },
            {
                "phase": "C3",
                "candidate_or_action": "model comparison decision gate",
                "initial_observation": "C1 remained a tempting candidate by initial validation AUC.",
                "validation_result": "C1 was kept only as a candidate requiring stability checks.",
                "stability_result": "Stability unresolved.",
                "failure_signal": "Promotion depended on validation and later stability, not test metrics.",
                "decision": "Run extended-seed stability checking.",
            },
            {
                "phase": "C4",
                "candidate_or_action": "C1 extended seed",
                "initial_observation": "Check whether C1 survives seed expansion.",
                "validation_result": "Mean 0.7718, std 0.0278, min/max 0.7379 / 0.8040.",
                "stability_result": "STABILITY_FAIL.",
                "failure_signal": "Seeds 1, 3, and 42 were below the strict MVP reference.",
                "decision": "Do not promote C1 based on single good seeds.",
            },
            {
                "phase": "C5",
                "candidate_or_action": "failure diagnosis",
                "initial_observation": "Bad seeds underperform with compressed probability separation.",
                "validation_result": "Good seed AUC 0.7933 vs bad seed AUC 0.7430.",
                "stability_result": "Failure localized to bad seeds.",
                "failure_signal": "Bad seeds helped negatives but harmed positives; residual shortcut signal was not the primary cause.",
                "decision": "Optimization stabilization only if any follow-up is approved.",
            },
            {
                "phase": "C6",
                "candidate_or_action": "low-weight and delayed BCE stabilization pilots",
                "initial_observation": "Try lighter/delayed text morphology BCE on bad seeds.",
                "validation_result": "Best partial AUC 0.7450 +/- 0.0070.",
                "stability_result": "No candidate reached PASS.",
                "failure_signal": "Positive preservation problem remained and strict MVP reference was not reached.",
                "decision": "No formal evaluation justified.",
            },
            {
                "phase": "C7",
                "candidate_or_action": "route correction",
                "initial_observation": "Consolidate C1-C6 evidence.",
                "validation_result": "Strict MVP remains the stable reference.",
                "stability_result": "Evidence BCE branch is unstable.",
                "failure_signal": "Direct weak text morphology BCE systematically risks positive-probability suppression.",
                "decision": "Demote evidence BCE branch to ablation-only.",
            },
        ]
    )


def build_ablation_status() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "branch": "strict_mvp",
                "last_phase_evaluated": "C3/C7",
                "best_validation_auc": STRICT_MVP_VAL_AUC,
                "best_decision": "current_main_path / stable_reference",
                "allowed_future_use": "main/reference",
                "forbidden_future_use": "none",
                "notes": "Use validation metrics for selection; test metrics remain reporting-only.",
            },
            {
                "branch": "text_morphology_bce",
                "last_phase_evaluated": "C6",
                "best_validation_auc": C1_EXTENDED_VAL_MAX,
                "best_decision": "ablation_only_unstable",
                "allowed_future_use": "ablation/report-only",
                "forbidden_future_use": "main-path promotion or formal training without a new positive-preserving formulation and gate",
                "notes": "C4 stability failed and C5/C6 showed positive-preservation concerns.",
            },
            {
                "branch": "text_image_morphology_bce",
                "last_phase_evaluated": "C3",
                "best_validation_auc": pd.NA,
                "best_decision": "failed_ablation",
                "allowed_future_use": "ablation/report-only",
                "forbidden_future_use": "direct BCE training target in the main path",
                "notes": "Do not enable image morphology BCE from this branch.",
            },
            {
                "branch": "text_evidence_anchor",
                "last_phase_evaluated": "C2/C3",
                "best_validation_auc": pd.NA,
                "best_decision": "failed_ablation",
                "allowed_future_use": "ablation/report-only",
                "forbidden_future_use": "promotion based on anchor variants that failed C3",
                "notes": "C2 did not rescue the evidence branch.",
            },
            {
                "branch": "c6_delayed_text_morphology",
                "last_phase_evaluated": "C6",
                "best_validation_auc": 0.7450,
                "best_decision": "partial_no_formal",
                "allowed_future_use": "diagnostic-only unless new positive-preservation gate is approved",
                "forbidden_future_use": "formal evaluation from current C6 candidate",
                "notes": "Best C6 candidate by validation AUC but still below strict MVP reference.",
            },
            {
                "branch": "c6_low_weight_text_morphology",
                "last_phase_evaluated": "C6",
                "best_validation_auc": 0.7438,
                "best_decision": "failed_or_partial_no_formal",
                "allowed_future_use": "diagnostic-only unless new positive-preservation gate is approved",
                "forbidden_future_use": "formal evaluation from current low-weight candidates",
                "notes": "Low-weight candidates did not pass the C6 stabilization gate.",
            },
        ]
    )


def build_positive_summary(delta: pd.DataFrame, c6_positive: pd.DataFrame, c6_summary: pd.DataFrame) -> pd.DataFrame:
    def c6_metric(candidate: str, column: str) -> float | None:
        if c6_summary.empty or "candidate_name" not in c6_summary.columns:
            return None
        rows = c6_summary[c6_summary["candidate_name"].astype(str) == candidate]
        if rows.empty or column not in rows.columns:
            return None
        return float(rows.iloc[0][column])

    rows = [
        {
            "candidate": "C1 text morphology only bad seeds",
            "positive_abs_error_delta_vs_mvp": aggregate_delta(delta, "bad", 1) or C1_BADSEED_POS_DELTA,
            "negative_abs_error_delta_vs_mvp": aggregate_delta(delta, "bad", 0) or C1_BADSEED_NEG_DELTA,
            "positive_negative_gap": C1_BADSEED_GAP,
            "sensitivity": pd.NA,
            "specificity": pd.NA,
            "positive_preservation_decision": "FAIL: helps negatives but substantially harms positives.",
        },
        {
            "candidate": "C1 text morphology only good seeds",
            "positive_abs_error_delta_vs_mvp": aggregate_delta(delta, "good", 1),
            "negative_abs_error_delta_vs_mvp": aggregate_delta(delta, "good", 0),
            "positive_negative_gap": C1_GOODSEED_GAP,
            "sensitivity": pd.NA,
            "specificity": pd.NA,
            "positive_preservation_decision": "WARNING: better AUC seeds still show positive harm in C5.",
        },
    ]
    for candidate, label in [
        ("delay_w001_start5", "C6 delay_w001_start5"),
        ("w001", "C6 w001"),
        ("w0005", "C6 w0005"),
    ]:
        rows.append(
            {
                "candidate": label,
                "positive_abs_error_delta_vs_mvp": c6_positive_delta(c6_positive, candidate, 1),
                "negative_abs_error_delta_vs_mvp": c6_positive_delta(c6_positive, candidate, 0),
                "positive_negative_gap": c6_metric(candidate, "val_pos_neg_gap_mean"),
                "sensitivity": c6_metric(candidate, "val_sensitivity_mean"),
                "specificity": c6_metric(candidate, "val_specificity_mean"),
                "positive_preservation_decision": "FAIL: no C6 candidate passed the positive-preservation/formal gate.",
            }
        )
    return pd.DataFrame(rows)


def discover_c6_csvs(phase_c6: Path, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not phase_c6.exists():
        input_rows.append({"path": str(phase_c6), "status": "missing_optional", "notes": "No Phase C6 directory found."})
        return pd.DataFrame()
    for path in sorted(phase_c6.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
            continue
        input_rows.append({"path": str(path), "status": "loaded_appendix", "notes": f"{len(frame)} rows"})
        for _, row in frame.head(200).iterrows():
            record = {"source_file": path.name}
            record.update(row.to_dict())
            rows.append(record)
    return pd.DataFrame(rows)


def write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C7 route-correction reports.")
    parser.add_argument("--project-root", default=".", help="DMEA-HT project root. Defaults to current directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to analysis_reports/phase_c7.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "analysis_reports" / "phase_c7"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, str]] = []

    phase_c3 = root / "analysis_reports" / "phase_c3"
    phase_c4 = root / "analysis_reports" / "phase_c4"
    phase_c5 = root / "analysis_reports" / "phase_c5"
    phase_c6 = root / "analysis_reports" / "phase_c6"

    c3_table = read_csv(phase_c3 / "model_comparison_table.csv", False, input_rows)
    read_csv(phase_c3 / "decision_gate_summary.csv", False, input_rows)
    c4_extended = read_csv(phase_c4 / "c1_extended_seed_summary.csv", False, input_rows)
    read_csv(phase_c4 / "c1_weight_pilot_summary.csv", False, input_rows)
    c5_summary = read_csv(phase_c5 / "c1_seed_failure_summary.csv", False, input_rows)
    c5_delta = read_csv(phase_c5 / "c1_vs_mvp_patient_delta_val.csv", False, input_rows)
    c5_residual = read_csv(phase_c5 / "c1_seed_shortcut_residual.csv", False, input_rows)
    c6_summary = read_csv(phase_c6 / "c6_badseed_pilot_summary.csv", False, input_rows)
    c6_positive = read_csv(phase_c6 / "c6_positive_preservation.csv", False, input_rows)
    read_text(phase_c6 / "c6_final_report.md", False, input_rows)
    c6_appendix = discover_c6_csvs(phase_c6, input_rows)

    main_summary = build_main_path_summary(c3_table, c4_extended, c6_summary)
    timeline = build_timeline()
    ablation_status = build_ablation_status()
    positive_summary = build_positive_summary(c5_delta, c6_positive, c6_summary)
    input_frame = pd.DataFrame(input_rows)

    main_summary.to_csv(out_dir / "main_path_decision_summary.csv", index=False)
    timeline.to_csv(out_dir / "evidence_bce_failure_timeline.csv", index=False)
    ablation_status.to_csv(out_dir / "ablation_status_table.csv", index=False)
    positive_summary.to_csv(out_dir / "positive_preservation_summary.csv", index=False)
    input_frame.to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    if not c6_appendix.empty:
        c6_appendix.to_csv(out_dir / "phase_c6_csv_appendix.csv", index=False)

    c5_good_auc = C1_GOODSEED_VAL_AUC
    c5_bad_auc = C1_BADSEED_VAL_AUC
    if not c5_summary.empty and {"seed_group", "val_auc"}.issubset(c5_summary.columns):
        good_auc = pd.to_numeric(c5_summary[c5_summary["seed_group"] == "good"]["val_auc"], errors="coerce").mean()
        bad_auc = pd.to_numeric(c5_summary[c5_summary["seed_group"] == "bad"]["val_auc"], errors="coerce").mean()
        c5_good_auc = float(good_auc) if not pd.isna(good_auc) else c5_good_auc
        c5_bad_auc = float(bad_auc) if not pd.isna(bad_auc) else c5_bad_auc

    c5_bad_residual = C1_BADSEED_RESIDUAL
    if not c5_residual.empty and {"seed_group", "abs_spearman"}.issubset(c5_residual.columns):
        bad_res = pd.to_numeric(c5_residual[c5_residual["seed_group"] == "bad"]["abs_spearman"], errors="coerce").max()
        c5_bad_residual = float(bad_res) if not pd.isna(bad_res) else c5_bad_residual

    write_text(
        out_dir / "evidence_bce_failure_report.md",
        [
            "# Evidence BCE Failure Report",
            "",
            "## Why C1 Initially Looked Promising",
            "",
            f"C1 text morphology BCE initially reported validation AUC {C1_INITIAL_VAL_AUC:.4f} +/- {C1_INITIAL_VAL_AUC_STD:.4f}, above the strict MVP validation reference {STRICT_MVP_VAL_AUC:.4f} +/- {STRICT_MVP_VAL_AUC_STD:.4f}. This made it a reasonable candidate for follow-up, but only under a validation/stability gate.",
            "",
            "## Why C1 Cannot Remain The Main Candidate",
            "",
            f"Phase C4 expanded the seed set and found mean validation AUC {C1_EXTENDED_VAL_AUC:.4f}, std {C1_EXTENDED_VAL_AUC_STD:.4f}, median {C1_EXTENDED_VAL_MEDIAN:.4f}, and min/max {C1_EXTENDED_VAL_MIN:.4f} / {C1_EXTENDED_VAL_MAX:.4f}. Multiple bad seeds fell below the strict MVP reference, so the branch failed stability checking.",
            "",
            "## Why C2 Does Not Rescue It",
            "",
            "C2 text-anchor variants did not replace C1 or strict MVP as a stable validation path. They remain failed ablations and should not be used to re-promote weak evidence supervision.",
            "",
            "## Why C6 Does Not Rescue It",
            "",
            "C6 tested lower and delayed text morphology BCE on bad seeds only. The best candidate, delay_w001_start5, reached validation AUC 0.7450 +/- 0.0070, below the strict MVP reference. No C6 candidate reached STABILIZATION_PASS_RECOMMEND_FORMAL.",
            "",
            "## Shortcut Residual Interpretation",
            "",
            f"C5 found bad-seed max shortcut residual Spearman around {c5_bad_residual:.4f}, while the prior good-seed reference was around {C1_GOODSEED_RESIDUAL:.4f}. This does not support residual selected-structure shortcut coupling as the primary failure cause.",
            "",
            "## Likely Failure Mode",
            "",
            f"C5 localized the failure to optimization/checkpoint instability and positive-probability suppression: good seed mean validation AUC {c5_good_auc:.4f} versus bad seed mean validation AUC {c5_bad_auc:.4f}. Bad seeds helped negatives relative to MVP but harmed positive-label patients.",
            "",
            "## Future Evidence Use",
            "",
            "Evidence labels should be used as analysis variables, explanation metadata, stratification fields, and patient-level evidence reporting aids. They should not be used as a direct BCE training target unless a new positive-preserving formulation is explicitly justified and pilot-gated.",
        ],
    )

    write_text(
        out_dir / "decision_gate_update.md",
        [
            "# Phase C7 Decision Gate Update",
            "",
            "Any candidate derived from weak evidence supervision must pass a positive-preservation gate before formal evaluation.",
            "",
            "Positive-preservation gate:",
            "",
            "1. It must not reduce positive-label predicted probabilities relative to strict MVP in a systematic way.",
            "2. It must not improve negative-label errors while substantially worsening positive-label errors.",
            "3. It must preserve or improve positive-negative prediction gap.",
            "4. It must not reduce validation sensitivity without a compensating validation AUC/AUPRC gain.",
            "5. It must pass bad-seed pilot evaluation before any formal three-seed run.",
            "",
            "Additional rules:",
            "",
            "- No candidate that fails extended-seed stability may be promoted based on a single good seed.",
            "- No candidate may enter formal evaluation if it is below strict MVP reference on bad-seed pilots.",
            "- No test metric may override validation or stability failure.",
        ],
    )

    write_text(
        out_dir / "phase_c7_final_report.md",
        [
            "# Phase C7 Final Report",
            "",
            "## Objective",
            "",
            "Phase C7 is a decision-consolidation and route-correction phase. It does not train a new model and does not promote a new candidate.",
            "",
            "## Inputs Used",
            "",
            frame_to_markdown(input_frame),
            "",
            "## Main Decision",
            "",
            "Current main path is `strict_structural_matched_dmea_mvp`. The weak text morphology BCE branch is demoted to ablation-only / unstable.",
            "",
            "## Why C1 Was Demoted",
            "",
            f"C1 initially looked promising at validation AUC {C1_INITIAL_VAL_AUC:.4f} +/- {C1_INITIAL_VAL_AUC_STD:.4f}, but Phase C4 extended-seed validation failed with mean {C1_EXTENDED_VAL_AUC:.4f}, std {C1_EXTENDED_VAL_AUC_STD:.4f}, and min/max {C1_EXTENDED_VAL_MIN:.4f} / {C1_EXTENDED_VAL_MAX:.4f}. Phase C5 then showed that bad seeds compressed positive-negative separation and harmed positive-label patients relative to MVP.",
            "",
            "## Why C2/C6 Were Not Promoted",
            "",
            "C2 text-anchor variants did not rescue the evidence branch. C6 low-weight and delayed text morphology BCE pilots produced no PASS candidate; the best partial C6 validation AUC was 0.7450 +/- 0.0070, still below the strict MVP reference.",
            "",
            "## Current Main Path",
            "",
            f"Strict structural matched DMEA-MVP remains the stable reference with validation AUC {STRICT_MVP_VAL_AUC:.4f} +/- {STRICT_MVP_VAL_AUC_STD:.4f}. Test AUC {STRICT_MVP_TEST_AUC:.4f} +/- {STRICT_MVP_TEST_AUC_STD:.4f} is reporting-only.",
            "",
            "## Updated Decision Gate",
            "",
            "Any future weak-evidence-supervised candidate must pass the positive-preservation gate and bad-seed pilot before formal evaluation. Validation and stability failures cannot be overridden by test metrics.",
            "",
            "## What Not To Do Next",
            "",
            "- Do not continue optimizing text morphology BCE as the main path.",
            "- Do not enable image morphology BCE, counterfactual loss, matched SupCon, new anchor-fusion losses, or other disabled evidence losses from this failed branch.",
            "- Do not launch formal training from C6 candidates.",
            "- Do not promote any model based on a single good seed.",
            "",
            "## Recommended Next Direction",
            "",
            "Recommended next direction: keep strict MVP as the current main path and use evidence labels only for diagnostics/explanation unless a new positive-preserving alignment formulation is separately proposed and pilot-gated.",
            "",
            "## Decision Tables",
            "",
            "### Main Path Decision Summary",
            "",
            frame_to_markdown(main_summary),
            "",
            "### Ablation Status Table",
            "",
            frame_to_markdown(ablation_status),
            "",
            "### Positive Preservation Summary",
            "",
            frame_to_markdown(positive_summary),
        ],
    )
    print(f"Wrote Phase C7 reports to {out_dir}")


if __name__ == "__main__":
    main()
