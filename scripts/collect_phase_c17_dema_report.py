#!/usr/bin/env python3
"""Collect the C17 validation-only residual audits and final decision report."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SHORTCUT_FIELDS = (
    "selected_n_visits",
    "used_images",
    "image_padding_count",
    "has_bio",
    "bio_missing_count",
    "report_length",
)
PATIENT_COLUMNS = (
    "seed",
    "patient_id",
    "label",
    "base_logit",
    "base_prob",
    "delta_logit",
    "final_logit",
    "final_prob",
    "patient_support_strength",
    "patient_opposition_strength",
    "patient_uncertainty_strength",
    "patient_conflict_score",
    "image_evidence_weight",
    "text_evidence_weight",
    "bio_evidence_weight",
    "morphology_alignment_cosine",
    "text_temporal_conflict_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-run-dir", required=True)
    parser.add_argument("--rank-run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c17_dema")
    parser.add_argument("--selected-route", choices=("auto", "DEMA-R", "DEMA-RP"), default="auto")
    parser.add_argument("--require-pilot-pass", action="store_true")
    return parser.parse_args()


def drop_forbidden_metric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["AUPRC"], errors="ignore")


def read_metrics(run_dir: Path, route: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_seed.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = drop_forbidden_metric(pd.read_csv(path))
    frame["route"] = route
    return frame


def read_epochs(run_dir: Path, route: str) -> pd.DataFrame:
    path = run_dir / "reports" / "metrics_by_epoch.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = drop_forbidden_metric(pd.read_csv(path))
    frame["route"] = route
    return frame


def read_predictions(run_dir: Path, route: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((run_dir / "predictions").glob("val_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            digits = "".join(character for character in path.stem if character.isdigit())
            frame["seed"] = int(digits) if digits else 0
        frame["route"] = route
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if array.size else float("nan")


def std_or_zero(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def select_route(core_metrics: pd.DataFrame, rank_metrics: pd.DataFrame, requested: str) -> str:
    if requested in {"DEMA-R", "DEMA-RP"}:
        return requested
    scores = {}
    for route, frame in (("DEMA-R", core_metrics), ("DEMA-RP", rank_metrics)):
        values = pd.to_numeric(frame.loc[frame["split"] == "val", "AUC"], errors="coerce") if not frame.empty and "split" in frame and "AUC" in frame else pd.Series(dtype=float)
        scores[route] = float(values.mean()) if not values.empty else -math.inf
    return max(scores, key=scores.get) if any(np.isfinite(value) for value in scores.values()) else "DEMA-R"


def patient_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=PATIENT_COLUMNS)
    result = pd.DataFrame(index=frame.index)
    for column in PATIENT_COLUMNS:
        if column in frame:
            result[column] = frame[column]
        else:
            result[column] = np.nan
    return result


def positive_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["route", "seed", "group", "n", "mean_probability_delta", "std_probability_delta", "mean_logit_delta", "std_logit_delta"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    prob_delta = finite_series(frame, "final_prob") - finite_series(frame, "base_prob")
    logit_delta = finite_series(frame, "final_logit") - finite_series(frame, "base_logit")
    labels = finite_series(frame, "label")
    rows: List[Dict[str, Any]] = []
    for group_name, mask in (("positive", labels == 1), ("negative", labels == 0)):
        p = prob_delta[mask].dropna().to_numpy(dtype=float)
        l = logit_delta[mask].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "route": frame["route"].iloc[0],
                "seed": int(pd.to_numeric(frame["seed"].iloc[0], errors="coerce")),
                "group": group_name,
                "n": int(mask.sum()),
                "mean_probability_delta": float(p.mean()) if p.size else 0.0,
                "std_probability_delta": float(p.std(ddof=1)) if p.size > 1 else 0.0,
                "mean_logit_delta": float(l.mean()) if l.size else 0.0,
                "std_logit_delta": float(l.std(ddof=1)) if l.size > 1 else 0.0,
            }
        )
    base_pred = finite_series(frame, "base_prob") >= 0.5
    final_pred = finite_series(frame, "final_prob") >= 0.5
    for transition, mask in {
        "FN_to_TP": (labels == 1) & (~base_pred) & final_pred,
        "TP_to_FN": (labels == 1) & base_pred & (~final_pred),
        "FP_to_TN": (labels == 0) & base_pred & (~final_pred),
        "TN_to_FP": (labels == 0) & (~base_pred) & final_pred,
    }.items():
        rows.append(
            {
                "route": frame["route"].iloc[0],
                "seed": int(pd.to_numeric(frame["seed"].iloc[0], errors="coerce")),
                "group": transition,
                "n": int(mask.sum()),
                "mean_probability_delta": float(prob_delta[mask].mean()) if bool(mask.any()) else 0.0,
                "std_probability_delta": float(prob_delta[mask].std(ddof=1)) if int(mask.sum()) > 1 else 0.0,
                "mean_logit_delta": float(logit_delta[mask].mean()) if bool(mask.any()) else 0.0,
                "std_logit_delta": float(logit_delta[mask].std(ddof=1)) if int(mask.sum()) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def pairwise_audit(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_columns = ["route", "seed", "positive_patient_id", "negative_patient_id", "base_margin", "final_margin", "base_inversion", "final_inversion", "margin_delta"]
    summary_columns = ["route", "seed", "base_inversions", "final_inversions", "repaired_inversions", "introduced_inversions", "net_inversion_change"]
    pairs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=pair_columns), pd.DataFrame(columns=summary_columns)
    for seed, seed_frame in frame.groupby("seed", dropna=False):
        positives = seed_frame[finite_series(seed_frame, "label") == 1]
        negatives = seed_frame[finite_series(seed_frame, "label") == 0]
        for _, positive in positives.iterrows():
            for _, negative in negatives.iterrows():
                base_margin = float(positive["base_logit"] - negative["base_logit"])
                final_margin = float(positive["final_logit"] - negative["final_logit"])
                pairs.append(
                    {
                        "route": frame["route"].iloc[0],
                        "seed": int(seed),
                        "positive_patient_id": positive["patient_id"],
                        "negative_patient_id": negative["patient_id"],
                        "base_margin": base_margin,
                        "final_margin": final_margin,
                        "base_inversion": int(base_margin <= 0),
                        "final_inversion": int(final_margin <= 0),
                        "margin_delta": final_margin - base_margin,
                    }
                )
        pair_frame = pd.DataFrame([row for row in pairs if row["seed"] == int(seed)], columns=pair_columns)
        base_inv = int(pair_frame["base_inversion"].sum()) if not pair_frame.empty else 0
        final_inv = int(pair_frame["final_inversion"].sum()) if not pair_frame.empty else 0
        repaired = int(((pair_frame["base_inversion"] == 1) & (pair_frame["final_inversion"] == 0)).sum()) if not pair_frame.empty else 0
        introduced = int(((pair_frame["base_inversion"] == 0) & (pair_frame["final_inversion"] == 1)).sum()) if not pair_frame.empty else 0
        summaries.append(
            {
                "route": frame["route"].iloc[0],
                "seed": int(seed),
                "base_inversions": base_inv,
                "final_inversions": final_inv,
                "repaired_inversions": repaired,
                "introduced_inversions": introduced,
                "net_inversion_change": final_inv - base_inv,
            }
        )
    return pd.DataFrame(pairs, columns=pair_columns), pd.DataFrame(summaries, columns=summary_columns)


def mechanism_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["route", "seed", "feature", "stratum", "n", "mean_delta_logit", "mean_probability_delta", "fraction_delta_negative"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    delta = finite_series(frame, "delta_logit")
    prob_delta = finite_series(frame, "final_prob") - finite_series(frame, "base_prob")
    rows: List[Dict[str, Any]] = []
    features = {
        "high_support": "patient_support_strength",
        "high_opposition": "patient_opposition_strength",
        "high_uncertainty": "patient_uncertainty_strength",
        "high_conflict": "patient_conflict_score",
        "high_temporal_conflict": "text_temporal_conflict_score",
        "high_morphology_alignment": "morphology_alignment_cosine",
    }
    for name, column in features.items():
        values = finite_series(frame, column)
        valid = values.notna()
        if int(valid.sum()) < 2:
            continue
        threshold = float(values[valid].median())
        for stratum, mask in (("high", valid & (values >= threshold)), ("low", valid & (values < threshold))):
            d = delta[mask].dropna().to_numpy(dtype=float)
            p = prob_delta[mask].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "route": frame["route"].iloc[0],
                    "seed": int(pd.to_numeric(frame["seed"].iloc[0], errors="coerce")),
                    "feature": name,
                    "stratum": stratum,
                    "n": int(mask.sum()),
                    "mean_delta_logit": float(d.mean()) if d.size else 0.0,
                    "mean_probability_delta": float(p.mean()) if p.size else 0.0,
                    "fraction_delta_negative": float((d < 0).mean()) if d.size else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def shortcut_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["route", "seed", "shortcut_field", "n", "shortcut_label_auc", "abs_spearman_with_final_prob", "linear_r2_label"]
    rows: List[Dict[str, Any]] = []
    labels = finite_series(frame, "label")
    final_prob = finite_series(frame, "final_prob")
    for field in SHORTCUT_FIELDS:
        values = finite_series(frame, field)
        valid = values.notna() & labels.notna() & final_prob.notna()
        x = values[valid].to_numpy(dtype=float)
        y = labels[valid].to_numpy(dtype=float)
        p = final_prob[valid].to_numpy(dtype=float)
        auc = 0.5
        if x.size and len(np.unique(y)) > 1 and len(np.unique(x)) > 1:
            from sklearn.metrics import roc_auc_score

            auc = max(float(roc_auc_score(y, x)), float(roc_auc_score(y, -x)))
        if x.size > 1 and np.std(x) > 0 and np.std(p) > 0:
            spearman = float(pd.Series(x).rank().corr(pd.Series(p).rank()))
        else:
            spearman = 0.0
        if x.size > 1 and np.std(y) > 0 and np.std(x) > 0:
            design = np.column_stack([np.ones(x.size), x])
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            prediction = design @ beta
            r2 = 1.0 - float(((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        else:
            r2 = 0.0
        rows.append(
            {
                "route": frame["route"].iloc[0] if not frame.empty else "DEMA-R",
                "seed": int(pd.to_numeric(frame["seed"].iloc[0], errors="coerce")) if not frame.empty else 0,
                "shortcut_field": field,
                "n": int(x.size),
                "shortcut_label_auc": auc,
                "abs_spearman_with_final_prob": abs(spearman) if np.isfinite(spearman) else 0.0,
                "linear_r2_label": r2,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def route_summary(metrics: pd.DataFrame, pair_summary: pd.DataFrame, route: str) -> Dict[str, float]:
    if metrics.empty:
        return {"route": route, "n_seeds": 0}
    val = metrics[metrics["split"] == "val"] if "split" in metrics else metrics
    out: Dict[str, float] = {"route": route, "n_seeds": int(val["seed"].nunique()) if "seed" in val else 0}
    for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "pos_neg_gap", "mean_positive_delta_logit", "std_delta_logit"):
        if key in val:
            values = pd.to_numeric(val[key], errors="coerce").dropna().to_numpy(dtype=float)
            out[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
            out[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if not pair_summary.empty:
        out["base_inversions_mean"] = float(pair_summary["base_inversions"].mean())
        out["final_inversions_mean"] = float(pair_summary["final_inversions"].mean())
    return out


def choose_decision(route: str, metrics: pd.DataFrame, predictions: pd.DataFrame, pair_summary: pd.DataFrame, shortcut: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if metrics.empty or predictions.empty:
        return "DEMA_C17_TRAINING_INVALID", {"reason": "missing validation evidence"}
    val = metrics[metrics["split"] == "val"] if "split" in metrics else metrics
    auc_values = pd.to_numeric(val.get("AUC", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    sensitivity = pd.to_numeric(val.get("Sensitivity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    specificity = pd.to_numeric(val.get("Specificity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    labels = finite_series(predictions, "label")
    positive_delta = finite_series(predictions, "delta_logit")[labels == 1].dropna().to_numpy(dtype=float)
    residual = finite_series(predictions, "delta_logit").dropna().to_numpy(dtype=float)
    final_gap = (finite_series(predictions, "final_prob")[labels == 1].mean() - finite_series(predictions, "final_prob")[labels == 0].mean())
    base_gap = (finite_series(predictions, "base_prob")[labels == 1].mean() - finite_series(predictions, "base_prob")[labels == 0].mean())
    pair_ok = pair_summary.empty or bool((pair_summary["final_inversions"] <= pair_summary["base_inversions"]).all())
    shortcut_ok = shortcut.empty or bool((shortcut["shortcut_label_auc"] < 0.80).all())
    checks = {
        "auc_at_least_c13_seed0": bool(auc_values.size and float(auc_values.min()) >= 0.8655500226),
        "sensitivity_at_least_0_55": bool(sensitivity.size and float(sensitivity.mean()) >= 0.55),
        "specificity_at_least_0_75": bool(specificity.size and float(specificity.mean()) >= 0.75),
        "gap_decrease_within_0_02": bool(np.isfinite(base_gap) and np.isfinite(final_gap) and final_gap >= base_gap - 0.02),
        "positive_delta_preserved": bool(positive_delta.size and float(positive_delta.mean()) >= -0.02 and float((positive_delta < -0.10).mean()) <= 0.25),
        "residual_nonzero": bool(residual.size > 1 and float(residual.std(ddof=1)) > 1e-8),
        "residual_not_saturated": bool(residual.size and float((np.abs(residual) >= 0.49999).mean()) < 0.25),
        "inversions_not_worse": pair_ok,
        "shortcut_audit_pass": shortcut_ok,
    }
    pilot_pass = all(checks.values())
    best_auc = float(auc_values.max()) if auc_values.size else 0.0
    if pilot_pass:
        decision = "PROMOTE_DEMA_C17_RESIDUAL_BCE" if route == "DEMA-R" else "PROMOTE_DEMA_C17_POSITIVE_PRESERVATION"
    elif best_auc > 0.8655500226 and (not checks["sensitivity_at_least_0_55"] or not checks["positive_delta_preserved"]):
        decision = "DEMA_C17_POSITIVE_SUPPRESSION_REMAINS"
    else:
        decision = "DEMA_C17_PILOT_FAIL_KEEP_C13"
    return decision, {"checks": checks, "best_auc": best_auc, "base_gap": float(base_gap), "final_gap": float(final_gap)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core_dir = Path(args.core_run_dir)
    rank_dir = Path(args.rank_run_dir)
    core_metrics = read_metrics(core_dir, "DEMA-R")
    rank_metrics = read_metrics(rank_dir, "DEMA-RP")
    selected_route = select_route(core_metrics, rank_metrics, args.selected_route)
    selected_dir = core_dir if selected_route == "DEMA-R" else rank_dir
    metrics = core_metrics if selected_route == "DEMA-R" else rank_metrics
    epochs = pd.concat([read_epochs(core_dir, "DEMA-R"), read_epochs(rank_dir, "DEMA-RP")], ignore_index=True)
    predictions = read_predictions(selected_dir, selected_route)

    all_metrics = drop_forbidden_metric(pd.concat([core_metrics, rank_metrics], ignore_index=True))
    all_metrics.to_csv(output_dir / "c17_metrics_by_seed.csv", index=False)
    if not all_metrics.empty and "split" in all_metrics:
        summary_rows: List[Dict[str, Any]] = []
        for route, frame in all_metrics.groupby("route"):
            val = frame[frame["split"] == "val"] if "split" in frame else frame
            row: Dict[str, Any] = {"route": route, "split": "val"}
            for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "pos_neg_gap", "mean_positive_delta_logit", "std_delta_logit"):
                if key in val:
                    values = pd.to_numeric(val[key], errors="coerce").dropna().to_numpy(dtype=float)
                    if values.size:
                        row[f"{key}_mean"] = float(values.mean())
                        row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "c17_metrics_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["route", "split"]).to_csv(output_dir / "c17_metrics_summary.csv", index=False)
    drop_forbidden_metric(epochs).to_csv(output_dir / "c17_metrics_by_epoch.csv", index=False)

    patient_diagnostics(predictions).to_csv(output_dir / "c17_patient_residual_diagnostics_val.csv", index=False)
    positive_audit(predictions).to_csv(output_dir / "c17_positive_preservation_audit.csv", index=False)
    pairs, pair_summary = pairwise_audit(predictions)
    pairs.to_csv(output_dir / "c17_pairwise_ranking_val.csv", index=False)
    pair_summary.to_csv(output_dir / "c17_pairwise_inversion_summary.csv", index=False)
    mechanism = mechanism_audit(predictions)
    mechanism.to_csv(output_dir / "c17_mechanism_residual_audit.csv", index=False)
    shortcut = shortcut_audit(predictions)
    shortcut.to_csv(output_dir / "c17_shortcut_residual_audit.csv", index=False)

    decision, gate = choose_decision(selected_route, metrics, predictions, pair_summary, shortcut)
    route_metrics = route_summary(metrics, pair_summary, selected_route)
    stability_text = "\n".join(
        [
            "# DEMA-HT C17 Seed Stability",
            "",
            f"Selected route: `{selected_route}`. Checkpoint selection is validation-AUC-only.",
            "",
            "```json",
            pd.Series(route_metrics).to_json(indent=2),
            "```",
            "",
            "Residual consistency is audited by per-seed positive delta, residual saturation, and inversion counts.",
            "",
        ]
    )
    (output_dir / "c17_seed_stability_report.md").write_text(stability_text, encoding="utf-8")

    comparison_text = "\n".join(
        [
            "# DEMA-HT C17 Model Comparison",
            "",
            "Official model name: `DEMA-HT`.",
            "Historical repository/package compatibility: `DMEA-HT` / `dmea_ht`.",
            "",
            "| Route | Validation AUC mean | Validation AUC std | Sensitivity mean | Specificity mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for route, frame in (("DEMA-R", core_metrics), ("DEMA-RP", rank_metrics)):
        summary = route_summary(frame, pd.DataFrame(), route)
        comparison_text += "\n| {route} | {auc:.6f} | {std:.6f} | {sens:.6f} | {spec:.6f} |".format(
            route=route,
            auc=summary.get("AUC_mean", 0.0),
            std=summary.get("AUC_std", 0.0),
            sens=summary.get("Sensitivity_mean", 0.0),
            spec=summary.get("Specificity_mean", 0.0),
        )
    comparison_text += "\n\nC13 strict-best reference mean validation AUC: `0.8664554097`.\n"
    (output_dir / "c17_model_comparison_report.md").write_text(comparison_text, encoding="utf-8")

    naming_text = "\n".join(
        [
            "# DEMA-HT Naming And Concept Correction",
            "",
            "Official model and research name: `DEMA-HT`.",
            "Historical repository/package identifiers remain `DMEA-HT` and `dmea_ht`.",
            "",
            "The alignment axis is HT pathological mechanism. Image, report-text, and biochemical evidence are the aligned objects. HT/non-HT is only the final binary prediction target.",
            "",
            "The corrected description is: align multimodal clinical evidence through HT pathological-mechanism relations and aggregate the mechanism evidence for HT risk prediction.",
            "",
            "C17 freezes C13 and learns only a bounded mechanism-evidence residual. Test is reporting-only and is not used for route selection.",
            "",
        ]
    )
    (output_dir / "c17_naming_and_concept_correction.md").write_text(naming_text, encoding="utf-8")

    final_text = "\n".join(
        [
            "# DEMA-HT Phase C17 Final Report",
            "",
            "## Contract",
            "",
            "- Official model name: `DEMA-HT`; historical repository/package identifiers remain `DMEA-HT` and `dmea_ht`.",
            "- The alignment axis is HT pathological mechanism; clinical evidence is the aligned object; HT/non-HT is the final prediction only.",
            "- C13 remains frozen. Route and checkpoint selection use validation AUC only. Test is reporting-only.",
            "",
            "## C16 Context",
            "",
            "- C16 improved seed-0 ranking AUC and reduced pairwise inversions, but failed its full safety gate because sensitivity and positive evidence preservation were inadequate.",
            "- C17 therefore uses a bounded residual and disables the former auxiliary and ranking objectives.",
            "",
            "## C17 Result",
            "",
            f"- Selected route: `{selected_route}`.",
            f"- Decision: `{decision}`.",
            f"- Best validation AUC observed for the selected route: `{gate.get('best_auc', 0.0):.6f}`.",
            f"- Base positive-negative gap: `{gate.get('base_gap', 0.0):.6f}`; final gap: `{gate.get('final_gap', 0.0):.6f}`.",
            "- Positive preservation, inversion, mechanism residual, and shortcut results are recorded in the accompanying CSV audits.",
            "",
            "## Gate",
            "",
        ]
    )
    for key, value in gate.get("checks", {}).items():
        final_text += f"- {key}: `{value}`.\n"
    final_text += "\nThe target validation AUC threshold is not claimed unless the complete formal promotion gate passes.\n"
    (output_dir / "phase_c17_dema_final_report.md").write_text(final_text, encoding="utf-8")

    print({"status": "PASS", "selected_route": selected_route, "decision": decision, "output_dir": str(output_dir)})
    if args.require_pilot_pass and not decision.startswith("PROMOTE_DEMA_C17_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
