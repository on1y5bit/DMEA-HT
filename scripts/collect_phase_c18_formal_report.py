#!/usr/bin/env python3
"""Collect both C18 formal routes and apply the validation-AUC-only decision gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_phase_c18_directional_residual import audit_run  # noqa: E402


EXPECTED_SEEDS = (0, 42, 3407)
C17_SEED_AUC = {0: 0.8700769579, 42: 0.8768673608, 3407: 0.8619284744}
C17_MEAN_AUC = 0.8696242644
AUDIT_FILES = (
    "c18_patient_directional_diagnostics_val.csv",
    "c18_positive_preservation_audit.csv",
    "c18_pairwise_ranking_val.csv",
    "c18_pairwise_inversion_summary.csv",
    "c18_directional_mechanism_audit.csv",
    "c18_gate_health_audit.csv",
    "c18_shortcut_residual_audit.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional-run-dir", default="runs/dema_ht_c18_directional_multiseed")
    parser.add_argument("--hardrank-run-dir", default="runs/dema_ht_c18_directional_hardrank_multiseed")
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c18_dema")
    parser.add_argument("--require-formal-pass", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, route: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).drop(columns=["AUPRC"], errors="ignore")
    if route is not None:
        frame["route"] = route
    return frame


def route_summary(summary: Dict[str, Any], metrics: pd.DataFrame) -> Dict[str, Any]:
    route = str(summary["route"])
    val = metrics[metrics["split"] == "val"] if "split" in metrics.columns else pd.DataFrame()
    val = val.copy()
    val["seed"] = pd.to_numeric(val.get("seed", pd.Series(dtype=float)), errors="coerce")
    auc_by_seed = {int(row["seed"]): float(row["AUC"]) for _, row in val.iterrows() if pd.notna(row.get("seed")) and pd.notna(row.get("AUC"))}
    auc_values = np.asarray(list(auc_by_seed.values()), dtype=float)
    seed_improvements = {seed: auc_by_seed.get(seed, float("nan")) - C17_SEED_AUC[seed] for seed in EXPECTED_SEEDS}
    pair_summary = pd.DataFrame(summary.get("pair_summary", []))
    positive = pd.DataFrame(summary.get("positive", []))
    health = pd.DataFrame(summary.get("health", []))
    repaired = int(pd.to_numeric(pair_summary.get("repaired_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pair_summary.empty else 0
    introduced = int(pd.to_numeric(pair_summary.get("introduced_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pair_summary.empty else 0
    inversion_reduced = int((pd.to_numeric(pair_summary.get("final_inversions", pd.Series(dtype=float)), errors="coerce") < pd.to_numeric(pair_summary.get("base_inversions", pd.Series(dtype=float)), errors="coerce")).sum()) if not pair_summary.empty else 0
    positive_mean = pd.to_numeric(positive.get("mean_positive_directional_delta", pd.Series(dtype=float)), errors="coerce")
    positive_fraction = pd.to_numeric(positive.get("fraction_positive_delta_below_minus_0_10", pd.Series(dtype=float)), errors="coerce")
    tp_to_fn = int(pd.to_numeric(positive.get("tp_to_fn", pd.Series(dtype=float)), errors="coerce").sum()) if not positive.empty else 0
    fn_to_tp = int(pd.to_numeric(positive.get("fn_to_tp", pd.Series(dtype=float)), errors="coerce").sum()) if not positive.empty else 0
    negative_delta = pd.to_numeric(positive.get("mean_negative_directional_delta", pd.Series(dtype=float)), errors="coerce")
    negative_prob_change = pd.to_numeric(positive.get("negative_probability_change", pd.Series(dtype=float)), errors="coerce")
    sensitivity = pd.to_numeric(val.get("Sensitivity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    specificity = pd.to_numeric(val.get("Specificity", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    shortcut_max = float(summary.get("shortcut_max_auc", float("nan")))
    health_ok = bool(
        not health.empty
        and health["finite"].astype(bool).all()
        and (pd.to_numeric(health["support_delta_std"], errors="coerce") > 1e-8).all()
        and (pd.to_numeric(health["opposition_delta_std"], errors="coerce") > 1e-8).all()
        and (pd.to_numeric(health["mean_support_gate"], errors="coerce").between(0.01, 0.99)).all()
        and (pd.to_numeric(health["mean_opposition_gate"], errors="coerce").between(0.01, 0.99)).all()
        and (pd.to_numeric(health["mean_conflict_suppression"], errors="coerce") > 1e-3).all()
    )
    checks = {
        "training_valid": bool(summary.get("valid", False)) and sorted(auc_by_seed) == list(EXPECTED_SEEDS),
        "mean_validation_auc_above_c17": bool(auc_values.size == 3 and float(auc_values.mean()) > C17_MEAN_AUC),
        "at_least_two_seed_auc_improvements": sum(value > 0.0 for value in seed_improvements.values() if np.isfinite(value)) >= 2,
        "no_seed_drop_over_0_005": bool(seed_improvements and min(seed_improvements.values()) >= -0.005),
        "validation_auc_std_le_0_02": bool(auc_values.size == 3 and float(auc_values.std(ddof=1)) <= 0.02),
        "at_least_two_seed_inversion_reduction": inversion_reduced >= 2,
        "introduced_less_than_repaired": introduced < repaired,
        "positive_preservation": bool(not positive_mean.empty and positive_mean.min() >= -0.02 and positive_fraction.max() <= 0.25 and tp_to_fn <= fn_to_tp),
        "no_negative_inflation": bool(not negative_delta.empty and negative_delta.max() <= 0.02 and negative_prob_change.max() <= 0.02),
        "sensitivity_mean_ge_0_55": bool(sensitivity.size == 3 and float(sensitivity.mean()) >= 0.55),
        "specificity_mean_ge_0_75": bool(specificity.size == 3 and float(specificity.mean()) >= 0.75),
        "branch_and_gate_health": health_ok,
        "shortcut_safety": bool(np.isfinite(shortcut_max) and shortcut_max <= 0.55),
        "test_is_reporting_only": True,
    }
    return {
        "route": route,
        "mean_validation_auc": float(auc_values.mean()) if auc_values.size else float("nan"),
        "std_validation_auc": float(auc_values.std(ddof=1)) if auc_values.size > 1 else float("nan"),
        "auc_by_seed": auc_by_seed,
        "seed_improvements": seed_improvements,
        "mean_test_auc_reporting_only": float(summary.get("mean_test_auc_reporting_only", float("nan"))),
        "base_inversions": int(pd.to_numeric(pair_summary.get("base_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pair_summary.empty else 0,
        "final_inversions": int(pd.to_numeric(pair_summary.get("final_inversions", pd.Series(dtype=float)), errors="coerce").sum()) if not pair_summary.empty else 0,
        "repaired_inversions": repaired,
        "introduced_inversions": introduced,
        "inversion_reduction_seed_count": inversion_reduced,
        "positive_tp_to_fn": tp_to_fn,
        "positive_fn_to_tp": fn_to_tp,
        "mean_sensitivity": float(sensitivity.mean()) if sensitivity.size else float("nan"),
        "mean_specificity": float(specificity.mean()) if specificity.size else float("nan"),
        "shortcut_max_auc": shortcut_max,
        "checks": checks,
        "safe": all(checks.values()),
    }


def merge_route_audits(route_outputs: Dict[str, Path], destination: Path) -> None:
    """Expose both route audits at the report root without losing route identity."""
    for filename in AUDIT_FILES:
        frames: List[pd.DataFrame] = []
        for route, source in route_outputs.items():
            path = source / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if "route" not in frame.columns:
                frame.insert(0, "route", route)
            else:
                frame["route"] = route
            frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(destination / filename, index=False)


def choose_decision(routes: Dict[str, Dict[str, Any]]) -> tuple[str, str, List[str]]:
    failures: List[str] = []
    for result in routes.values():
        if not result["checks"].get("training_valid", False):
            failures.append("DEMA_C18_TRAINING_INVALID")
        if not result["checks"].get("branch_and_gate_health", False):
            failures.append("DEMA_C18_DIRECTIONAL_BRANCH_COLLAPSE")
        if not result["checks"].get("introduced_less_than_repaired", False):
            failures.append("DEMA_C18_NEW_INVERSIONS_EXCEED_REPAIRS")
        if not result["checks"].get("positive_preservation", False):
            failures.append("DEMA_C18_POSITIVE_SUPPRESSION")
        if not result["checks"].get("no_negative_inflation", False):
            failures.append("DEMA_C18_NEGATIVE_INFLATION")
    failures = sorted(set(failures))
    valid_routes = [result for result in routes.values() if result["safe"]]
    if not valid_routes:
        any_gain = any(result["mean_validation_auc"] > C17_MEAN_AUC for result in routes.values() if np.isfinite(result["mean_validation_auc"]))
        core_stability = all(
            result["checks"].get("validation_auc_std_le_0_02", False)
            and result["checks"].get("no_seed_drop_over_0_005", False)
            for result in routes.values()
        )
        if any_gain and not core_stability:
            return "DEMA_C18_SMALL_GAIN_NOT_STABLE", "C17", failures
        if failures:
            return failures[0], "C17", failures
        return "DEMA_C18_FORMAL_FAIL_KEEP_C17", "C17", failures
    if len(valid_routes) == 2:
        directional = routes["C18-D"]
        hardrank = routes["C18-DH"]
        if abs(directional["mean_validation_auc"] - hardrank["mean_validation_auc"]) < 0.002:
            dh_clear = (
                hardrank["final_inversions"] < directional["final_inversions"]
                and hardrank["introduced_inversions"] <= directional["introduced_inversions"]
            )
            selected = hardrank if dh_clear else directional
        else:
            selected = max(valid_routes, key=lambda item: item["mean_validation_auc"])
    else:
        selected = valid_routes[0]
    label = "PROMOTE_DEMA_C18_DIRECTIONAL_HARDRANK" if selected["route"] == "C18-DH" else "PROMOTE_DEMA_C18_DIRECTIONAL"
    return label, selected["route"], failures


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_specs = {
        "C18-D": Path(args.directional_run_dir),
        "C18-DH": Path(args.hardrank_run_dir),
    }
    route_outputs: Dict[str, Path] = {
        route: output_dir / ("route_c18_d" if route == "C18-D" else "route_c18_dh")
        for route in route_specs
    }
    metrics_frames: List[pd.DataFrame] = []
    epoch_frames: List[pd.DataFrame] = []
    audit_summaries: Dict[str, Dict[str, Any]] = {}
    route_metrics: Dict[str, pd.DataFrame] = {}
    for route, run_dir in route_specs.items():
        try:
            audit_summaries[route] = audit_run(run_dir, route_outputs[route], route)
        except Exception as exc:
            audit_summaries[route] = {"route": route, "valid": False, "error": repr(exc), "mean_validation_auc": float("nan"), "safe": False, "checks": {"training_valid": False}}
        metrics = read_csv(run_dir / "reports" / "metrics_by_seed.csv", route)
        epochs = read_csv(run_dir / "reports" / "metrics_by_epoch.csv", route)
        route_metrics[route] = metrics
        if not metrics.empty:
            metrics_frames.append(metrics)
        if not epochs.empty:
            epoch_frames.append(epochs)

    metrics_all = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    epoch_all = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    metrics_all.to_csv(output_dir / "c18_metrics_by_seed.csv", index=False)
    epoch_all.to_csv(output_dir / "c18_metrics_by_epoch.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for route, frame in route_metrics.items():
        for split in ("val", "test"):
            subset = frame[frame["split"] == split] if "split" in frame.columns else pd.DataFrame()
            row: Dict[str, Any] = {"route": route, "split": split}
            for key in ("AUC", "Sensitivity", "Specificity", "Balanced_ACC", "positive_prob_mean", "negative_prob_mean", "mean_directional_delta", "mean_positive_directional_delta", "mean_negative_directional_delta"):
                if key in subset.columns:
                    values = pd.to_numeric(subset[key], errors="coerce").dropna().to_numpy(dtype=float)
                    if values.size:
                        row[f"{key}_mean"] = float(values.mean())
                        row[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "c18_metrics_summary.csv", index=False)

    route_gate_results = {
        route: route_summary(audit_summaries[route], route_metrics[route])
        for route in route_specs
    }
    decision, selected_route, failures = choose_decision(route_gate_results)
    merge_route_audits(route_outputs, output_dir)

    transition_script = Path(__file__).with_name("analyze_phase_c18_c17_inversion_transitions.py")
    transition_output = subprocess.run(
        [sys.executable, str(transition_script), "--prediction-dir", args.c17_prediction_dir, "--output-dir", str(output_dir)],
        text=True,
        capture_output=True,
    )
    transition_status = transition_output.returncode == 0
    gate = {
        "decision": decision,
        "selected_route": selected_route,
        "current_strict_best": selected_route if decision.startswith("PROMOTE_DEMA_C18_") else "DEMA-HT C17 Positive Preservation",
        "c17_mean_validation_auc": C17_MEAN_AUC,
        "routes": route_gate_results,
        "failure_labels": failures,
        "transition_analysis_status": transition_status,
        "static_synthetic_gate": "DIRECT_MULTI_SEED_AUTHORIZED",
        "no_smoke": True,
        "no_seed0_pilot": True,
        "direct_seeds": list(EXPECTED_SEEDS),
        "test_reporting_only": True,
        "auc_0_90_reached": bool(np.isfinite(route_gate_results.get(selected_route, {}).get("mean_validation_auc", np.nan)) and route_gate_results[selected_route]["mean_validation_auc"] >= 0.90),
    }
    (output_dir / "c18_formal_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    report_lines = [
        "# DEMA-HT Phase C18 Final Report",
        "",
        "- Official model name: `DEMA-HT`.",
        "- Frozen base: C13 temporal-focus logit; model selection uses validation AUC only.",
        "- C18-D and C18-DH were run directly with seeds `[0, 42, 3407]` after the static/synthetic gate.",
        "- No smoke run and no seed-0-only pilot were used.",
        "- Test predictions are reporting-only.",
        "",
        f"## Decision: `{decision}`",
        "",
        f"- Selected route: `{selected_route}`.",
        f"- C17 reference mean validation AUC: `{C17_MEAN_AUC:.10f}`.",
        f"- Validation AUC 0.90 reached: `{gate['auc_0_90_reached']}`.",
        "",
        "## Route Summary",
        "",
        "| Route | Validation AUC mean | Validation AUC std | Test AUC mean (reporting only) | Repaired | Introduced |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for route in ("C18-D", "C18-DH"):
        result = route_gate_results[route]
        report_lines.append(
            f"| {route} | {result['mean_validation_auc']:.10f} | {result['std_validation_auc']:.10f} | {result['mean_test_auc_reporting_only']:.10f} | {result['repaired_inversions']} | {result['introduced_inversions']} |"
        )
    report_lines += [
        "",
        "## Gate Details",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True),
        "```",
        "",
    ]
    (output_dir / "c18_seed_stability_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "c18_model_comparison_report.md").write_text("\n".join(report_lines[:14]) + "\n", encoding="utf-8")
    (output_dir / "phase_c18_dema_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True))
    if args.require_formal_pass and not decision.startswith("PROMOTE_DEMA_C18_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
