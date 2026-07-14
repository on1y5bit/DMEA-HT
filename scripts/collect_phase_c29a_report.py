#!/usr/bin/env python3
"""Collect the prespecified C29-A bottleneck attribution and route decision."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42, 3407)
P0 = "P0_official_C27"
P1 = "P1_patient_state"
P2 = "P2_pre_projection"
P3 = "P3_temporal_mechanisms"
P4 = "P4_conflicts_negative_control"
P5 = "P5_C17_mechanism_reference"
CANDIDATES = (P1, P2, P3)
MINOR_AUC = 0.003
MINOR_SENSITIVITY = 0.03
MINOR_TRANSITIONS = 2
MINOR_INVERSIONS = 3
MATERIAL_MEAN_AUC_GAIN = 0.005
MAX_GENERALIZATION_GAP = 0.15
MAX_RANDOM_LABEL_AUC = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_reports/phase_c29a_dema")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def read_csv(output: Path, name: str) -> pd.DataFrame:
    path = output / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"patient_id": str, "positive_patient_id": str, "negative_patient_id": str})


def safe_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], formats: Mapping[str, str] | None = None) -> List[str]:
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if column not in ("probe", "stage", "seed_pair", "object") else "---" for column in columns) + "|",
    ]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values: List[str] = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def add_materiality(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    official = frame[frame["probe"].eq(P0)].set_index("seed")
    gains: List[float] = []
    sensitivity_changes: List[float] = []
    inversion_changes: List[int] = []
    damage_changes: List[int] = []
    minor: List[bool] = []
    for row in frame.itertuples(index=False):
        base = official.loc[int(row.seed)]
        gain = float(row.validation_AUC - base.validation_AUC)
        sensitivity_change = float(row.Sensitivity - base.Sensitivity)
        inversion_change = int(row.pairwise_inversion_count - base.pairwise_inversion_count)
        damage_change = int(row.material_positive_damage_count - base.material_positive_damage_count)
        is_minor = bool(
            abs(gain) < MINOR_AUC
            and abs(sensitivity_change) < MINOR_SENSITIVITY
            and max(damage_change, 0) < MINOR_TRANSITIONS
            and abs(inversion_change) <= MINOR_INVERSIONS
        )
        gains.append(gain)
        sensitivity_changes.append(sensitivity_change)
        inversion_changes.append(inversion_change)
        damage_changes.append(damage_change)
        minor.append(is_minor)
    frame["validation_AUC_change_vs_official"] = gains
    frame["sensitivity_change_vs_official"] = sensitivity_changes
    frame["inversion_change_vs_official"] = inversion_changes
    frame["material_damage_change_vs_official"] = damage_changes
    frame["minor_variation"] = minor
    frame["variation_classification"] = np.where(frame["minor_variation"], "minor variation", "material or directional diagnostic change")
    return frame


def probe_summary(
    metrics: pd.DataFrame,
    generalization: pd.DataFrame,
    random_sanity: pd.DataFrame,
    shortcut: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    official = metrics[metrics["probe"].eq(P0)].sort_values("seed")
    official_auc_mean = float(official["validation_AUC"].mean())
    official_inversion_mean = float(official["pairwise_inversion_count"].mean())
    official_damage = int(official["material_positive_damage_count"].sum())
    official_by_seed = official.set_index("seed")
    for probe, frame in metrics.groupby("probe", sort=False):
        frame = frame.sort_values("seed")
        gains = np.asarray(
            [
                float(row.validation_AUC - official_by_seed.loc[int(row.seed), "validation_AUC"])
                for row in frame.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        sensitivity_drops = np.asarray(
            [
                float(official_by_seed.loc[int(row.seed), "Sensitivity"] - row.Sensitivity)
                for row in frame.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        damage = int(frame["material_positive_damage_count"].sum())
        damage_reduction = (official_damage - damage) / max(official_damage, 1)
        fitted = probe != P0
        if fitted:
            gen = generalization[generalization["probe"].eq(probe)]
            random = random_sanity[random_sanity["probe"].eq(probe)]
            generalization_pass = bool(as_bool(gen["pass"]).all())
            random_pass = bool(as_bool(random["pass"]).all())
            random_mean = float(random["random_label_validation_AUC"].mean())
            random_max = float(random["random_label_validation_AUC"].max())
            max_gap = float(gen["train_validation_AUC_gap"].max())
        else:
            generalization_pass = random_pass = True
            random_mean = random_max = max_gap = float("nan")
        shortcut_rows = shortcut[
            shortcut["object_type"].eq("probe")
            & shortcut["representation_or_swap"].eq(probe)
        ]
        shortcut_pass = bool(as_bool(shortcut_rows["object_shortcut_pass"]).all())
        rows.append(
            {
                "probe": probe,
                "authorization_candidate": probe in CANDIDATES,
                "validation_AUC_mean": float(frame["validation_AUC"].mean()),
                "validation_AUC_std": safe_std(frame["validation_AUC"]),
                "mean_AUC_gain_vs_official": float(frame["validation_AUC"].mean() - official_auc_mean),
                "material_improvement_seed_count": int((gains >= MINOR_AUC).sum()),
                "positive_direction_seed_count": int((gains > 0).sum()),
                "Sensitivity_mean": float(frame["Sensitivity"].mean()),
                "maximum_sensitivity_drop_vs_official": float(sensitivity_drops.max()),
                "pairwise_inversion_mean": float(frame["pairwise_inversion_count"].mean()),
                "mean_inversion_change_vs_official": float(frame["pairwise_inversion_count"].mean() - official_inversion_mean),
                "material_positive_damage_aggregate": damage,
                "severe_positive_damage_aggregate": int(frame["severe_positive_damage_count"].sum()),
                "material_damage_reduction_fraction": damage_reduction,
                "train_AUC_mean": float(frame["train_AUC"].mean()),
                "maximum_train_validation_AUC_gap": max_gap,
                "generalization_pass": generalization_pass,
                "random_label_AUC_mean": random_mean,
                "random_label_AUC_max": random_max,
                "random_label_pass": random_pass,
                "shortcut_pass": shortcut_pass,
                "p4_p5_authorization_excluded": probe in (P4, P5),
            }
        )
    return pd.DataFrame(rows)


def candidate_pass(summary: pd.DataFrame, probe: str, official: pd.Series) -> Dict[str, bool]:
    row = summary.set_index("probe").loc[probe]
    return {
        "mean_auc_gain": bool(row["mean_AUC_gain_vs_official"] >= MATERIAL_MEAN_AUC_GAIN),
        "two_seed_direction": bool(row["material_improvement_seed_count"] >= 2),
        "inversion_nonworsening": bool(row["mean_inversion_change_vs_official"] <= 0),
        "positive_damage_reduction": bool(row["material_damage_reduction_fraction"] >= 0.25),
        "sensitivity_safety": bool(row["maximum_sensitivity_drop_vs_official"] <= 0.05),
        "generalization": bool(row["generalization_pass"]),
        "random_label": bool(row["random_label_pass"]),
        "shortcut": bool(row["shortcut_pass"]),
    }


def attribute(
    gate: Mapping[str, Any],
    reproduction: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    classifier_swaps: pd.DataFrame,
    head_swaps: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> Tuple[str, str, Dict[str, Any]]:
    table = summary.set_index("probe")
    official = table.loc[P0]
    p1_checks = candidate_pass(summary, P1, official)
    p2_checks = candidate_pass(summary, P2, official)
    p2_minus_p1 = float(table.loc[P2, "validation_AUC_mean"] - table.loc[P1, "validation_AUC_mean"])
    p1_supported = all(p1_checks.values()) and p2_minus_p1 < 0.003
    p2_supported = all(p2_checks.values()) and p2_minus_p1 >= 0.005
    random_label_failure = any(
        not bool(table.loc[probe, "random_label_pass"]) for probe in CANDIDATES
    )
    generalization_warning = any(
        not bool(table.loc[probe, "generalization_pass"]) for probe in CANDIDATES
    )
    unsafe_apparent_signal = any(
        table.loc[probe, "mean_AUC_gain_vs_official"] >= MATERIAL_MEAN_AUC_GAIN
        and table.loc[probe, "material_improvement_seed_count"] >= 2
        and not bool(table.loc[probe, "generalization_pass"])
        for probe in CANDIDATES
    )
    # A gap warning cannot supersede a direct all-seed validation failure. It becomes
    # the primary risk label only when an apparent authorizing signal depends on it.
    probe_risk = random_label_failure or unsafe_apparent_signal

    off_classifier = classifier_swaps[~as_bool(classifier_swaps["diagonal"])].copy()
    off_head = head_swaps[~as_bool(head_swaps["diagonal"])].copy()
    official_auc = {
        int(row.seed): float(row.validation_AUC)
        for row in metrics[metrics["probe"].eq(P0)].itertuples(index=False)
    }
    classifier_offdiag_material_gains = sum(
        float(row.AUC - official_auc[int(row.representation_seed)]) >= MINOR_AUC
        for row in off_classifier.itertuples(index=False)
    )
    head_offdiag_material_gains = sum(
        float(row.AUC - official_auc[int(row.representation_seed)]) >= MINOR_AUC
        for row in off_head.itertuples(index=False)
    )
    coordinate_support = {
        stage: bool(as_bool(frame["global_coordinate_compatibility_supported"]).all())
        for stage, frame in coordinates.groupby("stage")
    }
    same_seed_signal = bool(
        (table.loc[P1, "mean_AUC_gain_vs_official"] >= MATERIAL_MEAN_AUC_GAIN and table.loc[P1, "material_improvement_seed_count"] >= 2)
        or (table.loc[P2, "mean_AUC_gain_vs_official"] >= MATERIAL_MEAN_AUC_GAIN and table.loc[P2, "material_improvement_seed_count"] >= 2)
    )
    swaps_fail = classifier_offdiag_material_gains == 0 and head_offdiag_material_gains == 0
    coordinates_fail = not all(coordinate_support.get(stage, False) for stage in ("S2_pre_projection", "S4_patient_state"))
    coordinate_mismatch = same_seed_signal and swaps_fail and coordinates_fail

    candidate_rows = table.loc[list(CANDIDATES)]
    no_material_probe_gain = bool((candidate_rows["mean_AUC_gain_vs_official"] < MATERIAL_MEAN_AUC_GAIN).all())
    improvements_one_seed_at_most = bool((candidate_rows["material_improvement_seed_count"] <= 1).all())
    no_stable_damage_reduction = bool((candidate_rows["material_damage_reduction_fraction"] < 0.25).all())
    visit_limitation = (no_material_probe_gain or improvements_one_seed_at_most) and no_stable_damage_reduction

    if not bool(gate.get("pass", False)):
        primary = "C29A_ANALYSIS_INVALID"
    elif not as_bool(reproduction["pass"]).all():
        primary = "C29A_REPRODUCTION_FAIL"
    elif probe_risk:
        primary = "C29A_PROBE_LEAKAGE_OR_OVERFIT_RISK"
    elif p1_supported:
        primary = "C29A_FINAL_CLASSIFIER_BOTTLENECK_SUPPORTED"
    elif p2_supported:
        primary = "C29A_PATIENT_PROJECTION_BOTTLENECK_SUPPORTED"
    elif coordinate_mismatch:
        primary = "C29A_CROSS_SEED_COORDINATE_MISMATCH"
    elif visit_limitation:
        primary = "C29A_VISIT_REPRESENTATION_LIMITATION_SUPPORTED"
    else:
        primary = "C29A_MIXED_OR_INCONCLUSIVE"

    authorization = {
        "C29A_FINAL_CLASSIFIER_BOTTLENECK_SUPPORTED": "C29B_FINAL_LINEAR_CLASSIFIER_AUTHORIZED",
        "C29A_PATIENT_PROJECTION_BOTTLENECK_SUPPORTED": "C29B_DIRECT_PATIENT_STATE_CLASSIFIER_AUTHORIZED",
    }.get(primary, "C29B_NOT_AUTHORIZED")
    evidence = {
        "p1_checks": p1_checks,
        "p2_checks": p2_checks,
        "p2_mean_auc_minus_p1": p2_minus_p1,
        "classifier_offdiagonal_material_gain_count": int(classifier_offdiag_material_gains),
        "head_offdiagonal_material_gain_count": int(head_offdiag_material_gains),
        "coordinate_support": coordinate_support,
        "coordinate_mismatch_rule": coordinate_mismatch,
        "visit_limitation_rule": visit_limitation,
        "probe_risk": probe_risk,
        "random_label_failure": random_label_failure,
        "generalization_warning": generalization_warning,
        "unsafe_apparent_signal": unsafe_apparent_signal,
    }
    return primary, authorization, evidence


def write_reproduction_report(output: Path, reproduction: pd.DataFrame, gate: Mapping[str, Any]) -> None:
    lines = [
        "# C29-A Reproduction Report",
        "",
        f"- gate: `{gate['status']}` (`{gate['passed_checks']}/{gate['total_checks']}`)",
        "- scope: fixed train-fit diagnostics with validation-only route decisions",
        "- official tolerances: max absolute logit error <= 1e-6 and probability error <= 1e-7",
        "- temporal weights are checked both against the exact official formula and saved latest-slot values",
        "- all C27 checkpoint tensors must remain bitwise unchanged",
        "",
        "| seed | train n | val n | max logit error | max prob error | temporal formula error | decomposition error | unchanged | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in reproduction.to_dict(orient="records"):
        lines.append(
            f"| {row['seed']} | {row['train_patient_count']} | {row['validation_patient_count']} | {row['max_abs_logit_error']:.12g} | "
            f"{row['max_abs_probability_error']:.12g} | {row['max_abs_temporal_weight_formula_error']:.12g} | "
            f"{row['max_abs_logit_decomposition_error']:.12g} | {row['checkpoint_state_unchanged']} | {row['pass']} |"
        )
    (output / "c29a_reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_representation_inventory(output: Path, shapes: pd.DataFrame) -> None:
    first = shapes[(shapes["seed"] == 0) & (shapes["split"] == "val")]
    lines = [
        "# C29-A Frozen Representation Inventory",
        "",
        "- S0-S5 were taken from the real frozen C27 forward path under inference mode.",
        "- Exact pre-projection order: five 256-dimensional mechanism states, five conflict scalars, then the 256-dimensional frozen fallback bio context.",
        "- Patient projection order: `Linear(1541, 256) -> GELU -> LayerNorm(256)`.",
        "- Classifier formula: `weight dot h_patient + bias`; the preceding Dropout is inactive in evaluation mode.",
        "- Only patient-level float representations and scalar diagnostics were retained.",
        "",
        "| stage | validation shape | description |",
        "|---|---|---|",
    ]
    for row in first.itertuples(index=False):
        lines.append(f"| {row.stage} | `{row.shape}` | {row.description} |")
    (output / "c29a_representation_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def swap_matrix(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot(index="representation_seed", columns=column, values="AUC").reindex(index=SEEDS, columns=SEEDS)


def write_swap_report(output: Path, classifier: pd.DataFrame, head: pd.DataFrame) -> None:
    classifier_matrix = swap_matrix(classifier, "classifier_seed")
    head_matrix = swap_matrix(head, "head_seed")
    lines = [
        "# C29-A Cross-Seed Head-Swap Report",
        "",
        "- Rows are representation seeds and columns are applied head seeds.",
        "- Diagonal cells reproduce the official C27 predictor; off-diagonal cells are diagnostics only.",
        "- No cell is an averaged prediction, deployment candidate, or seed-selection rule.",
        "",
        "## Classifier-Only Validation AUC",
        "",
        "| representation seed | 0 | 42 | 3407 |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        lines.append(f"| {seed} | {classifier_matrix.loc[seed, 0]:.10f} | {classifier_matrix.loc[seed, 42]:.10f} | {classifier_matrix.loc[seed, 3407]:.10f} |")
    lines.extend(
        [
            "",
            "## Projection Plus Classifier Validation AUC",
            "",
            "| representation seed | 0 | 42 | 3407 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        lines.append(f"| {seed} | {head_matrix.loc[seed, 0]:.10f} | {head_matrix.loc[seed, 42]:.10f} | {head_matrix.loc[seed, 3407]:.10f} |")
    lines.extend(
        [
            "",
            f"- maximum classifier diagonal logit error: `{classifier.loc[as_bool(classifier['diagonal']), 'max_abs_diagonal_logit_error'].max():.12g}`",
            f"- maximum full-head diagonal logit error: `{head.loc[as_bool(head['diagonal']), 'max_abs_diagonal_logit_error'].max():.12g}`",
        ]
    )
    (output / "c29a_head_swap_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_coordinate_report(output: Path, coordinates: pd.DataFrame) -> None:
    lines = [
        "# C29-A Coordinate Compatibility Report",
        "",
        "- Global compatibility requires validation linear CKA >= 0.70 and patient-distance Spearman >= 0.65.",
        "- kNN Jaccard and train-fitted orthogonal Procrustes validation error are supporting diagnostics and do not independently fail a stage.",
        "",
        "| stage | seed pair | CKA | distance Spearman | kNN Jaccard | Procrustes val relative error | compatible |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in coordinates.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.seed_pair} | {row.linear_CKA:.6f} | {row.patient_distance_spearman:.6f} | "
            f"{row.knn_jaccard_k10:.6f} | {row.procrustes_validation_relative_error:.6f} | {row.global_coordinate_compatibility_supported} |"
        )
    for stage, frame in coordinates.groupby("stage"):
        lines.append(f"- {stage} all-pair global compatibility: `{as_bool(frame['global_coordinate_compatibility_supported']).all()}`")
    (output / "c29a_coordinate_compatibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_reports(
    output: Path,
    summary: pd.DataFrame,
    geometry: pd.DataFrame,
    rescue: pd.DataFrame,
    shortcut: pd.DataFrame,
    primary: str,
    authorization: str,
    evidence: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    table = summary.set_index("probe")
    p1 = table.loc[P1]
    p2 = table.loc[P2]
    p3 = table.loc[P3]
    official = table.loc[P0]
    rescue_focus = rescue[rescue["diagnostic_name"].isin((P1, P2, P3))]
    aggregate_rescue = rescue_focus.groupby("diagnostic_name", as_index=True)["probability_rescue_count"].sum()
    audit_lines = [
        "# C29-A Bottleneck Attribution",
        "",
        f"- primary label: `{primary}`",
        f"- C29-B authorization: `{authorization}`",
        f"- P1 mean AUC gain: `{p1.mean_AUC_gain_vs_official:.10f}`; material-improvement seeds: `{int(p1.material_improvement_seed_count)}/3`",
        f"- P2 mean AUC gain: `{p2.mean_AUC_gain_vs_official:.10f}`; P2 minus P1: `{evidence['p2_mean_auc_minus_p1']:.10f}`",
        f"- P3 mean AUC gain: `{p3.mean_AUC_gain_vs_official:.10f}`; material-improvement seeds: `{int(p3.material_improvement_seed_count)}/3`",
        f"- official aggregate material positive damage: `{int(official.material_positive_damage_aggregate)}`",
        f"- P1/P2/P3 aggregate material damage: `{int(p1.material_positive_damage_aggregate)}` / `{int(p2.material_positive_damage_aggregate)}` / `{int(p3.material_positive_damage_aggregate)}`",
        f"- P1/P2/P3 probability rescue counts: `{int(aggregate_rescue.get(P1, 0))}` / `{int(aggregate_rescue.get(P2, 0))}` / `{int(aggregate_rescue.get(P3, 0))}`",
        f"- classifier/full-head off-diagonal material-gain directions: `{evidence['classifier_offdiagonal_material_gain_count']}` / `{evidence['head_offdiagonal_material_gain_count']}`",
        f"- coordinate support: `{json.dumps(evidence['coordinate_support'], sort_keys=True)}`",
        "",
        "## Prespecified Gates",
        "",
        f"- P1 final-classifier checks: `{json.dumps(evidence['p1_checks'], sort_keys=True)}`",
        f"- P2 patient-projection checks: `{json.dumps(evidence['p2_checks'], sort_keys=True)}`",
        f"- coordinate-mismatch rule: `{evidence['coordinate_mismatch_rule']}`",
        f"- visit-representation-limitation rule: `{evidence['visit_limitation_rule']}`",
        f"- probe leakage/overfit risk: `{evidence['probe_risk']}`",
        f"- train-validation generalization warning present: `{evidence['generalization_warning']}`",
        f"- random-label failure present: `{evidence['random_label_failure']}`",
        f"- apparent authorizing signal dependent on an unsafe gap: `{evidence['unsafe_apparent_signal']}`",
        "",
        "P4 conflict-only and P5 C17-reference probes are excluded from authorization by construction. Cross-seed swaps and classifier geometry are diagnostic and are not clinical causal evidence.",
    ]
    (output / "c29a_bottleneck_attribution.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    route_lines = [
        "# C29-A Route Decision",
        "",
        f"- primary bottleneck label: `{primary}`",
        f"- authorization: `{authorization}`",
        "- current strict best: `DEMA_C17_POSITIVE_PRESERVATION`",
        "- current route: `KEEP_DEMA_C17_STRICT_BEST`",
        "- `STOP_VTME_TEMPORAL_TUNING` remains binding.",
        "- C29-A does not launch, train, or create a checkpoint for C29-B.",
    ]
    if authorization == "C29B_FINAL_LINEAR_CLASSIFIER_AUTHORIZED":
        route_lines.append("- only authorized future change: frozen C27 patient state plus one normalized linear classifier.")
    elif authorization == "C29B_DIRECT_PATIENT_STATE_CLASSIFIER_AUTHORIZED":
        route_lines.append("- only authorized future change: remove the nonlinear patient projection and use a normalized single linear classifier on the exact pre-projection patient input.")
    else:
        route_lines.extend(["- `C29B_NOT_AUTHORIZED`", "- `KEEP_DEMA_C17_STRICT_BEST`"])
    (output / "c29a_route_decision.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")

    implementation_commit = git_output("rev-parse", "--short", "HEAD")
    shortcut_pass_count = int(as_bool(shortcut["object_shortcut_pass"]).sum())
    final_lines = [
        "# DEMA-HT Phase C29-A Final Report",
        "",
        "## Execution",
        "",
        "- canonical server directory: `/home/linruixin/chen/project/DMEA-HT`",
        "- branch: `main`",
        "- starting commit: `60d090b`",
        f"- analysis commit: `{implementation_commit}`",
        f"- runtime gate: `{gate['status']}` (`{gate['passed_checks']}/{gate['total_checks']}`)",
        "- analysis-only; fixed train-fit statistical probes; validation-only route decision",
        "- no neural parameter update, checkpoint write, calibration, prediction averaging, or seed selection",
        "",
        "## Probe Summary",
        "",
        "| probe | val AUC mean | std | gain vs official | improved seeds | sensitivity | inversions | material damage | max gap | random max | safety |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        safety = bool(row.generalization_pass and row.random_label_pass and row.shortcut_pass)
        final_lines.append(
            f"| {row.probe} | {row.validation_AUC_mean:.10f} | {row.validation_AUC_std:.10f} | {row.mean_AUC_gain_vs_official:+.10f} | "
            f"{int(row.material_improvement_seed_count)}/3 | {row.Sensitivity_mean:.6f} | {row.pairwise_inversion_mean:.3f} | "
            f"{int(row.material_positive_damage_aggregate)} | "
            f"{row.maximum_train_validation_AUC_gap:.6f} | {row.random_label_AUC_max:.6f} | {safety} |"
        )
    final_lines.extend(
        [
            "",
            "## Head Geometry And Safety",
            "",
            f"- classifier weight norms by seed: `{', '.join(f'{value:.6f}' for value in geometry['classifier_weight_norm'])}`",
            f"- classifier biases by seed: `{', '.join(f'{value:.6f}' for value in geometry['classifier_bias'])}`",
            f"- classifier-centroid direction cosine by seed: `{', '.join(f'{value:.6f}' for value in geometry['classifier_centroid_direction_cosine'])}`",
            f"- shortcut-safe formal objects: `{shortcut_pass_count}/{len(shortcut)}`; authorization is evaluated per candidate probe.",
            "- raw visit/image associations remain separate audit warnings and are not probe inputs.",
            "",
            "## Decision",
            "",
            f"- `{primary}`",
            f"- `{authorization}`",
            "- current strict best: `DEMA_C17_POSITIVE_PRESERVATION`",
            "- diagnostic probes and swaps are not formal models.",
        ]
    )
    (output / "phase_c29a_dema_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output_dir)
    gate_path = output / "c29a_runtime_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("pass", False) or int(gate.get("total_checks", 0)) != 35:
        raise RuntimeError("C29A collector requires a passing 35-check gate")

    reproduction = read_csv(output, "c29a_reproduction_by_seed.csv")
    shapes = read_csv(output, "c29a_representation_shapes.csv")
    metrics = add_materiality(read_csv(output, "c29a_probe_metrics_by_seed.csv"))
    generalization = read_csv(output, "c29a_probe_generalization_audit.csv")
    random_sanity = read_csv(output, "c29a_random_label_sanity.csv")
    classifier_swaps = read_csv(output, "c29a_classifier_swap_metrics.csv")
    head_swaps = read_csv(output, "c29a_head_swap_metrics.csv")
    coordinates = read_csv(output, "c29a_coordinate_compatibility.csv")
    rescue = read_csv(output, "c29a_positive_damage_rescue_summary.csv")
    shortcut = read_csv(output, "c29a_shortcut_audit.csv")
    geometry = read_csv(output, "c29a_classifier_geometry_by_seed.csv")

    metrics.to_csv(output / "c29a_probe_metrics_by_seed.csv", index=False)
    summary = probe_summary(metrics, generalization, random_sanity, shortcut)
    summary.to_csv(output / "c29a_probe_metrics_summary.csv", index=False)
    primary, authorization, evidence = attribute(
        gate, reproduction, metrics, summary, classifier_swaps, head_swaps, coordinates
    )
    write_reproduction_report(output, reproduction, gate)
    write_representation_inventory(output, shapes)
    write_swap_report(output, classifier_swaps, head_swaps)
    write_coordinate_report(output, coordinates)
    write_decision_reports(
        output,
        summary,
        geometry,
        rescue,
        shortcut,
        primary,
        authorization,
        evidence,
        gate,
    )
    print(
        json.dumps(
            {
                "status": "C29A_REPORT_COMPLETE",
                "primary": primary,
                "authorization": authorization,
                "strict_best": "DEMA_C17_POSITIVE_PRESERVATION",
            }
        )
    )


if __name__ == "__main__":
    main()
