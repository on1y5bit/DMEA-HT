#!/usr/bin/env python3
"""Consolidate C26-SM single-model outputs and apply the formal gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c20_common import (  # noqa: E402
    jaccard_by_patient,
    linear_cka,
    pairwise_distances,
    spearman,
    upper_triangle,
)

SEEDS = (0, 42, 3407)
MECHANISMS = ("M1", "M2", "M3", "M4", "M5")
RELATIONS = (
    "image_morphology", "text_morphology", "bio_immune", "bio_function",
    "text_opposition", "text_temporal",
)
SELECTED_SHORTCUT_FIELDS = (
    "selected_n_visits", "used_images", "image_padding_count", "has_bio",
    "bio_missing_count", "report_length",
)
RAW_SHORTCUT_FIELDS = ("raw_n_visits", "raw_n_images")
ALL_SHORTCUT_FIELDS = SELECTED_SHORTCUT_FIELDS + RAW_SHORTCUT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/dema_ht_c26sm_stable_mechanism_mixer_multiseed")
    parser.add_argument("--c17-run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--c22-comparison", default="analysis_reports/phase_c22_dema/c22_c13_c17_comparison.csv")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c26sm_dema")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    frames = []
    for path in sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else seed_from_name(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def probability_column(frame: pd.DataFrame) -> str:
    for name in ("prob", "final_prob", "pred_prob", "prediction", "y_prob"):
        if name in frame:
            return name
    raise RuntimeError(f"No probability column in {list(frame.columns)}")


def auc(labels: Iterable[int], probs: Iterable[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(list(labels), dtype=int), np.asarray(list(probs), dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0


def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def shortcut_auc(frame: pd.DataFrame, fields: Tuple[str, ...]) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    present = [field for field in fields if field in frame]
    if not present or frame["label"].nunique() < 2:
        return None
    matrix = pd.DataFrame(index=frame.index)
    for field in present:
        values = pd.to_numeric(frame[field], errors="coerce")
        matrix[field] = values.fillna(values.median() if not values.dropna().empty else 0.0)
    folds = min(5, int(frame["label"].value_counts().min()))
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"), matrix.to_numpy(),
        frame["label"].astype(int).to_numpy(),
        cv=StratifiedKFold(folds, shuffle=True, random_state=42), method="predict_proba",
    )[:, 1]
    return auc(frame["label"], probabilities)


def pairwise_table(frame: pd.DataFrame) -> pd.DataFrame:
    positives = frame[frame["label"].astype(int) == 1].sort_values("patient_id")
    negatives = frame[frame["label"].astype(int) == 0].sort_values("patient_id")
    seed = int(frame["seed"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for _, positive in positives.iterrows():
        for _, negative in negatives.iterrows():
            c17_margin = float(positive["c17_prob"]) - float(negative["c17_prob"])
            c26_margin = float(positive["final_prob"]) - float(negative["final_prob"])
            c17_inversion, c26_inversion = c17_margin < 0, c26_margin < 0
            rows.append({
                "seed": seed,
                "positive_patient_id": positive["patient_id"],
                "negative_patient_id": negative["patient_id"],
                "c17_positive_score": positive["c17_prob"],
                "c17_negative_score": negative["c17_prob"],
                "c17_margin": c17_margin,
                "c17_inversion": int(c17_inversion),
                "c26sm_positive_score": positive["final_prob"],
                "c26sm_negative_score": negative["final_prob"],
                "c26sm_margin": c26_margin,
                "c26sm_inversion": int(c26_inversion),
                "repaired": int(c17_inversion and not c26_inversion),
                "introduced": int((not c17_inversion) and c26_inversion),
            })
    return pd.DataFrame(rows)


def load_representation(run_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    path = run_dir / "representations" / f"val_mechanism_state_seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as payload:
        labels = payload["label"]
        mechanism_state = payload["mechanism_state"]
    predictions = pd.read_csv(
        run_dir / "predictions" / f"val_predictions_seed_{seed}.csv",
        dtype={"patient_id": str},
    ).sort_values("patient_id").reset_index(drop=True)
    prediction_labels = predictions["label"].to_numpy(dtype=np.int64)
    if len(predictions) != len(mechanism_state) or not np.array_equal(labels, prediction_labels):
        raise RuntimeError(f"C26-SM representation/prediction alignment failed for seed {seed}")
    return {
        "patient_id": np.asarray(predictions["patient_id"].astype(str).tolist(), dtype=np.str_),
        "label": labels,
        "mechanism_state": mechanism_state,
    }


def main() -> None:
    args = parse_args()
    run_dir, c17_run, output = map(resolve_path, (args.run_dir, args.c17_run_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    c26 = read_predictions(run_dir, "val")
    c17 = read_predictions(c17_run, "val")
    if c26.empty or c17.empty or set(c26["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Complete C26-SM and C17 validation predictions are required")
    c17 = c17.rename(columns={probability_column(c17): "c17_prob"})
    diagnostics = c26.merge(
        c17[["patient_id", "seed", "label", "c17_prob"]],
        on=["patient_id", "seed"], how="left", validate="one_to_one", suffixes=("", "_c17"),
    )
    if diagnostics["c17_prob"].isna().any() or not diagnostics["label"].astype(int).eq(diagnostics["label_c17"].astype(int)).all():
        raise RuntimeError("C17/C26-SM patient alignment failed")
    diagnostics.to_csv(output / "c26sm_patient_diagnostics_val.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    transition_frames: List[pd.DataFrame] = []
    pair_frames: List[pd.DataFrame] = []
    inversion_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    shortcut_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    mechanism_health_rows: List[Dict[str, Any]] = []
    node_frames: List[pd.DataFrame] = []
    metrics_by_seed = pd.read_csv(run_dir / "reports" / "metrics_by_seed.csv")

    for seed in SEEDS:
        frame = diagnostics[diagnostics["seed"].astype(int) == seed].copy()
        labels = frame["label"].astype(int).to_numpy()
        c13_prob = frame["base_prob"].to_numpy(dtype=float)
        c17_prob = frame["c17_prob"].to_numpy(dtype=float)
        c26_prob = frame["final_prob"].to_numpy(dtype=float)
        delta = frame["delta_logit"].to_numpy(dtype=float)
        positive = labels == 1
        c13_pred, c17_pred, c26_pred = c13_prob >= 0.5, c17_prob >= 0.5, c26_prob >= 0.5
        c17_auc, c26_auc = auc(labels, c17_prob), auc(labels, c26_prob)
        comparison_rows.append({
            "seed": seed, "c17_auc": c17_auc, "c26sm_auc": c26_auc,
            "c26sm_minus_c17_auc": c26_auc - c17_auc,
        })
        rescued_c17 = positive & ~c13_pred & c17_pred
        positive_rows.append({
            "seed": seed,
            "c13_tp_to_fn": int((positive & c13_pred & ~c26_pred).sum()),
            "c13_fn_to_tp": int((positive & ~c13_pred & c26_pred).sum()),
            "mean_positive_delta": float(delta[positive].mean()),
            "fraction_positive_delta_below_minus_0_10": float((delta[positive] < -0.10).mean()),
            "c17_rescues": int(rescued_c17.sum()),
            "retained_c17_rescues": int((rescued_c17 & c26_pred).sum()),
            "lost_c17_rescues": int((rescued_c17 & ~c26_pred).sum()),
            "new_rescues_vs_c13_not_c17": int((positive & ~c13_pred & ~c17_pred & c26_pred).sum()),
        })
        transition = frame[["patient_id", "seed", "label", "base_prob", "c17_prob", "final_prob", "delta_logit"]].copy()
        transition["c13_prediction"] = c13_pred.astype(int)
        transition["c17_prediction"] = c17_pred.astype(int)
        transition["c26sm_prediction"] = c26_pred.astype(int)
        transition_frames.append(transition)
        pairs = pairwise_table(frame)
        pair_frames.append(pairs)
        c17_inv, c26_inv = int(pairs["c17_inversion"].sum()), int(pairs["c26sm_inversion"].sum())
        inversion_rows.append({
            "seed": seed, "c17_inversions": c17_inv, "c26sm_inversions": c26_inv,
            "net_change": c26_inv - c17_inv,
            "repaired": int(pairs["repaired"].sum()), "introduced": int(pairs["introduced"].sum()),
        })
        residual_rows.append({
            "seed": seed, "mean_delta": float(delta.mean()), "std_delta": float(delta.std(ddof=1)),
            "min_delta": float(delta.min()), "max_delta": float(delta.max()),
            "fraction_near_negative_bound": float((delta <= -0.495).mean()),
            "fraction_near_positive_bound": float((delta >= 0.495).mean()),
            "nonzero_variance": bool(delta.std(ddof=1) > 1e-6),
        })
        shortcut_row: Dict[str, Any] = {"seed": seed, "selected_structure_shortcut_auc": shortcut_auc(frame, SELECTED_SHORTCUT_FIELDS)}
        for field in RAW_SHORTCUT_FIELDS:
            raw = pd.DataFrame({"label": frame["label"], "value": pd.to_numeric(frame[field], errors="coerce")}).dropna()
            raw_auc = auc(raw["label"], raw["value"])
            shortcut_row[f"{field}_orientation_invariant_label_auc_warning"] = max(raw_auc, 1.0 - raw_auc)
        for field in ALL_SHORTCUT_FIELDS:
            shortcut_row[f"delta_spearman_{field}"] = frame["delta_logit"].corr(pd.to_numeric(frame[field], errors="coerce"), method="spearman")
        shortcut_rows.append(shortcut_row)

        metric = metrics_by_seed[(metrics_by_seed["seed"].astype(int) == seed) & metrics_by_seed["split"].eq("val")].iloc[0]
        relation_row = {"seed": seed}
        for relation in RELATIONS:
            relation_row[f"relation_gate_{relation}"] = float(metric[f"relation_gate_{relation}"])
        relation_rows.append(relation_row)
        node_columns = [f"{name}_weight" for name in MECHANISMS]
        node_frame = frame[["patient_id", "seed", *node_columns, *[f"empty_slot_{name}" for name in MECHANISMS], *[f"{name}_norm" for name in MECHANISMS]]].copy()
        weights = node_frame[node_columns].to_numpy(dtype=float)
        node_frame["node_weight_entropy"] = -(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
        node_frame["max_weight"] = weights.max(axis=1)
        node_frames.append(node_frame)
        max_norm_index = frame[[f"{name}_norm" for name in MECHANISMS]].to_numpy(dtype=float).argmax(axis=1)
        empty_matrix = frame[[f"empty_slot_{name}" for name in MECHANISMS]].astype(bool).to_numpy()
        empty_max_norm_fraction = float(empty_matrix[np.arange(len(frame)), max_norm_index].mean())
        gate_values = np.asarray([relation_row[f"relation_gate_{relation}"] for relation in RELATIONS])
        mechanism_health_rows.append({
            "seed": seed,
            "mean_node_weight_entropy": float(node_frame["node_weight_entropy"].mean()),
            "fraction_max_node_weight_above_0_90": float((node_frame["max_weight"] > 0.90).mean()),
            "mean_absolute_deviation_from_uniform": float(np.abs(weights - 0.20).mean()),
            "all_relation_gates_in_range": bool(((gate_values >= 0.05) & (gate_values <= 0.95)).all()),
            "empty_slot_max_norm_fraction": empty_max_norm_fraction,
            "mechanism_norms_finite": bool(np.isfinite(frame[[f"{name}_norm" for name in MECHANISMS] + ["mechanism_final_norm"]].to_numpy(dtype=float)).all()),
        })

    comparison = pd.DataFrame(comparison_rows)
    positive_audit = pd.DataFrame(positive_rows)
    transitions = pd.concat(transition_frames, ignore_index=True)
    pairwise = pd.concat(pair_frames, ignore_index=True)
    inversions = pd.DataFrame(inversion_rows)
    residual = pd.DataFrame(residual_rows)
    shortcuts = pd.DataFrame(shortcut_rows)
    relation_audit = pd.DataFrame(relation_rows)
    node_audit = pd.concat(node_frames, ignore_index=True)
    mechanism_health = pd.DataFrame(mechanism_health_rows)
    positive_audit.to_csv(output / "c26sm_positive_preservation_audit.csv", index=False)
    transitions.to_csv(output / "c26sm_c17_transition_audit.csv", index=False)
    pairwise.to_csv(output / "c26sm_pairwise_ranking_val.csv", index=False)
    inversions.to_csv(output / "c26sm_pairwise_inversion_summary.csv", index=False)
    residual.to_csv(output / "c26sm_residual_health_audit.csv", index=False)
    shortcuts.to_csv(output / "c26sm_shortcut_audit.csv", index=False)
    relation_audit.to_csv(output / "c26sm_relation_gate_audit.csv", index=False)
    node_audit.to_csv(output / "c26sm_node_weight_audit.csv", index=False)
    mechanism_health.to_csv(output / "c26sm_mechanism_health_audit.csv", index=False)

    representations = {seed: load_representation(run_dir, seed) for seed in SEEDS}
    stability_rows: List[Dict[str, Any]] = []
    for left, right in ((0, 42), (0, 3407), (42, 3407)):
        left_rep, right_rep = representations[left], representations[right]
        if not np.array_equal(left_rep["patient_id"].astype(str), right_rep["patient_id"].astype(str)) or not np.array_equal(left_rep["label"], right_rep["label"]):
            raise RuntimeError(f"C26-SM representation alignment failed for {left}/{right}")
        left_state = left_rep["mechanism_state"].astype(np.float64)
        right_state = right_rep["mechanism_state"].astype(np.float64)
        left_frame = diagnostics[diagnostics["seed"].astype(int) == left].sort_values("patient_id")
        right_frame = diagnostics[diagnostics["seed"].astype(int) == right].sort_values("patient_id")
        left_weights = left_frame[[f"{name}_weight" for name in MECHANISMS]].to_numpy(dtype=float)
        right_weights = right_frame[[f"{name}_weight" for name in MECHANISMS]].to_numpy(dtype=float)
        stability_rows.append({
            "seed_left": left, "seed_right": right,
            "final_mechanism_linear_cka": linear_cka(left_state, right_state),
            "final_mechanism_distance_spearman": spearman(upper_triangle(pairwise_distances(left_state)), upper_triangle(pairwise_distances(right_state))),
            "final_mechanism_knn_jaccard": float(np.nanmean(jaccard_by_patient(left_state, right_state, k=10))),
            "final_probability_spearman": spearman(left_frame["final_prob"].to_numpy(), right_frame["final_prob"].to_numpy()),
            "delta_logit_spearman": spearman(left_frame["delta_logit"].to_numpy(), right_frame["delta_logit"].to_numpy()),
            "node_weight_spearman": spearman(left_weights.reshape(-1), right_weights.reshape(-1)),
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "c26sm_cross_seed_representation_stability.csv", index=False)
    gate_matrix = relation_audit.drop(columns="seed").to_numpy(dtype=float)
    gate_dispersion = float(gate_matrix.std(axis=0, ddof=1).mean())

    c22 = pd.read_csv(resolve_path(args.c22_comparison))[["seed", "c17_auc", "c22_auc"]]
    three_way = comparison.merge(c22, on=["seed", "c17_auc"], validate="one_to_one")
    three_way["c26sm_minus_c22_auc"] = three_way["c26sm_auc"] - three_way["c22_auc"]
    three_way.to_csv(output / "c26sm_c17_c22_comparison.csv", index=False)
    for source, target in (("metrics_by_epoch.csv", "c26sm_metrics_by_epoch.csv"), ("metrics_by_seed.csv", "c26sm_metrics_by_seed.csv"), ("metrics_summary.csv", "c26sm_metrics_summary.csv")):
        pd.read_csv(run_dir / "reports" / source).to_csv(output / target, index=False)

    c17_auc = comparison["c17_auc"].to_numpy(dtype=float)
    c26_auc = comparison["c26sm_auc"].to_numpy(dtype=float)
    auc_difference = c26_auc - c17_auc
    auc_gate = bool(c26_auc.mean() > c17_auc.mean() and int((auc_difference > 0).sum()) >= 2 and (auc_difference >= -0.005).all() and safe_std(c26_auc) <= 0.02)
    positive_gate = bool(int(positive_audit["c13_tp_to_fn"].sum()) == 0 and (positive_audit["mean_positive_delta"] >= -0.02).all() and (positive_audit["fraction_positive_delta_below_minus_0_10"] <= 0.10).all())
    inversion_gate = bool(int((inversions["net_change"] < 0).sum()) >= 2 and int(inversions["repaired"].sum()) > int(inversions["introduced"].sum()) and (inversions["net_change"] <= 3).all())
    stability_means = {
        "final_mechanism_linear_cka": float(stability["final_mechanism_linear_cka"].mean()),
        "final_mechanism_distance_spearman": float(stability["final_mechanism_distance_spearman"].mean()),
        "final_mechanism_knn_jaccard": float(stability["final_mechanism_knn_jaccard"].mean()),
    }
    stability_gate = bool(stability_means["final_mechanism_linear_cka"] >= 0.55 and stability_means["final_mechanism_distance_spearman"] >= 0.55 and stability_means["final_mechanism_knn_jaccard"] >= 0.40)
    mechanism_gate = bool(
        (mechanism_health["mean_node_weight_entropy"] > 0.50).all()
        and (mechanism_health["fraction_max_node_weight_above_0_90"] <= 0.20).all()
        and (mechanism_health["mean_absolute_deviation_from_uniform"] > 1e-4).all()
        and mechanism_health["all_relation_gates_in_range"].all()
        and (mechanism_health["empty_slot_max_norm_fraction"] <= 0.80).all()
        and mechanism_health["mechanism_norms_finite"].all()
    )
    residual_gate = bool(residual["nonzero_variance"].all() and (residual["fraction_near_negative_bound"] < 0.25).all() and (residual["fraction_near_positive_bound"] < 0.25).all())
    shortcut_max = float(pd.to_numeric(shortcuts["selected_structure_shortcut_auc"], errors="coerce").max())
    shortcut_gate = bool(np.isfinite(shortcut_max) and shortcut_max <= 0.55)
    training_valid = bool(np.isfinite(c26_auc).all() and all(diagnostics.groupby("seed")["final_prob"].nunique() > 1))

    if not training_valid:
        decision = "DEMA_C26SM_TRAINING_INVALID"
    elif not positive_gate:
        decision = "DEMA_C26SM_POSITIVE_SUPPRESSION"
    elif not inversion_gate:
        decision = "DEMA_C26SM_INVERSION_WORSENING"
    elif not stability_gate:
        decision = "DEMA_C26SM_MECHANISM_INSTABILITY"
    elif not mechanism_gate:
        decision = "DEMA_C26SM_MECHANISM_COLLAPSE"
    elif not residual_gate:
        decision = "DEMA_C26SM_RESIDUAL_COLLAPSE"
    elif not shortcut_gate or (auc_difference < -0.005).any() or safe_std(c26_auc) > 0.02:
        decision = "DEMA_C26SM_FORMAL_FAIL_KEEP_C17"
    elif auc_gate:
        decision = "PROMOTE_DEMA_C26SM_STABLE_MECHANISM_MIXER"
    else:
        decision = "DEMA_C26SM_NO_AUC_GAIN_KEEP_C17"

    promoted = decision == "PROMOTE_DEMA_C26SM_STABLE_MECHANISM_MIXER"
    test_metrics = metrics_by_seed[metrics_by_seed["split"].eq("test")].copy()
    if args.validation_only:
        test_summary = "not run; validation decision frozen first"
    elif set(test_metrics["seed"].astype(int)) == set(SEEDS):
        test_summary = f"{test_metrics['AUC'].mean():.10f} +/- {safe_std(test_metrics['AUC']):.10f}"
    else:
        raise RuntimeError("Complete reporting-only test metrics are required for final collection")
    common = [
        "- C26-E status: `C26E_WITHDRAWN_BY_USER`; no ensemble artifact exists",
        "- deployment contract: one checkpoint, one model, one forward",
        "- checkpoint selection: validation AUC only; test reporting-only",
        f"- C17 validation AUC mean/std: `{c17_auc.mean():.10f} +/- {safe_std(c17_auc):.10f}`",
        f"- C22 validation AUC mean/std: `{three_way['c22_auc'].mean():.10f} +/- {safe_std(three_way['c22_auc']):.10f}`",
        f"- C26-SM validation AUC mean/std: `{c26_auc.mean():.10f} +/- {safe_std(c26_auc):.10f}`",
        f"- C26-SM minus C17 mean: `{c26_auc.mean() - c17_auc.mean():+.10f}`; AUC gate=`{auc_gate}`",
        f"- positive preservation: `{positive_gate}`; aggregate TP->FN=`{int(positive_audit['c13_tp_to_fn'].sum())}`",
        f"- inversion gate: `{inversion_gate}`; repaired/introduced=`{int(inversions['repaired'].sum())}/{int(inversions['introduced'].sum())}`",
        f"- mechanism stability means: CKA=`{stability_means['final_mechanism_linear_cka']:.10f}`, distance Spearman=`{stability_means['final_mechanism_distance_spearman']:.10f}`, kNN Jaccard=`{stability_means['final_mechanism_knn_jaccard']:.10f}`; pass=`{stability_gate}`",
        f"- relation-gate cross-seed dispersion: `{gate_dispersion:.10f}`",
        f"- mechanism health: `{mechanism_gate}`; residual health=`{residual_gate}`",
        f"- selected-structure shortcut-only AUC: `{shortcut_max:.10f}`; pass=`{shortcut_gate}`",
        f"- reporting-only test AUC mean/std: `{test_summary}`",
        f"- decision: `{decision}`",
    ]
    (output / "c26sm_cross_seed_stability_report.md").write_text("# C26-SM Cross-Seed Stability\n\n" + "\n".join(common) + "\n", encoding="utf-8")
    (output / "c26sm_single_model_deployment_contract.md").write_text(
        "# C26-SM Single-Model Deployment Contract\n\n"
        "Each formal seed is an independent training replicate. Deployment inference loads exactly one C26-SM checkpoint, one model, and performs one forward pass. Checkpoints and predictions are never averaged, voted, stacked, or jointly loaded.\n\n"
        + "\n".join(common) + "\n", encoding="utf-8",
    )
    route_lines = ["# C26-SM Route Decision", "", *common]
    if not promoted:
        route_lines.extend(["- `KEEP_DEMA_C17_STRICT_BEST`", "- `STOP_C26SM_TUNING`"])
    (output / "c26sm_route_decision.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    (output / "phase_c26sm_dema_final_report.md").write_text(
        "# Phase C26-SM DEMA-HT Final Report\n\n"
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`\n"
        "- runtime: `/home/linruixin/chen/conda/envs/ma`\n"
        "- single-model route only; no ensemble or checkpoint averaging\n"
        + "\n".join(common) + "\n" + ("\nKEEP_DEMA_C17_STRICT_BEST\nSTOP_C26SM_TUNING\n" if not promoted else ""),
        encoding="utf-8",
    )
    payload = {
        "phase": "C26-SM", "decision": decision,
        "c17_mean_auc": float(c17_auc.mean()), "c26sm_mean_auc": float(c26_auc.mean()),
        "c26sm_std_auc": safe_std(c26_auc), "auc_gate": auc_gate,
        "positive_preservation_pass": positive_gate, "inversion_pass": inversion_gate,
        "mechanism_stability_pass": stability_gate, "mechanism_health_pass": mechanism_gate,
        "residual_health_pass": residual_gate, "shortcut_pass": shortcut_gate,
        "test_used_for_decision": False, "ensemble_used": False,
        "deployment_contract": "one_checkpoint_one_model_one_forward",
        "validation_decision_frozen_before_test": True,
        "keep_c17_strict_best": not promoted, "stop_c26sm_tuning": not promoted,
    }
    (output / "c26sm_final_decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if args.require_pass and not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
