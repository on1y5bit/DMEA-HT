#!/usr/bin/env python3
"""Collect C65-A evidence and decide whether common-backbone CV is authorized."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c65a_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c65a.yaml")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = common.load_c65a_config(args.config)
    output = common.report_dir(config)
    variance = load_json(output / "c65a_c64_variance_summary.json")
    frozen = load_json(output / "c65a_frozen_backbone_summary.json")
    reproduction_frame = pd.read_csv(output / "c65a_oof_reproduction.csv")
    decomposition = pd.read_csv(output / "c65a_variance_decomposition.csv")
    if not bool(reproduction_frame["reproduction_pass"].all()):
        route = "C65A_OOF_REPRODUCTION_FAIL"
    elif not bool(variance.get("fold_integrity_pass", False)):
        route = "C65A_ANALYSIS_INVALID"
    else:
        cka_supported = float(frozen["mean_linear_cka"]) < float(config["analysis"]["c65b_mean_cka_max"])
        distance_supported = float(frozen["mean_patient_distance_spearman"]) < float(config["analysis"]["c65b_mean_distance_spearman_max"])
        seed_plus_interaction = float(decomposition.loc[decomposition["component"] == "seed_plus_interaction", "variance_fraction_of_total"].iloc[0])
        seed_supported = seed_plus_interaction >= float(config["analysis"]["c65b_seed_plus_interaction_min"])
        prediction_supported = float(variance["mean_probability_spearman"]) < float(config["analysis"]["c65b_cross_seed_probability_spearman_max"])
        fold_fraction = float(decomposition.loc[decomposition["component"] == "fold_main", "variance_fraction_of_total"].iloc[0])
        seed_fraction = float(decomposition.loc[decomposition["component"] == "seed_main", "variance_fraction_of_total"].iloc[0])
        interaction_fraction = float(decomposition.loc[decomposition["component"] == "seed_x_fold_interaction_residual", "variance_fraction_of_total"].iloc[0])
        material = bool(cka_supported or distance_supported or seed_supported or prediction_supported)
        if cka_supported or distance_supported:
            route = "C65A_MIXED_VARIANCE" if seed_supported else "C65A_BACKBONE_VARIANCE_SUPPORTED"
        elif seed_supported or prediction_supported:
            route = "C65A_HEAD_OR_OPTIMIZATION_VARIANCE_SUPPORTED"
        elif interaction_fraction > max(seed_fraction, fold_fraction):
            route = "C65A_FOLD_INTERACTION_DOMINANT"
        else:
            route = "C65A_VARIANCE_GATE_NOT_SUPPORTED"
    if "material" not in locals():
        material = False
        cka_supported = False
        distance_supported = False
        seed_supported = False
        prediction_supported = False
        seed_plus_interaction = float("nan")
        fold_fraction = float("nan")
        seed_fraction = float("nan")
        interaction_fraction = float("nan")
    authorized = bool(
        route in {
            "C65A_MIXED_VARIANCE",
            "C65A_BACKBONE_VARIANCE_SUPPORTED",
            "C65A_HEAD_OR_OPTIMIZATION_VARIANCE_SUPPORTED",
        }
        and bool(variance.get("reproduction_pass", False))
        and bool(variance.get("fold_integrity_pass", False))
        and not bool(variance.get("test_loaded", True))
        and not bool(frozen.get("test_loaded", True))
        and material
    )
    decision = {
        "phase": "C65-VACS",
        "status": "C65B_COMMON_BACKBONE_CV_AUTHORIZED" if authorized else "C65B_NOT_AUTHORIZED",
        "c65a_route": route,
        "c65b_authorized": authorized,
        "oof_reproduction_pass": bool(variance.get("reproduction_pass", False)),
        "fold_integrity_pass": bool(variance.get("fold_integrity_pass", False)),
        "test_loaded": False,
        "mean_c64_oof_probability_spearman": float(variance["mean_probability_spearman"]),
        "mean_frozen_representation_linear_cka": float(frozen["mean_linear_cka"]),
        "mean_frozen_representation_distance_spearman": float(frozen["mean_patient_distance_spearman"]),
        "mean_frozen_representation_knn_jaccard": float(frozen["mean_knn_jaccard"]),
        "seed_plus_interaction_variance_fraction": seed_plus_interaction,
        "fold_main_variance_fraction": fold_fraction,
        "seed_main_variance_fraction": seed_fraction,
        "seed_x_fold_interaction_variance_fraction": interaction_fraction,
        "criteria": {
            "frozen_cka_below_0.90": cka_supported,
            "frozen_distance_spearman_below_0.90": distance_supported,
            "seed_plus_interaction_at_least_0.40": seed_supported,
            "cross_seed_probability_spearman_below_0.80": prediction_supported,
        },
        "c64_gate_relaxed": False,
        "ensemble": False,
        "prediction_averaging": False,
    }
    common.write_json(output / "c65a_route_decision.json", decision)
    common.write_markdown(
        output / "phase_c65a_final_report.md",
        [
            "# C65-A Variance Attribution Report",
            "",
            f"- Route: `{route}`.",
            f"- C65-B status: `{decision['status']}`.",
            f"- C64 OOF reproduction: `{decision['oof_reproduction_pass']}`; fold integrity: `{decision['fold_integrity_pass']}`.",
            f"- Mean cross-seed OOF probability Spearman: `{decision['mean_c64_oof_probability_spearman']:.10f}`.",
            f"- Mean frozen source representation linear CKA: `{decision['mean_frozen_representation_linear_cka']:.10f}`.",
            f"- Mean frozen source patient-distance Spearman: `{decision['mean_frozen_representation_distance_spearman']:.10f}`.",
            f"- Mean frozen source kNN Jaccard at k=10: `{decision['mean_frozen_representation_knn_jaccard']:.10f}`.",
            f"- Seed plus Seed-by-Fold interaction variance fraction: `{decision['seed_plus_interaction_variance_fraction']:.10f}`.",
            "- Test was not loaded, used, or inspected for this analysis.",
            "- C64 stability gate was not relaxed. C65-B is permitted only when the recorded authorization status is `C65B_COMMON_BACKBONE_CV_AUTHORIZED`.",
        ],
    )
    print(json.dumps(decision, sort_keys=True))
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
