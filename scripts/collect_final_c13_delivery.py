from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.config import load_config  # noqa: E402
from dmea_ht.data import read_manifest  # noqa: E402


DEFAULT_RUN_DIR = "runs/dmea_ht_v2_c13_temporal_focus_stress_seeds"
DEFAULT_CONFIG = "configs/dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml"
DEFAULT_MANIFEST = "/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl"
DEFAULT_OUTPUT_DIR = "analysis_reports/final_c13_delivery"
FORMAL_SEEDS = (0, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and verify the frozen C13 final delivery.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--phase-c13-dir", default="analysis_reports/phase_c13_stress")
    parser.add_argument("--phase-c14b-dir", default="analysis_reports/phase_c14b")
    parser.add_argument("--phase-c14c-dir", default="analysis_reports/phase_c14c")
    parser.add_argument("--phase-c14e-dir", default="analysis_reports/phase_c14e")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-root", default=".")
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
            text = "NA" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_inventory(rows: List[Dict[str, Any]], path: Path, kind: str, notes: str = "") -> None:
    exists = path.is_file()
    rows.append(
        {
            "kind": kind,
            "path": str(path),
            "exists": int(exists),
            "size_bytes": path.stat().st_size if exists else 0,
            "sha256": sha256_file(path) if exists else "",
            "notes": notes,
        }
    )


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
        return {
            "checkpoint_seed": checkpoint.get("seed", "") if isinstance(checkpoint, Mapping) else "",
            "checkpoint_best_epoch": checkpoint.get("best_epoch", "") if isinstance(checkpoint, Mapping) else "",
            "checkpoint_manifest": config.get("project", {}).get("manifest", "") if isinstance(config, Mapping) else "",
            "checkpoint_output_dir": config.get("project", {}).get("output_dir", "") if isinstance(config, Mapping) else "",
            "checkpoint_primary_metric": config.get("training", {}).get("primary_metric", "") if isinstance(config, Mapping) else "",
            "state_dict_keys": len(checkpoint.get("model", {})) if isinstance(checkpoint, Mapping) else 0,
            "load_status": "loaded",
        }
    except Exception as exc:
        return {
            "checkpoint_seed": "",
            "checkpoint_best_epoch": "",
            "checkpoint_manifest": "",
            "checkpoint_output_dir": "",
            "checkpoint_primary_metric": "",
            "state_dict_keys": 0,
            "load_status": f"error:{exc}",
        }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    phase_c13 = Path(args.phase_c13_dir)
    phase_c14b = Path(args.phase_c14b_dir)
    phase_c14c = Path(args.phase_c14c_dir)
    phase_c14e = Path(args.phase_c14e_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    metrics_by_seed = read_csv(phase_c13 / "c13_stress_metrics_by_seed.csv")
    metrics_summary = read_csv(phase_c13 / "c13_stress_metrics_summary.csv")
    confusion = read_csv(phase_c13 / "c13_stress_confusion_matrix_by_seed.csv")
    shortcut = read_csv(phase_c13 / "shortcut_residual" / "shortcut_residual_audit.csv")
    reproduction = read_csv(phase_c14b / "c14b_reproduction_check_by_seed.csv")
    c14c_gate = read_csv(phase_c14c / "c14c_route_gate_summary.csv")
    c14e_gate = read_csv(phase_c14e / "c14e_route_gate_summary.csv")

    manifest_rows = read_manifest(manifest_path)
    manifest_frame = pd.DataFrame(manifest_rows)
    manifest_frame["label"] = pd.to_numeric(manifest_frame["label"], errors="coerce").astype(int)
    manifest_frame["split"] = manifest_frame["split"].astype(str)
    manifest_counts = manifest_frame.groupby(["split", "label"]).size().reset_index(name="n_rows")
    manifest_counts.to_csv(out_dir / "final_manifest_split_label_counts.csv", index=False)

    inventory_rows: List[Dict[str, Any]] = []
    add_inventory(inventory_rows, config_path, "config", "Frozen C13 formal stress config")
    add_inventory(inventory_rows, manifest_path, "manifest", f"{len(manifest_rows)} patient rows")
    for report_path in (
        phase_c13 / "phase_c13_stress_decision_report.md",
        phase_c13 / "shortcut_residual" / "shortcut_residual_audit_report.md",
        phase_c14b / "phase_c14b_final_report.md",
        phase_c14c / "phase_c14c_final_report.md",
        phase_c14e / "phase_c14e_final_report.md",
    ):
        add_inventory(inventory_rows, report_path, "audit_report")

    selection_rows: List[Dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        checkpoint_path = run_dir / "checkpoints" / f"seed_{seed}_best.pt"
        val_prediction_path = run_dir / "predictions" / f"val_predictions_seed_{seed}.csv"
        test_prediction_path = run_dir / "predictions" / f"test_predictions_seed_{seed}.csv"
        add_inventory(inventory_rows, checkpoint_path, "checkpoint", f"seed={seed}; validation-AUC-selected")
        add_inventory(inventory_rows, val_prediction_path, "validation_predictions", f"seed={seed}")
        add_inventory(inventory_rows, test_prediction_path, "test_predictions_reporting_only", f"seed={seed}")
        metadata = load_checkpoint_metadata(checkpoint_path) if checkpoint_path.is_file() else load_checkpoint_metadata(checkpoint_path)
        val_metric = metrics_by_seed[(metrics_by_seed["seed"] == seed) & (metrics_by_seed["split"] == "val")].iloc[0]
        test_metric = metrics_by_seed[(metrics_by_seed["seed"] == seed) & (metrics_by_seed["split"] == "test")].iloc[0]
        val_rows = len(pd.read_csv(val_prediction_path)) if val_prediction_path.is_file() else 0
        test_rows = len(pd.read_csv(test_prediction_path)) if test_prediction_path.is_file() else 0
        selection_rows.append(
            {
                "route": "C13_TEMPORAL_FOCUS_DMEA_HT",
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
                "best_epoch_metrics_csv": int(val_metric["best_epoch"]),
                "best_epoch_checkpoint": metadata["checkpoint_best_epoch"],
                "val_auc": val_metric["AUC"],
                "val_auprc": val_metric["AUPRC"],
                "val_sensitivity": val_metric["Sensitivity"],
                "val_specificity": val_metric["Specificity"],
                "val_positive_negative_gap": val_metric["pos_neg_gap"],
                "test_auc_reporting_only": test_metric["AUC"],
                "test_auprc_reporting_only": test_metric["AUPRC"],
                "val_prediction_rows": val_rows,
                "test_prediction_rows_reporting_only": test_rows,
                **metadata,
            }
        )
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out_dir / "final_model_selection.csv", index=False)

    validation_metrics = metrics_by_seed[metrics_by_seed["split"] == "val"].copy()
    validation_metrics.to_csv(out_dir / "final_validation_metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(out_dir / "final_performance_summary.csv", index=False)
    confusion.to_csv(out_dir / "final_confusion_matrix_by_seed.csv", index=False)
    pooled_shortcut = shortcut[shortcut["seed"].astype(str) == "pooled"].copy()
    pooled_shortcut["selection_role"] = pooled_shortcut["split"].map({"val": "selection_safety", "test": "reporting_only"}).fillna("audit")
    pooled_shortcut.to_csv(out_dir / "final_shortcut_safety.csv", index=False)

    c14c_row = c14c_gate.iloc[0]
    c14e_row = c14e_gate.iloc[0]
    limitations = pd.DataFrame(
        [
            {
                "phase": "C14-C",
                "route": c14c_row["route"],
                "status": c14c_row["final_status"],
                "total_pairwise_rows": c14c_row["total_pairwise_rows"],
                "total_inversion_rows": c14c_row["total_inversion_rows"],
                "all_seed_inversion_pairs": c14c_row["all_seed_inversion_pairs"],
                "hard_positive_count": "",
                "hard_negative_count": "",
                "positive_matching_coverage": "",
                "negative_matching_coverage": "",
                "training_authorized": c14c_row["c15_authorized"],
                "decision_basis": c14c_row["decision_basis"],
            },
            {
                "phase": "C14-E",
                "route": c14e_row["route"],
                "status": c14e_row["allowed_next_step"],
                "total_pairwise_rows": "",
                "total_inversion_rows": "",
                "all_seed_inversion_pairs": "",
                "hard_positive_count": c14e_row["hard_positive_count"],
                "hard_negative_count": c14e_row["hard_negative_count"],
                "positive_matching_coverage": c14e_row["positive_matching_coverage"],
                "negative_matching_coverage": c14e_row["negative_matching_coverage"],
                "training_authorized": c14e_row["training_authorized"],
                "decision_basis": c14e_row["decision_basis"],
            },
        ]
    )
    limitations.to_csv(out_dir / "final_hard_subgroup_limitations.csv", index=False)

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(out_dir / "final_artifact_inventory.csv", index=False)

    expected_counts = {("train", 0): 301, ("train", 1): 301, ("val", 0): 47, ("val", 1): 47, ("test", 0): 42, ("test", 1): 42}
    actual_counts = {(str(row.split), int(row.label)): int(row.n_rows) for row in manifest_counts.itertuples()}
    val_summary = metrics_summary[metrics_summary["split"] == "val"].iloc[0]
    val_shortcut = pooled_shortcut[pooled_shortcut["split"] == "val"].iloc[0]
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    check("formal_seeds", list(config["training"]["seeds"]) == list(FORMAL_SEEDS), str(config["training"]["seeds"]))
    check("primary_metric", config["training"].get("primary_metric") == "val_AUC", str(config["training"].get("primary_metric")))
    check("manifest_path", str(config["project"].get("manifest")) == str(manifest_path), str(config["project"].get("manifest")))
    check("manifest_rows", len(manifest_rows) == 780, str(len(manifest_rows)))
    check("manifest_split_label_counts", actual_counts == expected_counts, json.dumps({f"{key[0]}_{key[1]}": value for key, value in actual_counts.items()}, sort_keys=True))
    check("checkpoints_exist", bool((inventory[inventory["kind"] == "checkpoint"]["exists"] == 1).all()), str(inventory[inventory["kind"] == "checkpoint"]["exists"].tolist()))
    check("checkpoints_load", bool((selection["load_status"] == "loaded").all()), str(selection["load_status"].tolist()))
    check("checkpoint_seed_metadata", bool((pd.to_numeric(selection["checkpoint_seed"]) == selection["seed"]).all()), str(selection[["seed", "checkpoint_seed"]].to_dict("records")))
    check("checkpoint_epoch_metadata", bool((pd.to_numeric(selection["checkpoint_best_epoch"]) == selection["best_epoch_metrics_csv"]).all()), str(selection[["seed", "best_epoch_checkpoint", "best_epoch_metrics_csv"]].to_dict("records")))
    check("validation_prediction_rows", bool((selection["val_prediction_rows"] == 94).all()), str(selection["val_prediction_rows"].tolist()))
    check("test_prediction_rows_reporting_only", bool((selection["test_prediction_rows_reporting_only"] == 84).all()), str(selection["test_prediction_rows_reporting_only"].tolist()))
    check("c14b_reproduction", bool((reproduction["reproduction_pass"].astype(int) == 1).all()), str(reproduction[["seed", "max_abs_prob_diff", "reproduction_pass"]].to_dict("records")))
    check("validation_auc_frozen", abs(float(val_summary["AUC_mean"]) - 0.8664554096876415) < 1e-12, str(val_summary["AUC_mean"]))
    check("shortcut_safety_recorded", float(val_shortcut["max_abs_spearman"]) < 0.20 and float(val_shortcut["linear_r2_prob_from_shortcuts"]) < 0.10, str(val_shortcut[["max_abs_spearman", "linear_r2_prob_from_shortcuts", "shortcut_only_label_auc_audit_only"]].to_dict()))
    check("c14e_training_blocked", int(c14e_row["training_authorized"]) == 0 and c14e_row["route"] == "DATA_LIMIT_NO_GENERAL_MODEL_FIX", str(c14e_row[["route", "allowed_next_step", "training_authorized"]].to_dict()))
    check("inventory_complete", bool((inventory["exists"] == 1).all()), str(inventory[inventory["exists"] == 0]["path"].tolist()))
    verification = pd.DataFrame(checks)
    verification.to_csv(out_dir / "final_delivery_verification.csv", index=False)
    delivery_pass = bool((verification["status"] == "PASS").all())

    environment: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "repo_commit_short": git_value(repo_root, "rev-parse", "--short", "HEAD"),
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "delivery_pass": delivery_pass,
    }
    try:
        import torch

        environment.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    except Exception as exc:
        environment["torch_status"] = f"unavailable:{exc}"
    (out_dir / "server_environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    commands = f"""# C13 Final Reproducibility Commands

Run from `/home/linruixin/chen/project/DMEA-HT` with the frozen environment:

```bash
PY=/home/linruixin/chen/conda/envs/ma/bin/python
MANIFEST={manifest_path}
RUN_DIR={run_dir}
CONFIG={config_path}
```

The original formal training command was:

```bash
$PY train.py --config $CONFIG
```

This command retrains the three formal seeds `[0, 42, 3407]`; it is recorded for reproducibility and is not authorized as a new experiment.

Verify the saved C13 checkpoints and inference contract without training:

```bash
$PY scripts/collect_phase_c14b_report.py \\
  --manifest $MANIFEST \\
  --run-dir $RUN_DIR \\
  --output-dir analysis_reports/phase_c14b \\
  --device auto --batch-size 4 --seeds 0,42,3407
```

Regenerate the final delivery inventory:

```bash
$PY scripts/collect_final_c13_delivery.py \\
  --run-dir $RUN_DIR \\
  --config $CONFIG \\
  --manifest $MANIFEST \\
  --output-dir analysis_reports/final_c13_delivery
```

Formal checkpoint paths:

- `{run_dir}/checkpoints/seed_0_best.pt`
- `{run_dir}/checkpoints/seed_42_best.pt`
- `{run_dir}/checkpoints/seed_3407_best.pt`

Checkpoint selection used validation AUC only. Test metrics are reporting-only. Shortcut fields are audit-only and are not classifier inputs.
"""
    (out_dir / "final_reproducibility_commands.md").write_text(commands, encoding="utf-8")

    model_card = f"""# Frozen C13 Model Card

## Model Identity

- Route: `C13_TEMPORAL_FOCUS_DMEA_HT`
- Config: `{config_path}`
- Manifest: `{manifest_path}`
- Formal seeds: `[0, 42, 3407]`
- Primary checkpoint-selection metric: validation AUC
- Delivery status: `{'PASS' if delivery_pass else 'FAIL'}`

The frozen delivery is the C13 route with three independently trained, validation-selected single-model checkpoints. The formal performance claim is the three-seed mean; no test-selected seed or ensemble is claimed.

## Validation Performance

- AUC: `{float(val_summary['AUC_mean']):.4f} +/- {float(val_summary['AUC_std']):.4f}`
- AUPRC: `{float(val_summary['AUPRC_mean']):.4f} +/- {float(val_summary['AUPRC_std']):.4f}`
- Sensitivity: `{float(val_summary['Sensitivity_mean']):.4f} +/- {float(val_summary['Sensitivity_std']):.4f}`
- Specificity: `{float(val_summary['Specificity_mean']):.4f} +/- {float(val_summary['Specificity_std']):.4f}`

Test AUC `{float(metrics_summary[metrics_summary['split'] == 'test'].iloc[0]['AUC_mean']):.4f} +/- {float(metrics_summary[metrics_summary['split'] == 'test'].iloc[0]['AUC_std']):.4f}` is reporting-only and was not used for selection.

## Shortcut Safety

- Validation pooled max absolute Spearman: `{float(val_shortcut['max_abs_spearman']):.4f}`
- Validation pooled shortcut linear R2: `{float(val_shortcut['linear_r2_prob_from_shortcuts']):.4f}`
- Validation shortcut-only label AUC, audit-only: `{float(val_shortcut['shortcut_only_label_auc_audit_only']):.4f}`

Shortcut and audit fields are never classifier inputs.

## Known Limitations

- The validation AUC target of 0.90 was not reached.
- Sensitivity is seed-sensitive.
- C14-C found concentrated hard-patient inversion structure.
- C14-E found insufficient matched controls and no generalizable correction mechanism; route: `DATA_LIMIT_NO_GENERAL_MODEL_FIX`.
- C15 training remains blocked. C13 is frozen as the current strict best and the limitation must be reported.
"""
    (out_dir / "final_c13_model_card.md").write_text(model_card, encoding="utf-8")

    delivery_report = f"""# DMEA-HT Final C13 Delivery Report

## Final Decision

`FREEZE_C13_AS_STRICT_BEST_AND_REPORT_LIMITATION`

Delivery verification: `{'PASS' if delivery_pass else 'FAIL'}`.

## Frozen Performance

{frame_to_markdown(selection[['seed', 'best_epoch_metrics_csv', 'val_auc', 'val_auprc', 'val_sensitivity', 'val_specificity', 'test_auc_reporting_only']])}

Three-seed validation AUC is `{float(val_summary['AUC_mean']):.4f} +/- {float(val_summary['AUC_std']):.4f}`. Three-seed validation AUPRC is `{float(val_summary['AUPRC_mean']):.4f} +/- {float(val_summary['AUPRC_std']):.4f}`.

## Shortcut Safety

{frame_to_markdown(pooled_shortcut[['split', 'selection_role', 'max_abs_spearman', 'linear_r2_prob_from_shortcuts', 'shortcut_only_label_auc_audit_only']])}

## Hard-Subgroup And Data Limitation

{frame_to_markdown(limitations)}

C14-E did not identify a broad, matched-control-supported mechanism. The final action is to keep C13 and report the data limitation, not to launch C15.

## Artifact Verification

{frame_to_markdown(verification)}

## Selection Integrity

- Patient-level labels, split assignment, task definition, C13 manifest, images, bio values, and report construction are frozen.
- Checkpoints were selected by validation AUC only.
- Test metrics are reporting-only.
- No shortcut field is a classifier input.
- No C15 or post-C14-E training was authorized.

The validation AUC 0.90 target was not reached. C13 remains the final strict best under the available evidence.
"""
    (out_dir / "final_delivery_report.md").write_text(delivery_report, encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "delivery_pass": delivery_pass, "repo_commit": environment["repo_commit_short"], "manifest_rows": len(manifest_rows), "validation_auc_mean": float(val_summary["AUC_mean"]), "validation_auc_std": float(val_summary["AUC_std"]), "final_decision": "FREEZE_C13_AS_STRICT_BEST_AND_REPORT_LIMITATION"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
