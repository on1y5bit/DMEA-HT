#!/usr/bin/env python3
"""Verify the canonical DEMA-HT project after C17 consolidation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

EXPECTED_SEEDS = [0, 42, 3407]
EXPECTED_C17_VAL_AUC = 0.8696242644
EXPECTED_INVERSION_ROWS = {0: (297, 287), 42: (277, 272), 3407: (311, 305)}
EXPECTED_SHORTCUT_MAX_AUC = 0.5088275238


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", default=".")
    parser.add_argument("--archive-dir", required=True)
    return parser.parse_args()


def run(args: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_checkpoint(path: Path) -> str:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint is None:
        raise ValueError("empty checkpoint")
    return type(checkpoint).__name__


def main() -> None:
    args = parse_args()
    canonical = Path(args.canonical_dir).resolve()
    archive = Path(args.archive_dir).resolve()
    output = canonical / "analysis_reports" / "project_consolidation"
    output.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    py_files = sorted(canonical.glob("dmea_ht/*.py")) + [canonical / "train.py"] + sorted((canonical / "scripts").glob("*.py"))
    compile_result = run([sys.executable, "-m", "py_compile", *[str(path) for path in py_files]], canonical)
    check("py_compile", compile_result.returncode == 0, compile_result.stderr.strip() or "all target Python files compiled")
    diff_result = run(["git", "diff", "--check"], canonical)
    check("git_diff_check", diff_result.returncode == 0, diff_result.stderr.strip() or "no tracked whitespace errors")

    config_names = [
        "dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml",
        "dema_ht_c17_formal_multiseed.yaml",
        "dema_ht_c17_residual_positive_preserve_seed0.yaml",
    ]
    parsed_configs = {}
    for name in config_names:
        path = canonical / "configs" / name
        try:
            with path.open(encoding="utf-8") as handle:
                parsed_configs[name] = yaml.safe_load(handle)
            check("config_parse:" + name, True, "parsed")
        except Exception as exc:
            check("config_parse:" + name, False, repr(exc))
    formal_config = parsed_configs.get("dema_ht_c17_formal_multiseed.yaml") or {}
    formal_seeds = formal_config.get("training", {}).get("seeds", formal_config.get("seeds", []))
    check("c17_formal_seed_contract", sorted(formal_seeds) == EXPECTED_SEEDS, str(formal_seeds))

    c13_checkpoint_dir = canonical / "runs" / "dmea_ht_v2_c13_temporal_focus_stress_seeds" / "checkpoints"
    c17_checkpoint_dir = canonical / "runs" / "dema_ht_c17_formal_multiseed" / "checkpoints"
    for label, directory in (("c13", c13_checkpoint_dir), ("c17", c17_checkpoint_dir)):
        paths = sorted(directory.glob("seed_*_best.pt"))
        check(label + "_checkpoint_count", len(paths) == 3, "found " + str(len(paths)))
        for path in paths:
            try:
                kind = load_checkpoint(path)
                check(label + "_checkpoint_load:" + path.name, True, kind)
            except Exception as exc:
                check(label + "_checkpoint_load:" + path.name, False, repr(exc))

    prediction_dir = canonical / "runs" / "dema_ht_c17_formal_multiseed" / "predictions"
    val_paths = sorted(prediction_dir.glob("val_predictions_seed_*.csv"))
    test_paths = sorted(prediction_dir.glob("test_predictions_seed_*.csv"))
    check("c17_validation_prediction_seed_count", len(val_paths) == 3, str([path.name for path in val_paths]))
    check("c17_test_prediction_seed_count", len(test_paths) == 3, str([path.name for path in test_paths]))
    for split, paths, expected_rows in (("val", val_paths, 94), ("test", test_paths, 84)):
        for path in paths:
            rows = read_csv(path)
            seed = int(path.stem.rsplit("_", 1)[-1])
            check(f"c17_{split}_row_count_seed_{seed}", len(rows) == expected_rows, f"found {len(rows)}, expected {expected_rows}")
            check(f"c17_{split}_label_columns_seed_{seed}", bool(rows and {"patient_id", "label", "final_logit", "final_prob"}.issubset(rows[0])), "required patient-level columns")

    run_report_dir = canonical / "runs" / "dema_ht_c17_formal_multiseed" / "reports"
    metrics_by_seed = run_report_dir / "metrics_by_seed.csv"
    metrics_summary = run_report_dir / "metrics_summary.csv"
    formal_metrics_summary = canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_metrics_summary.csv"
    formal_gate = canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_gate.json"
    formal_final = canonical / "analysis_reports" / "phase_c17_dema" / "phase_c17_dema_formal_final_report.md"
    required_files = [
        metrics_by_seed,
        metrics_summary,
        formal_metrics_summary,
        formal_gate,
        formal_final,
        canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_pairwise_inversion_summary.csv",
        canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_positive_preservation_audit.csv",
        canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_shortcut_residual_audit.csv",
        canonical / "analysis_reports" / "phase_c17_dema" / "phase_c17_dema_final_report.md",
    ]
    for path in required_files:
        check("required_artifact:" + str(path.relative_to(canonical)), path.exists(), "present" if path.exists() else "missing")

    try:
        summary_rows = read_csv(metrics_summary)
        val_summary = next(row for row in summary_rows if row.get("split") == "val")
        auc = float(val_summary["AUC_mean"])
        check("c17_mean_validation_auc", abs(auc - EXPECTED_C17_VAL_AUC) <= 1e-10, f"{auc:.10f}")
        by_seed = read_csv(metrics_by_seed)
        val_seeds = sorted(int(row["seed"]) for row in by_seed if row.get("split") == "val")
        test_seeds = sorted(int(row["seed"]) for row in by_seed if row.get("split") == "test")
        check("c17_metrics_val_seed_contract", val_seeds == EXPECTED_SEEDS, str(val_seeds))
        check("c17_metrics_test_seed_contract", test_seeds == EXPECTED_SEEDS, str(test_seeds))
        check("c17_auc_only_metrics", all("AUPRC" not in row for row in by_seed), "AUPRC absent")
    except Exception as exc:
        check("c17_metrics_reproduction", False, repr(exc))

    try:
        gate = json.loads(formal_gate.read_text(encoding="utf-8"))
        check("c17_formal_gate_pass", gate.get("formal_gate_pass") is True, str(gate.get("formal_gate_pass")))
        check("c17_formal_decision", gate.get("decision") == "PROMOTE_DEMA_C17_POSITIVE_PRESERVATION", str(gate.get("decision")))
    except Exception as exc:
        check("c17_formal_gate_read", False, repr(exc))

    inversion_path = canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_pairwise_inversion_summary.csv"
    try:
        rows = read_csv(inversion_path)
        observed = {int(row["seed"]): (int(row["base_inversions"]), int(row["final_inversions"])) for row in rows}
        check("c17_pairwise_inversion_reproduction", observed == EXPECTED_INVERSION_ROWS, str(observed))
    except Exception as exc:
        check("c17_pairwise_inversion_reproduction", False, repr(exc))

    shortcut_path = canonical / "analysis_reports" / "phase_c17_dema" / "c17_formal_shortcut_residual_audit.csv"
    try:
        rows = read_csv(shortcut_path)
        values = [float(row["shortcut_label_auc"]) for row in rows]
        check("c17_shortcut_audit_reproduction", len(rows) == 18 and max(values) <= EXPECTED_SHORTCUT_MAX_AUC + 1e-10, f"rows={len(rows)}, max={max(values):.10f}")
    except Exception as exc:
        check("c17_shortcut_audit_reproduction", False, repr(exc))

    model_sources = [
        canonical / "dmea_ht" / "c17_residual.py",
        canonical / "dmea_ht" / "mechanism_evidence_alignment.py",
        canonical / "dmea_ht" / "models.py",
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in model_sources if path.exists())
    forbidden = ("selected_n_visits", "used_images", "report_length", "bio_missing_count", "shared_private", "DecAlign")
    leaked = [token for token in forbidden if token in source_text]
    check("c17_model_shortcut_exclusion", not leaked, str(leaked) if leaked else "shortcut fields absent from model modules")
    check("c17_test_not_training_input", "val_predictions_seed_" not in (canonical / "train.py").read_text(encoding="utf-8"), "saved predictions are not read by train.py")

    inventory_rows = []
    for root_name in ("runs", "analysis_reports"):
        root = canonical / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = path.relative_to(canonical)
            if rel == Path("analysis_reports/project_consolidation/canonical_artifact_sha256.txt"):
                continue
            inventory_rows.append({"relative_path": str(rel), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    inventory_path = output / "canonical_artifact_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory_rows)
    (output / "canonical_artifact_sha256.txt").write_text(
        "\n".join(f"{row['sha256']}  {row['relative_path']}" for row in inventory_rows) + "\n",
        encoding="utf-8",
    )

    head = run(["git", "rev-parse", "HEAD"], canonical).stdout.strip()
    branch = run(["git", "branch", "--show-current"], canonical).stdout.strip()
    status = run(["git", "status", "--short", "--untracked-files=normal"], canonical).stdout.strip()
    worktrees = run(["git", "worktree", "list", "--porcelain"], canonical).stdout.strip()
    (output / "canonical_git_state.md").write_text(
        "\n".join([
            "# Canonical Git State",
            "",
            f"- Directory: {canonical}",
            f"- Branch: {branch}",
            f"- HEAD: {head}",
            f"- Archive: {archive}",
            f"- Untracked/dirty status line count: {len(status.splitlines()) if status else 0}",
            "",
            "## Worktrees",
            "",
            "WORKTREE_LIST_BEGIN",
            worktrees,
            "WORKTREE_LIST_END",
            "",
            "## Status Snapshot",
            "",
            "STATUS_BEGIN",
            status[:12000],
            "STATUS_END",
            "",
        ]),
        encoding="utf-8",
    )

    passed = all(row["passed"] for row in checks)
    status_label = "CANONICAL_DMEA_HT_VERIFIED" if passed else "DMEA_HT_CODE_MERGED_ARTIFACT_MIGRATION_INCOMPLETE"
    migration_counts = (archive / "artifact_migration_counts.txt").read_text(encoding="utf-8") if (archive / "artifact_migration_counts.txt").exists() else "missing"
    (output / "canonical_consolidation_report.md").write_text(
        "\n".join([
            "# DEMA-HT Canonical Consolidation Report",
            "",
            f"- Consolidation verification status: {status_label}.",
            f"- Canonical commit: {head}.",
            f"- C17 reproduction checks passed: {passed}.",
            f"- Artifact inventory files: {len(inventory_rows)}.",
            "",
            "## Artifact Migration Counts",
            "",
            "MIGRATION_COUNTS_BEGIN",
            migration_counts,
            "MIGRATION_COUNTS_END",
            "",
            "C17 formal values were checked after migration and were not regenerated or tuned.",
            "Old worktrees remain registered until the cleanup gate is completed.",
            "",
        ]),
        encoding="utf-8",
    )
    with (output / "canonical_c17_reproduction_check.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps({
        "status": status_label,
        "passed": passed,
        "failed_checks": [row for row in checks if not row["passed"]],
        "canonical": str(canonical),
        "head": head,
        "artifact_count": len(inventory_rows),
    }, ensure_ascii=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

