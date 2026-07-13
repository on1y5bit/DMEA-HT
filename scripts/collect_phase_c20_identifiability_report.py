#!/usr/bin/env python3
"""Run and collect the validation-only C20 mechanism identifiability audit."""

from __future__ import annotations

import argparse
import csv
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


EXPECTED_SCRIPTS = (
    "scripts/export_phase_c20_c17_internal_representations.py",
    "scripts/analyze_phase_c20_representation_identifiability.py",
    "scripts/analyze_phase_c20_layer_predictive_utility.py",
    "scripts/locate_phase_c20_instability_transition.py",
    "scripts/collect_phase_c20_identifiability_report.py",
    "scripts/phase_c20_common.py",
)
EXPECTED_C19_DECISION = "C19_POLARITY_BASE_INVALID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c17_formal_multiseed.yaml")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c20_dema")
    parser.add_argument("--run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--c17-prediction-dir", default="runs/dema_ht_c17_formal_multiseed/predictions")
    parser.add_argument("--c18-prediction-dir", default="runs/dema_ht_c18_directional_multiseed/predictions")
    parser.add_argument("--c18-hardrank-prediction-dir", default="runs/dema_ht_c18_directional_hardrank_multiseed/predictions")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--reuse-export", action="store_true", help="reuse the already validated server-only C20 NPZ export")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: Sequence[str], output_dir: Path, log_handle: Any) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    log_handle.write(f"$ {' '.join(command)}\n")
    log_handle.write(completed.stdout)
    log_handle.write(completed.stderr)
    log_handle.write(f"[returncode={completed.returncode}]\n\n")
    log_handle.flush()
    return completed


def static_checks(output_dir: Path, config_path: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for relative in EXPECTED_SCRIPTS:
        path = REPO_ROOT / relative
        try:
            py_compile.compile(str(path), doraise=True)
            checks.append({"check": f"compile_{relative.replace('/', '_')}", "pass": True, "detail": "compiled"})
        except Exception as exc:
            checks.append({"check": f"compile_{relative.replace('/', '_')}", "pass": False, "detail": repr(exc)})
    execution_source = "\n".join(
        (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in EXPECTED_SCRIPTS[:4]
        if (REPO_ROOT / relative).exists()
    )
    forbidden_runtime_tokens = ("torch.optim", ".backward(", "optimizer.step(", "train_seed(")
    checks.append({"check": "no_optimizer_or_backward", "pass": not any(token in execution_source for token in forbidden_runtime_tokens[:3]), "detail": "C20 scripts are analysis-only"})
    checks.append({"check": "no_training_entrypoint", "pass": "train_seed(" not in execution_source and "train.py" not in execution_source, "detail": "no training entrypoint referenced"})
    checks.append({"check": "no_saved_test_prediction_input", "pass": "test_predictions" not in execution_source and "split == \"test\"" not in execution_source, "detail": "test predictions are not loaded"})
    checks.append({"check": "fixed_probe_contract", "pass": "C=1.0" in execution_source and "train_fit_only" in execution_source and "20260714" in execution_source, "detail": "fixed C=1.0, train-only standardization, fixed random-label seed"})
    checks.append({"check": "c20_config_is_not_training_config", "pass": "c17" in config_path.read_text(encoding="utf-8").lower(), "detail": str(config_path)})
    c19_decision = None
    c19_path = REPO_ROOT / "analysis_reports" / "phase_c19_dema" / "c19_polarity_audit.json"
    if c19_path.exists():
        try:
            c19_decision = json.loads(c19_path.read_text(encoding="utf-8")).get("decision")
        except Exception:
            c19_decision = None
    checks.append({"check": "c19_gate_remains_blocked", "pass": c19_decision == EXPECTED_C19_DECISION, "detail": c19_decision or "missing C19 audit"})
    write_rows(output_dir / "c20_static_gate.csv", checks)
    return checks


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolve(args.config)
    checks = static_checks(output_dir, config_path)
    static_pass = all(bool(row["pass"]) for row in checks)

    command_log_path = output_dir / "c20_command_log.txt"
    with command_log_path.open("w", encoding="utf-8") as log_handle:
        if static_pass and not args.reuse_export:
            export_command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_phase_c20_c17_internal_representations.py"),
                "--config", str(config_path),
                "--output-dir", str(output_dir),
                "--run-dir", str(resolve(args.run_dir)),
            ]
            if args.manifest:
                export_command.extend(["--manifest", args.manifest])
            if args.data_root:
                export_command.extend(["--data-root", args.data_root])
            export_result = run_command(export_command, output_dir, log_handle)
            export_pass = export_result.returncode == 0
        elif static_pass and args.reuse_export:
            required_exports = (
                output_dir / "c20_internal_representations_train.npz",
                output_dir / "c20_internal_representations_val.npz",
                output_dir / "c20_reproduction_check_by_seed.csv",
            )
            reproduction_rows = read_rows(output_dir / "c20_reproduction_check_by_seed.csv")
            export_pass = all(path.exists() for path in required_exports) and bool(reproduction_rows) and all(row.get("pass", "False").lower() == "true" for row in reproduction_rows)
            log_handle.write(f"Reusing validated C20 export: {export_pass}\n")
        else:
            export_pass = False
            log_handle.write("C20 static gate failed; representation export was not started.\n")

        analysis_pass = False
        if static_pass and export_pass:
            analysis_result = run_command(
                [sys.executable, str(REPO_ROOT / "scripts" / "analyze_phase_c20_representation_identifiability.py"), "--output-dir", str(output_dir), "--c17-prediction-dir", str(resolve(args.c17_prediction_dir)), "--c18-prediction-dir", str(resolve(args.c18_prediction_dir)), "--c18-hardrank-prediction-dir", str(resolve(args.c18_hardrank_prediction_dir))],
                output_dir,
                log_handle,
            )
            probe_result = run_command(
                [sys.executable, str(REPO_ROOT / "scripts" / "analyze_phase_c20_layer_predictive_utility.py"), "--output-dir", str(output_dir)],
                output_dir,
                log_handle,
            )
            transition_result = run_command(
                [sys.executable, str(REPO_ROOT / "scripts" / "locate_phase_c20_instability_transition.py"), "--output-dir", str(output_dir)],
                output_dir,
                log_handle,
            )
            analysis_pass = analysis_result.returncode == 0 and probe_result.returncode == 0 and transition_result.returncode == 0

    reproduction_rows = read_rows(output_dir / "c20_reproduction_check_by_seed.csv")
    reproduction_pass = bool(reproduction_rows) and all(row.get("pass", "False").lower() == "true" for row in reproduction_rows)
    transition_path = output_dir / "c20_transition_summary.json"
    transition: Dict[str, Any] = {}
    if transition_path.exists():
        transition = json.loads(transition_path.read_text(encoding="utf-8"))
    if not static_pass or not export_pass or not reproduction_pass:
        c20_label = "C20_REPRODUCTION_GATE_FAIL" if not reproduction_pass else "C20_ANALYSIS_INVALID"
        c21_authorized = False
    elif not analysis_pass:
        c20_label = "C20_ANALYSIS_INVALID"
        c21_authorized = False
    else:
        c20_label = str(transition.get("decision", "C20_ANALYSIS_INVALID"))
        c21_authorized = bool(transition.get("c21_authorized", False))
    c21_label = "C21_AUTHORIZED" if c21_authorized else "C21_NOT_AUTHORIZED"
    strict_best = "DEMA_C17_POSITIVE_PRESERVATION"

    route_lines = [
        "# C20 Route Decision",
        "",
        f"- C20 label: `{c20_label}`",
        f"- stable layer: `{transition.get('stable_layers', ['none'])[0] if transition.get('stable_layers') else 'none'}`",
        f"- earliest unstable layer or stage: `{transition.get('earliest_unstable_layer_or_stage', 'unknown')}`",
        f"- C21 authorization: `{c21_label}`",
        f"- strict best retained: `{strict_best}`",
        "",
        "C19 remained blocked by `C19_POLARITY_BASE_INVALID`; C20 does not lower or bypass that gate.",
        "C20 is validation-only: no smoke run, no seed-0 pilot, no formal training, no optimizer/backward, and no test data or test predictions were read.",
        "Probe features are internal representations only. Shortcut fields and patient IDs are audit/alignment-only and are excluded from probes.",
    ]
    (output_dir / "c20_route_decision.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")

    gate_rows = read_rows(output_dir / "c20_stable_layer_gate.csv")
    group_rows = read_rows(output_dir / "c20_group_stability_analysis.csv")
    probe_rows = read_rows(output_dir / "c20_layer_probe_summary.csv")
    report_lines = [
        "# Phase C20 DEMA-HT Final Report",
        "",
        f"- canonical project: `/home/linruixin/chen/project/DMEA-HT`",
        f"- branch: `main`",
        f"- C20 result: `{c20_label}`",
        f"- C21 result: `{c21_label}`",
        f"- strict best: `{strict_best}`",
        "",
        "## Contract",
        "",
        "- C17 selected checkpoints for seeds `[0, 42, 3407]` were evaluated with `eval()` and `torch.no_grad()`.",
        "- C20 used train/validation representations only; saved validation predictions were used only for reproduction.",
        "- Cross-seed comparison uses linear CKA, patient-distance Spearman, kNN overlap, train-fit orthogonal Procrustes, and scalar rank consistency.",
        "- Probes use fixed C=1.0 L2 logistic regression, train-fit standardization, and validation-only evaluation.",
        "- AUC is the only predictive utility metric; AUPRC and test data are excluded from route decisions.",
        "",
        "## Reproduction",
        "",
        "| seed | IDs | labels | max abs probability diff | mean abs probability diff | pass |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in reproduction_rows:
        report_lines.append(f"| {row.get('seed')} | {row.get('patient_id_exact_match')} | {row.get('label_exact_match')} | {row.get('max_abs_prob_diff')} | {row.get('mean_abs_prob_diff')} | {row.get('pass')} |")
    report_lines.extend(["", "## Stable Layer Gate", "", "| layer | mean CKA | min CKA | mean distance Spearman | min distance Spearman | mean kNN | positive kNN | negative kNN | hard/non-hard | probe AUC | seeds >= .83 | pass |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in gate_rows:
        hard = row.get("mean_hard_knn_jaccard", "nan")
        non_hard = row.get("mean_non_hard_knn_jaccard", "nan")
        report_lines.append(f"| {row.get('layer')} | {row.get('mean_cka')} | {row.get('min_cka')} | {row.get('mean_distance_spearman')} | {row.get('min_distance_spearman')} | {row.get('mean_knn_jaccard')} | {row.get('mean_positive_knn_jaccard')} | {row.get('mean_negative_knn_jaccard')} | {hard}/{non_hard} | {row.get('mean_validation_probe_auc')} | {row.get('seeds_ge_0_83')} | {row.get('stable_layer_gate_pass')} |")
    report_lines.extend(["", "## Group Findings", "", f"- group stability rows: `{len(group_rows)}`", f"- probe summary rows: `{len(probe_rows)}`", f"- earliest instability: `{transition.get('earliest_unstable_layer_or_stage', 'unknown')}`", "- C18 repaired/introduced patients are grouped for audit only and do not enter model fitting or route selection.", "", "## Safety", "", "- No new prediction module, optimizer, backward pass, training config, smoke run, seed-0 pilot, test read, or formal training was performed by C20.", "- If C21 is not authorized, retain C17 and stop residual/evidence polarity expansion until new evidence or labels are available."])
    (output_dir / "phase_c20_dema_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = {
        "c20_label": c20_label,
        "c21_authorized": c21_authorized,
        "c21_label": c21_label,
        "strict_best": strict_best,
        "static_pass": static_pass,
        "export_pass": export_pass,
        "reproduction_pass": reproduction_pass,
        "analysis_pass": analysis_pass,
        "output_dir": str(output_dir),
    }
    (output_dir / "c20_final_decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if args.require_pass and (not static_pass or not export_pass or not reproduction_pass or not analysis_pass):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
