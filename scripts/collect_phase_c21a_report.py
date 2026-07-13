#!/usr/bin/env python3
"""Run and consolidate the C21-A validation-only mechanism propagation audit."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "phase_c21a_common.py",
    "export_phase_c21a_mechanism_propagation_trace.py",
    "analyze_phase_c21a_node_stability.py",
    "analyze_phase_c21a_edge_stability.py",
    "analyze_phase_c21a_inference_edge_ablation.py",
    "analyze_phase_c21a_inference_node_bypass.py",
    "score_phase_c21a_instability_responsibility.py",
    "collect_phase_c21a_report.py",
)
FORBIDDEN_PATTERNS = (
    r"torch\.optim",
    r"\.backward\s*\(",
    r"optimizer\.step\s*\(",
    r"loss\.backward\s*\(",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c17_formal_multiseed.yaml")
    parser.add_argument("--run-dir", default="runs/dema_ht_c17_formal_multiseed")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c21a_dema")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reuse-trace", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def run_command(command: Sequence[str], log_handle: Any) -> None:
    log_handle.write("$ " + " ".join(command) + "\n")
    log_handle.flush()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.stdout:
        log_handle.write(completed.stdout)
    if completed.stderr:
        log_handle.write(completed.stderr)
    log_handle.write(f"exit_code={completed.returncode}\n\n")
    log_handle.flush()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def static_gate() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    passed = True
    for name in SCRIPT_NAMES:
        path = REPO_ROOT / "scripts" / name
        exists = path.exists()
        violations: List[str] = []
        if exists:
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, text):
                    violations.append(pattern)
        item_pass = exists and not violations
        passed = passed and item_pass
        rows.append({"script": name, "exists": exists, "forbidden_runtime_patterns": ";".join(violations), "pass": item_pass})
    return {"pass": passed, "rows": rows}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    static = static_gate()
    with (output_dir / "c21a_static_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static["rows"][0]))
        writer.writeheader()
        writer.writerows(static["rows"])
    if not static["pass"]:
        raise SystemExit("C21A static gate failed")

    command_log_path = output_dir / "c21a_command_log.txt"
    python = sys.executable
    common_args = [
        "--config",
        args.config,
        "--run-dir",
        args.run_dir,
        "--output-dir",
        args.output_dir,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.manifest:
        common_args.extend(["--manifest", args.manifest])
    if args.data_root:
        common_args.extend(["--data-root", args.data_root])

    with command_log_path.open("w", encoding="utf-8") as log_handle:
        trace_files = [output_dir / "c21a_trace_train.npz", output_dir / "c21a_trace_val.npz", output_dir / "c21a_reproduction_check_by_seed.csv"]
        if not args.reuse_trace or not all(path.exists() for path in trace_files):
            run_command([python, "scripts/export_phase_c21a_mechanism_propagation_trace.py", *common_args], log_handle)
        run_command([python, "scripts/analyze_phase_c21a_node_stability.py", "--trace-dir", args.output_dir], log_handle)
        run_command([python, "scripts/analyze_phase_c21a_edge_stability.py", "--trace-dir", args.output_dir], log_handle)
        run_command([python, "scripts/analyze_phase_c21a_inference_edge_ablation.py", *common_args], log_handle)
        run_command([python, "scripts/analyze_phase_c21a_inference_node_bypass.py", *common_args], log_handle)
        run_command([python, "scripts/score_phase_c21a_instability_responsibility.py", "--trace-dir", args.output_dir], log_handle)

    summary_path = output_dir / "c21a_score_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reproduction_rows = read_csv(output_dir / "c21a_reproduction_check_by_seed.csv")
    reproduction_pass = bool(reproduction_rows) and all(row.get("pass", "false").lower() == "true" for row in reproduction_rows)
    c20_path = output_dir.parent / "phase_c20_dema" / "c20_final_decision.json"
    c20_summary: Dict[str, Any] = {}
    if c20_path.exists():
        try:
            c20_summary = json.loads(c20_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            c20_summary = {"path": str(c20_path), "readable": False}

    top = summary.get("top_responsibility") or {}
    second = summary.get("second_responsibility") or {}
    report_lines = [
        "# Phase C21-A DEMA-HT Final Report",
        "",
        "- canonical project: `/home/linruixin/chen/project/DMEA-HT`",
        "- phase type: validation-only mechanism propagation responsibility audit",
        f"- config: `{args.config}`",
        f"- run directory: `{args.run_dir}`",
        "- seeds: `0, 42, 3407`",
        "- splits read: `train` and `val` only; test data and test predictions were not read",
        "- runtime: `eval()` and `torch.no_grad()` only; no optimizer, backward pass, checkpoint update, or training was performed",
        "- branch/worktree policy: canonical `main` only; no branch or worktree created",
        "",
        "## Gates",
        "",
        f"- static gate: `{'PASS' if static['pass'] else 'FAIL'}`",
        f"- C17 forward reproduction gate: `{'PASS' if reproduction_pass else 'FAIL'}`",
        f"- C21-A route: `{summary.get('route')}`",
        f"- localized reproducible responsibility: `{summary.get('localized_reproducible')}`",
        f"- C22 design authorization: `{summary.get('c22_design_authorized')}`",
        "- training authorization: `False` (C21-A is analysis-only regardless of route)",
        "",
        "## Responsibility Ranking",
        "",
        f"- top: `{top.get('entity')}` ({top.get('category')}) score `{top.get('responsibility_score')}`",
        f"- second: `{second.get('entity')}` ({second.get('category')}) score `{second.get('responsibility_score')}`",
        f"- top ablation seed Spearman: `{summary.get('top_ablation_seed_spearman')}`",
        f"- top ablation direction consistency: `{summary.get('top_ablation_direction_consistency')}`",
        "",
        "The score combines min-max-normalized CKA drop, pairwise-distance Spearman drop, kNN-neighborhood drop, cross-seed ablation inconsistency, and message saturation/collapse using weights 0.30, 0.25, 0.20, 0.15, and 0.10.",
        "",
        "## Interpretation Boundary",
        "",
        "- A localized route identifies a reproducible responsibility candidate for a future design audit; it does not authorize training or change the task definition.",
        "- A diffuse route means mechanism propagation remains non-identifiable at the current evidence resolution; C22 is not authorized.",
        "- Shortcut values are validation-only audit diagnostics. They are excluded from the model and all representation probes.",
        "- The saved C17 strict-best result remains the reporting baseline unless a later authorized phase changes that decision.",
        "",
        "## Prior Context",
        "",
        f"- C20 decision artifact: `{c20_path}`" if c20_path.exists() else "- C20 decision artifact: unavailable in this checkout",
        f"- C20 summary: `{c20_summary.get('route', c20_summary.get('decision', 'not parsed'))}`" if c20_summary else "- C20 summary: not parsed",
        "",
        "## Artifacts",
        "",
        "- `c21a_environment_and_input_inventory.md`",
        "- `c21a_trace_tensor_inventory.csv`",
        "- `c21a_reproduction_check_by_seed.csv` and `c21a_reproduction_check_report.md`",
        "- `c21a_node_stability_by_stage.csv` and `c21a_edge_stability.csv`",
        "- `c21a_conflict_reliability_stability.csv` and `c21a_edge_weight_consistency.csv`",
        "- `c21a_edge_ablation_summary.csv` and `c21a_node_bypass_summary.csv`",
        "- `c21a_instability_responsibility_scores.csv` and `c21a_score_summary.json`",
        "- `c21a_shortcut_exclusion_audit.csv`",
        "- `c21a_command_log.txt`",
        "",
        "Large NPZ trace archives are server-only artifacts and are not committed to Git.",
    ]
    (output_dir / "phase_c21a_dema_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "phase": "C21-A",
        "static_gate_pass": bool(static["pass"]),
        "reproduction_pass": reproduction_pass,
        "route": summary.get("route"),
        "localized_reproducible": summary.get("localized_reproducible"),
        "c22_design_authorized": summary.get("c22_design_authorized"),
        "training_authorized": False,
        "test_data_read": False,
        "top_responsibility": top,
        "artifacts": [str(path) for path in sorted(output_dir.iterdir()) if path.is_file()],
    }
    (output_dir / "c21a_final_decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, default=str))
    if args.require_pass and (not static["pass"] or not reproduction_pass):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
