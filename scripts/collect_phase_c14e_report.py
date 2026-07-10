from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = "/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C14-E hard clinical evidence audit reports.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--c14a-dir", default="analysis_reports/phase_c14a")
    parser.add_argument("--c14b-dir", default="analysis_reports/phase_c14b")
    parser.add_argument("--c14c-dir", default="analysis_reports/phase_c14c")
    parser.add_argument("--c14d-dir", default="analysis_reports/phase_c14d")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c14e")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze_phase_c14e_hard_clinical_evidence.py")),
        "--manifest",
        args.manifest,
        "--c14a-dir",
        args.c14a_dir,
        "--c14b-dir",
        args.c14b_dir,
        "--c14c-dir",
        args.c14c_dir,
        "--c14d-dir",
        args.c14d_dir,
        "--output-dir",
        args.output_dir,
        "--bootstrap-iters",
        str(args.bootstrap_iters),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
