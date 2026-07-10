from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C14-D hard-patient subgroup audit reports.")
    parser.add_argument("--c14c-dir", default="analysis_reports/phase_c14c")
    parser.add_argument("--c14b-dir", default="analysis_reports/phase_c14b")
    parser.add_argument("--output-dir", default="analysis_reports/phase_c14d")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze_phase_c14d_hard_patient_audit.py")),
        "--c14c-dir",
        args.c14c_dir,
        "--c14b-dir",
        args.c14b_dir,
        "--output-dir",
        args.output_dir,
        "--top-k",
        str(args.top_k),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
