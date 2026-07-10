from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = "/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl"
DEFAULT_RUN_DIR = "runs/dmea_ht_v2_c13_temporal_focus_stress_seeds"
DEFAULT_OUTPUT_DIR = "analysis_reports/phase_c14c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase C14-C pairwise ranking inversion audit reports.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", default="0,42,3407")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze_phase_c14c_pairwise_ranking.py")),
        "--manifest",
        args.manifest,
        "--run-dir",
        args.run_dir,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--seeds",
        args.seeds,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
