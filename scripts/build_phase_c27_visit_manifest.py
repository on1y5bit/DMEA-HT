#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.visit_data import (  # noqa: E402
    build_visit_manifest,
    load_source_visits,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the C27 real visit-level manifest without resampling visits.")
    parser.add_argument("--data-root", default="/data/csb/DMEA-HT/HT_2025.12_25")
    parser.add_argument(
        "--base-manifest",
        default="/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c27_visit_level.jsonl",
    )
    parser.add_argument("--summary", default="analysis_reports/phase_c27_visit_design/c27_visit_manifest_build.json")
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_manifest)
    source_visits = load_source_visits(args.data_root)
    rows = build_visit_manifest(base_rows, source_visits)
    write_jsonl(args.output, rows)
    payload = {
        "phase": "C27-VTME",
        "patients": len(rows),
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "visit_manifest": str(Path(args.output).resolve()),
        "visit_manifest_sha256": sha256_file(args.output),
        "source": "all_patients.xlsx exact patient-date plus C13 selected dates/images",
        "test_used_for_rule_design": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
