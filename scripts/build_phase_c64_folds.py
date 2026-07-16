#!/usr/bin/env python3
"""Build the deterministic patient-level five-fold development split for C64."""

from __future__ import annotations

import argparse
import json
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c64_common as common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c64_cv.yaml")
    args = parser.parse_args()
    config = common.load_c64_config(args.config)
    rows = common.manifest_rows(config)
    assignments = common.write_fold_artifacts(config, rows)
    fold_counts = {str(fold): int(sum(value == fold for value in assignments.values())) for fold in range(common.FOLD_COUNT)}
    payload = {
        "phase": "C64-STCV",
        "status": "C64_FOLDS_BUILT",
        "fold_seed": common.FOLD_SEED,
        "fold_count": common.FOLD_COUNT,
        "development_patient_count": len(assignments),
        "fold_counts": fold_counts,
        "test_loaded": False,
        "test_used": False,
    }
    report_dir = common.resolve_path(config["project"]["report_dir"])
    common.write_status(report_dir / "c64_fold_build_status.json", payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
