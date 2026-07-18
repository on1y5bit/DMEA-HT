#!/usr/bin/env python3
"""Build C66's deterministic fold-local nested patient splits without opening Test."""

from __future__ import annotations

import argparse
import json
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    args = parser.parse_args()

    config = common.load_c66_config(args.config)
    inventory = common.read_c64_development_inventory(config)
    payload = common.build_nested_split_payload(config)
    integrity = common.validate_nested_split_payload(payload, inventory, config)
    if not integrity["all_pass"]:
        raise RuntimeError("C66 nested split construction failed its strict isolation checks")

    split_path = common.nested_split_path(config)
    common.write_json(split_path, payload)
    common.nested_split_summary(payload, inventory).to_csv(
        split_path.parent / "nested_split_summary.csv", index=False
    )
    common.write_json(common.report_dir(config) / "c66a_nested_split_integrity.json", integrity)
    print(
        json.dumps(
            {
                "phase": "C66-LFFC",
                "status": "C66_NESTED_SPLITS_BUILT",
                "split_path": str(split_path),
                "test_loaded": False,
                "test_rows_read": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
