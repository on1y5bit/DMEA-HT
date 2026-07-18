#!/usr/bin/env python3
"""Verify C66's exact public initialization before any patient-data training."""

from __future__ import annotations

import argparse
import json
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_training_common as common  # noqa: E402
from scripts import c66_common as protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    args = parser.parse_args()
    config = protocol.load_c66_config(args.config)
    payload = common.write_runtime_preflight(config)
    print(json.dumps({"phase": payload["phase"], "status": payload["status"], "test_loaded": False}))


if __name__ == "__main__":
    main()
