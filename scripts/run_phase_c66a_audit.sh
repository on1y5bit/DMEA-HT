#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linruixin/chen/conda/envs/ma/bin/python}"
CONFIG="${1:-configs/dema_ht_c66_source_learning.yaml}"

cd "$ROOT"
"$PYTHON_BIN" scripts/build_phase_c66_nested_splits.py --config "$CONFIG"
"$PYTHON_BIN" scripts/audit_phase_c66_prior_checkpoint_overlap.py --config "$CONFIG"
"$PYTHON_BIN" scripts/gate_phase_c66_nested_cv.py --config "$CONFIG"
