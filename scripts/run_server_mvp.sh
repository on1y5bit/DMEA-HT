#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ENV_PY="/home/linruixin/chen/conda/envs/ma/bin/python"
DATA_ROOT="${DATA_ROOT:-/data/csb/DMEA-HT/HT_2025.12_25}"
MANIFEST="${MANIFEST:-$DATA_ROOT/manifest.jsonl}"
OUT_DIR="${OUT_DIR:-runs/dmea_ht_mvp_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_DIR"

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Create a patient-level manifest first, then rerun." >&2
  exit 2
fi

"$ENV_PY" scripts/audit_shortcut_distribution.py \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR/reports/pretrain_shortcut_audit"

"$ENV_PY" train.py --config configs/dmea_ht_mvp.yaml \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR"
