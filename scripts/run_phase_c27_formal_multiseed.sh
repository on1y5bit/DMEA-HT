#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/linruixin/chen/project/DMEA-HT
PYTHON=/home/linruixin/chen/conda/envs/ma/bin/python
CONFIG=configs/dema_ht_c27_vtme_multiseed.yaml
RUN_DIR=runs/dema_ht_c27_vtme_multiseed
REPORT_DIR=analysis_reports/phase_c27_dema
MASTER_LOG=phase_c27_vtme_parallel_20260714.log

cd "$REPO_ROOT"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("analysis_reports/phase_c27_dema/c27_static_synthetic_gate.json").read_text())
visit = json.loads(Path("analysis_reports/phase_c27_visit_design/c27_visit_reconstruction_decision.json").read_text())
if gate.get("decision") != "C27_VTME_DIRECT_MULTI_SEED_AUTHORIZED":
    raise SystemExit("C27 full gate is not authorized")
if visit.get("decision") != "C27_VISIT_RECONSTRUCTION_PASS":
    raise SystemExit("C27 visit reconstruction is not authorized")
PY

if [[ -e "$RUN_DIR/reports/run_status.json" || -e "$RUN_DIR/seed_runs" ]]; then
  echo "C27 run output already exists; refusing to overwrite" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$REPORT_DIR"
declare -a pids=()
for seed in 0 42 3407; do
  seed_log="phase_c27_vtme_seed_${seed}_20260714.log"
  "$PYTHON" scripts/train_phase_c27.py \
    --config "$CONFIG" \
    --stage validation-seed \
    --seed "$seed" >"$seed_log" 2>&1 &
  pids+=("$!")
  echo "started seed=$seed pid=$! log=$seed_log" | tee -a "$MASTER_LOG"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more C27 validation shards failed" | tee -a "$MASTER_LOG" >&2
  exit 3
fi

"$PYTHON" scripts/train_phase_c27.py --config "$CONFIG" --stage validation-finalize | tee -a "$MASTER_LOG"
"$PYTHON" scripts/collect_phase_c27_formal_report.py --validation-only | tee -a "$MASTER_LOG"
"$PYTHON" scripts/train_phase_c27.py --config "$CONFIG" --stage reporting-test | tee -a "$MASTER_LOG"
"$PYTHON" scripts/collect_phase_c27_formal_report.py | tee -a "$MASTER_LOG"
echo "C27 formal workflow complete" | tee -a "$MASTER_LOG"
