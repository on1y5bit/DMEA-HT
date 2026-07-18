#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linruixin/chen/conda/envs/ma/bin/python}"
CONFIG="${1:-configs/dema_ht_c66_source_learning.yaml}"
LOG_DIR="$ROOT/runs/dema_ht_c66_nested_cv/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

wait_for_wave() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

"$PYTHON_BIN" scripts/gate_phase_c66_runtime_preflight.py --config "$CONFIG" \
  > "$LOG_DIR/runtime_preflight.log" 2>&1

# Each wave runs only the three formal seeds in parallel. No pilot or smoke stage exists.
for fold in 0 1 2 3 4; do
  pids=()
  for seed in 0 42 3407; do
    "$PYTHON_BIN" scripts/train_phase_c66_source.py --config "$CONFIG" --mode inner --fold "$fold" --seed "$seed" \
      > "$LOG_DIR/inner_source_fold_${fold}_seed_${seed}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_wave "${pids[@]}"

  for route in F E; do
    pids=()
    for seed in 0 42 3407; do
      "$PYTHON_BIN" scripts/train_phase_c66_inner_routes.py --config "$CONFIG" --fold "$fold" --route "$route" --seed "$seed" \
        > "$LOG_DIR/inner_route_${route}_fold_${fold}_seed_${seed}.log" 2>&1 &
      pids+=("$!")
    done
    wait_for_wave "${pids[@]}"
  done
  "$PYTHON_BIN" scripts/collect_phase_c66_inner_decision.py --config "$CONFIG" --fold "$fold" \
    > "$LOG_DIR/inner_decision_fold_${fold}.log" 2>&1
done

for fold in 0 1 2 3 4; do
  pids=()
  for seed in 0 42 3407; do
    "$PYTHON_BIN" scripts/train_phase_c66_outer_refit.py --config "$CONFIG" --fold "$fold" --seed "$seed" \
      > "$LOG_DIR/outer_refit_fold_${fold}_seed_${seed}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_wave "${pids[@]}"
done

# A nonzero OOF gate stops here. Test remains unread and no final run begins on failure.
"$PYTHON_BIN" scripts/collect_phase_c66_oof.py --config "$CONFIG" \
  > "$LOG_DIR/oof_collection.log" 2>&1

pids=()
for seed in 0 42 3407; do
  "$PYTHON_BIN" scripts/train_phase_c66_final.py --config "$CONFIG" --seed "$seed" \
    > "$LOG_DIR/final_train_seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
wait_for_wave "${pids[@]}"

# Test is opened only after all three final checkpoints have completed their frozen contracts.
"$PYTHON_BIN" scripts/collect_phase_c66_final.py --config "$CONFIG" \
  > "$LOG_DIR/final_test_collection.log" 2>&1
