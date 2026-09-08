#!/bin/bash
# run_bench.sh -- one arm of the SELL-C-sigma vs dense-batch comparison.
#
# Usage:  run_bench.sh <tag>
#   <tag>  "sellc" for the old kernel, "dense" for the new one. It only
#          names the output files; the code that runs is whatever is in
#          the tree right now.
#
# Runs strictly one job at a time, as the project owner requires. Each run
# also gets a GPU memory sampler, because the driver's own `rss` field is
# HOST memory only and says nothing about the card.
#
# Fixed across both arms, so the only difference is the kernel:
#   graph pubmed, pairs nbr_walk, law fdlinear, rule plain, lr 0.999,
#   200 epochs. 16 dimensions on 5 seeds, then 128 dimensions on 3 seeds.
#
# lr is 0.999 and never 1.0 -- the owner's global rule. The driver's own
# default of 1.0 breaks that rule, so this script always passes it.
set -u

TAG="${1:?usage: run_bench.sh <sellc|dense>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/${TAG}.tsv"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

EPOCHS=200
GRAPH=pubmed

run_one () {
  local dim=$1 seed=$2
  local name="${TAG}_d${dim}_s${seed}"
  local log="$LOGDIR/${name}.log"
  local gpulog="$LOGDIR/${name}.gpu"

  # Refuse to start while anything else is measuring.
  while pgrep -f "bench_fodiwalk.py" > /dev/null; do
    echo "[wait] a run is in flight; holding at $(date -u +%H:%M:%SZ)"
    sleep 30
  done

  echo "[run ] $name  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Sample the card every second for the life of the run.
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
      sleep 1
    done > "$gpulog" ) &
  local sampler=$!

  "$ROOT/.venv/bin/python" "$ROOT/experiments/fodiwalk/bench_fodiwalk.py" \
      --graph "$GRAPH" --dim "$dim" --epochs "$EPOCHS" --lr 0.999 \
      --seed "$seed" --pairs nbr_walk --device gpu \
      --out "$OUT" > "$log" 2>&1
  local rc=$?

  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  local gpupeak
  gpupeak=$(sort -n "$gpulog" 2>/dev/null | tail -1)
  echo "[done] $name rc=$rc gpu_peak_MiB=${gpupeak:-NA}"
  # Keep the card peak beside the driver's own RESULT line.
  echo -e "${TAG}\t${dim}\t${seed}\t${rc}\t${gpupeak:-NA}" >> "$HERE/${TAG}.gpu.tsv"
}

echo "=== $TAG  16 dimensions, 5 seeds  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for s in 42 43 44 45 46; do run_one 16 "$s"; done

echo "=== $TAG  128 dimensions, 3 seeds  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for s in 42 43 44; do run_one 128 "$s"; done

echo "=== $TAG complete  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
