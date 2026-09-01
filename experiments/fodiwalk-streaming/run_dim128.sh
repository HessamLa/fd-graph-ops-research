#!/bin/bash
# run_dim128.sh -- the dim-128 flights. 128 IS THE BASELINE (2026-08-29);
# 64 and 32 are only for quick convergence or learning-rate checks.
#
# Three tiers, run in this order so a complete tier lands early:
#     tier 1   5 epochs, const only
#     tier 2  50 epochs, const
#     tier 3  50 epochs, linear
# Within a tier the graphs go small to large, so a failure shows up cheap.
#
# `plain` only: it matched `velocity` to three decimals at 200 epochs on
# both large graphs and cost 44-66% LESS time (final_*.tsv).
#
# lr = 0.999 and never 1.0 (the global rule of 2026-08-29). NOTE this
# differs from the dim-64 campaign of 2026-08-28, which used lr = 1.0 and
# predates the rule -- so this is a re-baseline, not just a re-dimension.
#
# STRICTLY SEQUENTIAL. One python process at a time, and the script waits
# for any run already in flight before it starts.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
export PYTHONPATH=/home/h/gnn/fd-graph-embedding/fdmap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

OUT=experiments/fodiwalk-streaming/dim128.tsv
LOGS=experiments/fodiwalk-streaming/logs
mkdir -p "$LOGS"

# --- do not overlap with anything already running -----------------------
while pgrep -f "bench_stream.py" > /dev/null; do
  echo "[wait] a run is in flight; holding at $(date -u +%H:%M:%SZ)"
  sleep 120
done
echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)  the machine is free"

GRAPHS=(cora pubmed wordnet com_youtube as_skitter roadnet_ca ncbi_taxonomy)

batch_for () {   # cora is small enough that 1024 would be one batch
  [ "$1" = cora ] && echo 512 || echo 1024
}

run () {   # graph epochs decay
  g=$1; e=$2; d=$3; b=$(batch_for "$g")
  tag="d128_${g}_plain_lr0.999_${d}_e${e}"
  echo "=== $tag  $(date -u +%H:%M:%SZ)  free $(free -g|awk 'NR==2{print $7}')G ==="
  timeout 43200 .venv/bin/python experiments/fodiwalk-streaming/bench_stream.py \
      --graph "$g" --dim 128 --walks 5 --walk-len 20 --epochs "$e" \
      --batch "$b" --device gpu --seed 42 --score \
      --optim plain --lr 0.999 --lr-decay "$d" \
      --tag "$tag" --out "$OUT" > "$LOGS/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
      echo "  FAILED rc=$rc  $(grep -oE '[A-Za-z]+Error.*' "$LOGS/$tag.log" | tail -1 | cut -c1-80)"
  else
      grep -E "TOTAL |peak RSS .*embedding AND scoring|accuracy  |f1  |auc  " \
        "$LOGS/$tag.log" | sed 's/^/  /'
  fi
}

echo "########## TIER 1: 5 epochs, const ##########"
for g in "${GRAPHS[@]}"; do run "$g" 5 const; done
echo "########## TIER 2: 50 epochs, const ##########"
for g in "${GRAPHS[@]}"; do run "$g" 50 const; done
echo "########## TIER 3: 50 epochs, linear ##########"
for g in "${GRAPHS[@]}"; do run "$g" 50 linear; done
echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
