#!/bin/bash
# fdhop_all, made stable. See run.log: k2=1.0 with k1=0.999 DIVERGED on both
# graphs, and the reason is measured, not guessed.
#
# Fa = k1 * x * exp(-k2 * h) is LINEAR in x, so a row behaves like a spring
# of constant  k1 * sum_v exp(-k2 h) / deg(u).  Above 1.0 the map expands
# and Z runs to non-finite. Measured worst-row gain at k2 = 1.0:
#   cora 2.85, pubmed 3.58  ->  k1 must be under 0.350 and 0.279.
# `fdhop` sits at exactly 1.00 by construction: it attracts on h == 1 only
# and the engine divides by that same count.
#
# Two stable arms:
#   A  k2 = 2.0, k1 = 0.999   worst-row gain 0.47 / 0.56, k1 unchanged
#   B  k2 = 1.0, k1 = 0.25    the specified k2, k1 under both bounds
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python; S=experiments/embeddings/make_embedding.py
LOG=experiments/fdhop-all/run.log; RUNS=experiments/embeddings/runs.log
FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  $PY $S "$@" >> "$LOG" 2>&1; local rc=$?
  echo "--- exit $rc ---" >> "$LOG"
  [ "$rc" -ne 0 ] && printf '%s\tFAIL\tscheduler saw exit %s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$*" >> "$RUNS"
}

for G in cora pubmed; do
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop_all --k2 2.0 --k1 0.999 --k4 1.0 \
      --walks 10 --walk-len 20 --window 5 --score
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop_all --k2 1.0 --k1 0.25 --k4 1.0 \
      --walks 10 --walk-len 20 --window 5 --score
done
echo "FDHOP_ALL_2 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
