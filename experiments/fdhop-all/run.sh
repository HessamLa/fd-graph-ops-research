#!/bin/bash
# fdhop_all on cora and pubmed.
#
#   Fr = -kr * h * exp(-k4 * x)   for EVERY h
#   Fa = k1 * x  * exp(-k2 * h)   for EVERY h
#
# Every other law of this project attracts at h == 1 and nowhere else.
# This one lets a far pair pull as well, and `k2` sets how fast that pull
# dies with the hop number.
#
# `fdhop2` is the control: same repulsion, attraction at h == 1 only. It is
# already in the store for both graphs, so it is not re-run here.
# Everything else matches: dim 128, 200 epochs, walk_edges, plain,
# lr 0.999 const, k1 0.999, k4 1.0, kr 1.0, min_gap, 10x20/w5, seed 42.
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
      --force fdhop_all --k2 1.0 --k4 1.0 \
      --walks 10 --walk-len 20 --window 5 --score
done
echo "FDHOP_ALL DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
