#!/bin/bash
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python; S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/fdhopmin.log; RUNS=experiments/embeddings/runs.log
FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"
run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  $PY $S "$@" >> "$LOG" 2>&1; local rc=$?
  echo "--- exit $rc ---" >> "$LOG"
  [ "$rc" -ne 0 ] && printf '%s\tFAIL\texit %s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$*" >> "$RUNS"
}
for G in cora pubmed; do
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop_min --k4 1.0 --walks 10 --walk-len 20 --window 5 --score
done
echo "FDHOPMIN DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
