#!/bin/bash
# The two com_youtube cells the interruption took. cora and pubmed are done.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python; S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/fdhop_pq.log; RUNS=experiments/embeddings/runs.log
FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"
run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  $PY $S "$@" >> "$LOG" 2>&1; local rc=$?
  echo "--- exit $rc ---" >> "$LOG"
  [ "$rc" -ne 0 ] && printf '%s\tFAIL\tscheduler saw exit %s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$*" >> "$RUNS"
}
for Q in 0.5 2.0; do
  run --method fodiwalk --graph com_youtube --dim 128 --epochs 200 $FD \
      --force fdhop --k4 1.0 --walks 10 --walk-len 20 --window 5 \
      --p 1.0 --q $Q --score
done
echo "FDHOP_PQ_CY DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
