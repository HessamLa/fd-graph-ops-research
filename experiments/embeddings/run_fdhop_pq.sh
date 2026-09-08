#!/bin/bash
# fdhop with a BIASED walk. The `walk_edges` policy calls the same
# `make_walker` node2vec uses, so p and q apply to fodiwalk too; only the
# driver was not passing them.
#
#   q = 0.5   in-out < 1, the walk moves away from where it came, DFS-like
#   q = 2.0   in-out > 1, the walk stays near its origin, BFS-like
#
# p = 1.0 throughout. Everything else matches the stored fdhop rows:
# dim 128, 200 epochs, plain, lr 0.999 const, k4 1.0, 10x20/w5, min_gap.
# No timeout: a run finishes or it dies.
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

for G in cora pubmed com_youtube; do
  for Q in 0.5 2.0; do
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force fdhop --k4 1.0 --walks 10 --walk-len 20 --window 5 \
        --p 1.0 --q $Q --score
  done
done
echo "FDHOP_PQ DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
