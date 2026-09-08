#!/bin/bash
# node2vec WITH its search bias. Everything else matches the p=q=1 runs:
# 10 walks x 80 steps, window 10, 1 epoch, d=128, negative sampling.
#
#   q = 0.5   in-out < 1, the walk moves AWAY from where it came, DFS-like
#   q = 2.0   in-out > 1, the walk stays near its origin, BFS-like
#
# This is the first use of `node2vec_walks` in this store. Every earlier
# node2vec run took the uniform branch, because p = q = 1 collapses the
# second-order rule to a first-order walk.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix5.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 14400 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

for G in cora pubmed wordnet; do
  for Q in 0.5 2.0; do
    run --method node2vec --graph $G --dim 128 \
        --walks 10 --walk-len 80 --window 10 --n2v-epochs 1 \
        --p 1.0 --q $Q --score
  done
done

echo "MATRIX5 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
