#!/bin/bash
# Paper parameters, plus the fodiwalk walk/window arm.
#
#   node2vec  Grover & Leskovec 2016: 10 walks x 80 steps, window 10,
#             1 epoch, p=q=1, negative sampling, d=128
#   deepwalk  Perozzi et al. 2014: 80 walks x 40 steps, window 10,
#             hierarchical softmax, 1 epoch, d=128
#   fodiwalk  fdhop, at node2vec's 10 x 80 / window 10. The 10 x 20 /
#             window 5 arm is already in the store and is not re-run.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix3.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 14400 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

for G in cora pubmed wordnet; do
  run --method node2vec --graph $G --dim 128 \
      --walks 10 --walk-len 80 --window 10 --n2v-epochs 1 --p 1.0 --q 1.0 --score
  run --method deepwalk --graph $G --dim 128 \
      --walks 80 --walk-len 40 --window 10 --n2v-epochs 1 --score
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop --k4 1.0 --walks 10 --walk-len 80 --window 10 --score
done

echo "MATRIX3 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
