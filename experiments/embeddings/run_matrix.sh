#!/bin/bash
# One job at a time. The two top optimisers, the two picked walk methods,
# the one force law, plus node2vec at the same dimension.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 3600 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

for G in cora pubmed; do
  for OPT in plain velocity; do
    for P in walk_edges nbr_walk; do
      run --method fodiwalk --graph $G --dim 128 --epochs 200 \
          --pairs $P --optim $OPT --force fdlinear --weight min_gap \
          --lr 0.999 --device cpu --score
    done
  done
done

run --method node2vec --graph pubmed --dim 128 --score
echo "MATRIX DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
