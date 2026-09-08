#!/bin/bash
# pubmed and wordnet at dim 128. One job at a time.
# Reused from the store, not re-run: pubmed node2vec, pubmed fodiwalk+fdlinear.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix2.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 7200 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

# pubmed: the two missing arms
run --method deepwalk --graph pubmed --dim 128 --score
run --method fodiwalk --graph pubmed --dim 128 --epochs 200 $FD --force fdhop --k4 1.0 --score

# wordnet, 82,115 nodes: all four
run --method deepwalk --graph wordnet --dim 128 --score
run --method node2vec --graph wordnet --dim 128 --score
run --method fodiwalk --graph wordnet --dim 128 --epochs 200 $FD --force fdlinear --score
run --method fodiwalk --graph wordnet --dim 128 --epochs 200 $FD --force fdhop --k4 1.0 --score

echo "MATRIX2 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
