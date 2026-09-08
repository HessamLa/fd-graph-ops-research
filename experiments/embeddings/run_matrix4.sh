#!/bin/bash
# The missing fodiwalk arm: fdlinear at 10 x 80 / window 10.
# fdlinear at 10 x 20 / window 5 and fdhop at both settings are already in
# the store, or are being run by matrix3. Nothing here is a re-run.
#
# Waits for matrix3 first: one job at a time, so the recorded seconds and
# peak_rss mean something.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix4.log

while pgrep -f run_matrix3.sh > /dev/null; do
  echo "[wait] matrix3 in flight at $(date -u +%H:%M:%SZ)" >> "$LOG"
  sleep 120
done

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 14400 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

for G in cora pubmed wordnet; do
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdlinear --walks 10 --walk-len 80 --window 10 --score
done

echo "MATRIX4 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
