#!/bin/bash
# The same 8 runs, on the four large graphs. One job at a time.
#
# ORDER IS DELIBERATE. Graphs go smallest first, and inside each graph the
# cheap runs go first, so a graph that cannot finish still leaves results
# behind instead of nothing. `deepwalk` is last of each graph: at 80 walks
# x 40 steps its corpus is the largest thing any run builds.
#
# EXPECTED FAILURES, written down before the fact. About 5 GB of RAM is
# free. The fodiwalk PRECOMPUTED path stores every pair, thus at 10 walks
# x 80 steps a million-node graph needs roughly 900 million pairs and will
# not fit. Those runs are scheduled anyway, they fail fast on allocation,
# and the driver writes FAIL to runs.log with the reason.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix6.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  timeout 14400 $PY $S "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

for G in com_youtube as_skitter roadnet_ca ncbi_taxonomy; do
  echo "########## $G $(date -u +%H:%M:%SZ) ##########" >> "$LOG"

  for Q in 1.0 0.5 2.0; do
    run --method node2vec --graph $G --dim 128 \
        --walks 10 --walk-len 80 --window 10 --n2v-epochs 1 \
        --p 1.0 --q $Q --score
  done

  for LAW in fdhop fdlinear; do
    K4=0.01; [ "$LAW" = fdhop ] && K4=1.0
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force $LAW --k4 $K4 --walks 10 --walk-len 20 --window 5 --score
  done

  for LAW in fdhop fdlinear; do
    K4=0.01; [ "$LAW" = fdhop ] && K4=1.0
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force $LAW --k4 $K4 --walks 10 --walk-len 80 --window 10 --score
  done

  run --method deepwalk --graph $G --dim 128 \
      --walks 80 --walk-len 40 --window 10 --n2v-epochs 1 --score
done

echo "MATRIX6 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
