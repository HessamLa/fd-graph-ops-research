#!/bin/bash
# fdhop_all_freq at k1 = 0.999.
#
#   Fa =  freq * k1 * x * exp(-k2 * h)
#   Fr = -freq * kr * h * exp(-k4 * x)
#
# `freq` multiplies the attraction, which is linear in x, so it multiplies
# the row spring constant. Measured worst-row gain at k1 = 1.0:
#   k2=1.0  cora 375.2  pubmed 264.2
#   k2=6.5  cora   1.53 pubmed   0.90
#   k2=7.0  cora   0.93 pubmed   0.55
# k1 = 0.999 scales all of these by 0.999, thus k2 >= 7.0 is still the
# first stable point on cora. Two arms, 7.0 and 8.0.
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
  for K2 in 7.0 8.0; do
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force fdhop_all_freq --k1 0.999 --k2 $K2 --k4 1.0 --kr 1.0 \
        --walks 10 --walk-len 20 --window 5 --score
  done
done
echo "FDHOP_ALL_FREQ DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
