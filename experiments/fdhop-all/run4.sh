#!/bin/bash
# fdhop_all with k1 = 1.0 and NOTHING else changed from the stable arm:
# k2 = 2.0, k4 = 1.0, kr = 1.0. At k2 = 2.0 the worst-row spring constant
# measures 0.47 on cora and 0.56 on pubmed, so k1 = 1.0 leaves it under the
# 1.0 edge and this should run.
#
# lr 0.999 constant, rule `plain`, as every fodiwalk run in this store.
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
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop_all --k1 1.0 --k2 2.0 --k4 1.0 --kr 1.0 \
      --walks 10 --walk-len 20 --window 5 --score
done
echo "FDHOP_ALL_K1ONLY DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
