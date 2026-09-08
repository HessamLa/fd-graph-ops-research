#!/bin/bash
# fdhop_all with every force constant at 1.0: k1 = k2 = k4 = kr = 1.0.
#
# PREDICTED TO DIVERGE, and the prediction is measured, not guessed. The
# attraction is linear in x, so a row is a spring of constant
# k1 * sum_v exp(-k2 h) / deg(u). At k2 = 1.0 the worst row measures 2.85
# on cora and 3.58 on pubmed; k1 = 1.0 leaves it there. The run is made
# anyway, because a prediction that is never tested is an opinion.
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
      --force fdhop_all --k1 1.0 --k2 1.0 --k4 1.0 --kr 1.0 \
      --walks 10 --walk-len 20 --window 5 --score
done
echo "FDHOP_ALL_K1 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
