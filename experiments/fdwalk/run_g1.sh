#!/bin/bash
# Gate G1: every variant of axis A and axis B, on Cora, at three seeds.
#
# Axis C (the update rule) does not run here. PLAN.md section 4 says that it
# runs on the winner of G1 only.
#
# The ball takes only 'flat' and 'min_gap'. The PMI of a ball has no
# meaning: a ball has no visit count, thus `cnt` is only a rank of the hops.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results
mkdir -p $OUT
SUM=$OUT/g1_cora.tsv
: > $SUM

for seed in 42 56 88; do
  for pairs in walk walk_edges ball; do
    if [ "$pairs" = "ball" ]; then WEIGHTS="flat min_gap";
    else WEIGHTS="flat min_gap mean_gap pmi"; fi
    for w in $WEIGHTS; do
      LOG=$OUT/g1_cora_${pairs}_${w}_plain_s${seed}.log
      echo "== $pairs/$w/plain seed=$seed -> $LOG"
      .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
        --graph cora --pairs "$pairs" --weight "$w" --optim plain \
        --seed "$seed" > "$LOG" 2>&1
      grep -h "RESULT" "$LOG" >> $SUM || echo "FAILED $LOG"
    done
  done
done
echo "summary: $SUM"
