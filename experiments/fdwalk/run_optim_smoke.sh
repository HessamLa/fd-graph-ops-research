#!/bin/bash
# Smoke test: every rule of the roster runs and converges on Cora.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/optim
mkdir -p $OUT
for o in plain velocity sgd sqn momentum nesterov adam fa2; do
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py \
    --graph cora --pairs walk --weight min_gap --optim "$o" \
    --dim 32 --epochs 60 --lr 1.0 --seed 42 > $OUT/smoke_$o.log 2>&1
  printf "%-9s %s\n" "$o" "$(grep -oE 'auc=[0-9.]+|r2_dist=[0-9.-]+|final \|\|dZ\|\| avg [0-9.e+-]+|state arrays' $OUT/smoke_$o.log | tr '\n' ' ')"
done
