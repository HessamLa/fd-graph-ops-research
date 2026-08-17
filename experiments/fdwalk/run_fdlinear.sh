#!/bin/bash
# fdlinear against the three best augmentations, at two learning rates.
#
#   nbr_walk    the specification of 2026-08-17: EVERY neighbour at h=1,
#               plus every node that a walk from the row reached. No cap,
#               no window, D is directed.
#   walk        the main line: walk pairs capped at m for each node, plus
#               far pairs at h=100.
#   walk_edges  the same, plus every original edge forced to h=1.
#
# `k4 = 1.0` comes from the sweep of 2026-08-17: `exp(-x)` beats
# `exp(-0.01x)` on every row of it. `--no-deg-norm` diverges at lr >= 0.1,
# thus the engine keeps its division by deg1(u).
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/fdlinear
mkdir -p $OUT
SUM=$OUT/fdlinear_${1:-cora}.tsv
: > $SUM
G=${1:-cora}
EP=${2:-2000}
for seed in 42 56 88; do
  for src in nbr_walk walk walk_edges; do
    for lr in 1.0 0.1; do
      L=$OUT/fdl_${G}_${src}_lr${lr}_s${seed}.log
      echo "== $G $src lr=$lr seed=$seed"
      .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
        --graph "$G" --pairs "$src" --weight min_gap --force fdlinear \
        --k4 1.0 --kr 1.0 --lr "$lr" --walks 5 --len 20 --window 5 \
        --epochs "$EP" --seed "$seed" > "$L" 2>&1
      grep -h RESULT "$L" >> $SUM || echo "FAILED $L"
    done
  done
done
echo "summary: $SUM"
