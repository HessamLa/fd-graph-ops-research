#!/bin/bash
# The off-branch experiment: a new force law and a new augmentation budget.
#
#   force v2   h == 1 -> Fa + Fr with Fr = -k3*h*exp(-k4*x)
#              h  > 1 -> Fr = -k3*h, thus NO decay with the distance
#   buckets    every original edge at h = 1, plus n*log10(n) pairs at
#              50% h=2, 25% h=3, 25% h>=4, and NO far pairs
#
# The ball needs k = 4 and the walk needs window = 5, because the policy
# asks for a bucket of h >= 4 and neither source gives one otherwise.
#
# `--cap 0` turns the per-node cap off, thus the buckets draw from every
# pair that the source found. The 1.13M graph keeps the cap, because the
# accumulator needs the bound; that is a deviation, and REPORT.md says so.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/offbranch
mkdir -p $OUT
SUM=$OUT/offbranch.tsv
: > $SUM

for g in cora pubmed; do
  for src in ball walk; do
    EXTRA="--k 4"
    [ "$src" = "walk" ] && EXTRA="--window 5 --walks 10 --len 20"
    LOG=$OUT/ob_${g}_${src}_s42.log
    echo "== $g / $src"
    .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
      --graph "$g" --pairs "$src" --weight min_gap \
      --force v2 --policy buckets --cap 0 $EXTRA \
      --seed 42 --epochs 2000 > "$LOG" 2>&1
    grep -h RESULT "$LOG" >> $SUM || echo "FAILED $LOG"
  done
done
echo "summary: $SUM"
