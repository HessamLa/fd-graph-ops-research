#!/bin/bash
# Roster item 1: does a degree-biased far pair repair the link prediction?
#
# The hypothesis (log entry 2026-08-16 02:30 and the analysis before it):
# fodined draws its far pairs UNIFORMLY, thus every node receives about the
# same number of repulsive partners. node2vec draws a negative sample in
# proportion to `deg^0.75`, thus a hub of com_youtube is about 636 times
# more likely to be a negative for node2vec than for a uniform draw. A hub
# of fdwalk therefore receives almost no repulsion, and its attraction is
# divided by its degree two times. Link-prediction pairs are degree-biased,
# thus a badly placed hub costs the AUC and it does not cost the hop R2.
#
# The test needs HUBS, thus Cora and PubMed cannot show it: their maximum
# degrees are 168 and about 171. It uses a BFS ball of com_youtube, which
# keeps the hubs and runs in minutes instead of half an hour.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results
mkdir -p $OUT
SUM=$OUT/farbias_youtube150k.tsv
: > $SUM
N=${1:-150000}
EPOCHS=${2:-500}

for seed in 42 56 88; do
  for a in 0.0 0.75; do
    LOG=$OUT/farbias_yt${N}_a${a}_s${seed}.log
    echo "== alpha=$a seed=$seed"
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
      --graph com_youtube --max-nodes "$N" \
      --pairs walk --weight min_gap --cap 16 --prune-factor 1 \
      --walks 5 --len 20 --window 5 \
      --far-bias "$a" --dim 128 --epochs "$EPOCHS" --chunks 4 \
      --seed "$seed" > "$LOG" 2>&1
    grep -h RESULT "$LOG" >> $SUM || echo "FAILED $LOG"
  done
done
echo "summary: $SUM"
