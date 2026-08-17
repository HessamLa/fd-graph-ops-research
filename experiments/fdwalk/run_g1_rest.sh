#!/bin/bash
# The runs that the first pass of run_g1.sh did not give:
#   * every 'walk_edges' variant, which stopped with a length error
#   * 'mean_gap' and 'pmi', which gave a FLOAT weight and collapsed
# Both defects are repaired. FINDINGS.md holds the reason for each.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results
SUM=$OUT/g1_cora.tsv

for seed in 42 56 88; do
  for w in flat min_gap mean_gap pmi; do
    LOG=$OUT/g1_cora_walk_edges_${w}_plain_s${seed}.log
    echo "== walk_edges/$w/plain seed=$seed"
    .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
      --graph cora --pairs walk_edges --weight "$w" --optim plain \
      --seed "$seed" > "$LOG" 2>&1
    grep -h "RESULT" "$LOG" >> $SUM || echo "FAILED $LOG"
  done
  for w in mean_gap pmi; do
    LOG=$OUT/g1_cora_walk_${w}_plain_s${seed}.log
    echo "== walk/$w/plain seed=$seed"
    .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
      --graph cora --pairs walk --weight "$w" --optim plain \
      --seed "$seed" > "$LOG" 2>&1
    grep -h "RESULT" "$LOG" >> $SUM || echo "FAILED $LOG"
  done
done
echo "done"
