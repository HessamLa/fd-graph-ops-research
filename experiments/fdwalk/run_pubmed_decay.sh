#!/bin/bash
# Close the gap: the decaying schedule was run on PubMed for `plain` only,
# yet on Cora it CHANGED the ranking -- adam went 0.414 -> 0.616 and became
# the best point of the whole branch. Excluding a rule on its constant-lr
# number alone would be wrong.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/grid_pubmed
BASE="--graph pubmed --pairs nbr_walk --weight min_gap --force fdlinear \
      --k4 1.0 --kr 1.0 --dim 64 --epochs 2000 --far-bias 0.75 \
      --far-weight 100 --seed 42 --lr-decay linear"
# momentum is omitted: it diverged with decay from BOTH 0.999 and 0.9 on
# Cora, thus it has already failed this test.
for o in adam sqn nesterov velocity sgd fa2; do
  L=$OUT/A_decay_${o}_lr0.999.log
  [ -s "$L" ] && grep -q RESULT "$L" && continue
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE --optim $o --lr 0.999 > "$L" 2>&1
  .venv/bin/python experiments/fdwalk/update_results.py > /dev/null
done
echo "=== pubmed decay done ==="
