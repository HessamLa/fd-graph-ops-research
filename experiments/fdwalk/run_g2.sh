#!/bin/bash
# Gate G2: PubMed, 19,717 nodes. Only the variants that passed G1.
#
#   bash run_g2.sh "walk:min_gap ball:min_gap"
#
# The default list is the winner of G1 and its control. PLAN.md section 7
# holds the numbers that a variant must reach here.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results
mkdir -p $OUT
SUM=$OUT/g2_pubmed.tsv
VARIANTS=${1:-"walk:min_gap ball:min_gap walk:flat"}

for seed in 42 56 88; do
  for v in $VARIANTS; do
    p=${v%%:*}; w=${v##*:}
    LOG=$OUT/g2_pubmed_${p}_${w}_plain_s${seed}.log
    echo "== $p/$w/plain seed=$seed"
    .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
      --graph pubmed --pairs "$p" --weight "$w" --optim plain \
      --seed "$seed" > "$LOG" 2>&1
    grep -h "RESULT" "$LOG" >> $SUM || echo "FAILED $LOG"
  done
done
echo "summary: $SUM"
