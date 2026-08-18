#!/bin/bash
# The grids of 2026-08-18. One graph per invocation:
#   bash run_grids_2026-08-18.sh cora   2000
#   bash run_grids_2026-08-18.sh pubmed 2000
#
# Grid A  the gradient optimizer roster        (CLAUDE.md, IMPORTANT)
# Grid B  cap x buckets  X  symmetric x directed   (requested 2026-08-18)
# Grid C  does `buckets` lose the hop R2 only for want of far pairs?
# Grid D  the second-order node2vec walk, p and q
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
G=${1:-cora}; EP=${2:-2000}
OUT=experiments/fdwalk/results/grid_$G
mkdir -p $OUT
BASE="--graph $G --weight min_gap --dim 64 --epochs $EP --far-bias 0.75 --far-weight 100"
FDL="--force fdlinear --k4 1.0 --kr 1.0 --lr 0.1"

run () { name=$1; shift
  L=$OUT/$name.log
  [ -s "$L" ] && grep -q RESULT "$L" && { echo "skip $name"; return; }
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE "$@" > "$L" 2>&1 \
    || echo "FAILED $name"
}

echo "=== Grid A: the optimizer roster ==="
# The learning rate is NOT shared across rules: `momentum` has a
# steady-state gain of 1/(1-beta) = 10 and `velocity` has 1, thus one lr
# would test the lr and not the rule. Each rule gets its own scan first.
for o in plain velocity sgd sqn momentum nesterov adam fa2; do
  for lr in 1.0 0.1 0.01; do
    run "A_lrscan_${o}_lr${lr}" --pairs nbr_walk $FDL --optim "$o" --lr "$lr" --seed 42
  done
done

echo "=== Grid B: cap x buckets  X  symmetric x directed ==="
for s in 42 56 88; do
  run "B_cap_sym_s$s"      --pairs walk     --cap 16 --seed $s $FDL
  run "B_cap_dir_s$s"      --pairs nbr_walk --row-cap 16 --seed $s $FDL
  run "B_buckets_sym_s$s"  --pairs walk     --policy buckets --seed $s $FDL
  run "B_buckets_dir_s$s"  --pairs nbr_walk --policy buckets --seed $s $FDL
done

echo "=== Grid C: buckets, with and without far pairs ==="
for s in 42 56 88; do
  run "C_buckets_nofar_s$s" --pairs walk --policy buckets --seed $s $FDL
  run "C_buckets_far_s$s"   --pairs walk --policy buckets --far-with-buckets --seed $s $FDL
done
# and is the 73% a BUDGET effect rather than a POLICY effect?
for bt in 5000 20000 80000; do
  run "C_budget_${bt}" --pairs walk --policy buckets --bucket-total $bt --seed 42 $FDL
done

echo "=== Grid D: the second-order walk ==="
for s in 42 56 88; do
  for pq in "1.0 1.0" "1.0 0.5" "1.0 2.0" "0.5 1.0" "2.0 1.0"; do
    set -- $pq; P=$1; Q=$2
    run "D_p${P}_q${Q}_s$s" --pairs nbr_walk $FDL --p $P --q $Q --seed $s
  done
done
echo "=== done $G ==="
