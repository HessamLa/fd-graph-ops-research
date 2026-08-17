#!/bin/bash
# Mix and match: two force laws x three augmentations, at three seeds.
#
# force v1        the law of the package, lr 1.0
# force fdlinear  the law of 2026-08-17, k4=1.0, lr 0.1
#
# nbr_walk/both     every edge in BOTH rows, plus the walk of the row
# nbr_walk/low_deg  an edge only in the row of the lower-degree node
# walk_edges        the capped window policy, plus every edge
#
# Every variant gets far pairs at deg^0.75, because that gain is measured.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/mixmatch
mkdir -p $OUT
G=${1:-cora}; EP=${2:-2000}; FAR=${3:-9295}
for seed in 42 56 88; do
  for force in v1 fdlinear; do
    LR=1.0; EX=""
    [ "$force" = "fdlinear" ] && { LR=0.1; EX="--k4 1.0 --kr 1.0"; }
    for cfg in "nbr_walk both" "nbr_walk low_deg" "walk_edges both"; do
      set -- $cfg; P=$1; ER=$2
      L=$OUT/mm_${G}_${force}_${P}_${ER}_s${seed}.log
      echo "== $G $force $P/$ER seed=$seed"
      .venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
        --graph "$G" --pairs "$P" --edge-rule "$ER" --weight min_gap \
        --force "$force" $EX --lr $LR \
        --far "$FAR" --far-bias 0.75 --far-weight 100 \
        --walks 5 --len 20 --window 5 --cap 16 \
        --epochs "$EP" --seed "$seed" > "$L" 2>&1 || echo "FAILED $L"
    done
  done
done
echo done
