#!/bin/bash
# Smoke: every NEW augmentation path runs and records what it did.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/newsmoke
mkdir -p $OUT
run () { name=$1; shift
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py --graph cora \
    --weight min_gap --dim 32 --epochs 40 --seed 42 "$@" > $OUT/$name.log 2>&1
  # CLAUDE.md: peak memory and the runtime of each stage are REQUIRED in
  # every report, thus they are in this line and cannot be left out of it.
  printf "%-26s %s\n" "$name" "$(grep -oE 'dnnz=[0-9]+|t_aug=[0-9.]+|t_embed=[0-9.]+|rss=[0-9]+|auc=[0-9.]+|r2_dist=[0-9.-]+|dz=[0-9.]+|policy=[a-z]+' $OUT/$name.log | tr '\n' ' ')"
  grep -qE "Error|Traceback" $OUT/$name.log && echo "   ^^ FAILED: $(grep -m1 -E 'Error|error:' $OUT/$name.log)"
}
run nbrwalk_base      --pairs nbr_walk --force fdlinear --lr 0.1
run nbrwalk_buckets   --pairs nbr_walk --force fdlinear --lr 0.1 --policy buckets
run nbrwalk_rowcap8   --pairs nbr_walk --force fdlinear --lr 0.1 --row-cap 8
run walk_buckets      --pairs walk --policy buckets
run walk_buckets_far  --pairs walk --policy buckets --far-with-buckets
run walk_pq_dfs       --pairs walk --p 1.0 --q 0.25
run walk_pq_bfs       --pairs walk --p 1.0 --q 4.0
run nbrwalk_pq_dfs    --pairs nbr_walk --force fdlinear --lr 0.1 --q 0.25
