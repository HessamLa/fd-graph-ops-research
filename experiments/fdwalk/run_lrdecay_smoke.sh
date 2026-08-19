#!/bin/bash
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/lrdecay; mkdir -p $OUT
BASE="--graph cora --pairs nbr_walk --weight min_gap --force fdlinear --k4 1.0 --kr 1.0 --dim 64 --epochs 2000 --far-bias 0.75 --far-weight 100 --seed 42"
for o in momentum nesterov; do
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE --optim $o --lr 0.1              > $OUT/${o}_lr0.1_const.log  2>&1
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE --optim $o --lr 1.0 --lr-decay linear > $OUT/${o}_lr1.0_linear.log 2>&1
done
printf "%-26s %9s %9s %9s %9s %8s %8s\n" variant t_aug t_embed peakMB AUC hopR2 dz diverged
for f in $OUT/*.log; do
  printf "%-26s %s\n" "$(basename $f .log)" "$(grep -oE 't_aug=[0-9.]+|t_embed=[0-9.]+|rss=[0-9]+|auc=[0-9.a-z]+|r2_dist=[-0-9.a-z]+|dz=[0-9.a-z]+|diverged=1' $f | tr '\n' ' ')"
done
