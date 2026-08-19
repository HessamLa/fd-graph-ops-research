#!/bin/bash
# The learning-rate ladder of 2026-08-18 (instruction of the user).
#
#   Every lr is BELOW 1.0. Start at 0.999. If that diverges, take 0.9.
#   If that diverges too, take 0.1.
#   For a decaying lr, start from 0.999 or from 0.9.
#
# Cora is a small graph, thus this may share the machine (CLAUDE.md).
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/lrladder; mkdir -p $OUT
BASE="--graph cora --pairs nbr_walk --weight min_gap --force fdlinear \
      --k4 1.0 --kr 1.0 --dim 64 --epochs 2000 --far-bias 0.75 \
      --far-weight 100 --seed 42"

run () { # name, then flags
  n=$1; shift
  [ -s $OUT/$n.log ] && grep -q RESULT $OUT/$n.log && return
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE "$@" > $OUT/$n.log 2>&1
}
diverged () { grep -q "diverged=1" $OUT/$1.log 2>/dev/null; }

for o in plain velocity sgd sqn momentum nesterov adam fa2; do
  # --- the constant ladder: 0.999 -> 0.9 -> 0.1, stopping at the first
  #     rate that does not diverge. Every rung is still RUN and recorded,
  #     because "it diverged here" is itself a measurement.
  for lr in 0.999 0.9 0.1; do
    run "const_${o}_lr${lr}" --optim $o --lr $lr
    diverged "const_${o}_lr${lr}" || break
  done
  # --- the decaying ladder, from 0.999 and from 0.9
  for lr in 0.999 0.9; do
    run "decay_${o}_lr${lr}" --optim $o --lr $lr --lr-decay linear
  done
done
echo "=== ladder done ==="
