#!/bin/bash
# run_optim.sh -- the optimiser grid, under the streaming path.
#
# GLOBAL RULE (2026-08-29): no learning rate is 1.0. Every lr here is
# 0.999 or less, and a decay schedule starts at 0.999 or less.
#
# `nesterov` runs at lr 0.0999 and not 0.999 because its DC gain is
# 1/(1-beta) = 10: the EFFECTIVE learning rate is lr*10, and 0.0999*10 =
# 0.999 puts it on the same effective footing as every other row.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
export PYTHONPATH=/home/h/gnn/fd-graph-embedding/fdmap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
G=$1; E=$2; B=${3:-1024}
OUT=experiments/fodiwalk-streaming/optim_${G}.tsv
LOGS=experiments/fodiwalk-streaming/logs

#     optim     lr      decay
# The grid is BALANCED: every rule appears with const AND linear, so the
# schedule can be read off without a confound. `lr` is chosen so the
# EFFECTIVE learning rate (lr x the DC gain of the rule) is ~0.999 for
# every row:
#   plain, velocity, sgd, adam   gain 1        -> lr 0.999
#   nesterov                     gain 1/(1-b)=10 -> lr 0.0999
#   fa2                          speed capped at k_max=10, so the gain
#                                reaches 10   -> lr 0.0999
# fa2 at lr 0.999 DIVERGED at epoch 15 on pubmed (891,520 non-finite
# values), which is the same gain-10 failure the classic momentum has.
CFG=("plain     0.999   const"
     "plain     0.999   linear"
     "velocity  0.999   const"
     "velocity  0.999   linear"
     "sgd       0.999   const"
     "sgd       0.999   linear"
     "adam      0.999   const"
     "adam      0.999   linear"
     "nesterov  0.0999  const"
     "nesterov  0.0999  linear"
     "fa2       0.0999  const"
     "fa2       0.0999  linear")

for c in "${CFG[@]}"; do
  set -- $c; o=$1; lr=$2; d=$3
  tag="${G}_${o}_lr${lr}_${d}_e${E}"
  printf "  %-38s " "$tag"
  timeout 10800 .venv/bin/python experiments/fodiwalk-streaming/bench_stream.py \
      --graph "$G" --dim 64 --walks 5 --walk-len 20 --epochs "$E" \
      --batch "$B" --device gpu --seed 42 --score \
      --optim "$o" --lr "$lr" --lr-decay "$d" --beta 0.9 \
      --tag "$tag" --out "$OUT" > "$LOGS/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED rc=$rc"
  else grep -oE "dz \(mean[^,]*, *max\|Z\| = [0-9.e+]*" "$LOGS/$tag.log" | tail -1; fi
done
