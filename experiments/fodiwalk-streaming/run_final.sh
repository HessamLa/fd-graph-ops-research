#!/bin/bash
# run_final.sh -- plain vs velocity, const vs linear, on a large graph.
#
# The four configurations the pubmed grid left standing. Every lr is 0.999
# (the global rule: never 1.0), and both rules have DC gain 1, so the
# effective learning rate is 0.999 for all four and the ONLY things varying
# are the rule and the schedule.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
export PYTHONPATH=/home/h/gnn/fd-graph-embedding/fdmap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
G=$1; E=${2:-200}; B=${3:-1024}
OUT=experiments/fodiwalk-streaming/final_${G}.tsv
LOGS=experiments/fodiwalk-streaming/logs

CFG=("plain     0.999   const"
     "plain     0.999   linear"
     "velocity  0.999   const"
     "velocity  0.999   linear")

for c in "${CFG[@]}"; do
  set -- $c; o=$1; lr=$2; d=$3
  tag="${G}_${o}_lr${lr}_${d}_e${E}"
  echo "=== $tag  $(date -u +%H:%M:%SZ)  free $(free -g|awk 'NR==2{print $7}')G ==="
  timeout 21600 .venv/bin/python experiments/fodiwalk-streaming/bench_stream.py \
      --graph "$G" --dim 64 --walks 5 --walk-len 20 --epochs "$E" \
      --batch "$B" --device gpu --seed 42 --score \
      --optim "$o" --lr "$lr" --lr-decay "$d" \
      --tag "$tag" --out "$OUT" > "$LOGS/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc"
  else grep -E "TOTAL|peak RSS .*embedding AND scoring|accuracy  |f1  |auc  " \
         "$LOGS/$tag.log" | sed 's/^/  /'; fi
done
