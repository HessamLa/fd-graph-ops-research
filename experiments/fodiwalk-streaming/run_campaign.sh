#!/bin/bash
# run_campaign.sh -- streaming fodiwalk over every graph the loader supports.
#
# One process per (graph, epochs). Each writes its own log under logs/ and
# appends one RESULT line to results.tsv, so a run that dies leaves the
# earlier rows intact and names itself in the gap.
#
# The card is 2048 MiB and JAX preallocates 75% by default, which is not
# enough for a graph of this size: MEM_FRACTION=0.95 is what makes the
# large graphs run at all (experiments/fodiwalk/REPORT_1M.md section 3).
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
export PYTHONPATH=/home/h/gnn/fd-graph-embedding/fdmap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

OUT=experiments/fodiwalk-streaming/results.tsv
LOGS=experiments/fodiwalk-streaming/logs
BIN=experiments/fodiwalk-streaming/bench_stream.py

run () {   # graph epochs batch device
  g=$1; e=$2; b=$3; d=$4
  tag="${g}_e${e}_b${b}_${d}"
  echo "=== $tag  ($(date -u +%H:%M:%SZ), free $(free -g|awk 'NR==2{print $7}')G) ==="
  timeout 10800 .venv/bin/python "$BIN" \
      --graph "$g" --dim 64 --walks 5 --walk-len 20 --epochs "$e" \
      --batch "$b" --device "$d" --seed 42 --score --tag "$tag" \
      --out "$OUT" > "$LOGS/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
      echo "  FAILED rc=$rc -- $(grep -oE '[A-Za-z]*Error[^\"]*' "$LOGS/$tag.log" | tail -1)"
  else
      grep -E "^\[stream\] peak RSS .* embedding AND scoring|TOTAL" "$LOGS/$tag.log" | sed 's/^/  /'
  fi
}

for spec in "$@"; do
  IFS=: read -r g e b d <<< "$spec"
  run "$g" "$e" "${b:-1024}" "${d:-gpu}"
done
