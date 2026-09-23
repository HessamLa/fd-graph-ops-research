#!/bin/bash
# Run every comparison method serially -- ONE embedding at a time (the box
# has 7 GB RAM and 8 cores; overlapping runs contend). Each created run
# directory is appended to a run list for the scorer. Each run has a
# timeout so one hang cannot block the queue.
#
# Waits first for the baseline fill batch, so nothing overlaps it.
#
# Usage:  bash other-methods/run_all.sh [fill_pid]
set -u
ROOT=/home/h/gm/fd-graph-ops-research
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
KPY="$ROOT/other-methods/karateclub/.venv-karate/bin/python"
ADJ="$ROOT/other-methods/results/adj"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNLIST="$ROOT/other-methods/results/runlist_${STAMP}.txt"
LOG="$ROOT/other-methods/results/run_all_${STAMP}.log"
: > "$RUNLIST"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_all start $(date -u) -> $RUNLIST ==="

FILL_PID="${1:-}"
if [ -n "$FILL_PID" ]; then
  echo "waiting for fill batch pid $FILL_PID ..."
  while kill -0 "$FILL_PID" 2>/dev/null; do sleep 15; done
  echo "fill batch done, starting."
fi

record() { grep -oE '] stored .*' | sed 's/] stored /\//; s#^#'"$ROOT"'#' >> "$RUNLIST"; }

# --- Stage A: shared-venv method (landmark MDS), 3 graphs x 3 seeds -------
# Laplacian Eigenmaps moved to Stage B (karateclub) -- sklearn's arpack
# hangs on these disconnected graphs; karateclub solves them in seconds.
for G in cora citeseer pubmed; do
  for S in 42 43 44; do
    echo "=== $(date -u +%H:%M:%SZ) landmark_mds $G seed=$S ==="
    timeout 1800 "$PY" other-methods/landmark_mds/run.py --graph "$G" --seed "$S" --dim 128 2>&1 | grep -vi CUDA | record
  done
done

# --- Stage B: karateclub / nodevectors, emit then score ------------------
kc() {  # method graph seed
  local M=$1 G=$2 S=$3 H
  H=$(mktemp -d /tmp/kc_XXXX)
  echo "=== $(date -u +%H:%M:%SZ) $M $G seed=$S (emit) ==="
  timeout 1800 "$KPY" other-methods/karateclub/run.py --emit --method "$M" \
      --graph "$G" --seed "$S" --dim 128 --out "$H" --adj "$ADJ/$G.npz" 2>&1 | grep -vi CUDA
  if [ -f "$H/Z.npy" ]; then
    "$PY" other-methods/common/store_handoff.py "$H" 2>&1 | grep -vi CUDA | record
  else
    echo "SKIP $M $G seed=$S -- no Z emitted (OOM or error above)"
  fi
  rm -rf "$H"
}
for G in cora citeseer pubmed; do            # sparse / scalable: all graphs
  for S in 42 43 44; do
    kc lapeig "$G" "$S"; kc prone "$G" "$S"; kc randne "$G" "$S"; kc line "$G" "$S"
  done
done
for G in cora citeseer; do                   # dense n x n: small graphs only
  for S in 42 43 44; do kc netmf "$G" "$S"; kc grarep "$G" "$S"; kc hope "$G" "$S"; done
done

# --- Stage C: Force2Vec family (binary has no seed -> one run each) -------
f2v() { # option graph timeout
  echo "=== $(date -u +%H:%M:%SZ) force2vec opt$1 $2 ==="
  timeout "$3" "$PY" other-methods/force2vec/run.py --graph "$2" --option "$1" \
      --dim 128 --iter 1200 --threads 8 2>&1 | grep -vi CUDA | record
}
for G in cora citeseer pubmed; do f2v 5 "$G" 7200; f2v 7 "$G" 7200; done  # t- , r-
for G in cora citeseer; do f2v 1 "$G" 5400; done                          # base O(n^2)

echo "=== run_all done $(date -u).  runs: $(wc -l < "$RUNLIST") ==="
echo "RUNLIST=$RUNLIST"
