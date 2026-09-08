#!/bin/bash
# The rest of the large-graph set, with NO TIMEOUT. A run finishes or it
# dies; it is never cut off at the clock.
#
# Replaces run_matrix6.sh, whose scheduler was stopped after the deepwalk
# run of com_youtube. Two com_youtube cells are re-queued here: both
# fodiwalk 10 x 80 runs were killed at the 4-hour mark and stored nothing.
#
# A non-zero exit now writes its own FAIL line. `timeout` and the OOM
# killer both send a signal that the driver cannot catch, so a run killed
# from outside used to leave NOTHING in runs.log. That gap is closed here,
# in the scheduler, where the exit code is visible.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/matrix7.log
RUNS=experiments/embeddings/runs.log

# One embedding at a time: wait out anything matrix6 left in flight.
while pgrep -f "make_embedding.py" > /dev/null; do
  echo "[wait] an embedding is in flight at $(date -u +%H:%M:%SZ)" >> "$LOG"
  sleep 120
done

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  $PY $S "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "--- exit $rc ---" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    printf '%s\tFAIL\tscheduler saw exit %s (a signal the driver cannot catch, thus no entry of its own)\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$*" >> "$RUNS"
  fi
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

# com_youtube: the two cells the 4-hour cut destroyed.
for LAW in fdhop fdlinear; do
  K4=0.01; [ "$LAW" = fdhop ] && K4=1.0
  run --method fodiwalk --graph com_youtube --dim 128 --epochs 200 $FD \
      --force $LAW --k4 $K4 --walks 10 --walk-len 80 --window 10 --score
done

for G in as_skitter roadnet_ca ncbi_taxonomy; do
  echo "########## $G $(date -u +%H:%M:%SZ) ##########" >> "$LOG"

  for Q in 1.0 0.5 2.0; do
    run --method node2vec --graph $G --dim 128 \
        --walks 10 --walk-len 80 --window 10 --n2v-epochs 1 \
        --p 1.0 --q $Q --score
  done

  for LAW in fdhop fdlinear; do
    K4=0.01; [ "$LAW" = fdhop ] && K4=1.0
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force $LAW --k4 $K4 --walks 10 --walk-len 20 --window 5 --score
  done

  for LAW in fdhop fdlinear; do
    K4=0.01; [ "$LAW" = fdhop ] && K4=1.0
    run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
        --force $LAW --k4 $K4 --walks 10 --walk-len 80 --window 10 --score
  done

  run --method deepwalk --graph $G --dim 128 \
      --walks 80 --walk-len 40 --window 10 --n2v-epochs 1 --score
done

echo "MATRIX7 DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
