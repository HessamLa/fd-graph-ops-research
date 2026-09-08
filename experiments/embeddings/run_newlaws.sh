#!/bin/bash
# fdhop2 and fdunit on cora and pubmed. One job at a time.
#
# fdhop2 is registered, so it runs on the KERNEL and compares directly with
# the stored fdhop rows. fdunit cannot run on the kernel -- its direction is
# a per-row aggregate -- so it runs in the host loop, and the loop's own
# fdhop run is its control. The loop and the kernel do NOT agree to the
# last digit (see RESULTS), thus a fdunit number is only comparable with a
# loop number.
cd /home/h/gnn/fd-graph-embedding/fdmap
PY=.venv/bin/python
S=experiments/embeddings/make_embedding.py
LOG=experiments/embeddings/newlaws.log
RUNS=experiments/embeddings/runs.log

run() {
  echo "=== $(date -u +%H:%M:%SZ) $* ===" >> "$LOG"
  $PY $S "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "--- exit $rc ---" >> "$LOG"
  [ "$rc" -ne 0 ] && printf '%s\tFAIL\tscheduler saw exit %s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$*" >> "$RUNS"
}

FD="--pairs walk_edges --optim plain --weight min_gap --lr 0.999 --device cpu"

for G in cora pubmed; do
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdhop2 --k4 1.0 --walks 10 --walk-len 20 --window 5 --score
  run --method fodiwalk --graph $G --dim 128 --epochs 200 $FD \
      --force fdunit --k4 1.0 --walks 10 --walk-len 20 --window 5 --score
done

# the loop control for pubmed; cora's was run by hand already
run --method fodiwalk --graph pubmed --dim 128 --epochs 200 $FD \
    --force fdhop --k4 1.0 --custom-loop --walks 10 --walk-len 20 --window 5 --score

echo "NEWLAWS DONE $(date -u +%H:%M:%SZ)" >> "$LOG"
