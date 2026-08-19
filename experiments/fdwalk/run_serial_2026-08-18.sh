#!/bin/bash
# SERIAL. One measurement run at a time, nothing else on the machine.
# Rule of CLAUDE.md, 2026-08-18: a concurrent run makes the runtime and the
# peak memory of BOTH runs meaningless.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
if pgrep -f "bench_fdwalk" | grep -qv $$; then
  echo "REFUSING: a bench_fdwalk process is already running."; exit 1
fi
# 1. grids B and C, re-measured alone (their quality numbers were valid;
#    their time and memory overlapped the lr-decay smoke).
rm -f experiments/fdwalk/results/grid_cora/B_*.log \
      experiments/fdwalk/results/grid_cora/C_*.log
bash experiments/fdwalk/run_grids_2026-08-18.sh cora 2000
# 2. the lr-decay runs, re-measured alone.
rm -f experiments/fdwalk/results/lrdecay/*.log
bash experiments/fdwalk/run_lrdecay_smoke.sh > experiments/fdwalk/results/lrdecay.txt 2>&1
echo "=== serial batch done ==="
