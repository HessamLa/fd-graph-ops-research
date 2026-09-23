#!/bin/bash
# Score new ICLR 2027 runs every 5 minutes. Stop: touch data_cache/evaluator/for-iclr2027/STOP
# One thread, nice 19: Runner times the large-graph embeddings beside this.
# eval_runs.py itself skips a large graph while a large embedding is in flight.
cd /home/h/gnn/fd-graph-embedding/fdmap
D=data_cache/evaluator/for-iclr2027
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ ! -e $D/STOP ]; do
  n=$(.venv/bin/python $D/eval_runs.py --list | tail -1 | cut -d' ' -f1)
  if [ "$n" != "0" ]; then
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $n pending ===" >> $D/eval_runs.log
    nice -n 19 .venv/bin/python $D/eval_runs.py >> $D/eval_runs.log 2>&1
  fi
  sleep 300
done
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) watch stopped ===" >> $D/eval_runs.log
