#!/bin/bash
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
bash experiments/fdwalk/run_new_smoke.sh > experiments/fdwalk/results/newsmoke.txt 2>&1
echo "== new-path smoke done =="
bash experiments/fdwalk/run_grids_2026-08-18.sh cora 2000
