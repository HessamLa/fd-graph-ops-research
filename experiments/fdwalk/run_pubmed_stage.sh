#!/bin/bash
# Stage 2 of the ladder of 2026-08-18: PubMed, 19,717 nodes.
#
# Instructions honoured here:
#   * Wait for every Cora run to finish. NOTHING runs in parallel.
#   * Grid A carries only the (optim, lr) points that did NOT diverge on
#     Cora, taken from `survivors.py`.
#   * Every lr is below 1.0.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/grid_pubmed; mkdir -p $OUT

# --- wait for the machine to be free -------------------------------------
while pgrep -f "bench_fdwalk" > /dev/null; do sleep 30; done
echo "machine free at $(date -Iseconds); starting PubMed"

G=pubmed; EP=2000
BASE="--graph $G --weight min_gap --dim 64 --epochs $EP --far-bias 0.75 --far-weight 100"
FDL="--force fdlinear --k4 1.0 --kr 1.0"
run () { n=$1; shift
  L=$OUT/$n.log
  [ -s "$L" ] && grep -q RESULT "$L" && { echo "skip $n"; return; }
  .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE "$@" > "$L" 2>&1 \
    || echo "FAILED $n"
  .venv/bin/python experiments/fdwalk/update_results.py > /dev/null
}

echo "=== Grid A: only the optimizers that survived on Cora ==="
.venv/bin/python experiments/fdwalk/survivors.py | while read -r o lr; do
  run "A_${o}_lr${lr}" --pairs nbr_walk $FDL --optim "$o" --lr "$lr" --seed 42
done
# the decaying schedule, from the two starts the user named
for lr in 0.999 0.9; do
  run "A_decay_plain_lr${lr}" --pairs nbr_walk $FDL --optim plain --lr $lr --lr-decay linear --seed 42
done

echo "=== Grid B: cap x buckets X symmetric x directed ==="
for s in 42 56 88; do
  run "B_cap_sym_s$s"     --pairs walk     --cap 16        --seed $s $FDL --lr 0.1
  run "B_cap_dir_s$s"     --pairs nbr_walk --row-cap 16    --seed $s $FDL --lr 0.1
  run "B_buckets_sym_s$s" --pairs walk     --policy buckets --seed $s $FDL --lr 0.1
  run "B_buckets_dir_s$s" --pairs nbr_walk --policy buckets --seed $s $FDL --lr 0.1
done

echo "=== Grid C: buckets, far pairs and budget ==="
for s in 42 56 88; do
  run "C_buckets_nofar_s$s" --pairs walk --policy buckets --seed $s $FDL --lr 0.1
  run "C_buckets_far_s$s"   --pairs walk --policy buckets --far-with-buckets --seed $s $FDL --lr 0.1
done

echo "=== Grid D: the second-order walk ==="
for s in 42 56 88; do
  for pq in "1.0 1.0" "1.0 0.5" "1.0 2.0" "0.5 1.0" "2.0 1.0"; do
    set -- $pq
    run "D_p${1}_q${2}_s$s" --pairs nbr_walk $FDL --lr 0.1 --p $1 --q $2 --seed $s
  done
done
echo "=== PubMed stage done $(date -Iseconds) ==="
