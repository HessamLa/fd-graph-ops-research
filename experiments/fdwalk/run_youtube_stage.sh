#!/bin/bash
# com_youtube, 1,134,890 nodes. The 1.13M stage of 2026-08-18.
#
# Decisions of the user, recorded here because they shape the list:
#   * `sqn` is DROPPED. It needs 2*memory+2 = 8 arrays of (n,d), which is
#     2324 MB at dim 64, and this card is a GTX 950 with 2048 MiB total.
#     Even memory=1 does not fit. H6 -- "the optimizer state is the limit
#     at 1M" -- is thus confirmed by arithmetic, not by a crash.
#   * SMALL CONFIGURATIONS RUN FIRST, ordered by expected D.nnz, so that
#     the memory results exist even if a later, larger run is killed. Only
#     ~5 GB of system RAM is free and the best 1.13M run peaked at 4374 MB.
#   * One seed (42) for the whole grid; three seeds afterwards for the
#     winners only.
#
# STRICTLY SERIAL. com_youtube is a large graph: CLAUDE.md forbids sharing
# the machine, and this is the only stage where time and peak memory are
# real measurements rather than noise on a ~1 GB fixed cost.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/grid_youtube; mkdir -p $OUT

# The guard counts REAL interpreter processes by their exe symlink, not by
# a pattern over command lines. A `pgrep -f bench_fdwalk` matches any shell
# whose command line merely CONTAINS that string -- including the tool call
# that writes this file -- and it refused a launch that way on 2026-08-18.
busy=$(ls -l /proc/*/exe 2>/dev/null | grep -c "\.venv/bin/python")
if [ "$busy" -gt 0 ]; then
  echo "REFUSING: $busy venv python process(es) already running."
  ls -l /proc/*/exe 2>/dev/null | grep "\.venv/bin/python"
  exit 1
fi

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
BASE="--graph com_youtube --weight min_gap --dim 64 --epochs 500 \
      --far 2270000 --far-bias 0.75 --far-weight 100 \
      --chunks 8 --chunk-host --force fdlinear --k4 1.0 --kr 1.0"
# The policy grid holds the optimizer fixed at `plain` with linear decay:
# zero state arrays, thus the smallest device footprint, and the decay
# repaired `plain` by +72% on PubMed.
POL="--optim plain --lr 0.999 --lr-decay linear"

run () { n=$1; shift
  L=$OUT/$n.log
  [ -s "$L" ] && grep -q RESULT "$L" && { echo "skip $n"; return; }
  echo "--- $n  $(date -Iseconds)"
  /usr/bin/time -v -o $OUT/$n.time \
    .venv/bin/python experiments/fdwalk/bench_fdwalk.py $BASE "$@" > "$L" 2>&1 \
    || echo "FAILED $n"
  .venv/bin/python experiments/fdwalk/update_results.py > /dev/null
}

# ---- ordered by expected D.nnz, smallest first ---------------------------
echo "=== Y-B / Y-C: the policy grid (smallest matrices) ==="
run Y_B_buckets_dir_s42  --pairs nbr_walk --policy buckets $POL --seed 42
run Y_B_buckets_sym_s42  --pairs walk     --policy buckets $POL --seed 42
run Y_C_buckets_far_s42  --pairs walk     --policy buckets --far-with-buckets $POL --seed 42
run Y_B_cap_dir_s42      --pairs nbr_walk --row-cap 16     $POL --seed 42
run Y_B_cap_sym_s42      --pairs walk     --cap 16         $POL --seed 42

echo "=== Y-D: p = 0.5, the smaller walk ==="
run Y_D_p0.5_q1.0_s42    --pairs nbr_walk --p 0.5 --q 1.0  $POL --seed 42

echo "=== Y-A: the optimizer sweep on nbr_walk/both ==="
run Y_A_plain_const_s42     --pairs nbr_walk --optim plain    --lr 0.999 --seed 42
run Y_A_plain_decay_s42     --pairs nbr_walk --optim plain    --lr 0.999 --lr-decay linear --seed 42
run Y_A_adam_decay_s42      --pairs nbr_walk --optim adam     --lr 0.999 --lr-decay linear --seed 42
run Y_A_velocity_const_s42  --pairs nbr_walk --optim velocity --lr 0.999 --seed 42
run Y_A_momentum_const_s42  --pairs nbr_walk --optim momentum --lr 0.1   --seed 42
run Y_A_nesterov_const_s42  --pairs nbr_walk --optim nesterov --lr 0.1   --seed 42

echo "=== Y-D: p = 2.0, the larger walk (last, it is the biggest) ==="
run Y_D_p2.0_q1.0_s42    --pairs nbr_walk --p 2.0 --q 1.0  $POL --seed 42

echo "=== com_youtube seed-42 pass done $(date -Iseconds) ==="
