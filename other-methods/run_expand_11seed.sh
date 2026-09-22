#!/bin/bash
# Extend every method from 3 seeds (42,43,44) to 11 seeds by adding seeds
# 45..52. ONE embedding at a time. Method order is fast -> slow and
# method-outer, so each row reaches 11 seeds before the next method starts;
# if the run dies late, only the two slow rows (deepwalk, force2vec base)
# are left short.
#
# Usage:  bash other-methods/run_expand_11seed.sh [wait_pid]
set -u
ROOT=/home/h/gm/fd-graph-ops-research
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
KPY="$ROOT/other-methods/karateclub/.venv-karate/bin/python"
ADJ="$ROOT/other-methods/results/adj"
ME="experiments/embeddings/make_embedding.py"
SEEDS="45 46 47 48 49 50 51 52"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RL="$ROOT/other-methods/results/expand11_runlist_${STAMP}.txt"
LOG="$ROOT/other-methods/results/expand11_${STAMP}.log"
: > "$RL"
exec > >(tee -a "$LOG") 2>&1
echo "=== expand-to-11 start $(date -u) seeds:$SEEDS -> $RL ==="

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
  echo "prior batch done, starting."
fi

record(){ grep -oE '] stored .*' | sed 's/] stored /\//; s#^#'"$ROOT"'#' >> "$RL"; }

kc(){ local M=$1 G=$2 S=$3 H; H=$(mktemp -d /tmp/kc_XXXX)
  echo "=== $(date -u +%H:%M:%SZ) $M $G seed=$S ==="
  timeout 2400 "$KPY" other-methods/karateclub/run.py --emit --method "$M" \
      --graph "$G" --seed "$S" --dim 128 --out "$H" --adj "$ADJ/$G.npz" 2>&1 | grep -vi CUDA
  [ -f "$H/Z.npy" ] && "$PY" other-methods/common/store_handoff.py "$H" 2>&1 | grep -vi CUDA | record \
      || echo "SKIP $M $G seed=$S"
  rm -rf "$H"; }

lm(){ local G=$1 S=$2
  echo "=== $(date -u +%H:%M:%SZ) landmark_mds $G seed=$S ==="
  timeout 1200 "$PY" other-methods/landmark_mds/run.py --graph "$G" --seed "$S" --dim 128 2>&1 | grep -vi CUDA | record; }

f2v(){ local OPT=$1 G=$2 S=$3 T=$4
  echo "=== $(date -u +%H:%M:%SZ) force2vec opt$OPT $G seed=$S ==="
  timeout "$T" "$PY" other-methods/force2vec/run.py --graph "$G" --option "$OPT" \
      --dim 128 --iter 1200 --threads 8 --seed "$S" 2>&1 | grep -vi CUDA | record; }

fw(){ local G=$1 S=$2 DS=""
  [ "$G" = citeseer ] && DS="--deg-source A"     # I5: citeseer isolated nodes
  echo "=== $(date -u +%H:%M:%SZ) fodiwalk $G seed=$S ==="
  timeout 1200 "$PY" "$ME" --method fodiwalk --graph "$G" --dim 128 --epochs 200 \
      --pairs walk_edges --weight min_gap --force fdhop --optim sqn --lr 0.999 \
      --lr-decay const --k1 0.999 --k4 1.0 --kr 1.0 --walks 10 --walk-len 20 --window 5 \
      $DS --seed "$S" --device cpu --protocol fodiwalk_dist --score 2>&1 | grep -vi CUDA | record; }

dw(){ local G=$1 S=$2
  echo "=== $(date -u +%H:%M:%SZ) deepwalk $G seed=$S ==="
  timeout 5400 "$PY" "$ME" --method deepwalk --graph "$G" --dim 128 --walks 80 --walk-len 40 \
      --window 10 --seed "$S" --protocol fodiwalk_dist --score 2>&1 | grep -vi CUDA | record; }

n2v(){ local G=$1 S=$2
  echo "=== $(date -u +%H:%M:%SZ) node2vec $G seed=$S ==="
  timeout 2400 "$PY" "$ME" --method node2vec --graph "$G" --dim 128 --walks 10 --walk-len 80 \
      --window 10 --p 1.0 --q 1.0 --seed "$S" --protocol fodiwalk_dist --score 2>&1 | grep -vi CUDA | record; }

# fast -> slow, method-outer. small = cora citeseer only (dense n x n / O(n^2))
for S in $SEEDS; do for G in cora citeseer; do kc netmf "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer; do kc grarep "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer; do kc hope "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do lm "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do kc randne "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do fw "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do kc prone "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do kc lapeig "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do f2v 7 "$G" "$S" 2400; done; done  # rforce
for S in $SEEDS; do for G in cora citeseer pubmed; do f2v 5 "$G" "$S" 2400; done; done  # tforce
for S in $SEEDS; do for G in cora citeseer pubmed; do n2v "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer pubmed; do kc line "$G" "$S"; done; done
# slow last
for S in $SEEDS; do for G in cora citeseer pubmed; do dw "$G" "$S"; done; done
for S in $SEEDS; do for G in cora citeseer; do f2v 1 "$G" "$S" 5400; done; done          # base O(n^2)

echo "=== expand-to-11 DONE $(date -u).  new records: $(wc -l < "$RL") ==="
