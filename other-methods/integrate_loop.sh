#!/bin/bash
# Integrate-as-you-go. For each remaining embedding: generate it, evaluate
# it, update the results tables/files, commit and push the branch, then
# merge to main and push. One embedding at a time.
#
# main is merged in a dedicated worktree (/home/h/gm/fd-main-wt) so the
# primary tree stays on other-comparisons and generation is never disturbed.
set -u
ROOT=/home/h/gm/fd-graph-ops-research
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
MAINWT=/home/h/gm/fd-main-wt
ME=experiments/embeddings/make_embedding.py
TABLES=other-methods/results/comparison_tables_11seed
RL=other-methods/results/collect_runlist.txt
LOG="$ROOT/other-methods/results/integrate_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== integrate_loop start $(date -u) ==="

# remaining embeddings: "type graph seed", generated in this order
JOBS=(
  "deepwalk pubmed 51" "deepwalk pubmed 52"
  "f2v1 cora 45" "f2v1 cora 46" "f2v1 cora 47" "f2v1 cora 48"
  "f2v1 cora 49" "f2v1 cora 50" "f2v1 cora 51" "f2v1 cora 52"
  "f2v1 citeseer 45" "f2v1 citeseer 46" "f2v1 citeseer 47" "f2v1 citeseer 48"
  "f2v1 citeseer 49" "f2v1 citeseer 50" "f2v1 citeseer 51" "f2v1 citeseer 52"
)

generate() { # type graph seed
  case $1 in
    deepwalk) timeout 5400 "$PY" "$ME" --method deepwalk --graph "$2" --dim 128 \
              --walks 80 --walk-len 40 --window 10 --seed "$3" \
              --protocol fodiwalk_dist --score 2>&1 | grep -vi cuda ;;
    f2v1)     timeout 5400 "$PY" other-methods/force2vec/run.py --graph "$2" \
              --option 1 --dim 128 --iter 1200 --threads 8 --seed "$3" 2>&1 | grep -vi cuda ;;
  esac
}

merge_to_main() { # commit message tail
  git -C "$MAINWT" fetch -q origin
  # bring any upstream main commits (other sessions) first
  if ! git -C "$MAINWT" merge -q --ff-only origin/main; then
    echo "  WARN: main worktree not ff to origin/main; skipping main merge this round"
    return 1
  fi
  if git -C "$MAINWT" merge -q --no-edit other-comparisons \
       && git -C "$MAINWT" push -q origin main; then
    echo "  main updated -> $(git -C "$MAINWT" rev-parse --short HEAD)"
  else
    git -C "$MAINWT" merge --abort 2>/dev/null
    echo "  WARN: main merge/push failed; left for manual reconcile"
    return 1
  fi
}

for job in "${JOBS[@]}"; do
  set -- $job; TYPE=$1; G=$2; S=$3
  echo "=== $(date -u +%H:%M:%SZ) generate $TYPE $G seed=$S ==="
  generate "$TYPE" "$G" "$S"
  echo "--- evaluate (incremental) + update tables ---"
  "$PY" other-methods/eval_sequencer.py "$G" 2>&1 | grep -viE "cuda|all cached, skip"
  "$PY" other-methods/collect_store.py "$RL" >/dev/null 2>&1
  "$PY" other-methods/score_report.py "$RL" "$TABLES.md" >/dev/null 2>&1
  echo "--- commit + push branch ---"
  git add "$TABLES.md" "$TABLES.json"
  if git commit -q -m "results: $TYPE $G seed=$S ($(date -u +%H:%MZ))"; then
    git push -q origin other-comparisons && echo "  branch pushed -> $(git rev-parse --short HEAD)"
    merge_to_main "$TYPE $G seed=$S"
  else
    echo "  no table change to commit"
  fi
done
echo "=== integrate_loop DONE $(date -u) ==="
