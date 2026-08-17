#!/bin/bash
# Gate G3: com_youtube, 1,134,890 nodes. One seed only, because of the cost.
#
#   bash run_g3.sh <pairs> <weight> <cap> <dim> <epochs> <rss_gb> <timeout_s>
#
# The guard watches the RSS of the python process and it kills only that
# process. `../modular-graphs/run_guarded.sh` holds the warning about the
# PID: `$!` after `timeout ... &` gives the PID of `timeout`, and the guard
# then reads 2 MB while python takes 4.8 GB.
#
# PLAN.md section 9 predicts the cost, and the deviations that G3 may take:
# EPOCHS = 500 or M = 8, because 36M cells at the measured rate need 33
# minutes for 2000 epochs.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
PAIRS=${1:-walk}; W=${2:-min_gap}; CAP=${3:-16}; DIM=${4:-128}
EPOCHS=${5:-500}; RSSGB=${6:-2.6}; TMO=${7:-2400}; FAR=${8:-0}
CHUNKS=${9:-1}; PF=${10:-4}; HOST=${11:-0}; CACHE=${12:-}
OUT=experiments/fdwalk/results
mkdir -p $OUT
LOG=$OUT/g3_com_youtube_${PAIRS}_${W}_cap${CAP}_d${DIM}_e${EPOCHS}_c${CHUNKS}_pf${PF}.log

# `--far` at 0 asks for `n * log10(n)` = 6.87M pairs, thus 13.7M more
# entries of D. That is the budget that this machine cannot hold. See
# FINDINGS.md, the deviations of G3.
.venv/bin/python -u experiments/fdwalk/bench_fdwalk.py \
    --graph com_youtube --pairs "$PAIRS" --weight "$W" --cap "$CAP" \
    --dim "$DIM" --epochs "$EPOCHS" --seed 42 --far "$FAR" \
    --chunks "$CHUNKS" --prune-factor "$PF" \
    $( [ "$HOST" = "1" ] && echo --chunk-host ) \
    $( [ -n "$CACHE" ] && echo --cache-d "$CACHE" ) \
    --walks 5 --len 20 --window 3 \
    --hop-sources 200 --hop-pairs 20000 > "$LOG" 2>&1 &
PID=$!
LIMIT=$(python3 -c "print(int($RSSGB*1024*1024))")
START=$SECONDS
PEAK=0

while kill -0 $PID 2>/dev/null; do
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    HWM=$(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    [ "${HWM:-0}" -gt "$PEAK" ] && PEAK=$HWM
    if [ "${RSS:-0}" -gt "$LIMIT" ]; then
        echo "" >> "$LOG"
        echo "[guard] KILLED at $((SECONDS-START))s: RSS $((RSS/1024)) MB passed ${RSSGB} GB" >> "$LOG"
        kill -9 $PID 2>/dev/null
        break
    fi
    if [ $((SECONDS-START)) -gt "$TMO" ]; then
        echo "" >> "$LOG"
        echo "[guard] KILLED: the deadline of ${TMO}s passed" >> "$LOG"
        kill -9 $PID 2>/dev/null
        break
    fi
    sleep 2
done
wait $PID 2>/dev/null
echo "[guard] peak RSS $((PEAK/1024)) MB, $((SECONDS-START))s" >> "$LOG"
tail -5 "$LOG"
