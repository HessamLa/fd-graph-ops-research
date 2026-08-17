#!/bin/bash
# Run fodined/modular.py on one graph, with a guard on the memory.
#
# This machine has about 3 GB of free RAM. A graph of a million nodes can ask
# for much more, and the Linux OOM killer then stops a random process, which
# can be the desktop. The watchdog below kills only the child.
#
#   $1  graph name          $2  MAX_NODES (0 = the whole graph)
#   $3  RSS limit in GB     $4  timeout in seconds     $5  log file
set -u
GRAPH=$1; MAXN=$2; RSSGB=$3; TMO=$4; LOG=$5
cd /home/h/gnn/fd-graph-embedding/fdmap

# NO `timeout` wrapper around the python process: `$!` then gives the PID of
# `timeout`, and the watchdog reads the memory of THAT process, which stays
# at 2 MB. A first version did this, and the python process reached 4.8 GB
# with 2 GB of free RAM before a manual kill stopped it. The deadline below
# uses SECONDS instead, thus `$!` is the python process itself.
FODINED_GRAPH="$GRAPH" FODINED_MAX_NODES="$MAXN" \
    .venv/bin/python -u fodined/modular.py > "$LOG" 2>&1 &
PID=$!
LIMIT=$(python3 -c "print(int($RSSGB*1024*1024))")   # kB
START=$SECONDS

while kill -0 $PID 2>/dev/null; do
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    PEAK=$(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null || echo "$PEAK")
    if [ "${RSS:-0}" -gt "$LIMIT" ]; then
        echo "" >> "$LOG"
        echo "[guard] KILLED at $((SECONDS-START))s: RSS $((RSS/1024)) MB passed the limit of ${RSSGB} GB" >> "$LOG"
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
RC=$?
echo "[guard] exit code $RC, peak RSS ${PEAK:-?} kB" >> "$LOG"
tail -3 "$LOG"
