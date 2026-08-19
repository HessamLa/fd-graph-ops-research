#!/bin/bash
# A light sampler, so that an OOM has CONTEXT and not only a signal number.
# One line every 10 s: time, system available MB, the RSS of the running
# bench, and which configuration it is. ~0 CPU, no measurable memory.
set -u
cd /home/h/gnn/fd-graph-embedding/fdmap
OUT=experiments/fdwalk/results/grid_youtube/memwatch.csv
[ -s "$OUT" ] || echo "ts,avail_mb,bench_rss_mb,config" > "$OUT"
while true; do
  read -r a < <(free -m | awk '/^Mem:/{print $7}')
  line=$(ps -eo rss=,args= | awk '/[b]ench_fdwalk\.py/ && !/awk/ {print; exit}')
  rss=$(echo "$line" | awk '{printf "%.0f", $1/1024}')
  cfg=$(echo "$line" | grep -oE '\-\-pairs [a-z_]+|--policy [a-z]+|--optim [a-z]+|--row-cap [0-9]+|--p [0-9.]+|--far-with-buckets' | tr '\n' ' ')
  echo "$(date +%H:%M:%S),${a:-},${rss:-0},\"${cfg:-idle}\"" >> "$OUT"
  sleep 10
done
