#!/usr/bin/env bash
sep(){ printf '\n== %s ==\n' "$1"; }

sep "host"
uname -a
ldd --version 2>/dev/null | head -1

sep "cpu"
lscpu | grep -E 'Model name|^CPU\(s\)|Core\(s\)|Thread|max MHz|L2|L3'

sep "memory"
free -h

sep "disk (home)"
df -h "$HOME" | tail -1

sep "gpu / driver"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv 2>/dev/null || true

sep "cuda toolkits on disk (optional; wheels bundle their own CUDA)"
ls -d /usr/local/cuda* 2>/dev/null || echo "none under /usr/local"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 || echo "nvcc not on PATH (fine)"

sep "pythons"
command -v python3 >/dev/null && { command -v python3; python3 --version; }
FDMAP_PY="$HOME/gnn/fd-graph-embedding/fdmap/.venv/bin/python"
[ -x "$FDMAP_PY" ] && { echo "$FDMAP_PY"; "$FDMAP_PY" --version; } \
  || echo "FDMap venv python not found at $FDMAP_PY (adjust path if moved)"
