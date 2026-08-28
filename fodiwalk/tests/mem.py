"""mem.py -- the memory gate for `nbr_walk`. Peak RSS against a recorded
ceiling, so the 2026-08-28 fix cannot rot back toward what it replaced.

`golden.py` proves the NUMBERS did not move; this proves the PEAK did not
creep back. `agentic-log/10.mem-agent/` cut it from 3067 MB to ~1154 MB at
150,000 nodes of com_youtube, default `nbr_walk` (`walks=10, walk_len=20`).
This runs the SAME production path -- `augment_graph.policies.build` -- and
never touches `stats["key"]`, which only a reader outside the hot path may
ask for (`golden.py`, `check_api.py`); touching it here would measure that
reader's compatibility shim and not the fix.

Use::

    .venv/bin/python -m fodiwalk.tests.mem --check
    .venv/bin/python -m fodiwalk.tests.mem --check --nodes 300000

`--check` fails (exit 1) when the peak exceeds the recorded ceiling of
`--nodes`, and fails loudly (exit 2, not a false pass) when `--nodes` names
a size with no recorded ceiling -- record one in `CEILING_MB` first.

Peak is sampled on a background thread, every 5 ms, from
`/proc/self/statm` -- the measure `agentic-log/10.mem-agent/`'s profiler
used. `ru_maxrss` is also reported, as a cross-check and not the gate: it
is the OS's own high-water mark of PAGES resident at any point in the
process's life, and it never falls, so a second augmentation in the same
process would inflate it for a reason this file does not control.
"""
from __future__ import annotations

import argparse
import resource
import sys
import threading
import time

import numpy as np

# Recorded 2026-08-28 at 150,000 nodes of com_youtube, default `nbr_walk`
# (walks=10, walk_len=20, seed=42): peak RSS 1154 MB, down from the
# pre-fix 3067 MB (`agentic-log/10.mem-agent/`). The ceiling here is the
# task's own target (<= 1200 MB) -- tight against the measurement, not
# loose slack toward the number this fix removed. A run that creeps past
# it is a regression, not jitter: the sampled peak repeated within 1 MB
# across separate runs while this gate was written.
CEILING_MB = {150_000: 1200}
DEFAULT_NODES = 150_000


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def peak_rss(n_nodes: int, dataset: str = "com_youtube", seed: int = 42):
    """Run one `nbr_walk` augmentation. Returns `(peak_mb, ru_maxrss_mb, D.nnz)`.

    The production path, start to finish: `make_graph.load`, then
    `augment_graph.policies.build` with the package defaults of
    `AugmentSpec(pairs="nbr_walk")` -- the exact configuration
    `agentic-log/10.mem-agent/` measured.
    """
    from fodiwalk.augment_graph import policies
    from fodiwalk.augment_graph.result import AugmentSpec
    from fodiwalk.make_graph import load

    peak = [0.0]
    stop = [False]

    def sampler():
        while not stop[0]:
            peak[0] = max(peak[0], _rss_mb())
            time.sleep(0.005)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    peak[0] = max(peak[0], _rss_mb())

    A, n = load(dataset, max_nodes=n_nodes, seed=seed)
    spec = AugmentSpec(pairs="nbr_walk")
    rng = np.random.default_rng(seed)
    aug = policies.build(A, n, spec, rng)

    stop[0] = True
    t.join(timeout=1.0)
    peak[0] = max(peak[0], _rss_mb())
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return peak[0], ru_maxrss, int(aug.D.nnz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    ap.add_argument("--dataset", default="com_youtube")
    a = ap.parse_args(argv)

    peak, ru_maxrss, dnnz = peak_rss(a.nodes, a.dataset)
    ceiling = CEILING_MB.get(a.nodes)
    print(f"[mem] n={a.nodes} D.nnz={dnnz} peak_rss={peak:.0f} MB "
         f"ru_maxrss={ru_maxrss:.0f} MB"
         + (f" ceiling={ceiling} MB" if ceiling is not None
            else " (no recorded ceiling at this size)"))
    if not a.check:
        return 0
    if ceiling is None:
        print(f"[mem] no recorded ceiling at n={a.nodes}; record one in "
             f"CEILING_MB before using --check at this size.")
        return 2
    if peak > ceiling:
        print(f"MEM FAIL: peak {peak:.0f} MB > ceiling {ceiling} MB")
        return 1
    print("MEM OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
