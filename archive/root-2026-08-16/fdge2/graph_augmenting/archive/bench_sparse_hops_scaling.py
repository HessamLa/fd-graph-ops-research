"""bench_sparse_hops_scaling.py -- nnz(D) vs n scaling for the sparse policy.

WHAT THIS IS
------------
A throwaway benchmark script (kept in ``graph_augmenting/archive/`` rather than
alongside the real code, because it is not part of the pipeline and not a
test) that empirically confirms the central claim of T2.2: that
``graph_augmenting.sparse_hops.augment_graph`` produces a ``D`` whose number of
stored entries (``nnz``) grows roughly LINEARLY with ``n`` at fixed
average degree and radius -- i.e. genuinely sub-``O(n^2)`` -- unlike the
dense ``hopfill`` reference whose ``nnz`` is ``~n^2``.

WHY IT EXISTS (rationale)
-------------------------
ARCHITECTURE.md's "Known limitation" note says hop-fill is ``O(n^2)``
regardless of source sparsity, and RPD.md task T2.2 asks for a sparse
policy whose sub-quadratic growth is *measured, not asserted*. This script
is that measurement. The logic:

* If ``D`` were dense (``O(n^2)``), then ``nnz / n`` would grow linearly
  with ``n`` -- e.g. ~16x larger going from n=500 to n=8000.
* Bounded-hop-radius BFS stores, per node, only the nodes within ``radius``
  hops (~``1 + d + d^2 + ... + d^radius`` for average degree ``d``), a
  count that does NOT depend on ``n``. So ``nnz / n`` should stay ~flat as
  ``n`` grows, which is the signature of ``O(n)`` total storage.

The script prints ``nnz``, ``nnz/n``, the dense ``n^2`` it avoids, the
ratio ``sparse_nnz / n^2``, and wall-clock build time, across several
graph sizes and two radii; then a direct head-to-head against dense
``hopfill`` at the smaller sizes (hop-fill is not run at n=8000 because its
``O(n^2)`` intermediate would be large/slow -- which is itself the point).

HOW TO RUN
----------
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    cd /home/h/gnn/fd-graph-embedding/fdmap        # repo root, so fdge2 imports
    python -m fdge2.graph_augmenting.archive.bench_sparse_hops_scaling

Graphs are ``networkx.gnm_random_graph`` at a fixed average degree (the
task explicitly permits plain networkx random graphs). Seed is fixed so
the numbers are reproducible.

RELATION TO THE TEST SUITE
--------------------------
``test_sparse_hops_smoke.py::test_nnz_scales_subquadratic`` asserts a
coarse version of this (nnz/n spread < 3x) so CI catches a regression;
this script is the richer, human-readable table used to write up the
scaling numbers in the T2.2 report. It intentionally shares no code with
the test so it can be run/edited independently.
"""
from __future__ import annotations

import time

import networkx as nx

from fdge2.graph_augmenting import sparse_hops, hopfill


def bench(avg_deg: int = 8, sizes=(500, 2000, 8000), radii=(2, 3)) -> None:
    print(f"Erdos-Renyi (gnm_random_graph), avg_degree={avg_deg}, seed=0\n")
    hdr = (f"{'n':>6} {'radius':>6} {'sparse_nnz':>11} {'nnz/n':>7} "
           f"{'dense_n^2':>12} {'sparse/dense':>12} {'build_s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for n in sizes:
        G = nx.gnm_random_graph(n, n * avg_deg // 2, seed=0)
        for r in radii:
            t = time.perf_counter()
            D = sparse_hops.augment_graph(G, is_sparse=True, radius=r)
            dt = time.perf_counter() - t
            dense = n * n
            print(f"{n:>6} {r:>6} {D.nnz:>11} {D.nnz / n:>7.1f} "
                  f"{dense:>12} {D.nnz / dense:>12.4f} {dt:>8.3f}")

    print("\nHead-to-head vs dense hopfill (fills every reachable pair, ~n^2):")
    for n in (500, 2000):
        G = nx.gnm_random_graph(n, n * avg_deg // 2, seed=0)
        Dh = hopfill.augment_graph(G, is_sparse=True)
        Ds = sparse_hops.augment_graph(G, is_sparse=True, radius=2)
        print(f"  n={n:>5}: hopfill nnz={Dh.nnz:>9} (~n^2={n * n:>9}), "
              f"sparse_hops(r=2) nnz={Ds.nnz:>7}  "
              f"reduction={Dh.nnz / Ds.nnz:>6.1f}x")


if __name__ == "__main__":
    bench()
