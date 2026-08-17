"""Perf benchmark: fdge_jax's ``ShellForce`` (JAX/GPU) vs. fdge2's
``ShellForce`` (Numba) -- the hot-loop kernel each package's whole design
exists to make fast (docs/DESIGN.md: augmentation runs once, ``forces``
runs every epoch).

Uses the genuinely sparse ``sparse_hops`` augmentation policy (bounded
radius) on both sides so the comparison scales to large ``n`` -- dense
hop-fill's ``D`` is ``O(n^2)`` by construction on *either* engine, which
would benchmark memory bandwidth more than kernel throughput at large n.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax.validation.bench_embedding_perf     # from repo root (fdmap/)
"""
from __future__ import annotations

import time

import numpy as np
import networkx as nx

from fdge2.graph_augmenting.sparse_hops import augment_graph as numba_augment
from fdge2.embedding.shell_force import ShellForce as NumbaShellForce

from fdge_jax.graph_augmenting.sparse_hops import augment_graph as jax_augment
from fdge_jax.embedding.shell_force import ShellForce as JaxShellForce

import jax

K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01


def _make_graph(n, avg_degree, seed=0):
    k = max(2, avg_degree - (avg_degree % 2))  # nx requires even k
    return nx.watts_strogatz_graph(n, k=k, p=0.1, seed=seed)


def _bench_numba(G, Z, D, epochs):
    sf = NumbaShellForce(k1=K1, k2=K2, k3=K3, k4=K4, random_drop_rate=0.0)
    sf(Z, D, 0, Z.shape[0], rng=np.random.default_rng(0))  # warm up njit compile
    t0 = time.perf_counter()
    for _ in range(epochs):
        sf(Z, D, 0, Z.shape[0], rng=np.random.default_rng(0))
    t1 = time.perf_counter()
    return (t1 - t0) / epochs


def _bench_jax(G, Z, D, epochs):
    sf = JaxShellForce(k1=K1, k2=K2, k3=K3, k4=K4, random_drop_rate=0.0)
    key = jax.random.PRNGKey(0)
    sf(Z, D, 0, Z.shape[0], key=key)  # warm up jit compile
    jax.block_until_ready(sf(Z, D, 0, Z.shape[0], key=key))
    t0 = time.perf_counter()
    out = None
    for _ in range(epochs):
        out = sf(Z, D, 0, Z.shape[0], key=key)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / epochs


def bench(n, d=16, avg_degree=10, radius=2, epochs=200, seed=0):
    G = _make_graph(n, avg_degree, seed=seed)
    rng = np.random.default_rng(seed)
    Z = np.ascontiguousarray(rng.standard_normal((n, d)))

    D_numba = numba_augment(G, is_sparse=True, radius=radius)
    D_jax = jax_augment(G, is_sparse=True, radius=radius)
    nnz = D_jax.nnz

    numba_ms = _bench_numba(G, Z, D_numba, epochs) * 1e3
    jax_ms = _bench_jax(G, Z, D_jax, epochs) * 1e3
    print(f"n={n:>7} nnz={nnz:>9} d={d:<3} "
          f"numba={numba_ms:8.4f} ms/epoch  jax[gpu]={jax_ms:8.4f} ms/epoch  "
          f"speedup={numba_ms / jax_ms:5.2f}x")
    return n, nnz, numba_ms, jax_ms


if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}")
    # NOTE: this machine's GPU has only ~2GB VRAM (checked via nvidia-smi) --
    # n is capped well below where a real deployment GPU would need to
    # stop; JAX's caching GPU allocator also doesn't return freed blocks
    # to the OS between sizes in this loop, so even n values that fit in
    # isolation can OOM after a few prior iterations have fragmented the
    # 2GB pool. Scale n up freely on a GPU with more (or dedicated) memory.
    results = []
    for n in (1_000, 5_000, 10_000, 20_000, 30_000):
        results.append(bench(n=n, d=16, avg_degree=10, radius=2, epochs=200))

    print("\nSummary:")
    print(f"{'n':>8} {'nnz':>10} {'numba (ms)':>12} {'jax[gpu] (ms)':>14} {'speedup':>9}")
    for n, nnz, numba_ms, jax_ms in results:
        print(f"{n:>8} {nnz:>10} {numba_ms:>12.4f} {jax_ms:>14.4f} "
              f"{numba_ms / jax_ms:>8.2f}x")
