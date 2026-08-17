"""
model_204_shell.py -- Numba rewrite of the shell-averaged FD model.

Physics (identical to the legacy torch model, per node pair (u, v) with
hop distance h = hops[u, v] > 0 and embedding distance x = ||Z[v] - Z[u]||):

    Fa(u,v) =  k1 * (1/|S_h(u)|) * x * exp(-k2 * (h - 1))     attractive
    Fr(u,v) = -k3 * h * exp(-k4 * x)                          repulsive

where S_h(u) is the set of nodes exactly h hops from u (the "shell").
The scalar force acts along the unit vector (Z[v]-Z[u])/x. Per-node force
sums are divided by the node degree, then a steady-rate random drop is
applied to the resulting dZ rows.

Deliberate fixes relative to the legacy implementation:

* Hops are stored as an int32 (n, n) matrix, not float64 -- 2x less memory.
* Shell sizes are kept as a compact (n, maxhops+2) count table indexed by
  hop value, instead of a dense float64 (n, n) occurrence matrix.
* The kernel never materializes the (batch, n, d) pairwise-difference
  tensor: forces are accumulated per row. Peak extra memory is O(n*d).
* Random drop is per NODE (whole dZ row), not per coordinate. The legacy
  per-coordinate dropout introduced axis-aligned anisotropy.
* Disconnected pairs keep the legacy convention h = n_nodes (so behavior
  on multi-component graphs is unchanged and comparable).

Researcher notes -- to try different force laws, edit `_scalar_force`
below. It is a scalar function of (x, h, shell_coeff, k1..k4) and is
inlined into the parallel kernel by Numba; nothing else needs to change.
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit, prange

from .ForceDirected import ForceDirected


# ---------------------------------------------------------------------------
# Graph utilities (NetworkX in -> CSR arrays -> Numba)
# ---------------------------------------------------------------------------
def graph_to_csr(Gx):
    """Convert an (undirected) NetworkX graph to CSR adjacency arrays.

    Nodes are indexed 0..n-1 in the order of Gx.nodes().
    Returns (indptr, indices, degrees).
    """
    nodes = list(Gx.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    degrees = np.zeros(n, dtype=np.int64)
    for u, d in Gx.degree():
        degrees[index[u]] = d
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    indices = np.empty(indptr[-1], dtype=np.int64)
    cursor = indptr[:-1].copy()
    for u, v in Gx.edges():
        iu, iv = index[u], index[v]
        indices[cursor[iu]] = iv; cursor[iu] += 1
        indices[cursor[iv]] = iu; cursor[iv] += 1
    return indptr, indices, degrees


@njit(parallel=True, cache=True)
def get_hops_csr(indptr, indices, n, unreachable):
    """All-pairs hop distances by parallel BFS, one source per thread.

    Returns int32 (n, n); hops[u, u] = 0; unreachable pairs get the
    sentinel value `unreachable` (legacy convention: n).
    """
    hops = np.full((n, n), unreachable, dtype=np.int32)
    for src in prange(n):
        dist = hops[src]
        dist[src] = 0
        frontier = np.empty(n, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        frontier[0] = src
        f_len = 1
        level = 0
        while f_len > 0:
            level += 1
            nxt_len = 0
            for fi in range(f_len):
                u = frontier[fi]
                for p in range(indptr[u], indptr[u + 1]):
                    v = indices[p]
                    if dist[v] == unreachable:
                        dist[v] = level
                        nxt[nxt_len] = v
                        nxt_len += 1
            frontier, nxt = nxt, frontier
            f_len = nxt_len
    return hops


def get_shell_counts(hops, n_bins):
    """counts[u, h] = |S_h(u)| = number of nodes exactly h hops from u.

    Compact (n, n_bins) int64 table; replaces the legacy dense float64
    (n, n) occurrence matrix. n_bins must be > max value present in hops.
    """
    n = hops.shape[0]
    counts = np.zeros((n, n_bins), dtype=np.int64)
    for u in range(n):
        binc = np.bincount(hops[u], minlength=n_bins)
        counts[u, :] = binc[:n_bins]
    return counts


def generate_random_points(n_points, n_dim, rng, radius=1.0):
    """Uniformly distributed points inside a d-ball of the given radius."""
    directions = rng.standard_normal((n_points, n_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(n_points) ** (1.0 / n_dim)
    return directions * radii[:, None]


# ---------------------------------------------------------------------------
# Force law + kernel
# ---------------------------------------------------------------------------
@njit(inline="always")
def _scalar_force(x, h, shell_coeff, k1, k2, k3, k4):
    """Scalar force magnitude for one (u, v) pair. EDIT ME to try new laws.

    x           : embedding-space Euclidean distance ||Z[v] - Z[u]||
    h           : hop distance (>= 1; disconnected pairs carry h = n)
    shell_coeff : 1/|S_h(u)|, the shell-averaging factor
    Positive values attract (pull u toward v), negative repel.
    """
    Fa = k1 * shell_coeff * x * np.exp(-k2 * (h - 1.0))
    Fr = -k3 * h * np.exp(-k4 * x)
    return Fa + Fr


# ---------------------------------------------------------------------------
# Random drop
# ---------------------------------------------------------------------------
class DropSteadyRate:
    """Zero out whole dZ rows (nodes) with a fixed probability each epoch.

    Per-node (row-wise) drop, unlike the legacy per-coordinate dropout,
    which biased updates along coordinate axes (axis-aligned anisotropy).
    """

    def __init__(self, drop_rate: float = 0.5, name: str = "steady-rate"):
        self.name = name
        self.drop_rate = float(drop_rate)

    def __call__(self, dZ: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.drop_rate <= 0.0:
            return dZ
        keep = rng.random(dZ.shape[0]) >= self.drop_rate
        dZ *= keep[:, None]
        return dZ


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FDModel(ForceDirected):
    """Force Directed Model -- shell averaging (Numba backend).

    f(x) = 1/|S_h(u)| * k1 * x * e^{-k2 (h-1)}  -  k3 * h * e^{-k4 x}
    """

    VER_MIN = "04"
    DESCRIPTION = ("The basic FD model with shell averaging: "
                   "f(x) = 1/|S_h(u)| * ( x e^(-h+1) - h e^(-x) ). Numba backend.")

    def __init__(self, Gx, n_dim,
                 k1: float = 0.999, k2: float = 1.0, k3: float = 10.0, k4: float = 0.01,
                 lr: float = 1.0, random_drop_rate: float = 0.5,
                 **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.VERSION = f"{self.VER_MAJ}{self.VER_MIN}"
        self.k1, self.k2, self.k4 = k1, k2, k4
        self.n_dim = n_dim
        self.Gx = Gx

        self.n_nodes = n = Gx.number_of_nodes()
        self.k3 = float(n) if k3 is None else k3

        # --- graph structure -> CSR -> hop distances
        indptr, indices, degrees = graph_to_csr(Gx)
        self.degrees = degrees.astype(np.float64)

        hops = get_hops_csr(indptr, indices, n, unreachable=np.int32(n))
        finite = hops[hops < n]
        self.maxhops = int(finite.max()) if finite.size else 0
        self.hops = hops  # int32 (n, n); disconnected pairs = n (legacy convention)

        # --- shell sizes |S_h(u)| as a compact lookup table
        # bins 0..maxhops plus (possibly) the sentinel value n.
        if (hops == n).any():
            # remap sentinel n -> maxhops+1 for compact indexing; keep the
            # numeric value h = n inside the force law for legacy parity.
            self._sentinel_idx = self.maxhops + 1
            hops_idx = hops.copy()
            hops_idx[hops_idx == n] = self._sentinel_idx
            self._h_of_idx = np.arange(self._sentinel_idx + 1, dtype=np.float64)
            self._h_of_idx[self._sentinel_idx] = float(n)
        else:
            self._sentinel_idx = self.maxhops
            hops_idx = hops
            self._h_of_idx = np.arange(self.maxhops + 1, dtype=np.float64)
        self._hops_idx = np.ascontiguousarray(hops_idx, dtype=np.int32)
        self.shell_counts = get_shell_counts(self._hops_idx, self._sentinel_idx + 1)

        if self.verbosity >= 1:
            print("max hops:", self.maxhops)
        if self.verbosity >= 3:
            print("hops.shape", self.hops.shape)
            print("shell_counts.shape", self.shell_counts.shape)

        # --- initial embedding
        self.Z = np.ascontiguousarray(
            generate_random_points(n, n_dim, self.rng), dtype=np.float64)

        self.random_drop = DropSteadyRate(drop_rate=random_drop_rate)

    # ------------------------------------------------------------------ core
    def forward(self, row_start: int, row_end: int, **kwargs) -> np.ndarray:
        out = np.empty((row_end - row_start, self.n_dim), dtype=np.float64)
        _forces_204_hidx(self.Z, self._hops_idx, self._h_of_idx,
                         self.shell_counts, self.degrees,
                         self.k1, self.k2, self.k3, self.k4,
                         row_start, row_end, out)
        if self.verbosity >= 3:
            magnitudes = np.linalg.norm(out, axis=-1)
            mag_sum = magnitudes.sum()
            mag_mean = magnitudes.mean()
            print(f"  |dZ| sum: {mag_sum:.3f}, mean: {mag_mean:.8f}")
        return self.random_drop(out, self.rng)


@njit(parallel=True, fastmath=True, cache=True)
def _forces_204_hidx(Z, hops_idx, h_of_idx, shell_counts, degrees,
                     k1, k2, k3, k4, row_start, row_end, out):
    """Same as _forces_204 but hop values are index-compressed:
    hops_idx[u, v] indexes both shell_counts[u, .] and h_of_idx[.],
    where h_of_idx maps the index back to the numeric hop value
    (the sentinel index maps to h = n_nodes, the legacy convention).
    """
    n, d = Z.shape
    for r in prange(row_end - row_start):
        u = row_start + r
        hrow = hops_idx[u]
        crow = shell_counts[u]
        acc = np.zeros(d)
        for v in range(n):
            hi = hrow[v]
            if hi <= 0:
                continue
            s = 0.0
            for t in range(d):
                diff = Z[v, t] - Z[u, t]
                s += diff * diff
            x = np.sqrt(s)
            if x == 0.0:
                continue
            shell_coeff = 1.0 / crow[hi]
            f = _scalar_force(x, h_of_idx[hi], shell_coeff, k1, k2, k3, k4)
            scale = f / x
            for t in range(d):
                acc[t] += scale * (Z[v, t] - Z[u, t])
        deg = degrees[u]
        if deg > 0:
            for t in range(d):
                out[r, t] = acc[t] / deg
        else:
            for t in range(d):
                out[r, t] = 0.0
