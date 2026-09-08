#!/bin/env python3
"""evaluator.io -- read a graph, read an embedding, write a result record.

`load_graph` dispatches on the TYPE of `src`: a `scipy.sparse` matrix, an
`(m, 2)` integer array, a path (the suffix picks the reader), or a bare
name (a registry lookup). Every path ends in `_finish`, thus `A` always
returns a symmetric CSR of 1.0, sorted indices, zero diagonal, ids
contiguous on `0..n-1`. Renumbering uses a mask and a scatter, never
`A[keep][:, keep]`, which overflows scipy's int32 result count on a large
graph (`fodiwalk/make_graph/datasets.py`).

This module does NOT own the dataset registry. A bare name defers to
`fodiwalk.make_graph.datasets.load`, imported INSIDE the function, thus
`import evaluator` works when `fodiwalk` is absent.

Import discipline: `numpy` and `scipy` only. `fodiwalk` is optional and
local to one function.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


# -- graph normalization -----------------------------------------------


def _finish(A):
    """A square matrix into a symmetric CSR of 1.0: sorted, zero diagonal."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    A.sort_indices()
    return A


def _edges_to_csr(edges):
    """An `(m, 2)` edge array into a CSR on `0..n-1`. ids made contiguous.

    A mask-and-scatter renumber, in O(m): `np.unique` gives each distinct
    id one slot. `_finish` symmetrizes and drops the diagonal, thus a
    self-loop or a duplicate edge here is harmless.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), 0
    ids, flat = np.unique(edges.ravel(), return_inverse=True)
    e = flat.reshape(edges.shape).astype(np.int64)
    n = ids.size
    data = np.ones(e.shape[0], dtype=np.float64)
    A = sp.csr_matrix((data, (e[:, 0], e[:, 1])), shape=(n, n))
    return A, n


def _read_edge_list(path):
    """A whitespace- or comma-separated edge list. `#` lines are comments."""
    with open(path) as f:
        first = next((ln for ln in f if not ln.lstrip().startswith("#")), "")
    delim = "," if "," in first else None
    edges = np.loadtxt(path, dtype=np.int64, comments="#", delimiter=delim)
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    return _edges_to_csr(edges[:, :2])


def _read_npz_graph(path):
    """A scipy sparse `.npz`, or a plain `.npz` with an `edges` or `A` key."""
    try:
        return sp.load_npz(path), None
    except Exception:
        pass
    with np.load(path, allow_pickle=True) as z:
        if "edges" in z.files:
            return _edges_to_csr(z["edges"])
        if "A" in z.files:
            raw = z["A"]
            A = raw.item() if raw.dtype == object else raw
            return sp.csr_matrix(A), None
    raise ValueError(
        f"load_graph: {path} is not a scipy sparse .npz, and it has no "
        "'edges' key and no 'A' key.")


def _read_mtx_graph(path):
    return sio.mmread(str(path)), None


def _load_registry(name, max_nodes, seed):
    """Delegate to the dataset registry. `fodiwalk` is optional."""
    try:
        from fodiwalk.make_graph import datasets
    except ImportError as exc:
        raise ImportError(
            f"load_graph({name!r}): fodiwalk is not installed, and "
            f"{name!r} is not an existing path. Install fodiwalk, or pass "
            "a path to a graph file instead of a registry name."
        ) from exc
    A, n = datasets.load(name, max_nodes, seed)
    return A, n


def load_graph(src, max_nodes=0, seed=42):
    """Read a graph. Returns `(A, n, info)`.

    `src` is a `scipy.sparse` matrix, an `(m, 2)` int array, a path whose
    suffix picks the reader (`.npz`, `.mtx`, else an edge list), or a bare
    name that is not an existing path (a registry lookup, via
    `fodiwalk.make_graph.datasets.load`, which also applies `max_nodes`
    and `seed`).

    `A` is a symmetric CSR of 1.0, sorted indices, a zero diagonal, and
    ids contiguous on `0..n-1`. `info` records the source, `n`, the
    undirected edge count, and the average degree.
    """
    if sp.issparse(src):
        A, source = src, "<sparse>"
    elif isinstance(src, np.ndarray):
        A, _ = _edges_to_csr(src)
        source = "<array>"
    elif isinstance(src, Path) or (isinstance(src, str) and Path(src).exists()):
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"load_graph: {path} does not exist.")
        source = str(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            A, _ = _read_npz_graph(path)
        elif suffix == ".mtx":
            A, _ = _read_mtx_graph(path)
        else:
            A, _ = _read_edge_list(path)
    elif isinstance(src, str):
        A, _ = _load_registry(src, max_nodes, seed)
        source = src
    else:
        raise TypeError(f"load_graph: unsupported source type {type(src)!r}")

    A = _finish(A)
    n = A.shape[0]
    n_edges = int(A.nnz // 2)
    info = {
        "source": source,
        "n": n,
        "n_edges": n_edges,
        "avg_degree": (A.nnz / n) if n else 0.0,
    }
    return A, n, info


# -- embedding reading ---------------------------------------------------


def _read_npz_embedding(path):
    with np.load(path) as z:
        if "Z" in z.files:
            return z["Z"]
        if len(z.files) == 1:
            return z[z.files[0]]
    raise ValueError(
        f"load_embedding: {path} has no 'Z' key and more than one array.")


def _read_csv_embedding(path, suffix, n):
    """A delimited matrix, with an optional leading column of node ids.

    The first column is read as ids when every value is a whole number
    and the column holds no duplicate. Then row `i` of `Z` is the vector
    of the line whose id column reads `i`; a missing id stays a zero row.
    """
    delim = "\t" if suffix == ".tsv" else ","
    raw = np.loadtxt(path, delimiter=delim, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    first = raw[:, 0]
    has_ids = (np.all(first == np.round(first))
               and np.unique(first).size == first.size)
    if not has_ids:
        return raw
    ids = first.astype(np.int64)
    vecs = raw[:, 1:]
    rows = n if n is not None else int(ids.max()) + 1
    Z = np.zeros((rows, vecs.shape[1]), dtype=vecs.dtype)
    Z[ids] = vecs
    return Z


def _read_w2v_embedding(path, n):
    """word2vec text: a header `count dim`, then `id v1 .. vd` lines.

    A missing id keeps a zero row -- the vocabulary gap of a word2vec run
    on a graph, one row for each node the training never visited.
    """
    with open(path) as f:
        count, dim = (int(x) for x in f.readline().split())
        rows = n if n is not None else count
        Z = np.zeros((rows, dim), dtype=np.float32)
        for line in f:
            parts = line.split()
            if not parts:
                continue
            i = int(parts[0])
            if i < rows:
                Z[i] = np.asarray(parts[1:dim + 1], dtype=np.float32)
    return Z


def load_embedding(src, n=None):
    """Read an embedding. Returns `(Z, info)`.

    `src` is a numpy array, or a path: `.npy`; `.npz` (key `Z`, else the
    only key); `.csv`/`.tsv` (an optional leading id column); otherwise a
    word2vec text file (`count dim` header).

    `info` records the source, the shape, the dtype, and `zero_rows` --
    the count of all-zero rows. A benchmark script leaves such a row for
    a node the word2vec vocabulary misses; a guard later reads this
    count.
    """
    if isinstance(src, np.ndarray):
        Z, source = src, "<array>"
    else:
        path = Path(src)
        suffix = path.suffix.lower()
        source = str(path)
        if suffix == ".npy":
            Z = np.load(path)
        elif suffix == ".npz":
            Z = _read_npz_embedding(path)
        elif suffix in (".csv", ".tsv"):
            Z = _read_csv_embedding(path, suffix, n)
        else:
            Z = _read_w2v_embedding(path, n)

    Z = np.asarray(Z)
    zero_rows = int(np.count_nonzero(~Z.any(axis=1)))
    info = {
        "source": source,
        "shape": tuple(Z.shape),
        "dtype": str(Z.dtype),
        "zero_rows": zero_rows,
    }
    return Z, info


# -- report writing --------------------------------------------------------


def write_report(report, path):
    """Write one result record. `.jsonl` appends a line; `.json` writes one object.

    `report` is a dict, or an object with a `to_dict()` method.
    """
    if isinstance(report, dict):
        record = report
    elif hasattr(report, "to_dict"):
        record = report.to_dict()
    else:
        raise TypeError(
            f"write_report: {type(report)!r} is not a dict and has no "
            "to_dict() method.")

    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".jsonl":
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    elif suffix == ".json":
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
    else:
        raise ValueError(
            f"write_report: unsupported suffix {suffix!r} (use .json or "
            ".jsonl).")
