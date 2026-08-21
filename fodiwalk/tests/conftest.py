"""conftest.py -- the fixtures the tests share, and the JAX backend.

The backend must be chosen BEFORE `import jax`, thus it is set here and not
in a test. `pytest` imports this file first.
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[2]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "parity: reproduces a recorded run. Slow, needs a GPU.")
    config.addinivalue_line(
        "markers", "big: needs com_youtube, 1.13M nodes and several GB.")
    config.addinivalue_line(
        "markers", "slow: reads a dataset larger than Cora.")


@pytest.fixture(scope="session")
def cora():
    """`(A, n)` of Cora: 2708 nodes, symmetric, no self loop."""
    from fodiwalk.make_graph import load
    return load("cora")


@pytest.fixture
def tiny():
    """A small connected graph with one hub, as a symmetric CSR.

    Node 0 is the hub. Every test that needs a graph and not a dataset uses
    this one, thus a test costs milliseconds and no file on disk.
    """
    rng = np.random.default_rng(0)
    n = 60
    rows, cols = [], []
    for v in range(1, n):                      # a star, thus it is connected
        rows += [0, v]
        cols += [v, 0]
    for _ in range(120):                       # and some random chords
        u, v = rng.integers(1, n, 2)
        if u != v:
            rows += [int(u), int(v)]
            cols += [int(v), int(u)]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data[:] = 1.0
    A.sort_indices()
    return A, n
