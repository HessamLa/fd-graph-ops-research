"""
Strategy registry and public dispatcher for graph construction.

New strategies register themselves via the @register_strategy decorator
(see strategies/ for examples) and become reachable through make_graph
without any changes to this file.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

_STRATEGIES: dict = {}


def register_strategy(name):
    """Decorator: register a graph-construction strategy under `name`."""

    def decorator(fn):
        if name in _STRATEGIES:
            raise ValueError(f"Strategy {name!r} is already registered.")
        _STRATEGIES[name] = fn
        return fn

    return decorator


def make_graph(data, strategy, node_labels=None, weighted=False, **kwargs):
    """Build a graph over the rows of `data` using a named strategy.

    Parameters
    ----------
    data : array-like, shape (n, d)
        Coordinates of n points in d-dimensional space.
    strategy : str
        One of the registered strategy names (see `available_strategies()`).
    node_labels : sequence, optional
        A label for each row of `data` (length n), e.g. class labels for
        coloring or classification. Labels need not be unique. Stored on
        each node as the `label` attribute; node names remain the
        positional indices 0..n-1, aligned with the rows of `data`.
        Access via `G.nodes[i]['label']` or
        `nx.get_node_attributes(G, 'label')`.
    weighted : bool, default False
        Reserved for future weighted variants; must be False for now.
    **kwargs
        Forwarded to the strategy function (e.g. k_neighbors=10).

    Returns
    -------
    networkx.Graph
        Simple undirected graph on nodes 0..n-1, connected.
    """
    if weighted:
        raise NotImplementedError(
            "Weighted graphs are not implemented yet; use weighted=False."
        )
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Available: {available_strategies()}"
        )

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n, d), got shape {data.shape}.")
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points to build a graph.")

    if node_labels is not None:
        node_labels = list(node_labels)
        if len(node_labels) != n:
            raise ValueError(
                f"node_labels has length {len(node_labels)}, expected {n} "
                f"(one label per row of data)."
            )

    G = _STRATEGIES[strategy](data, **kwargs)

    if node_labels is not None:
        nx.set_node_attributes(
            G, dict(enumerate(node_labels)), name="target"
        )

    return G


def available_strategies(description=False):
    """List all registered strategy names, optionally with their docstrings.

    Parameters
    ----------
    description : bool, default False
        If False, return a sorted list of strategy names (unchanged from
        before -- callers like `make_graph`'s error message depend on this
        shape).
        If True, return a dict mapping each strategy name to the first
        line of its docstring (or "(no docstring)" if it has none).

    Returns
    -------
    list[str] or dict[str, str]
    """
    names = sorted(_STRATEGIES)
    if not description:
        return names

    return {
        name: (_STRATEGIES[name].__doc__ or "(no docstring)").strip().splitlines()[0]
        for name in names
    }