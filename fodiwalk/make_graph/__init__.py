"""fodiwalk.make_graph -- stage 1: the graph.

In this project the stage is a READER and not a builder: the graphs arrive
as edge lists on disk. `Fodiwalk.make_graph` is a stub that returns what it
is given, and `datasets.load` is what a caller actually uses.

    from fodiwalk.make_graph import load, read_edges, to_csr, induced
"""
from __future__ import annotations

from .datasets import (read_edges, to_csr, induced, load, remap,
                       subtree, DATA, SNAP, TREE)

__all__ = ["read_edges", "to_csr", "induced", "load", "remap",
           "subtree", "DATA", "SNAP", "TREE"]
