"""embed.planner -- `D` and its planes into the batch plan, and the JAX plumbing.

One plan, or `chunks` plans of a ROW RANGE each. The whole plan of a graph
of a million nodes does not fit beside `Z` and `dZ` on a 2 GB card. A chunk
holds the rows `[a, b)` only, thus the device holds ONE chunk at a time.
The rows outside `[a, b)` become empty, and `make_plan` gives an isolated
row no virtual row at all, thus an empty row costs nothing in the plan of
another chunk.

WHY A ROW RANGE, and never a set of pairs (trap 8 of REFACTOR.md, I6):
`forcedirected.step` writes `dZ.at[rows].add(...)`, thus only DISJOINT
rows make the parts additive. `ForceDirected.embed` slices `dZ` by
the same range for its batches, thus the chunk and the batch are the same
object.

THE GLOBAL QUANTITIES STAY GLOBAL. `degrees` is counted over the WHOLE `D`
and a chunk only slices the planes. A degree counted on one chunk is not
the degree of the node.

Provenance: `Fodiwalk._build_plan` of `fodiwalk/fodiwalk.py` (2026-08-20),
with the `jax.jit`, `jax.device_put` and `functools.partial` plumbing moved
out of the model class (defect D7).

Import discipline: numpy, scipy, jax, the sibling modules of `embed`
and `forcedirected` only.
"""
from __future__ import annotations

import dataclasses
import functools

import numpy as np
import scipy.sparse as sp
import jax

from forcedirected import make_plan, step

from . import plan_contract


@dataclasses.dataclass(frozen=True)
class PlanSpec:
    """The narrow stage-3 config of the LAYOUT. No physics, no walk knob."""

    b_cells: int = 16_384
    k_max: int = 256
    ladder_base: float = 1.5
    chunks: int = 1
    chunk_host: bool = False
    check_planes: bool = True      # I1, I2, I4, I5 at the seam
    check_padding: bool = True     # I3, one pass over the built tiles

    @classmethod
    def from_config(cls, cfg) -> "PlanSpec":
        names = [f.name for f in dataclasses.fields(cls)]
        return cls(**{k: getattr(cfg, k) for k in names if hasattr(cfg, k)})


@dataclasses.dataclass
class PlanSet:
    """What stage 3 needs for every epoch. The seam of the planner to the model.

    `plans`        one plan for each chunk, on the device when `resident`.
    `steps`        the jitted kernel of each chunk.
    `inv_deg_ext`  `(n + 1,)` on the device. Global, thus every chunk has
                   the same array.
    `chunk_rows`   the row count of a chunk. `forces` picks the plan with it.
    `resident`     the plans stay on the device.
    `stats`        the plan statistics. The model exposes it as `plan_stats`.
    """

    plans: list
    steps: list
    inv_deg_ext: object
    chunk_rows: int
    resident: bool
    stats: dict


def build_plans(D, planes, degrees, spec: PlanSpec, force_fn) -> PlanSet:
    """The plan of `D`, in `spec.chunks` row ranges, and the jitted steps.

    `planes` are the `(nnz,)` arrays of `planes.build_planes`, aligned to
    `D.indices`. A chunk slices them with the CSR span of its rows.
    """
    n = D.shape[0]
    resident = not spec.chunk_host
    chunk_rows = max(1, (n + max(1, spec.chunks) - 1) // max(1, spec.chunks))
    plans, steps, cells = [], [], 0
    stats = None
    inv_deg_ext = None
    # ONE scratch buffer for the padded per-chunk `indptr`, reused for
    # every chunk instead of a fresh `np.zeros(n + 1)` each time: at
    # 1.13M nodes over 8 chunks that was eight (n + 1) allocations, ~72 MB
    # of churn, to hold data `make_plan` reads and returns from before the
    # next chunk starts. `Dc` is a loop-local variable that nothing keeps
    # past this iteration -- `make_plan`'s OWN `D = to_csr(Dc)` is a no-op
    # (`Dc` is already a CSR) and everything it reads from `indptr` is
    # consumed before `make_plan` returns -- so overwriting the buffer on
    # the next iteration is safe.
    scratch = np.zeros(n + 1, dtype=D.indptr.dtype)
    for a in range(0, n, chunk_rows):
        b = min(a + chunk_rows, n)
        lo, hi = int(D.indptr[a]), int(D.indptr[b])
        np.subtract(D.indptr, lo, out=scratch)
        np.clip(scratch, 0, hi - lo, out=scratch)
        Dc = sp.csr_matrix((D.data[lo:hi], D.indices[lo:hi], scratch),
                           shape=D.shape)
        plan, inv_deg_ext, stats = make_plan(
            Dc, tuple(p[lo:hi] for p in planes), degrees=degrees,
            b_cells=spec.b_cells, k_max=spec.k_max,
            ladder_base=spec.ladder_base)
        cells += stats["cells"]
        # `resident` keeps the plan on the device, which is the fast form
        # and the one that needs the memory. Otherwise the plan stays in
        # the host memory and it moves for each use.
        plans.append(jax.tree_util.tree_map(jax.device_put, plan)
                     if resident else plan)
        steps.append(jax.jit(functools.partial(
            step, n=n, force_fn=force_fn)))
        if spec.check_padding:
            # I3, on the HOST tiles and before the device copy: a pad cell
            # carries 0 in every plane. One pass for each chunk, one time
            # for each D.
            plan_contract.check_plan(plan, len(planes))
    return PlanSet(
        plans=plans, steps=steps,
        inv_deg_ext=jax.device_put(inv_deg_ext),
        chunk_rows=chunk_rows, resident=resident,
        stats=dict(stats, cells=cells, chunks=len(plans),
                   rows_per_chunk=chunk_rows))
