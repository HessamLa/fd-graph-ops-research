"""embed.plan_contract -- the asserter of the plane contract.

THIS IS THE ONLY NEW MODULE OF THE PACKAGE, and it is the reason the
refactor pays.

`make_plan` takes a SEQUENCE of planes and `step` gives that sequence to
the law, which unpacks it POSITIONALLY. Nothing in either file checks the
count, the order, or the meaning of a plane. Every defect that came from
this seam was SILENT: the run gave a number, and not an error.

    2026-08-17  a third plane given to a two-plane law.
    2026-08-18  `fdlinear` with a missing `freq` read a coefficient plane
                as `h`. The physics was the physics of another law under
                an `fdlinear` label, the run went to NaN at 2000 epochs,
                and only a crash inside sklearn showed it. That plane is
                removed from the package (2026-08-19), but the defect
                CLASS -- one plane read as another -- is what the value
                tests below still guard against.
    2026-08-18  a policy that emptied the `h = 1` rows froze them, because
                `degrees_from_D` gave 0 and every law turns that to
                0.0.
    2026-08-16  a continuous weight gave every pair its own shell, thus
                `degrees_from_D` gave 0 for every row and the WHOLE force
                vanished. AUC 0.55, `||dZ|| = 0.000`, no error.

The invariants, quoted from PRD section 7:

    I1  planes are `(D.nnz,)` and aligned to `D.indices`
    I2  the plane COUNT and ORDER match the law's registry entry
    I3  a pad cell has every plane at 0
    I4  stored weights are integers, and 1 means adjacency
    I5  a row with no `h = 1` entry gets an explicit degree

`check` asserts I1, I2 and I4. `check_plan` asserts I3 on a built plan.
`check_degrees` asserts I5.

**It RAISES. It never warns, and it never falls back to other physics.** A
missing plane is a defect of the augmentation, and different physics under
the same label is the failure this module exists to stop.

How I2 is caught when the planes are unnamed arrays: a plane name is a
PROMISE about the values, and this module tests the promise.

    `h`      is `D.data` itself -- the stored weight of the pair. Tested
             by elementwise equality.
    `freq`   is a count of visits, thus it is `>= 1`.
    `w`      is the fused plane: exactly `-1` at `h = 1`, and `> 0` above.
             A value in `(-1, 0)` or below `-1` cannot be decoded.

Thus `check("fdlinear", (freq, h), D)` RAISES: `fdlinear` reads
`(h, freq)`, and a `freq` array in the `h` position is not `D.data`. That
is the 2026-08-18 defect, caught at the seam.

Import discipline: numpy and `embed.forces` only.

Provenance: this file was `fodiwalk/core/plan_contract.py` until
2026-08-28. It moved here with `forces.py`, whose registry it reads; a
`core` that read `embed` would have turned the dependency upside down.
No line of a body changed.
"""
from __future__ import annotations

import numpy as np

from .forces import FORCE_PLANES, planes_of


class PlaneContractError(ValueError):
    """A plane does not keep the promise of its name. Always fatal."""


def _fail(msg: str):
    raise PlaneContractError(msg)


# ---------------------------------------------------------------------------
# What each plane name promises
# ---------------------------------------------------------------------------
def _check_h(p, D, law):
    d = np.asarray(D.data, dtype=np.float64)
    if not np.array_equal(np.asarray(p, dtype=np.float64), d):
        _fail(f"{law}: the `h` plane is not `D.data`. A plane in the `h` "
              f"position must be the stored weight of the pair, aligned "
              f"1:1 with D.indices. Got a plane in "
              f"[{float(np.min(p)):.6g}, {float(np.max(p)):.6g}] against "
              f"D.data in [{d.min():.6g}, {d.max():.6g}]. This is the "
              f"2026-08-18 defect: a law reading one plane as another.")
    if d.size and (d.min() < 1.0 or not np.array_equal(d, np.rint(d))):
        _fail(f"{law}: a stored weight must be an INTEGER >= 1 (I4). "
              f"Got [{d.min():.6g}, {d.max():.6g}]. A continuous weight "
              f"gives every pair its own shell, `degrees_from_D` then "
              f"gives 0 for every row, and the whole force vanishes with "
              f"no error.")


def _check_freq(p, D, law):
    if p.size and p.min() < 1.0:
        _fail(f"{law}: the `freq` plane counts visits, thus it is >= 1. "
              f"Got a minimum of {float(p.min()):.6g}.")


def _check_w(p, D, law):
    if p.size:
        bad = (p < 0) & (p != -1.0)
        if bad.any():
            _fail(f"{law}: the fused `w` plane holds the sentinel -1 at "
                  f"h = 1 and `h / freq > 0` above. {int(bad.sum())} "
                  f"values are negative and not -1, thus the two branches "
                  f"cannot be decoded.")
        if (p == 0).any():
            _fail(f"{law}: `w = 0` marks a PAD cell, thus no stored pair "
                  f"may carry it. {int((p == 0).sum())} stored pairs do.")


def _check_deg_le(p, D, law):
    v = np.unique(np.asarray(p, dtype=np.float64))
    if v.size and not np.all(np.isin(v, (-1.0, 1.0, 0.0))):
        _fail(f"{law}: the `deg_le` plane must hold only +1.0, -1.0 and "
              f"0.0. Got {v.size} distinct values in "
              f"[{v.min():.6g}, {v.max():.6g}]. It is a SIGNED MASK: +1 "
              f"means `deg(u) <= deg(v)`, -1 means it does not, and 0.0 is "
              f"a pad cell and nothing else. A 1/0 encoding breaks I3, "
              f"because a real pair would then carry 0 in this plane and "
              f"a non-zero `h`, which `check_plan` reads as a corrupt pad.")


PLANE_CHECKS = {"h": _check_h, "freq": _check_freq, "w": _check_w,
                "deg_le": _check_deg_le}


# ---------------------------------------------------------------------------
# The three public assertions
# ---------------------------------------------------------------------------
def check(law: str, planes, D) -> None:
    """Assert I1, I2 and I4 for `planes` against the registry entry of `law`.

    Raises `PlaneContractError` -- a subclass of `ValueError` -- on the
    first violation. Returns `None` when every plane keeps its promise.

    Call it in `augment_graph`, between the build of the planes and
    `make_plan`. It costs one pass over each `(nnz,)` array, one time for
    each `D`, thus it never touches the epoch loop.
    """
    names = planes_of(law)                       # raises on an unknown law
    planes = tuple(planes)
    if len(planes) != len(names):
        _fail(f"{law} reads {len(names)} planes {names}, and it got "
              f"{len(planes)} (I2). The kernel unpacks POSITIONALLY, thus "
              f"a wrong count is a wrong law, not a warning.")
    # `nbr_walk` builds `D` as a plain `augment_graph.rows.RowCSR`
    # (2026-08-28), not a `scipy.sparse.csr_matrix` -- `sp.issparse` is
    # False for it, and the ORIGINAL check here fell to `np.count_nonzero`,
    # the DENSE branch, which does not know what to do with it. `.nnz` is
    # what both a real sparse matrix and `RowCSR` carry; a genuine dense
    # `ndarray` has neither, so it still takes the `count_nonzero` branch.
    nnz = int(D.nnz) if hasattr(D, "nnz") else int(np.count_nonzero(D))
    for i, (name, p) in enumerate(zip(names, planes)):
        p = np.asarray(p)
        if p.shape != (nnz,):
            _fail(f"{law}: plane {i} (`{name}`) has shape {p.shape}, and "
                  f"the contract is ({nnz},) -- one value for each stored "
                  f"pair, aligned to D.indices in D's own pre-split CSR "
                  f"order (I1).")
        if not np.isfinite(p).all():
            _fail(f"{law}: plane {i} (`{name}`) holds "
                  f"{int((~np.isfinite(p)).sum())} non-finite values.")
        # `asarray` and not `astype`: a plane that is already float64 --
        # every `h` plane is, it IS `D.data` -- passes through with no
        # copy. At 23M stored pairs a needless copy is 185 MB.
        PLANE_CHECKS[name](np.asarray(p, dtype=np.float64), D, law)


def check_degrees(degrees, D) -> None:
    """Assert I5: no row may reach the force law with a degree of 0.

    Every law divides a degree of 0 by 1, thus the row keeps its WHOLE force
    of the row -- the repulsion too. The row then never moves, in silence.
    A row with no `h = 1` entry must therefore get an explicit degree, from
    the true degree of `A` or from `degrees = 1`.

    A row with NO stored entry at all is not affected: `make_plan` gives an
    isolated row no virtual row, thus the law never reads it.
    """
    degrees = np.asarray(degrees, dtype=np.float64)
    width = np.diff(D.indptr)
    frozen = (degrees == 0) & (width > 0)
    if frozen.any():
        i = int(np.flatnonzero(frozen)[0])
        _fail(f"{int(frozen.sum())} rows hold stored pairs and a degree of "
              f"0 (row {i} holds {int(width[i])} pairs). Every force law "
              f"turns that into 0.0 and the rows freeze for the whole run, "
              f"with no error (I5). Pass an explicit `degrees` array -- "
              f"the true degree of A, or 1.")


def check_plan(plan, n_planes: int) -> None:
    """Assert I3: every plane of a PAD cell is exactly 0.

    A pad cell also carries the row's own id as its neighbour, thus `x = 0`
    and the kernel zeroes the cell a second time. The double safety is
    deliberate; this assertion tests the first layer, which is the one a
    law with a constant term depends on.

    A pad cell is one that no real stored entry filled. `make_plan` builds
    the tiles with `np.where(valid, p[src], 0.0)`, thus this reads the
    tiles and finds the cells where EVERY plane is 0 -- which is what the
    contract promises.
    """
    for r, rung in enumerate(plan):
        rows, nbrs, *tiles = rung
        if len(tiles) != n_planes:
            _fail(f"rung {r} carries {len(tiles)} plane tiles and the law "
                  f"reads {n_planes} (I2).")
        # A pad ROW is the sentinel owner id; a pad CELL of a real row has
        # its own row id as the neighbour. Both must give 0 in every tile.
        pad = np.ones(tiles[0].shape, dtype=bool)
        for t in tiles:
            pad &= (np.asarray(t) == 0.0)
        # Every cell that is padding must be 0 in EVERY tile: a cell that
        # is 0 in one tile and not in another is a partly-filled pad.
        for j, t in enumerate(tiles):
            t = np.asarray(t)
            zero_here = (t == 0.0)
            mixed = zero_here & ~pad
            if mixed.any() and n_planes > 1:
                # No plane name of the registry lets a real stored pair
                # carry 0 (`h` cannot, `freq` cannot, `w` cannot), thus
                # this is a defect for every plane the registry has.
                _fail(f"rung {r}, plane tile {j}: {int(mixed.sum())} cells "
                      f"are 0 in this plane and not in every plane. A pad "
                      f"cell carries 0 in ALL of them (I3).")
