"""augment_graph.planes -- the planes of a law, from a REGISTRY and not an
`if` chain.

A plane is a `(nnz,)` array with one value for each stored pair of `D`, in
`D`'s own pre-split CSR order. `make_plan` pads and tiles it, and the law
unpacks the tiles POSITIONALLY. Nothing in the kernel knows the count, the
order or the meaning of a plane, thus a wrong plane is a wrong physics and
not an error (`core/plan_contract.py` records four such defects).

THIS MODULE IS STAGE 2. `dev-docs/fodiwalk-module.md` says stage 3 does no
graph analysis and no data preparation, and a plane IS a data-preparation
choice: which values a law reads is a property of the RECIPE (the pair
policy plus the law), not of the kernel that later consumes them. A
different augmentation builds a different plane set for the same law, thus
the plane builder moves with the augmentation and not with the kernel.

THE ONE RULE THIS MODULE KEEPS. The plane NAMES come from
`core.forces.planes_of(law)` and never from a branch on the law. A plane
that the augmentation did not build RAISES `PlaneContractError`. It never
falls back to the planes of another law: that fallback made `fdlinear` read
a coefficient plane as `h` on 2026-08-18, and the run went to NaN under an
`fdlinear` label with no error.

THE PAIRING, which is now visible:

    PLANE_BUILDERS   here                    makes  the values of a plane name
    PLANE_CHECKS     core/plan_contract.py   asserts what the name promises

To add a plane, a person edits those two tables and `FORCE_PLANES` in
`core/forces.py`, and nothing else. The three keys must stay equal.

Provenance: `Fodiwalk._build_planes` and `Fodiwalk._plane` of
`fodiwalk/fodiwalk.py` (2026-08-20); `fodiwalk/embed/planes.py`
(2026-08-20, the first split); moved into `augment_graph` (2026-08-21) when
the stage boundary moved -- see `dev-docs/CATALOG.md`, the RECIPE entry.
The error messages are verbatim across both moves.

Import discipline: numpy and `core` only. `augment_graph` reads `core` for
the plane contract (`planes_of`, `fuse`, `PLANE_CHECKS`); it still imports
NO `embed` and NO model.
"""
from __future__ import annotations

import dataclasses

from ..core.forces import fuse, planes_of
from ..core.plan_contract import PLANE_CHECKS, PlaneContractError


# ---------------------------------------------------------------------------
# ForceSpec -- the narrow spec of the physics, read by the augmentation that
# prepares its planes and by the kernel that consumes them.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ForceSpec:
    """The knobs of the law, the degree divisor and the drop.

    `Config` stays FLAT and public; this spec is the narrowing at the seam,
    thus a stage cannot reach a knob outside its own recipe. `from_config`
    reads the attributes by NAME, thus this package never imports the model
    layer.

    `edge_rule` is a pair-policy knob and it is here for ONE reason: the
    `auto` degree source reads it (see `degrees.resolve_degrees`). A
    `low_deg` edge rule can empty the `h = 1` entries of a hub, thus the
    degree must then come from `A`.
    """

    force: str = "fdlinear"
    fuse_planes: bool = False
    k1: float = 0.999
    k4: float = 0.01
    kr: float = 1.0
    fdlinear_sign: float = -1.0
    no_deg_norm: bool = False
    deg_source: str = "auto"        # auto | D | A
    edge_rule: str = "both"         # both | low_deg -- the `auto` rule reads it
    random_drop_rate: float = 0.5
    drop_strategy: str = "random_rows"

    @classmethod
    def from_config(cls, cfg) -> "ForceSpec":
        names = [f.name for f in dataclasses.fields(cls)]
        return cls(**{k: getattr(cfg, k) for k in names if hasattr(cfg, k)})

    @property
    def law(self) -> str:
        """The law name, resolved ONE time. `fuse_planes` picks the fused
        form of `fdlinear`, thus the plane list follows from one name."""
        return ("fdlinear_fused"
                if self.fuse_planes and self.force == "fdlinear"
                else self.force)


def force_params(spec: ForceSpec) -> dict:
    """The traced scalars the law reads. `params` of `forcedirected.step`.

    The kernel only APPLIES these; it never chooses them. Choosing them is
    part of the recipe, thus this stays beside `build_planes` and not in
    the kernel-consuming stage.
    """
    return dict(k1=spec.k1, k4=spec.k4, kr=spec.kr, sign=spec.fdlinear_sign)


# ---------------------------------------------------------------------------
# THE REGISTRY. One entry for each key of `plan_contract.PLANE_CHECKS`.
# ---------------------------------------------------------------------------
# Each builder takes the same arguments, thus `build_planes` needs no
# branch. `pairs` and `policy` are there for the message of a missing
# plane: it must name the augmentation that did not build one.
def _build_h(D, freq, law, pairs, policy):
    return D.data


def _build_freq(D, freq, law, pairs, policy):
    if freq is None:
        raise PlaneContractError(
            f"{law} reads the plane `freq` and the "
            f"augmentation for pairs={pairs!r} "
            f"policy={policy!r} did not build one. This "
            f"is a defect of the augmentation, not a configuration "
            f"error, and it is never a reason to run the planes of "
            f"another law.")
    return freq


def _build_w(D, freq, law, pairs, policy):
    if freq is None:
        raise PlaneContractError(
            f"{law} reads the fused plane `w = h/freq`, and "
            f"the augmentation built no `freq`.")
    return fuse(D.data, freq)


PLANE_BUILDERS = {"h": _build_h, "freq": _build_freq, "w": _build_w}

# The two tables are one contract: a name that one holds and the other does
# not is a plane that is built and not asserted, or asserted and not built.
assert set(PLANE_BUILDERS) == set(PLANE_CHECKS)


def build_planes(law: str, D, freq, pairs=None, policy=None) -> tuple:
    """The planes of `law`, in the order the registry gives.

    `law`     a key of `core.forces.FORCE_PLANES`.
    `D`       the augmented matrix. `D.data` is the `h` plane.
    `freq`    the `(nnz,)` frequency data aligned to `D.indices`, or None.
    `pairs`,
    `policy`  the augmentation that built `D`. They only name the defect in
              the message of a missing plane.

    No branch on the law name. A plane that the augmentation did not build
    stops the run here, with the name of the plane and the name of the law
    -- never with different physics.
    """
    out = []
    for name in planes_of(law):                  # raises on an unknown law
        build = PLANE_BUILDERS.get(name)
        if build is None:
            raise PlaneContractError(
                f"{law} declares a plane `{name}` that this class cannot "
                f"build. Add it here and to core.forces.FORCE_PLANES together.")
        out.append(build(D, freq, law, pairs, policy))
    return tuple(out)
