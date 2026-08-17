"""model_fdmap_auxdim.model -- ``EuclideanAuxDimModel``: auxiliary embedding
dimensions that are annealed away over training.

The idea (full design record, including the rejected/deferred alternatives
and the validation plan, lives in ``model_fdmap_auxdim/idea.md`` -- read it
before changing anything here): run the ordinary force law in a
``d + aux``-dimensional space so points can route *around* each other early
in training instead of getting stuck behind a topological obstruction, then
shrink the ``aux`` columns toward zero on a schedule so the FINAL embedding
is genuinely ``d``-dimensional. Only the first ``d`` columns are ever
returned as the embedding; the ``aux`` ones are scaffolding.

This module implements **Step 1** of ``idea.md`` only -- "directly shrink
Z's aux columns after each update". Step 2 (fading the aux dims' weight
*inside the distance metric*, which requires touching the shared
``embedding/sell_c_sigma.py`` kernel that every model in the package uses)
is deliberately NOT implemented here: it is gated on Step 1 being
empirically validated first, so that there is a real baseline to compare
it against.

Step 1 needs zero engine changes. ``EuclideanDistanceModel``'s force law
and the SELL-C-sigma kernel are already dimension-agnostic (``diff``,
``x = norm(diff)`` and the padding contract all work over whatever ``Z``'s
last axis happens to be), so the whole mechanism fits in the
``updateZ`` override seam that ``core/force_directed.py`` already documents
for this class of experiment.

Import discipline note: ``model_fdmap_auxdim`` is a standalone research
add-on *next to* ``fdge_jax_sell_c_sigma``, not a stage package inside it,
so the "stage packages never import each other" rule in that package's
``docs/DESIGN.md`` does not apply -- importing the composition root
(``fdge_jax_sell_c_sigma.models``) directly is the intended thing to do
here.

Run the tests:
    /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/python -m model_fdmap_auxdim.test_model
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import jax.numpy as jnp

from fdge_jax_sell_c_sigma.models import EuclideanDistanceModel


# ---------------------------------------------------------------------------
# Decay schedules
#
# Every schedule is a plain ``(epoch, max_epochs) -> coeff`` function, with
# the contract: ``coeff(0, E) ~= 1`` (aux dims at full strength) and
# ``coeff(E, E) ~= 0`` (aux dims collapsed). They are approximations of that
# contract, not exact identities -- ``sigmoid`` and ``exponential`` in
# particular only get *near* the endpoints (~0.993 / ~0.0067 and 1.0 /
# ~4.5e-5 respectively), which is fine: the point is the shape of the ramp
# in between, and those two are the shapes that spend most of the budget at
# full strength / collapse early.
#
# They are defined for ``epoch`` in ``[0, max_epochs]``. Outside that range
# the formulas are simply extrapolated as written (``linear`` goes negative,
# ``cosine`` comes back up) -- see ``EuclideanAuxDimModel.updateZ`` for the
# one situation where that can be reached in practice.
# ---------------------------------------------------------------------------
def linear_decay(epoch: int, max_epochs: int) -> float:
    """``1 - epoch/max_epochs`` -- constant shrink pressure, hits exactly 0."""
    return 1.0 - epoch / max(max_epochs, 1e-9)


def cosine_decay(epoch: int, max_epochs: int) -> float:
    """``0.5 * (1 + cos(pi * epoch/max_epochs))`` -- slow start, slow finish.

    Flat at both ends, so the aux dims stay near full strength while the
    layout is still coarse and are eased to zero rather than yanked.
    """
    return 0.5 * (1.0 + math.cos(math.pi * epoch / max(max_epochs, 1e-9)))


def linear_sine_decay(epoch: int, max_epochs: int, frequency: float = math.pi/4) -> float:
    """Product of ``linear_decay`` and a sine wave.
    The sine wave is sin(frequency*epoch), or sin(wt)

    ``idea.md`` lists this schedule without naming it. It is strictly below
    both factors, i.e. the most aggressive of the three polynomial-ish
    shapes, while keeping cosine's soft landing at the end.
    """
    return linear_decay(epoch, max_epochs) * math.sin(frequency * epoch)


def sigmoid_decay(epoch: int, max_epochs: int) -> float:
    """``1 / (1 + exp((epoch - max_epochs/2) / (max_epochs/10)))``.

    Holds the aux dims at ~full strength for the first ~40% of the budget,
    then drops them over a narrow window around the midpoint. The schedule
    to pick if the hypothesis is "the extra room only matters early, and a
    long tail of half-strength aux dims just adds noise".
    """
    # max_epochs/10 is the transition width; a short smoke-test run
    # (max_epochs <= 10) would make it <= 1 and max_epochs == 0 would make it
    # exactly 0, so floor it rather than divide by zero.
    denom = max(max_epochs / 10.0, 1e-9)
    t = (epoch - max_epochs / 2.0) / denom
    # Evaluate the logistic in whichever direction keeps exp()'s argument
    # negative: with a floored denom, t can be enormous (a resumed embed()
    # far past the budget, or max_epochs == 0), and math.exp raises
    # OverflowError rather than saturating the way numpy does.
    if t >= 0.0:
        e = math.exp(-t)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(t))


def exponential_decay(epoch: int, max_epochs: int) -> float:
    """``exp(-epoch / (max_epochs/10))`` -- collapses fastest of the five.

    Down to ~37% of full strength by 10% of the budget. Useful as the
    "does the aux scaffolding matter at all?" control: if this performs the
    same as the no-aux baseline, the extra dims are not doing any work.
    """
    denom = max(max_epochs / 10.0, 1e-9)   # same floor + reason as sigmoid_decay
    return math.exp(-epoch / denom)


#: Named schedules accepted by ``EuclideanAuxDimModel(decay_schedule=...)``.
#: ``linear`` is the default: it is the only one of the five that reaches
#: exactly 0 at the budget with no free shape parameter, so it is the
#: easiest baseline to reason about when comparing runs.
DECAY_SCHEDULES: dict[str, Callable[[int, int], float]] = {
    "linear": linear_decay,
    "cosine": cosine_decay,
    "linear_sine": linear_sine_decay,
    "sigmoid": sigmoid_decay,
    "exponential": exponential_decay,
}


class EuclideanAuxDimModel(EuclideanDistanceModel):
    """``EuclideanDistanceModel`` embedded in ``d + aux`` dims, aux annealed away.

    Everything about the pipeline -- graph building, the weighted
    shortest-path augmentation, the shell-averaged force law, the bucketed
    SELL-C-sigma engine -- is inherited unchanged from
    ``EuclideanDistanceModel``. The only additions are:

    1. ``self.n_dim`` is ``d + aux`` rather than ``d``, which is the single
       knob controlling the width of the lazily-initialized ``Z``
       (``core/force_directed.py``'s ``embed()``). Nothing downstream needs
       to know the model is augmented.
    2. Once per epoch, immediately after the base ``Z = Z + lr*dZ``, the
       trailing ``aux`` columns are multiplied by ``decay_fn(epoch,
       max_epochs)``. As the coefficient goes to 0 every point converges to
       a shared ~zero aux coordinate, so the aux dims' contribution to every
       pairwise distance fades out on a common schedule.
    3. ``get_embeddings()`` (and therefore ``get_embeddings_df()``, which
       delegates to it) returns only the leading ``d`` columns.

    Parameters
    ----------
    d : int
        The real, final embedding dimensionality -- what
        ``get_embeddings()`` returns. Required; there is no sensible default.
    aux : int, default 0
        Number of auxiliary scaffolding dimensions. ``aux=0`` is a fully
        supported degenerate case: the shrink step is skipped entirely and
        the model is behaviourally identical to a plain
        ``EuclideanDistanceModel(n_dim=d)`` (asserted by the parity test in
        ``test_model.py``).
    decay_schedule : str, default ``"linear"``
        One of ``DECAY_SCHEDULES``.
    decay_fn : callable ``(epoch, max_epochs) -> float``, optional
        Custom schedule; takes precedence over ``decay_schedule`` when given.

    Do NOT pass ``n_dim=`` -- it is computed as ``d + aux`` and forwarded to
    ``ForceDirected.__init__``. Passing it as well raises Python's own
    "multiple values for keyword argument" ``TypeError``, which says exactly
    what went wrong; no custom handling is layered on top of that.

    Known risks (from ``idea.md`` section 3, Step 1 -- these are hypotheses
    to *measure*, not settled behaviour; this is an unvalidated research
    idea, not a known win):

    - **Tug-of-war.** The force law pushes the aux columns around every
      epoch under the undamped physics, then the shrink step yanks them back
      down. Depending on ``lr`` versus the decay rate this can work as
      intended, collapse so fast the aux dims are irrelevant within a few
      epochs (defeating the purpose), or oscillate. Plot ``||Z[:, d:]||``
      over epochs to see which regime a given configuration is in.
    - **k1..k4 scale interaction.** Pairwise distance ``x`` is computed over
      all ``d + aux`` dims and is therefore inflated early on by the random
      aux columns. The ``k1..k4`` defaults were tuned for ``x`` on the
      ``d``-dim scale, so they may need re-tuning per ``aux`` count --
      especially at the small ``d`` (2, 3) this project visualizes.
    - **Aux init scale.** The base class's lazy ``Z`` init draws all columns
      from the same ``jax.random.normal``, so the aux columns start at the
      same magnitude as the real ones, feeding straight into the previous
      risk. A separate, smaller aux init scale is the first thing to try if
      that misbehaves; it is not implemented here because it needs its own
      override of the lazy-init path and should not be added speculatively.
    - **Compute cost.** Every per-cell tensor in the bucketed engine scales
      with the full ``d + aux`` width. At ``d=128`` an ``aux=8`` is ~6%
      overhead; at ``d=2`` the same ``aux=8`` is a 5x cost increase. Report
      this explicitly in any validation write-up.
    """

    VER_MAJ = "00"
    VER_MIN = "01"
    DESCRIPTION = ("Euclidean-distance force-directed model with annealed "
                   "auxiliary dimensions (fdmap auxdim, Step 1)")

    def __init__(self, *args,
                 d: int,
                 aux: int = 0,
                 decay_schedule: str = "linear",
                 decay_fn: Callable[[int, int], float] | None = None,
                 **kwargs) -> None:
        if int(d) < 1:
            raise ValueError(f"d must be >= 1 (the real embedding width), got {d!r}")
        if int(aux) < 0:
            raise ValueError(f"aux must be >= 0, got {aux!r}")

        if decay_fn is None:
            if decay_schedule not in DECAY_SCHEDULES:
                raise ValueError(
                    f"Unknown decay_schedule {decay_schedule!r}. "
                    f"Valid names: {sorted(DECAY_SCHEDULES)}. "
                    f"Alternatively pass decay_fn=<callable (epoch, max_epochs) -> coeff>.")
            decay_fn = DECAY_SCHEDULES[decay_schedule]

        # n_dim is the ONE place that sets Z's actual width (the base class's
        # lazy init in embed()), so this is the whole of "make the model
        # augmented" as far as the engine is concerned.
        super().__init__(*args, n_dim=int(d) + int(aux), **kwargs)

        self.d = int(d)
        self.aux = int(aux)
        self.decay_schedule = decay_schedule
        self.decay_fn = decay_fn

        # Own epoch counter, NOT self.latest_epoch -- see updateZ().
        self._epoch_counter = 0
        self._max_epochs = None    # captured by embed()

    # ------------------------------------------------------------- main loop
    def embed(self, G, epochs: int = 1000, **kwargs):
        """Capture the epoch budget, then run the inherited loop unchanged.

        This override exists for exactly one reason: ``updateZ(self, lr=None)``
        never receives ``epochs``, so without stashing it here there is
        nothing for the schedule to normalize against. Everything else is
        ``ForceDirected.embed``.
        """
        self._max_epochs = epochs
        return super().embed(G, epochs=epochs, **kwargs)

    # ------------------------------------------------------------ integrator
    def updateZ(self, lr: float | None = None) -> None:
        """Base position update, then shrink the aux columns by ``coeff(epoch)``.

        Ordering matters: the normal ``Z = Z + lr*dZ`` runs first over all
        ``d + aux`` columns (the aux dims move under the *undamped* force
        law -- that is the point, they need freedom to route points around
        obstructions), and only then are they pulled back toward zero. This
        composes cleanly with the other override seam: ``updateGradient``
        acts on the force side, ``updateZ`` on the integration side.

        The epoch fed to the schedule is ``self._epoch_counter``, not
        ``self.latest_epoch``: the base loop assigns ``self.latest_epoch``
        *after* calling ``updateZ``, so reading it from in here would be off
        by one. A counter incremented once per call is exact regardless of
        that ordering and regardless of ``batch_count`` (``updateZ`` runs
        exactly once per epoch either way, however many batches ran).

        The counter deliberately does not reset between ``embed()`` calls,
        matching the base class's own resume semantics (``end_epoch``). A
        resumed run therefore continues past ``max_epochs``, where the
        schedules are extrapolations of their formulas rather than the
        [1 -> 0] ramp they are specified for; with the default ``linear``
        that means a small negative coefficient (aux columns flip sign each
        epoch while staying near zero). If that matters for an experiment,
        pass a ``decay_fn`` that clamps.
        """
        super().updateZ(lr=lr)
        if self.aux <= 0:
            return                     # no scaffolding: exact no-op wrapper
        if self._max_epochs is None:
            raise RuntimeError(
                "EuclideanAuxDimModel.updateZ needs the epoch budget captured by "
                "embed(); call embed(G, epochs=...) rather than driving updateZ "
                "directly, or set self._max_epochs yourself first.")
        self._epoch_counter += 1
        coeff = self.decay_fn(self._epoch_counter, self._max_epochs)
        self.Z = self.Z.at[:, self.d:].multiply(coeff)

    # ------------------------------------------------------------ embeddings
    def get_embeddings(self) -> np.ndarray:
        """The ``(n, d)`` embedding -- real dims only, never the scaffolding.

        ``get_embeddings_df()`` is deliberately NOT overridden: it builds
        its frame from ``self.get_embeddings()``, so it picks up this slice
        (and its ``z0..z{d-1}`` column names) for free -- there is nothing
        aux-specific left for an override to fix. (Independently of this
        model, that base method reads ``self.Gx`` for the node labels, an
        attribute nothing in ``fdge_jax_sell_c_sigma`` sets -- models set
        ``self.G``. That is a pre-existing base-class wart, out of scope
        here, and not something an override in *this* class should quietly
        paper over for one model only.)
        """
        return np.asarray(self.Z[:, :self.d])

    # ------------------------------------------------------- convergence stat
    def Th(self, dZ) -> float:
        """Mean row norm of ``dZ`` over the REAL dims only.

        Overridden as an instance method (the base declares it a
        ``@staticmethod``; normal attribute lookup on ``self.Th(self.dZ)``
        resolves to this either way) because it needs ``self.d``.

        Why slice: the aux columns are pushed by the undamped force law and
        then externally reset every epoch, so their contribution to ``dZ``'s
        norm need never settle -- leaving them in could stop
        ``epsilon``-based early stopping from ever triggering, or trigger it
        on a signal that has nothing to do with the embedding actually being
        returned. This resolves one of ``idea.md`` section 4's open
        questions; the alternative (all ``d + aux`` dims) is the base
        class's behaviour and is one line away if a run wants to compare.
        """
        return float(jnp.linalg.norm(dZ[:, :self.d], axis=-1).mean())
