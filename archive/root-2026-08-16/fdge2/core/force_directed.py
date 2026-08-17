"""core.force_directed -- the fdge2 ``ForceDirected`` base class.

This is a faithful port of the loop machinery in
``fdge_numba/forcedirected_numba/ForceDirected.py`` (callbacks, contiguous
row batching, convergence test), adapted to the fdge2 three-stage
skeleton (ARCHITECTURE.md / API_DESIGN.md). The physics and the loop
semantics are *not* redesigned -- only the seams the pipeline needs.

What changed from the reference base class, and why (API_DESIGN.md):

* ``embed(self, G, ...)`` takes the **un-augmented** graph ``G`` and its
  first step is ``D = self.augment_graph(G, ...)``. The reference took no
  graph argument at all (the model held ``self.hops`` from __init__); the
  fdge2 skeleton makes augmentation an explicit, swappable first step of
  ``embed`` (RPD.md §3: callers never call ``augment_graph`` directly in
  the normal ``fit`` / ``embed`` flow).

* ``updateGradient(self, Z, D, **kwargs)`` is the new seam between the raw
  force law and the step applied to ``Z``. Default is a pure passthrough
  ``return self.forces(Z, D, **kwargs)`` -- plain gradient descent,
  numerically identical to the reference with ``beta == 0``. Momentum /
  velocity / gradient-clipping is now expressed by a subclass **overriding
  updateGradient**, not by a ``beta`` term hardcoded in the loop
  (``updateZ`` in the reference). See the worked example in API_DESIGN.md.

* ``forces`` replaces the reference's ``forward``: same role (the one
  function to edit for a new force law), same batching contract -- it
  takes ``row_start`` / ``row_end`` and returns the dZ rows for that range
  (Q3). It arrives here as an abstract hook; the concrete kernel lives in
  ``embedding/``.

Abstract hooks -- ``make_graph``, ``augment_graph``, ``forces`` -- raise
``NotImplementedError`` here. Their concrete implementations live in the
*other* packages (``graph_building/``, ``graph_augmenting/``, ``embedding/``),
which by import discipline (RPD.md §5) never import each other.

Composition root
----------------
A single runnable model needs all three hooks implemented at once, but the
three stage packages must never import one another. The resolution: the
composition happens **one level up**, in a future ``fdge2/models.py``
(NOT part of this ``core`` task) that is allowed to import ``core`` and all
three stage packages and wire them together::

    from fdge2.core import ForceDirected
    from fdge2 import graph_building, graph_augmenting, embedding

    class ReferenceFDModel(ForceDirected):
        def make_graph(self, data, **kw):
            return graph_building.make_graph(data, **kw)
        def augment_graph(self, G, **kw):
            return graph_augmenting.hopfill.augment_graph(G, **kw)
        def forces(self, Z, D, row_start, row_end, **kw):
            return embedding.shell_force.forces(Z, D, row_start, row_end, **kw)

Nothing in ``core`` assumes or hardcodes a specific augmentation or force
implementation -- the abstract-hook design gives that for free. ``core``
knows the three stages' *contracts*, never their internals.

Import discipline: this module imports **numpy only** (plus the sibling
``core.csr``).
"""
from __future__ import annotations

import numpy as np

from .csr import n_rows


# ---------------------------------------------------------------------------
# Callbacks (ported verbatim from the reference base class)
# ---------------------------------------------------------------------------
class Callback_Base:
    """Subclass and override any of the hooks you need.

    Every hook receives the model as first argument, plus the loop kwargs
    (epoch, epochs, batch, batch_count, batch_size, ...).
    """

    # The complete lifecycle event vocabulary -- one place, in sync with the
    # six on_* hooks below by construction, so ForceDirected.notify_callback
    # can reject an unknown event name immediately (even with zero callbacks
    # attached) instead of silently doing nothing.
    _EVENTS = ("train_begin", "train_end", "epoch_begin", "epoch_end",
               "batch_begin", "batch_end")

    def on_train_begin(self, model, **kwargs): pass
    def on_train_end(self, model, **kwargs): pass
    def on_epoch_begin(self, model, **kwargs): pass
    def on_epoch_end(self, model, **kwargs): pass
    def on_batch_begin(self, model, **kwargs): pass
    def on_batch_end(self, model, **kwargs): pass


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class ForceDirected:
    """Force-Directed base model for the fdge2 three-stage pipeline.

    Subclasses (concrete models, composed in ``fdge2/models.py``) provide
    ``make_graph`` / ``augment_graph`` / ``forces``; the epoch loop,
    batching, callbacks, and convergence test live here and are inherited
    unchanged.
    """

    VER_MAJ = "00"
    VER_MIN = "01"
    DESCRIPTION = "fdge2 Force-Directed Base Model (three-stage pipeline)"

    def __init__(self,
                 n_dim: int | None = None,
                 lr: float = 1.0,
                 beta: float = 0.0,
                 epsilon: float | None = None,
                 verbosity: int = 2,
                 seed: int | None = None,
                 **kwargs) -> None:
        """
        n_dim     : embedding dimensionality; used to lazily initialize Z
                    inside ``embed`` when neither ``Z=`` nor a preset
                    ``self.Z`` is provided. May be None if the model sets
                    ``self.Z`` some other way before embedding.
        lr        : learning rate (step size), default 1.0
        beta      : momentum coefficient. NOT used by the base loop -- kept
                    as a convenience attribute for a ``updateGradient``
                    override that wants to implement momentum (API_DESIGN.md).
        epsilon   : optional convergence threshold on Th(dZ) (mean row norm).
                    None == run for the full ``epochs`` count.
        verbosity : 0 silent, 1 epoch line, 2 +dZ stats, 3 +batch detail
        seed      : seed for the model RNG (Z init, dropout, ...)
        """
        self.n_dim = n_dim
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.verbosity = verbosity
        self.rng = np.random.default_rng(seed)

        self.Z = None        # (n, d) embedding
        self.dZ = None       # (n, d) per-epoch step buffer
        self.V = None        # (n, d) velocity buffer, for updateGradient overrides

        self.callbacks = []
        self.stop_training = False
        self.latest_epoch = 0
        self.end_epoch = 0   # advanced by ``epochs`` each embed() call (resume)

    # ------------------------------------------------------------------ misc
    def __str__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN} - {self.DESCRIPTION}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN}"

    # ---------------------------------------------------------- stage hooks
    # These three are the swappable stages (ARCHITECTURE.md). Concrete
    # implementations live in graph_building/, graph_augmenting/, embedding/ and
    # are wired onto a subclass in fdge2/models.py (see module docstring).
    def make_graph(self, data, **kwargs):
        """Stage 1: build a graph ``G`` from raw ``data`` (MST, kNN, ...).

        Returns any indexable weighted/unweighted graph. Left to the
        researcher / experiment. (Q2: ``make_graph`` itself takes no
        ``is_sparse`` flag -- it just returns a graph; density of the
        eventual ``D`` is ``augment_graph``'s decision.)
        """
        raise NotImplementedError("make_graph(.) is not implemented")

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        """Stage 2: turn ``G`` into a weighted matrix ``D`` (add/reweight edges).

        ``D`` must satisfy the indexability contract (``D[i, j]`` works),
        dense ndarray or ``scipy.sparse`` (Q2: ``is_sparse`` is honored
        *here*, not in ``make_graph``). ``D`` is a constant input to the
        embedding stage -- nothing in ``forces`` / ``updateGradient`` /
        ``embed`` knows how it was produced (RPD.md §3).
        """
        raise NotImplementedError("augment_graph(.) is not implemented")

    def forces(self, Z, D, row_start: int, row_end: int, **kwargs) -> np.ndarray:
        """Stage 3: the force law. Compute dZ rows for nodes [row_start, row_end).

        The one function to edit for a new force law (plays the role of
        ``_forces_204_hidx`` / ``forward`` in the reference). Returns an
        array of shape ``(row_end - row_start, n_dim)``. Keeps row-range
        batching support (Q3) so ``embed``'s batch loop can call it per
        batch.
        """
        raise NotImplementedError("forces(.) is not implemented")

    # ---------------------------------------------------------- gradient seam
    def updateGradient(self, Z, D, **kwargs) -> np.ndarray:
        """Seam between the raw force law and the step applied to ``Z``.

        Default: pure passthrough to ``forces`` -- plain gradient descent,
        numerically identical to the reference model's ``beta == 0`` path.
        ``row_start`` / ``row_end`` flow through ``**kwargs`` to ``forces``.

        Override this (not ``forces``) for momentum / velocity / gradient
        clipping, e.g.::

            def updateGradient(self, Z, D, **kwargs):
                self.v = self.alpha * self.v + self.beta * self.forces(Z, D, **kwargs)
                return self.v

        Note: a whole-array velocity override like the above assumes it
        sees the whole graph at once, so use it with ``batch_count == 1``
        (the default). Batching invariance is a guarantee of the default
        passthrough path only.
        """
        return self.forces(Z, D, **kwargs)

    # ------------------------------------------------------------- callbacks
    def attach_callback(self, callback: Callback_Base) -> None:
        if not isinstance(callback, Callback_Base):
            raise TypeError(
                f"callback must be a 'Callback_Base' subclass, got {type(callback)}")
        self.callbacks.append(callback)

    def notify_callback(self, event: str, **kwargs) -> None:
        if event not in Callback_Base._EVENTS:
            raise ValueError(
                f"Unknown callback event {event!r}. "
                f"Valid events: {Callback_Base._EVENTS}")
        for cb in self.callbacks:
            getattr(cb, f"on_{event}")(self, **kwargs)

    # ------------------------------------------------------------ embeddings
    def get_embeddings(self) -> np.ndarray:
        """Returns embeddings as a (n, d) numpy array."""
        return np.asarray(self.Z)

    def get_embeddings_df(self, columns=None):
        """Returns embeddings as a pandas dataframe, first column = node label."""
        import pandas as pd
        node_labels = list(self.Gx.nodes())
        emb = self.get_embeddings()
        data = np.column_stack([node_labels, emb])
        if columns is None:
            columns = ["node"] + [f"z{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(data, columns=columns)

    # ------------------------------------------------------------ integrator
    @staticmethod
    def Th(dZ: np.ndarray) -> float:
        """Convergence statistic: mean Euclidean norm of the dZ rows."""
        return float(np.linalg.norm(dZ, axis=-1).mean())

    def updateZ(self, lr: float | None = None) -> None:
        """Apply the assembled step ``self.dZ`` to ``self.Z``.

        Plain ``Z += lr * dZ``. Momentum is no longer applied here (it
        moved to ``updateGradient``); this is a thin override seam for a
        custom integrator if a researcher wants one.
        """
        if lr is None:
            lr = self.lr
        self.Z += lr * self.dZ

    # ------------------------------------------------------------- main loop
    def embed(self,
              G,
              epochs: int = 1000,
              lr: float | None = None,
              Z: np.ndarray | None = None,
              batch_count: int = 1,
              epsilon: float | None = None,
              **kwargs) -> np.ndarray:
        """Augment ``G`` into ``D`` and run the force-directed relaxation.

        This method is computationally heavy -- the epoch loop is the hot
        path (API_DESIGN.md).

        G           : the **un-augmented** graph. First step is
                      ``D = self.augment_graph(G, ...)`` (RPD.md §3).
        epochs      : maximum number of epochs
        lr          : learning rate override (defaults to self.lr)
        Z           : optional (n, d) array to start / continue an embedding
        batch_count : number of contiguous row batches per epoch (Q1/Q3).
                      With the default passthrough ``updateGradient``, the
                      result is identical for any ``batch_count`` (batching
                      invariance) because forces are batch-local and ``Z``
                      is updated once per epoch.
        epsilon     : convergence threshold override (defaults to self.epsilon)

        Returns ``self.Z``.
        """
        # ---- stage 2: augment first, then loop against D (never G) --------
        D = self.augment_graph(G, **kwargs)

        if Z is not None:
            self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        if self.Z is None:
            # lazy random init now that n is known from D
            if self.n_dim is None:
                raise RuntimeError(
                    "Z is not initialized. Pass Z=..., set self.Z, or give n_dim=.")
            n = n_rows(D)
            self.Z = np.ascontiguousarray(
                self.rng.standard_normal((n, self.n_dim)), dtype=np.float64)
        if epsilon is None:
            epsilon = self.epsilon

        n, d = self.Z.shape
        if self.dZ is None or self.dZ.shape != (n, d):
            self.dZ = np.zeros((n, d), dtype=np.float64)

        kwargs.update(epochs=epochs)
        self.stop_training = False
        self.notify_callback("train_begin", **kwargs)

        # contiguous row ranges: [0, b), [b, 2b), ...  (Q1: batching kept)
        batch_count = max(1, min(int(batch_count), n))
        batch_size = (n + batch_count - 1) // batch_count
        bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]

        start_epoch = self.end_epoch + 1
        for epoch in range(start_epoch, start_epoch + epochs):
            if self.stop_training:
                break
            kwargs.update(epoch=epoch, batch_count=len(bounds), batch_size=batch_size)
            self.notify_callback("epoch_begin", **kwargs)

            if self.verbosity >= 1:
                print(f"Epoch {epoch}/{epochs}", end="")

            self.dZ.fill(0.0)
            for b, (row_start, row_end) in enumerate(bounds):
                kwargs.update(batch=b + 1)
                self.notify_callback("batch_begin", **kwargs)
                if self.verbosity >= 3 and len(bounds) > 1:
                    print(f"  batch {b + 1}/{len(bounds)}")
                # ------------- gradient seam -> force law -------------
                self.dZ[row_start:row_end] = self.updateGradient(
                    self.Z, D, row_start=row_start, row_end=row_end, **kwargs)
                self.notify_callback("batch_end", **kwargs)

            dZ_norm_avg = self.Th(self.dZ)
            if self.verbosity >= 2:
                if len(bounds) > 1:
                    print(f" ({len(bounds)} batches)", end="")
                print(f"  dZ norm avg: {dZ_norm_avg:.6f}", end="")
            if self.verbosity >= 1:
                print("")

            self.updateZ(lr=lr)
            self.latest_epoch = epoch
            self.notify_callback("epoch_end", **kwargs)

            if epsilon is not None and dZ_norm_avg <= epsilon:
                if self.verbosity >= 1:
                    print(f"Converged at epoch {epoch}: "
                          f"Th(dZ)={dZ_norm_avg:.6g} <= {epsilon}")
                break

            self.end_epoch += 1  # advance for resume

        self.notify_callback("train_end", **kwargs)
        return self.Z

    def fit(self, data, epochs: int = 1000, **kwargs) -> np.ndarray:
        """Convenience: ``make_graph(data)`` then ``embed(G)``.

        Thin, unchanged from the reference intent (API_DESIGN.md).
        """
        G = self.make_graph(data, **kwargs)
        return self.embed(G, epochs=epochs, **kwargs)
