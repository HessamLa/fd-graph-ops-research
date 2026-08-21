"""core.force_directed -- the ``ForceDirected`` base class, and the callbacks.

It owns the epoch loop, the batching, the callback events and the ``Z``
update. It owns NO augmentation and NO force law: ``forces`` dispatches to
a law that the subclass supplies, and ``updateGradient`` fills ``dZ`` for a
ROW RANGE.

Callback events, in order::

    train_begin, then per epoch: epoch_begin,
                                 per batch: batch_begin, batch_end,
                                 epoch_end,
    train_end

THE CHUNK CONTRACT, and it is the reason ``forces`` takes a row range and
not a set of pairs: ``sell_c_sigma._step`` writes ``dZ.at[rows].add(...)``,
and only DISJOINT rows make the parts additive. ``embed`` slices ``dZ`` by
the same row range that the plan chunk uses, thus the batch and the chunk
are the same object. A chunk of arbitrary pairs sums ``dZ`` twice, in
silence (I6).

I7, and it is why a diverged run does not crash: a non-finite ``Z`` is
RECORDED on the model (``self.diverged``) and returned. It is never raised
through an evaluator. Seven runs of the Cora grid of 2026-08-18 died
inside sklearn with "Input X contains NaN", and a crash carries no time, no
memory and no RESULT line -- thus a real measurement, "this rate diverges",
was lost as a stack trace.

Provenance: a verbatim move of ``fodined/core/fodined.py``, itself a copy
of ``fdge_jax_sell_c_sigma/core/force_directed.py``. Three changes, each
one marked in place:

  1. the class is ``ForceDirected`` again, and not ``Fodined``;
  2. ``updateZ`` dispatches through ``misc/optim.py`` and defaults to
     ``plain``, which is the line the base class had;
  3. ``embed`` records a non-finite ``Z`` (I7).

Import discipline: numpy + jax + ``core.csr`` at module level. ``misc.optim``
arrives through a function-local import inside ``set_rule``, thus the
module-level dependency still runs one way: ``core`` imports nothing of the
package except ``core``.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from .csr import n_rows


# ---------------------------------------------------------------------------
# Callbacks (ported verbatim from fdge2 -- pure plumbing, no array backend
# involved at all)
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
    """Force-directed base model for the three-stage pipeline.

    Subclasses provide ``make_graph`` / ``augment_graph`` / ``forces``; the
    epoch loop, batching, callbacks, and convergence test live here and are
    inherited unchanged from fdge2's shape.
    """

    VER_MAJ = "00"
    VER_MIN = "03"
    DESCRIPTION = "Force-Directed Base Model (three-stage pipeline, JAX)"

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
                    override that wants to implement momentum.
        epsilon   : optional convergence threshold on Th(dZ) (mean row norm).
                    None == run for the full ``epochs`` count.
        verbosity : 0 silent, 1 epoch line, 2 +dZ stats, 3 +batch detail
        seed      : seed for both ``self.rng`` (numpy) and ``self.key`` (jax)
        """
        self.n_dim = n_dim
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.verbosity = verbosity
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(seed if seed is not None else 0)

        self.Z = None        # (n, d) embedding, jnp.ndarray
        self.dZ = None       # (n, d) per-epoch step buffer, jnp.ndarray
        self.V = None        # (n, d) velocity buffer, for updateGradient overrides

        self.G = None        # the un-augmented graph, set by embed()
        self.D = None        # the augmented distance matrix, set by embed()

        self.callbacks = []
        self.stop_training = False
        self.latest_epoch = 0
        self.end_epoch = 0   # advanced by ``epochs`` each embed() call (resume)

        self.epoch = 0       # the epoch that ``updateZ`` sees
        self.epochs = 0      # the count that a decaying lr divides by
        self.rule = None     # the update rule; ``set_rule`` binds it
        self.rule_name = "plain"
        self.opt_state = {}  # the state of the rule, owned by this class
        self.lr_schedule = None   # (lr, epoch, epochs) -> lr, or None
        self.diverged = False     # I7: recorded, never raised
        self.diverged_epoch = None
        self.n_nonfinite = 0

    # ------------------------------------------------------------------ misc
    def __str__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN} - {self.DESCRIPTION}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN}"

    # ---------------------------------------------------------- stage hooks
    # These three are the swappable stages. Concrete implementations live in
    # graph_building/, graph_augmenting/, embedding/ and are wired onto a
    # subclass in fdge_jax_sell_c_sigma/models.py (see module docstring).
    def make_graph(self, data, **kwargs):
        """Stage 1: build a graph ``G`` from raw ``data`` (MST, kNN, ...).

        Returns any indexable weighted/unweighted graph. Density of the
        eventual ``D`` is ``augment_graph``'s decision, not this one's.
        """
        raise NotImplementedError("make_graph(.) is not implemented")

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        """Stage 2: turn ``G`` into a weighted matrix ``D`` (add/reweight edges).

        ``D`` must satisfy the indexability contract (``D[i, j]`` works) --
        a dense ndarray or a ``scipy.sparse.csr_matrix``. ``D`` is a constant
        input to the embedding stage -- nothing in ``forces`` /
        ``updateGradient`` / ``embed`` knows how it was produced.
        """
        raise NotImplementedError("augment_graph(.) is not implemented")

    def forces(self, Z, D, row_start: int, row_end: int, **kwargs):
        """Stage 3: the force law. Compute dZ rows for nodes [row_start, row_end).

        The one function to edit for a new force law. Returns an array of
        shape ``(row_end - row_start, n_dim)``. Keeps row-range batching
        support so ``embed``'s batch loop can call it per batch.
        """
        raise NotImplementedError("forces(.) is not implemented")

    # ---------------------------------------------------------- gradient seam
    def updateGradient(self, Z, D, **kwargs):
        """Seam between the raw force law and the step applied to ``Z``.

        Default: pure passthrough to ``forces`` -- plain gradient descent,
        numerically identical to fdge2's ``beta == 0`` path.
        ``row_start`` / ``row_end`` / ``key`` flow through ``**kwargs`` to
        ``forces``.

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
        """Returns embeddings as a (n, d) numpy array (host copy of self.Z)."""
        return np.asarray(self.Z)

    def get_embeddings_df(self, columns=None):
        """Returns embeddings as a pandas dataframe, first column = node label."""
        import pandas as pd
        if self.G is None:
            raise RuntimeError(
                "No graph on this model yet -- call embed(G, ...) or fit(data, ...) "
                "before get_embeddings_df().")
        # A CSR carries no node labels, thus the row index IS the label.
        node_labels = (list(self.G.nodes()) if hasattr(self.G, "nodes")
                       else list(range(self.Z.shape[0])))
        emb = self.get_embeddings()
        data = np.column_stack([node_labels, emb])
        if columns is None:
            columns = ["node"] + [f"z{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(data, columns=columns)

    # ------------------------------------------------------------ integrator
    @staticmethod
    def Th(dZ) -> float:
        """Convergence statistic: mean Euclidean norm of the dZ rows.

        ``float(...)`` is a deliberate host sync -- needed every epoch for
        the print/epsilon check regardless of backend, same cost/role as
        fdge2's plain-numpy version.
        """
        return float(jnp.linalg.norm(dZ, axis=-1).mean())

    # -- CHANGE 2 of the move: the update rule is a dispatch, and no longer
    #    one line. `plain` IS that one line, thus the default is unchanged.
    def set_rule(self, name: str = "plain", lr: float | None = None,
                 **rule_kw) -> None:
        """Choose the update rule of ``updateZ``. ``plain`` is the default.

        The rules live in ``misc/optim.py`` and each one is pure: it takes
        ``(Z, dZ, lr, state, epoch)`` and gives ``(Z_new, state)``. The
        state stays in ``self.opt_state``, a plain dict that this class
        keeps and no rule owns.

        ``rule_kw`` binds the parameters of one rule -- ``eta`` for
        ``velocity``, ``frac`` and ``seed`` for ``sgd``, ``memory`` for
        ``sqn``.

        The import is function-local, and deliberately: it keeps the
        module-level import graph one-way (``core`` never imports another
        stage of the package), while ``updateZ`` still dispatches through
        ``misc/optim.py`` as the design asks.
        """
        from ..misc import optim

        base = optim.RULES[name]
        self.rule_name = name
        self.rule = functools.partial(base, **rule_kw) if rule_kw else base
        self.opt_state = {}
        if lr is not None:
            self.lr = lr

    def updateZ(self, lr: float | None = None) -> None:
        """Apply the assembled step ``self.dZ`` to ``self.Z``.

        ``plain`` is ``Z = Z + lr * dZ`` -- no in-place ``+=``,
        ``jnp.ndarray`` is immutable. Every other rule of ``misc/optim.py``
        replaces that one line, and it keeps its state in
        ``self.opt_state``.

        ``lr_schedule`` is an optional callable ``(lr, epoch, epochs) ->
        lr``. ``None`` holds ``lr`` constant, which is the behaviour of
        every run before 2026-08-18.
        """
        if lr is None:
            lr = self.lr
        if self.lr_schedule is not None:
            lr = self.lr_schedule(lr, self.epoch, self.epochs)
        if self.rule is None:
            self.set_rule("plain")
        self.Z, self.opt_state = self.rule(
            self.Z, self.dZ, lr, self.opt_state, self.epoch)

    # ------------------------------------------------------------- main loop
    def embed(self,
              G,
              epochs: int = 1000,
              lr: float | None = None,
              Z=None,
              batch_count: int = 1,
              epsilon: float | None = None,
              **kwargs):
        """Augment ``G`` into ``D`` and run the force-directed relaxation.

        This method is computationally heavy -- the epoch loop is the hot
        path.

        G           : the **un-augmented** graph. First step is
                      ``D = self.augment_graph(G, ...)``.
        epochs      : maximum number of epochs
        lr          : learning rate override (defaults to self.lr)
        Z           : optional (n, d) array to start / continue an embedding
                      (coerced to jnp.ndarray, float32)
        batch_count : number of contiguous row batches per epoch. With the
                      default passthrough ``updateGradient``, the result is
                      identical for any ``batch_count`` (batching
                      invariance) because forces are batch-local and ``Z``
                      is updated once per epoch.
        epsilon     : convergence threshold override (defaults to self.epsilon)

        Returns ``self.Z``.
        """
        # ---- stage 2: augment first, then loop against D (never G) --------
        self.epochs = epochs

        # Keep both on the model: embed(G) is a valid entry point on its own
        # (no fit()/make_graph() call), so this is the only place self.G is
        # guaranteed to be set -- get_embeddings_df() reads it for node ids.
        self.G = G
        D = self.D = self.augment_graph(G, **kwargs)

        if Z is not None:
            self.Z = jnp.asarray(Z, dtype=jnp.float32)
        if self.Z is None:
            # lazy random init now that n is known from D
            if self.n_dim is None:
                raise RuntimeError(
                    "Z is not initialized. Pass Z=..., set self.Z, or give n_dim=.")
            n = n_rows(D)
            self.key, sub = jax.random.split(self.key)
            self.Z = jax.random.normal(sub, (n, self.n_dim), dtype=jnp.float32)
        if epsilon is None:
            epsilon = self.epsilon

        n, d = self.Z.shape
        if self.dZ is None or self.dZ.shape != (n, d):
            self.dZ = jnp.zeros((n, d), dtype=jnp.float32)

        kwargs.update(epochs=epochs)
        self.stop_training = False
        self.notify_callback("train_begin", **kwargs)

        # contiguous row ranges: [0, b), [b, 2b), ...
        batch_count = max(1, min(int(batch_count), n))
        batch_size = (n + batch_count - 1) // batch_count
        bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]

        start_epoch = self.end_epoch + 1
        for epoch in range(start_epoch, start_epoch + epochs):
            self.epoch = epoch
            if self.stop_training:
                break
            kwargs.update(epoch=epoch, batch_count=len(bounds), batch_size=batch_size)
            self.notify_callback("epoch_begin", **kwargs)

            if self.verbosity >= 1:
                print(f"Epoch {epoch}/{epochs}", end="")

            self.dZ = jnp.zeros_like(self.dZ)
            for b, (row_start, row_end) in enumerate(bounds):
                self.key, batch_key = jax.random.split(self.key)
                kwargs.update(batch=b + 1)
                self.notify_callback("batch_begin", **kwargs)
                if self.verbosity >= 3 and len(bounds) > 1:
                    print(f"  batch {b + 1}/{len(bounds)}")
                # ------------- gradient seam -> force law -------------
                rows = self.updateGradient(
                    self.Z, D, row_start=row_start, row_end=row_end,
                    key=batch_key, **kwargs)
                self.dZ = self.dZ.at[row_start:row_end].set(rows)
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

        # CHANGE 3 of the move -- I7. A non-finite `Z` is RECORDED here and
        # returned. It is never raised, and it never reaches an evaluator
        # by accident: the caller reads `self.diverged` and writes a
        # measurement instead of a stack trace.
        if not bool(jnp.isfinite(self.Z).all()):
            self.diverged = True
            self.diverged_epoch = self.latest_epoch
            self.n_nonfinite = int(jnp.sum(~jnp.isfinite(self.Z)))
            if self.verbosity >= 1:
                print(f"DIVERGED at epoch {self.latest_epoch}: Z holds "
                      f"{self.n_nonfinite} non-finite values")
        return self.Z

    def fit(self, data, epochs: int = 1000, **kwargs):
        """Convenience: ``make_graph(data)`` then ``embed(G)``."""
        G = self.make_graph(data, **kwargs)
        return self.embed(G, epochs=epochs, **kwargs)
