"""core.force_directed -- the fdge_jax_sell_c_sigma ``ForceDirected`` base class.

Port of ``fdge2/core/force_directed.py``. The loop *shape* (callbacks,
contiguous row batching, epsilon convergence) is unchanged -- it's
plumbing, not physics. What changes is entirely a consequence of
``jnp.ndarray`` being immutable and JAX's RNG being explicit-key rather
than global-state (docs/DESIGN.md "core.ForceDirected -- what changes
from fdge2, concretely"):

* ``self.Z`` / ``self.dZ`` are ``jnp.ndarray`` (float32 by default -- this
  package's whole point is throughput, not numeric parity with the Numba
  ``fdge2`` sibling, so nothing here calls
  ``jax.config.update("jax_enable_x64", True)`` or otherwise forces
  float64. A researcher who wants float64 precision can still get it:
  call ``jax.config.update("jax_enable_x64", True)`` before import and
  pass float64 arrays in explicitly via ``Z=`` -- but the default
  construction here (lazy random init, ``dZ`` zeros) is float32).
* ``self.rng`` (a ``np.random.Generator``) is kept, unchanged, for any
  non-hot-path numpy randomness (e.g. a ``graph_building`` strategy that
  wants a numpy ``random_state``). Alongside it, ``self.key`` is a
  ``jax.random.PRNGKey`` -- split once per batch and threaded into
  ``updateGradient`` / ``forces`` as a ``key=`` kwarg, the JAX analogue of
  fdge2 passing ``rng=self.rng`` through (see below: fdge2's base loop
  actually never threads ``rng`` through kwargs itself -- individual force
  laws that want randomness pull ``self.rng`` directly off the model. The
  JAX RNG-is-explicit discipline means ``key`` genuinely has to flow
  through the kwargs chain instead, since ``forces`` is meant to be a
  pure function of its arguments, not a method reading ``self``).
* Writes into ``dZ`` use ``.at[...].set(...)``; ``updateZ`` uses
  ``self.Z = self.Z + lr * self.dZ``. No in-place mutation.
* ``get_embeddings()`` still returns a **numpy** array
  (``np.asarray(self.Z)``) -- callers (pandas, plotting) shouldn't need to
  know the backend.

Abstract hooks -- ``make_graph``, ``augment_graph``, ``forces`` -- raise
``NotImplementedError`` here, exactly as in fdge2. Their concrete
implementations live in ``graph_building/`` / ``graph_augmenting/`` /
``embedding/``, which by import discipline (docs/DESIGN.md) never import
each other; composition happens one level up, in
``fdge_jax_sell_c_sigma/models.py``.

Import discipline: this module imports **numpy + jax + core.csr only**,
never a stage package.
"""
from __future__ import annotations

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
    """Force-Directed base model for the fdge_jax_sell_c_sigma three-stage pipeline.

    Subclasses (concrete models, composed in
    ``fdge_jax_sell_c_sigma/models.py``) provide ``make_graph`` /
    ``augment_graph`` / ``forces``; the epoch loop, batching, callbacks,
    and convergence test live here and are inherited unchanged from
    fdge2's shape.
    """

    VER_MAJ = "00"
    VER_MIN = "02"
    DESCRIPTION = "fdge_jax_sell_c_sigma Force-Directed Base Model (three-stage pipeline, JAX)"

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
        node_labels = list(self.G.nodes())
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

    def updateZ(self, lr: float | None = None) -> None:
        """Apply the assembled step ``self.dZ`` to ``self.Z``.

        ``Z = Z + lr * dZ`` -- no in-place ``+=``, ``jnp.ndarray`` is
        immutable. Momentum is not applied here (it lives in
        ``updateGradient``); this is a thin override seam for a custom
        integrator if a researcher wants one.
        """
        if lr is None:
            lr = self.lr
        self.Z = self.Z + lr * self.dZ

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
        return self.Z

    def fit(self, data, epochs: int = 1000, **kwargs):
        """Convenience: ``make_graph(data)`` then ``embed(G)``."""
        G = self.make_graph(data, **kwargs)
        return self.embed(G, epochs=epochs, **kwargs)
