"""
ForceDirected.py -- Numba/NumPy rewrite of the Force-Directed base model.

This is a torch-free rewrite of forcedirected/models/ForceDirected.py.

Design notes (deviations from the legacy torch version, all deliberate):

* No torch. The model is an n-body relaxation, not a neural network:
  there is no autograd, no learned parameters, no reason to carry
  torch.nn.Module. State is plain float64 NumPy arrays.

* No `self.train = self.embed` alias. In the legacy code this shadowed
  nn.Module.train and caused subtle breakage. Use `embed()`.

* Batches are contiguous row ranges [row_start, row_end). This is the
  natural unit for a Numba kernel (contiguous memory, prange over rows)
  and is equivalent to the legacy `batchify(range(n))` behavior.
  `batch_count` is kept as a parameter for memory/progress granularity;
  with a prange kernel, batch_count=1 is the performance default on CPU.

* Optional convergence test. If `epsilon` is given, training stops when
  Th(dZ) <= epsilon, where Th defaults to the mean row-norm of dZ
  (the same statistic the legacy loop printed but never acted on).

* Optional momentum. `beta=0.0` (default) reproduces the legacy update
  Z += lr * dZ exactly. With beta > 0 a velocity buffer V is kept and
  the update becomes  V = beta*V + dZ ;  Z += lr*V.

Subclasses implement `forward(row_start, row_end, **kwargs)` and return
the dZ rows for that range, shape (row_end - row_start, n_dim).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class Callback_Base:
    """Subclass and override any of the hooks you need.

    Every hook receives the model as first argument, plus the loop kwargs
    (epoch, epochs, batch, batch_count, batch_size, ...).
    """

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
    """Force-Directed Base Model (NumPy + Numba backends in subclasses)."""

    VER_MAJ = "03"
    VER_MIN = "00"
    DESCRIPTION = "Force-Directed Base Model (torch-free, Numba-ready)"

    def __init__(self,
                 lr: float = 1.0,
                 beta: float = 0.0,
                 epsilon: float | None = None,
                 verbosity: int = 2,
                 seed: int | None = None,
                 **kwargs) -> None:
        """
        lr        : learning rate (step size), default 1.0
        beta      : momentum coefficient in [0, 1). 0.0 == legacy behavior.
        epsilon   : optional convergence threshold on Th(dZ) (mean row norm).
                    None == run for the full `epochs` count (legacy behavior).
        verbosity : 0 silent, 1 epoch line, 2 +dZ stats, 3 +batch/model detail
        seed      : seed for the model RNG (init, dropout, ...)
        """
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.verbosity = verbosity
        self.rng = np.random.default_rng(seed)

        self.Z = None        # (n, d) embedding, set by subclass or embed(Z=...)
        self.dZ = None       # (n, d) per-epoch force/step buffer
        self.V = None        # (n, d) velocity buffer, used only if beta > 0

        self.callbacks = []
        self.stop_training = False
        self.latest_epoch = 0

        self.end_epoch = 0  # for resume, set by subclass if needed. Everytime calling embed(epochs=...) 
                            # advances self.end_epoch by epochs.


    # ------------------------------------------------------------------ misc
    def __str__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN} - {self.DESCRIPTION}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN}"

    # ------------------------------------------------------------- callbacks
    def attach_callback(self, callback: Callback_Base) -> None:
        if(not isinstance(callback, Callback_Base)):
            raise TypeError(f"callback must be a 'Callback_Base' subclass, got {type(callback)}")
        self.callbacks.append(callback)

    def _notify(self, event: str, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, f"on_{event}")(self, **kwargs)
            
    # Thin wrappers kept for API parity with the legacy Model_Base
    def notify_train_begin_callbacks(self, **kw): self._notify("train_begin", **kw)
    def notify_train_end_callbacks(self, **kw): self._notify("train_end", **kw)
    def notify_epoch_begin_callbacks(self, **kw): self._notify("epoch_begin", **kw)
    def notify_epoch_end_callbacks(self, **kw): self._notify("epoch_end", **kw)
    def notify_batch_begin_callbacks(self, **kw): self._notify("batch_begin", **kw)
    def notify_batch_end_callbacks(self, **kw): self._notify("batch_end", **kw)

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

    # --------------------------------------------------------------- physics
    def forward(self, row_start: int, row_end: int, **kwargs) -> np.ndarray:
        """Compute dZ rows for nodes [row_start, row_end).

        Must be implemented by the derived model. Returns an array of
        shape (row_end - row_start, n_dim).
        """
        raise NotImplementedError("forward(.) is not implemented")

    # ------------------------------------------------------------ integrator
    @staticmethod
    def Th(dZ: np.ndarray) -> float:
        """Convergence statistic: mean Euclidean norm of the dZ rows."""
        return float(np.linalg.norm(dZ, axis=-1).mean())

    def updateZ(self, lr: float | None = None) -> None:
        if lr is None:
            lr = self.lr
        if self.beta > 0.0:
            self.V *= self.beta
            self.V += self.dZ
            self.Z += lr * self.V
        else:
            self.Z += lr * self.dZ

    # ------------------------------------------------------------- main loop
    def embed(self,
              epochs: int = 100,
              lr: float | None = None,
              Z: np.ndarray | None = None,
              batch_count: int = 1,
              epsilon: float | None = None,
              **kwargs) -> None:
        """Run the relaxation.

        epochs      : maximum number of epochs
        lr          : learning rate override (defaults to self.lr)
        Z           : optional (n, d) array to continue an existing embedding
        start_epoch : first epoch index (for resuming)
        batch_count : number of contiguous row batches per epoch
        epsilon     : convergence threshold override (defaults to self.epsilon)
        """
        if Z is not None:
            self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        if self.Z is None:
            raise RuntimeError("Z is not initialized. Pass Z=... or initialize in the model.")
        if epsilon is None:
            epsilon = self.epsilon

        n, d = self.Z.shape
        if self.dZ is None:
            self.dZ = np.zeros((n, d), dtype=np.float64)
        if self.beta > 0.0 and self.V is None:
            self.V = np.zeros((n, d), dtype=np.float64)

        kwargs.update(epochs=epochs)
        self.stop_training = False
        self.notify_train_begin_callbacks(**kwargs)

        # contiguous row ranges: [0, b), [b, 2b), ...
        batch_count = max(1, min(int(batch_count), n))
        batch_size = (n + batch_count - 1) // batch_count
        bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
        start_epoch = self.end_epoch + 1
        for epoch in range(start_epoch, start_epoch + epochs):
            if self.stop_training:
                break
            kwargs.update(epoch=epoch, batch_count=len(bounds), batch_size=batch_size)
            self.notify_epoch_begin_callbacks(**kwargs)

            if self.verbosity >= 1:
                print(f"Epoch {epoch}/{epochs}", end="")

            self.dZ.fill(0.0)
            for b, (row_start, row_end) in enumerate(bounds):
                kwargs.update(batch=b + 1)
                self.notify_batch_begin_callbacks(**kwargs)
                if self.verbosity >= 3 and len(bounds) > 1:
                    print(f"  batch {b + 1}/{len(bounds)}")
                # ---------------- forward pass ----------------
                self.dZ[row_start:row_end] = self.forward(row_start, row_end, **kwargs)
                self.notify_batch_end_callbacks(**kwargs)

            dZ_norm_avg = self.Th(self.dZ)
            if self.verbosity >= 2:
                if len(bounds) > 1:
                    print(f" ({len(bounds)} batches)", end="")
                print(f"  dZ norm avg: {dZ_norm_avg:.6f}", end="")
            if self.verbosity >= 1:
                print("")

            self.updateZ(lr=lr)
            self.latest_epoch = epoch
            self.notify_epoch_end_callbacks(**kwargs)

            if epsilon is not None and dZ_norm_avg <= epsilon:
                if self.verbosity >= 1:
                    print(f"Converged at epoch {epoch}: Th(dZ)={dZ_norm_avg:.6g} <= {epsilon}")
                break

            self.end_epoch += 1  # advance end_epoch for resume

        self.notify_train_end_callbacks(**kwargs)
