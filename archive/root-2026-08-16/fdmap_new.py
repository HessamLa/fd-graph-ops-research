# %%
"""fdmap_new.py -- fdge2 rewrite of fdge_numba/famap.py.

Same workflow (load digits -> build a kNN/MST graph -> force-directed
embed to 2D), rewritten against the new `fdge2` three-stage pipeline
instead of the old `graphmaking` + `forcedirected_numba` modules. See
fdge2/docs/ARCHITECTURE.md for the three-stage design this follows.

Run: python fdmap_new.py   (from the repo root, fdmap/)
"""
import numpy as np
import networkx as nx
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# %%

data, target = load_digits(return_X_y=True)

# %%
## ---------------------------------
## Graph Building
## ---------------------------------
from fdge2 import graph_building

Gx = graph_building.make_graph(
    data, strategy="union_mst_nndescent", node_labels=target, k_neighbors=9)
# nx.draw(Gx, node_size=1, width=0.05, )

nx.draw(Gx, node_size=1, width=0.05, node_color=target, cmap='Spectral')
# %%
## ---------------------------------
## Graph Embedding
## ---------------------------------
from fdge2.core import Callback_Base
from fdge2.models import ReferenceFDModel, EuclideanDistanceModel


_snapshots = {}  # epoch -> (n, n_dim) embedding snapshot, filled every 10 epochs


class Rec(Callback_Base):
    # def on_train_begin(self, model, **kw): events.append("tb")
    def on_epoch_end(self, model, **kw):
        """Callback to print epoch info and stash a snapshot every 10 epochs."""
        print(f"epoch {kw['epoch']:4d} ||dZ|| avg: {np.linalg.norm(model.dZ, axis=1).mean():.3f}")
        if kw['epoch'] % 10 == 0:  # snapshot every 10 epochs
            _snapshots[kw['epoch']] = model.Z.copy()

    def on_train_end(self, model, **kw):
        # One np.savez call, many named arrays -- NOT repeated np.savez(f, ...)
        # calls on a file opened in "ab" mode. Each savez call writes an
        # independent zip archive; concatenating them corrupts the offsets,
        # so np.load on the result raises BadZipFile. Writing once here with
        # all snapshots as separate keys is what actually produces a valid,
        # multi-snapshot embeddings.npz.
        np.savez("embeddings.npz",
                 **{f"epoch_{e:04d}": Z for e, Z in _snapshots.items()})
        print(f"[fdmap_new] saved {len(_snapshots)} snapshots to embeddings.npz")


# fdobj = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
# fdobj = ReferenceFDModel(n_dim=2, verbosity=0, seed=42)
fdobj = EuclideanDistanceModel(n_dim=2, verbosity=0, seed=42)
fdobj.attach_callback(Rec())
# %%
fdobj.embed(Gx, epochs=2000, verbosity=3)
embedding = fdobj.get_embeddings()

plt.scatter(embedding[:,0], embedding[:,1], c=target, cmap='Spectral', s=8, alpha=0.9)
plt.legend(np.unique(target))
plt.show()

# # %%
# nx.draw(Gx, node_size=1, width=0.05, node_color=target, cmap='Spectral', pos=embedding)
#generate a motion gif
import matplotlib.animation as animation

snapshots = np.load("embeddings.npz")
epoch_keys = sorted(snapshots.files, key=lambda k: int(k.split("_")[1]))

fig, ax = plt.subplots(figsize=(6, 6))


def _draw_frame(i):
    ax.clear()
    Z_i = snapshots[epoch_keys[i]]
    pos = dict(zip(Gx.nodes(), Z_i))
    nx.draw(Gx, pos=pos, ax=ax, node_size=8, width=0.05,
            node_color=target, cmap="Spectral")
    ax.set_title(epoch_keys[i].replace("_", " "))


ani = animation.FuncAnimation(fig, _draw_frame, frames=len(epoch_keys), interval=200)
ani.save("embedding_motion.gif", writer=animation.PillowWriter(fps=5))
plt.close(fig)
print(f"[fdmap_new] saved motion gif with {len(epoch_keys)} frames to embedding_motion.gif")

# %%

# %%
if __name__ == "__main__":
    print(f"[fdmap_new] embedding shape: {embedding.shape}, "
          f"finite: {np.isfinite(embedding).all()}")

    import os
    out_path = os.environ.get("FDMAP_NEW_OUT")
    if out_path:
        np.savez(out_path, embedding=embedding, target=target,
                 edges=np.array(list(Gx.edges())))
        print(f"[fdmap_new] saved embedding + graph to {out_path}")
