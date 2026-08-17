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
from fdge_jax import graph_building

Gx = graph_building.make_graph(
    data, strategy="union_mst_nndescent", node_labels=target, k_neighbors=9)
# nx.draw(Gx, node_size=1, width=0.05, )

nx.draw(Gx, node_size=1, width=0.05, node_color=target, cmap='Spectral')
# %%
## ---------------------------------
## Graph Embedding
## ---------------------------------
from fdge_jax.core import Callback_Base
from fdge_jax.models import ReferenceFDModel, EuclideanDistanceModel


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
ani.save("embedding_motion_jax.gif", writer=animation.PillowWriter(fps=5))
plt.close(fig)
print(f"[fdmap_new] saved motion gif with {len(epoch_keys)} frames to embedding_motion.gif")

# %%
# to interactive html.
# and html, with all the nodes and the edges, 
# position of nodes from the embedding, and color of nodes from the target labels.
# mouse hover on each node: show its original image (8x8) from the digits dataset.

# %%
# to interactive html.
# nodes positioned by the embedding, colored by target label;
# hovering a node shows its original 8x8 digit image.
import base64
from io import BytesIO

from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.resources import INLINE

# --- node order guard -------------------------------------------------
# Everything below (and the gif cell above) assumes row i of `embedding`
# describes node i of Gx. Make that explicit instead of trusting the
# iteration order of Gx.nodes(): a silent mismatch here does not raise,
# it just produces a plausible-looking but wrong picture.
node_list = list(Gx.nodes())
row_of = {n: i for i, n in enumerate(node_list)}
assert node_list == list(range(len(data))), \
    "Gx node keys are not 0..n-1; embedding rows cannot be matched by position"

# labels: prefer whatever make_graph stored on the graph, fall back to target
labels = [str(Gx.nodes[n].get("label", target[row_of[n]])) for n in node_list]

# --- 8x8 digit thumbnails as base64 PNG data URIs ---------------------
def _thumb(row):
    buf = BytesIO()
    plt.imsave(buf, row.reshape(8, 8), cmap="gray_r",
               vmin=0, vmax=16, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

imgs = [_thumb(data[row_of[n]]) for n in node_list]

# --- data sources ------------------------------------------------------
nodes_src = ColumnDataSource(dict(
    x=embedding[:, 0], y=embedding[:, 1],
    label=labels, img=imgs, idx=list(range(len(node_list))),
))

# one multi_line glyph for all edges, drawn once from the final embedding
edge_xs, edge_ys = [], []
for u, v in Gx.edges():
    i, j = row_of[u], row_of[v]
    edge_xs.append([embedding[i, 0], embedding[j, 0]])
    edge_ys.append([embedding[i, 1], embedding[j, 1]])
edges_src = ColumnDataSource(dict(xs=edge_xs, ys=edge_ys))

# --- figure ------------------------------------------------------------
p = figure(width=900, height=900, title="fdmap embedding (digits)",
           tools="pan,wheel_zoom,box_zoom,reset,save",
           active_scroll="wheel_zoom", match_aspect=True)
p.grid.visible = False
p.axis.visible = False

p.multi_line("xs", "ys", source=edges_src,
             line_color="#999999", line_width=0.3, line_alpha=0.25)

node_r = p.scatter("x", "y", source=nodes_src, size=7,
                   line_color=None,
                   fill_color=factor_cmap("label", palette=Spectral10,
                                          factors=sorted(set(labels))),
                   legend_field="label")
p.legend.title = "digit"
p.legend.click_policy = "hide"

# hover on nodes only -- attaching it to the edge glyph as well makes the
# tooltip flicker as the cursor crosses line segments
p.add_tools(HoverTool(renderers=[node_r], tooltips="""
    <div style="padding:4px">
      <img src="@img" style="width:96px;height:96px;image-rendering:pixelated">
      <div><b>digit:</b> @label</div>
      <div><b>node:</b> @idx</div>
    </div>
"""))

output_file("embedding_interactive.html",
            title="fdmap embedding", mode="inline")
save(p, resources=INLINE)   # INLINE -> single self-contained file, works offline
print("[fdmap_new] wrote embedding_interactive.html")


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
