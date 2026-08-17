# %%
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
import time

from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

# %%

data, target = load_digits(return_X_y=True)
# data = digits.data

# %%
from graphmaking import make_graph, available_strategies
# %%
## ---------------------------------
## Graph Building
## ---------------------------------
# Gx = make_graph(digits.data, "mst")
Gx = make_graph(data, strategy="union_mst_nndescent", node_labels=target, k_neighbors=7)
nx.draw(Gx, node_size=1, width=0.05)

# %%
## ---------------------------------
## Graph Embedding
## ---------------------------------
from forcedirected_numba.ForceDirected import Callback_Base
from forcedirected_numba.ForceDirected import ForceDirected
from forcedirected_numba.model_204_shell import (
    FDModel, graph_to_csr, get_hops_csr, _forces_204_hidx)

class Rec(Callback_Base):
    # def on_train_begin(self, model, **kw): events.append("tb")
    def on_epoch_end(self, model, **kw): 
        """Callback to print epoch info."""
        print(f"epoch {kw['epoch']:4d} ||dZ|| avg: {np.linalg.norm(model.dZ).mean():.3f}, "
            f"mean: {np.abs(model.dZ).mean():.8f}")

    # def on_train_end(self, model, **kw): events.append("te")



# fdobj = FDModel(Gx, n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
fdobj = FDModel(Gx, n_dim=2, verbosity=0, seed=42)
# fdobj.attach_callback(Rec())

t0=time.time()
fdobj.embed(epochs=2000, verbosity=3)
t1=time.time()
print(f"[fdmap_new] embed() finished in {t1-t0:.3f}s")
embedding = fdobj.get_embeddings()
# plt.scatter(embedding[:,0], embedding[:,1], c=target, cmap='Spectral', s=8, alpha=0.9)
# plt.legend(np.unique(target))
# plt.show()

# # %%
# nx.draw(Gx, node_size=1, width=0.05, node_color=target, cmap='Spectral', pos=embedding)