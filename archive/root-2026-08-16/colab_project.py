# %%
from google.colab import drive
drive.mount('/content/drive')

print("Hi")
# %%
! pwd
! test -d forecedirected || git clone https://github.com/HessamLa/forcedirected.git
%cd forcedirected
! pip show forcedirected > /dev/null 2>&1 || pip install -e .
%cd /content
# %%
! test -d Multicore-TSNE || git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
! ls
%cd './Multicore-TSNE/'
! ls
! pip show MulticoreTSNE > /dev/null 2>&1 || pip install .
%cd /content

! pip install umap-learn
! pip install datashader bokeh holoviews scikit-image colorcet
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
from MulticoreTSNE import MulticoreTSNE
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.set(context='notebook',
        rc={'figure.figsize':(12,10)},
        palette=sns.color_palette('tab10', 10))

mnist = fetch_openml('Fashion-MNIST', version=1)

def data_size_scaling(algorithm, data, sizes=[100, 200, 400, 800, 1600], n_runs=5):
    result = []
    for size in sizes:
        for run in range(n_runs):
            subsample = resample(data, n_samples=size)
            start_time = time.time()
            algorithm.fit(subsample)
            elapsed_time = time.time() - start_time
            del subsample
            result.append((size, elapsed_time))
    return pd.DataFrame(result, columns=('dataset size', 'runtime (s)'))

# %%
# load the data and create the graph
digits, target = load_digits(return_X_y=True)
digits = load_digits()
reducer = UMAP(random_state=42)
reducer.fit(digits.data)
# %%

embedding = reducer.transform(digits.data)
plt.scatter(embedding[:,0], embedding[:,1], c=digits.target, cmap='Spectral', s=8, alpha=0.9)


from umap.utils import (
    ts,
    csr_unique,
    fast_knn_indices,
)
ts()
# %%

from graphmaking import make_graph, available_strategies

# Gx = make_graph(digits.data, "mst")
Gx = make_graph(digits.data, strategy="union_mst_nndescent", node_labels=digits.target, k_neighbors=7)
nx.draw(Gx, node_size=1, width=0.05)
# %%
from forcedirected.models import model_204_shell
from forcedirected.models import model_224_random_landmarks
# from forcedirected.models.model_214_targets import FDModel
# %%
n_dim = 2
epochs = 1000
device = 'cpu'
fdobj = model_204_shell.FDModel(Gx, n_dim, k3=1000)#, **kwargs)
# fdobj = model_224_random_landmarks.FDModel(Gx, n_dim)#, **kwargs)
fdobj.embed(epochs=epochs, device=device)
embedding = fdobj.get_embeddings()
plt.scatter(embedding[:,0], embedding[:,1], c=digits.target, cmap='Spectral', s=8, alpha=0.9)
plt.legend(digits.target)
plt.show()

# %%
fdobj.embed(epochs=epochs, device=device)
embedding = fdobj.get_embeddings()
plt.scatter(embedding[:,0], embedding[:,1], c=digits.target, cmap='Spectral', s=8, alpha=0.9)
# plt.scatter(embedding[:,0], embedding[:,1], cmap='Spectral', s=8, alpha=0.9)
plt.legend()
plt.show()

# %%
nx.draw(Gx, node_size=1, width=0.05, pos=embedding)


# %%
#################################################
# MNIST
import numpy as np
import sklearn.datasets
import umap
import umap.plot
data, labels = sklearn.datasets.fetch_openml(
    'mnist_784', version=1, return_X_y=True
)
# %%
mapper1 = umap.UMAP().fit(data)
umap.plot.points(mapper1, labels=labels)

# %%
from graphmaking import make_graph, available_strategies
Gx = make_graph(data, strategy="union_mst_nndescent", node_labels=labels, k_neighbors=7)
nx.draw(Gx, node_size=1, width=0.05)

from forcedirected.models import model_204_shell
from forcedirected.models import model_224_random_landmarks
# from forcedirected.models.model_214_targets import FDModel
# %%
n_dim = 2
epochs = 1000
device = 'cpu'
fdobj = model_204_shell.FDModel(Gx, n_dim, k3=1000)#, **kwargs)
# fdobj = model_224_random_landmarks.FDModel(Gx, n_dim)#, **kwargs)
fdobj.embed(epochs=epochs, device=device)
embedding = fdobj.get_embeddings()
plt.scatter(embedding[:,0], embedding[:,1], c=digits.target, cmap='Spectral', s=8, alpha=0.9)
plt.legend(digits.target)
plt.show()

# %%
