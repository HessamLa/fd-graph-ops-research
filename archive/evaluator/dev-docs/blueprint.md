This module is for the downstream ML tasks used to evaluate the quality of a graph embedding or dimensionality reduction operation.

For the evaluation, after we available libraries and frameworks such as scikit-learn 

The evaluation criteria are the following and can be extended
- **link prediction:** \
  In a graph, given two node embeddings, are they neighbors in the graph?

- **distance approximation:** \
  In a graph, given two node embeddings, what is their hop distance? 

- ...

The return is accuracy, precision, recall, f1 score, auc.

The module is in such a way that can be used as a tool 

```
.venv/bin/python -m evaluator <input graph> <input embeddings> --link-prediction(split=20, negposratio=1,...) --dist-aproximation(...) --results=<resultfile>
```

Or can be imported and called through APIs

```
import evaluator as eval
...
lp_result = eval.link_prediction(G, embeddings, ...)
da_result = eval.dist_approxmation(G, embeddings, ...)
```