Paper

Now we want to finalize fodiwalk, with fdhop as its force function. Using the name fdhop, and only use the force function as an integral part of the method.

Before all, write a paper about fodiwalk under @papers/fodiwalk/ in markdown format. Describe the walk method and the force calculation. Talk about the gradient and lr values. Finally, cover the metrology and report the results.



Discuss our forcedirected graph embedding consisting of graph augmentation and graph embedding. 

Describe the objectives in graph augmentation and how we use walks, and what graph is passed down to graph embedding.

Desribe the objectives of graph embedding

In graph augmentation, we use a walk method to find nodes for pairing them up with a source node, $u$, to calculate the force observed from them by $u$. In other words, during graph augmentation, we generate a directed asymetric weighted graph. The weight of each edge from $v$ to $u$ is used as a parameter in the force function from $v$ to $u$. Also, all immediate neighbors of $u$ form a directed edge to and from it. Point out the value of weight on each edge.

In graph embeddging we use the 
