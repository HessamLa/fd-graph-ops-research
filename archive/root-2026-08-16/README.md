FDMap is a dimentionality reduction method, which borrows from the force-directed graph embedding approach available at `https://github.com/HessamLa/forcedirected/` or `/home/h/gnn/fd-graph-embedding/forcedirected`. Basically, we would perform dimensionality reduction using an embedding method.

n is the available data points
m is the dimensions of each data point
d is the target or embedding space dimensions, such that d << m

The reduction $n \times m \rightarrow n \times d$

## Steps

1. Load the data and build a weighted graph by connecting nodes using a kNN algorithm.

    $d(u,v)$ or $d_{uv}$ is a positive value and is the distance between the nodes $u$ and $v$ in the original space, we can start with the Euclidean distance. The distance metric in the original space can differ. Nevertheless, the original distances will be computed only once and will be constants to the FDMap.

    If two nodes are connected in the kNN generated graph, then weight of the edge is $d_{uv}$.

1. Generate a sparse distance matrix $D \in R^{n \times n}$ such that $D[i,j] = shortest weighted path between u_i and u_j$.
   
   We can use different policies for generating $D$ which will be discussed later.
   
   Some other constant matrices also exist, such as $C1$ and $C2$, whose elements correspond to that of $D$.
   The calculations only consider the indices with non-zero values in $D$.

1. Initialize the embedding values $Z \in R^{n \times d}$ using random values.

1. Initialize the gradient values $dZ \in R^{n \times d}$ to 0.
   
1. Run the following algorithm:

   1. $dZ \in R^{n \times d} = some large values such as 1_000_000$
   2. $\text {For}  0 \le i < n:$ # LOOP STARTS
      1. $F[i] = \sum_{j} F_{ij}(Z[i],Z[j])$: \
         
         Calculate per the following over all available $j$ such that $D[i,j] != 0$ \
         $z_{diff} = Z[j] - Z[i]$ \
         $z_{norm} = ||z_{diff}||$ \
         $z_{unit} = z_{diff}/z_{norm}$ \
         $f_{ij-attr} = C1[i,j].z_{norm}.e^{-D[i,j]}$ # scalar attractive force \
         $f_{ij-repl} = C2[i,k].D[i,j].e^{-z_{norm}}$ # scalar repulsive force \
         $F_{ij} = z_{unit}*f_{ij}$ # force vector between them \
         
         Finally: $F[i] = \sum_{j} F_{ij}$
       2. $dZ[i] = \beta dZ[i] + \eta F[i]$ # $0 \le \beta < 1$
    4. $Z = Z + dZ$
    5. $Th(dZ) > \epsilon$ Go to "LOOP START" # $Th:R^{n \times d}\rightarrow R$ is a function that calculates the epsilong. In the simlest form, it averages over all $dZ$

## Requirements

**This code must be developed for research. Therefore, readability and developer experience is an utmost priority.**

The section for making the graph must be a function like the one described in `/home/h/gnn/fd-graph-embedding/fdmap/graphmaking/README.md` 

The section for generating the sparce $D$ matrix must be a different function. The functions `f_{ij-attr}` and `f_{ij-repl}` are tentative and can be changed. It must be very easy for a researcher to modify the code block to try different functions. See the functions under `/home/h/gnn/fd-graph-embedding/forcedirected/forcedirected/models`.

But the most important section of this project is the inner loop to calculate $dZ[i]$ values. This section can be implemented using parallel or vectorization schemes. The function for handling the loop must also accept a `batch_count` parameter, and calculate in batches. 

Consider the following function from the file `/home/h/gnn/fd-graph-embedding/forcedirected/forcedirected/models/ForceDirected.py` as a reference. Omit the function `optimize_batch_count` at this moment. 

```
    @torch.no_grad()
    def embed(self, epochs=100, device='cpu', row_batch_size='auto', lr=None, Z=None, start_epoch=1, **kwargs):
        """
        Train the model to embed the graph.
        """
        # train begin
        if(start_epoch > epochs):
            raise ValueError(f"start_epoch should be <= epochs. start_epoch: {start_epoch}, epochs: {epochs}")
        kwargs = rns(kwargs)
        kwargs.epochs = epochs
        kwargs.start_epoch = start_epoch
        
        self.notify_train_begin_callbacks(**kwargs)

        # continue on an existing embedding
        if(Z is not None): 
            self.Z = Z 
        
        if(self.dZ is None):
            self.dZ = torch.nn.Parameter(
                        # torch.zeros_like(self.Z, device=device),
                        torch.zeros_like(self.Z),
                        requires_grad=False)            
        
        self.to(device)

        from forcedirected.utilities import optimize_batch_count
        @optimize_batch_count(max_batch_count=self.Z.shape[0])
        def run_batches(batch_count=1, **kwargs):
            kwargs = rns(kwargs)
            kwargs.batch_count = batch_count
            # print(f"run_batches: batch count: {kwargs['batch_count']}")
            n = self.Z.shape[0]
            batch_size = int(n/batch_count + 0.5) # ceiling of total/count
            for i, bmask in enumerate (batchify(list(range(n)), batch_size=batch_size)):
                # batch begin
                kwargs.batch = i+1
                kwargs.batch_size = batch_size
                self.notify_batch_begin_callbacks(**kwargs)
                if(self.verbosity >=3 and kwargs.batch_count > 1):
                    print(f"  batch {kwargs.batch}/{kwargs.batch_count}")
                ###################################
                # this is the forward pass
                self.dZ[bmask] = self.forward(bmask, **kwargs)
                # batch ends
                self.notify_batch_end_callbacks(**kwargs)    
            return batch_count, batch_size
        
        for epoch in range(start_epoch, epochs+1):
            if(self.stop_training): break
            
            # epoch begin
            # kwargs['epoch'] = epoch  
            kwargs.epoch = epoch
            self.notify_epoch_begin_callbacks(**kwargs)
            if(self.verbosity>=1): 
                print(f"Epoch {epoch}/{epochs}", end='')
            self.dZ.zero_() # fill self.dZ with zeros

            # kwargs['batch_count'], kwargs['batch_size'] = run_batches(**kwargs)
            kwargs.batch_count, kwargs.batch_size = run_batches(**kwargs)
            # get the average dZ
            dZ_norm_avg = torch.norm(self.dZ, dim=-1).mean().item()
            if(self.verbosity>=2):
                if(kwargs.batch_count > 1):
                    print(f"({kwargs.batch_count} batches)", end='')
                print(f"  dZ norm avg: {dZ_norm_avg:.4f}", end='')
                print("")
            self.updateZ(lr=lr)
            self.notify_epoch_end_callbacks(**kwargs)
            # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass
```

At this moment, we want to implement it using Numba.

Simplicy is very important.

The development of this project must be performed at different stages. After each stage, the developed code must be used and evaluated by a human user. After user's verification, we might move to develop the next stage.