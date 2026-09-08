# Embedding results, all graphs, d=128, seed 42

36 embeddings in `data_cache/embeddings/`. `hop R2` is the mlp regressor of `evaluator.dist_approx`.
Bold marks the best cell of its graph. **Scores are NOT comparable
across graphs** -- the hop sample and the pair budget differ by size.


## cora  (n = 2,708)

| method | setting | walks | acc | f1_score | auc | hop R2 | s | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepwalk | hier.softmax | 80x40x10 | 0.9744 | 0.9740 | 0.9980 | +0.6086 | 21 | 276 |
| fodiwalk+fdhop | walk_edges/plain | 10x20x5 | 0.9882 | 0.9882 | 0.9990 | **+0.7119** | 6 | 826 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x20x5 | 0.9882 | 0.9882 | 0.9990 | **+0.7119** | 13 | 916 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x80x10 | **0.9905** | 0.9905 | **0.9993** | +0.7064 | 11 | 846 |
| fodiwalk+fdlinear | nbr_walk/plain/k4=0.01 | 10x20x5 | 0.9730 | 0.9726 | 0.9954 | +0.2537 | 20 | 907 |
| fodiwalk+fdlinear | nbr_walk/velocity/k4=0.01 | 10x20x5 | 0.9711 | 0.9706 | 0.9949 | +0.2604 | 13 | 918 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x20x5 | 0.9692 | 0.9691 | 0.9936 | +0.7073 | 14 | 935 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x80x10 | 0.9664 | 0.9661 | 0.9935 | +0.6922 | 21 | 983 |
| fodiwalk+fdlinear | walk_edges/velocity/k4=0.01 | 10x20x5 | 0.9697 | 0.9695 | 0.9941 | +0.7103 | 6 | 890 |
| node2vec | q=0.5 | 10x80x10 | 0.9697 | 0.9692 | 0.9953 | +0.5301 | 5 | 276 |
| node2vec | q=1.0 | 10x80x10 | 0.9740 | 0.9737 | 0.9970 | +0.4674 | 4 | 277 |
| node2vec | q=2.0 | 10x80x10 | 0.9702 | 0.9697 | 0.9980 | +0.4056 | 5 | 278 |

## pubmed  (n = 19,717)

| method | setting | walks | acc | f1_score | auc | hop R2 | s | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepwalk | hier.softmax | 80x40x10 | 0.9736 | 0.9731 | 0.9978 | +0.5221 | 177 | 295 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x20x5 | **0.9915** | 0.9915 | 0.9989 | **+0.6194** | 41 | 1132 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x80x10 | 0.9912 | 0.9912 | **0.9992** | +0.6171 | 101 | 1084 |
| fodiwalk+fdlinear | nbr_walk/plain/k4=0.01 | 10x20x5 | 0.9851 | 0.9850 | 0.9986 | +0.4268 | 115 | 1108 |
| fodiwalk+fdlinear | nbr_walk/velocity/k4=0.01 | 10x20x5 | 0.9824 | 0.9822 | 0.9986 | +0.4371 | 115 | 1163 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x20x5 | 0.9712 | 0.9710 | 0.9955 | +0.5170 | 46 | 1088 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x80x10 | 0.9720 | 0.9719 | 0.9950 | +0.5176 | 159 | 1133 |
| fodiwalk+fdlinear | walk_edges/velocity/k4=0.01 | 10x20x5 | 0.9709 | 0.9707 | 0.9949 | +0.5197 | 40 | 1092 |
| node2vec | q=0.5 | 10x80x10 | 0.9667 | 0.9660 | 0.9960 | +0.2361 | 59 | 313 |
| node2vec | q=1.0 | 10x80x10 | 0.9680 | 0.9675 | 0.9961 | +0.2080 | 33 | 310 |
| node2vec | q=2.0 | 10x80x10 | 0.9667 | 0.9661 | 0.9959 | +0.1670 | 60 | 308 |

## wordnet  (n = 82,115)

| method | setting | walks | acc | f1_score | auc | hop R2 | s | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepwalk | hier.softmax | 80x40x10 | 0.9892 | 0.9891 | 0.9991 | +0.0647 | 671 | 344 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x20x5 | **0.9964** | 0.9964 | **0.9999** | +0.1339 | 161 | 1100 |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x80x10 | 0.9954 | 0.9954 | 0.9996 | +0.1279 | 272 | 1108 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x20x5 | 0.9929 | 0.9929 | 0.9990 | **+0.1841** | 158 | 1118 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x80x10 | 0.9930 | 0.9930 | 0.9992 | +0.1807 | 440 | 1121 |
| node2vec | q=0.5 | 10x80x10 | 0.9940 | 0.9940 | 0.9993 | +0.0678 | 223 | 356 |
| node2vec | q=1.0 | 10x80x10 | 0.9933 | 0.9933 | 0.9993 | +0.0614 | 175 | 356 |
| node2vec | q=2.0 | 10x80x10 | 0.9946 | 0.9946 | 0.9994 | +0.0649 | 228 | 357 |

## com_youtube  (n = 1,134,890)

| method | setting | walks | acc | f1_score | auc | hop R2 | s | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fodiwalk+fdhop | walk_edges/plain/k4=1.0 | 10x20x5 | **0.9856** | 0.9855 | **0.9983** | +0.4658 | 9256 | 1410 |
| fodiwalk+fdlinear | walk_edges/plain/k4=0.01 | 10x20x5 | 0.9681 | 0.9678 | 0.9941 | **+0.5420** | 8858 | 1591 |
| node2vec | q=0.5 | 10x80x10 | 0.9348 | 0.9316 | 0.9870 | +0.1650 | 4966 | 1065 |
| node2vec | q=1.0 | 10x80x10 | 0.9292 | 0.9255 | 0.9851 | +0.1375 | 4130 | 1063 |
| node2vec | q=2.0 | 10x80x10 | 0.9334 | 0.9302 | 0.9866 | +0.1019 | 5227 | 1067 |
