# gipa_wide_deep

## Dependencies
+ cuda=10.2
+ pytorch=1.12.1 
+ torch-cluster=1.6.0+pt112cu102
+ torch-scatter=2.1.0+pt112cu102 
+ torch-sparse=0.6.16+pt112cu102 
+ dgl-cu102=0.9.1.post1
+ ogb=1.3.5
+ numpy=1.20.2

## Basic commandline arguments
$DARA_DIR : Your data path to store ogbn-proteins dataset

## Commandline
```bash
python -u ./train_gipa.py --root $DARA_DIR --advanced-optimizer --use-sparse-fea 
```
the ogbn-protein dataset would be downloaded into $DARA_DIR.

Or just
```bash
./run_deep_wide.sh
```
the ogbn-protein dataset would be downloaded into current path.

## Performance

| Model              |Valid rocauc  | Test rocauc   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| GIPA     | 0.9478 | 0.8917 | 17.4M  | Tesla V100 (32GB GPU) |

## References
+ Houyi Li, Zhihong Chen, Zhao Li, Qinkai Zheng, Peng Zhang, Shuigeng Zhou. (2023). GIPA++: A General Information Propagation Algorithm for Graph Learning
 https://arxiv.org/abs/2301.08209
+ Qinkai Zheng, Houyi Li, Peng Zhang, Zhixiong Yang, Guowei Zhang, Xintan Zeng, Yongchao Liu. (2021). GIPA: General Information Propagation Algorithm for Graph Learning. https://arxiv.org/abs/2105.06035