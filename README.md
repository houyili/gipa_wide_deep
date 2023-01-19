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
or just
```bash
./run_deep_wide.sh
```
the ogbn-protein dataset would be downloaded into current path.

## Performance

| Model              |Valid rocauc  | Test rocauc   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| GIPA     | 0.9478 | 0.8917 | 17.4M  | Tesla V100 (32GB GPU) |

## References
