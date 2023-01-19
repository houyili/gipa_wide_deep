# gipa_wide_deep

## Dependencies


## Basic commandline arguments
$DARA_DIR : Your data path to store ogbn-proteins data sets

## Commandline
```bash
python -u ./train_gipa.py --root $DARA_DIR --advanced-optimizer --use-sparse-fea 
```

## Performance

| Model              |Valid rocauc  | Test rocauc   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| GIPA     | 0.9478 | 0.8917 | 17.4M  | Tesla V100 (32GB GPU) |

## References
