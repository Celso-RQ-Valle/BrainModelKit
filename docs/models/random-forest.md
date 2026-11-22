# Random forest: parameter reference

[Model examples](README.md#random-forest) | [Search-space syntax](../optimization.md#search-spaces) | [Training and refitting](../training.md)

Omit a key to keep its default; an empty list or an empty string does not request
the default. `[value]` fixes a scalar, and `[minimum, maximum]` tunes a numeric
interval. `None` is an explicit value, not an instruction to use the default.
The catalog below is commented out so that optional settings are not passed as
explicit nulls. Uncomment only the entries you want to adjust, and supply at
least one entry to `optimize`. For `train_model(model_params=...)`, use scalar
values directly, without the search-space brackets.

These catalogs include all explicit constructor parameters in the verified local
versions, except Spark columns managed by BrainModelKit and the internal Java
handle. Some parameters control execution or prediction rather than learning.
Defaults may change with dependency versions; the linked API explains each
parameter, its valid values, and compatibility constraints. Custom estimators
retain their configured values when an entry is omitted. Built-in model strings
use the optimization seed (42 by default) for `random_state` / `seed`.

## pandas

[Parameter meanings and valid combinations](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Catalog: 1.6.1.

```python
search_space = {
    # "n_estimators": [100],  # Default if omitted: 100
    # "criterion": ['gini'],  # Default if omitted: 'gini'
    # "max_depth": [None],  # Default if omitted: None
    # "min_samples_split": [2],  # Default if omitted: 2
    # "min_samples_leaf": [1],  # Default if omitted: 1
    # "min_weight_fraction_leaf": [0.0],  # Default if omitted: 0.0
    # "max_features": ['sqrt'],  # Default if omitted: 'sqrt'
    # "max_leaf_nodes": [None],  # Default if omitted: None
    # "min_impurity_decrease": [0.0],  # Default if omitted: 0.0
    # "bootstrap": [True],  # Default if omitted: True
    # "oob_score": [False],  # Default if omitted: False
    # "n_jobs": [None],  # Default if omitted: None
    # "random_state": [None],  # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "verbose": [0],  # Default if omitted: 0
    # "warm_start": [False],  # Default if omitted: False
    # "class_weight": [None],  # Default if omitted: None
    # "ccp_alpha": [0.0],  # Default if omitted: 0.0
    # "max_samples": [None],  # Default if omitted: None
    # "monotonic_cst": [None],  # Default if omitted: None
}
```

## Spark

[Parameter meanings and valid combinations](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html). Catalog: 4.0.2.

```python
search_space = {
    # "maxDepth": [5],  # Default if omitted: 5
    # "maxBins": [32],  # Default if omitted: 32
    # "minInstancesPerNode": [1],  # Default if omitted: 1
    # "minInfoGain": [0.0],  # Default if omitted: 0.0
    # "maxMemoryInMB": [256],  # Default if omitted: 256
    # "cacheNodeIds": [False],  # Default if omitted: False
    # "checkpointInterval": [10],  # Default if omitted: 10
    # "impurity": ['gini'],  # Default if omitted: 'gini'
    # "numTrees": [20],  # Default if omitted: 20
    # "featureSubsetStrategy": ['auto'],  # Default if omitted: 'auto'
    # "seed": [None],  # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "subsamplingRate": [1.0],  # Default if omitted: 1.0
    # "leafCol": [''],  # Default if omitted: ''
    # "minWeightFractionPerNode": [0.0],  # Default if omitted: 0.0
    # "weightCol": [None],  # Default if omitted: None
    # "bootstrap": [True],  # Default if omitted: True
}
```
