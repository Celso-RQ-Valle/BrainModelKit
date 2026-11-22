# Logistic regression: parameter reference

[Model examples](README.md#logistic-regression) | [Search-space syntax](../optimization.md#search-spaces) | [Training and refitting](../training.md)

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

[Parameter meanings and valid combinations](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html). Catalog: 1.6.1.

```python
search_space = {
    # "penalty": ['l2'],  # Default if omitted: 'l2'
    # "dual": [False],  # Default if omitted: False
    # "tol": [0.0001],  # Default if omitted: 0.0001
    # "C": [1.0],  # Default if omitted: 1.0
    # "fit_intercept": [True],  # Default if omitted: True
    # "intercept_scaling": [1],  # Default if omitted: 1
    # "class_weight": [None],  # Default if omitted: None
    # "random_state": [None],  # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "solver": ['lbfgs'],  # Default if omitted: 'lbfgs'
    # "max_iter": [100],  # Default if omitted: 100
    # "multi_class": ['deprecated'],  # Default if omitted: 'deprecated'
    # "verbose": [0],  # Default if omitted: 0
    # "warm_start": [False],  # Default if omitted: False
    # "n_jobs": [None],  # Default if omitted: None
    # "l1_ratio": [None],  # Default if omitted: None
}
```

## Spark

[Parameter meanings and valid combinations](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html). Catalog: 4.0.2.

```python
search_space = {
    # "maxIter": [100],  # Default if omitted: 100
    # "regParam": [0.0],  # Default if omitted: 0.0
    # "elasticNetParam": [0.0],  # Default if omitted: 0.0
    # "tol": [1e-06],  # Default if omitted: 1e-06
    # "fitIntercept": [True],  # Default if omitted: True
    # "threshold": [0.5],  # Default if omitted: 0.5
    # "thresholds": [None],  # Default if omitted: None
    # "standardization": [True],  # Default if omitted: True
    # "weightCol": [None],  # Default if omitted: None
    # "aggregationDepth": [2],  # Default if omitted: 2
    # "family": ['auto'],  # Default if omitted: 'auto'
    # "lowerBoundsOnCoefficients": [None],  # Default if omitted: None
    # "upperBoundsOnCoefficients": [None],  # Default if omitted: None
    # "lowerBoundsOnIntercepts": [None],  # Default if omitted: None
    # "upperBoundsOnIntercepts": [None],  # Default if omitted: None
    # "maxBlockSizeInMB": [0.0],  # Default if omitted: 0.0
}
```

For the scaling pipeline in the model guide, prefix every pandas key with
`logisticregression__`. That example explicitly sets `max_iter=1000`, so omitting
its search entry keeps 1000 rather than the constructor default of 100. Solver,
penalty, dual, and elastic-net settings must be compatible; consult the API above.
