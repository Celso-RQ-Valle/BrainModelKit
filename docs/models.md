# Models and hyperparameter optimization

[Main workflow](../README.md) | [Optimization contract](optimization.md)

All built-in models below are binary classifiers: targets are 0/1 and predictions
provide a class-1 probability. Logistic regression is a classifier here; the
optimization API does not currently support continuous-target regression metrics.

## Model and parameter index

Choose your backend, then open the official parameter reference directly. Model
names jump to search-space examples on this page. Versions below identify the
linked documentation, not a guarantee of compatibility with every installed
version. For `latest` or `stable` references, select your installed version in
the upstream documentation before checking defaults.

### Spark parameters

| Model / implementation | Documentation version | Parameter reference |
| --- | --- | --- |
| [Logistic regression](#logistic-regression) | Spark latest | [Official parameters](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html) |
| [Decision tree](#decision-tree) | Spark latest | [Official parameters](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html) |
| [Random forest](#random-forest) | Spark latest | [Official parameters](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html) |
| [Gradient boosting](#gradient-boosting) | Spark latest | [Official parameters](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html) |
| [LightGBM (SparkLightGBM)](#lightgbm) | SparkLightGBM 0.9.2 / LightGBM stable | [Bridge API](https://pypi.org/project/sparklightgbm/0.9.2/), [native parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) |
| [LightGBM (SynapseML)](#lightgbm) | SynapseML 1.0.14 | [Official parameters](https://mmlspark.blob.core.windows.net/docs/1.0.14/pyspark/synapse.ml.lightgbm.html#synapse.ml.lightgbm.LightGBMClassifier.LightGBMClassifier) |

### pandas parameters

| Model / implementation | Documentation version | Parameter reference |
| --- | --- | --- |
| [Logistic regression](#logistic-regression) | scikit-learn 1.6.1 | [Official parameters](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html) |
| [Decision tree](#decision-tree) | scikit-learn 1.6.1 | [Official parameters](https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html) |
| [Random forest](#random-forest) | scikit-learn 1.6.1 | [Official parameters](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| [Gradient boosting](#gradient-boosting) | scikit-learn 1.6.1 | [Official parameters](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| [LightGBM](#lightgbm) | LightGBM stable | [Official parameters](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) |

## Optimization example

Use `[minimum, maximum]` for intervals: `[200, 1000]` samples integers throughout
that interval, including its bounds. If either bound is a float, sampling is
continuous; `[100]` fixes a parameter. Trials explore combinations within
the budget rather than exhaustively testing every possible value. See
[search-space rules](optimization.md#search-spaces) for categorical choices
and logarithmic sampling.

Choose one model and its search space below, then pass them to `optimize`:

### Spark

```python
from brainmodelkit.optimization.pyspark import optimize
```

### Pandas

```python
from brainmodelkit.optimization.pandas import optimize
```

### Run the search

After choosing the import for your backend, run the same call with its DataFrames.

```python
# Define model and search_space using one backend-specific example below.
tuning = optimize(
    train_df=train_df,
    oot_df=validation_df,
    target_col="target",
    feature_cols=selected_features,
    model=model,
    search_space=search_space,
    objective_metric="oot_auc",
    direction="maximize",
    target_metric="train_oot_ks_abs_shift",
    target_limit=0.05,
    target_operator="<=",
    n_trials=20,
    cv=3,
    save_trials_path="optimization_trials.csv",
)
if not tuning.constraint_satisfied:
    raise ValueError("No feasible trial; inspect tuning.trials before final training.")
print(tuning.best_params)
print(tuning.trials_path)  # CSV with each trial's parameters and metrics.
```

Install `brainmodelkit[pandas,optimization]` or
`brainmodelkit[pyspark,optimization]`. Pass development validation as `oot_df`
during optimization; keep final OOT untouched until the configuration is frozen.
Refit with `train_model(..., model=model, model_params=tuning.best_params)` using
the same model or pipeline and backend. See the main README for the full refit
and [trial comparison](optimization.md#save-and-compare-trials) to choose a
configuration yourself from the saved DataFrame.

Omit a hyperparameter entry to keep its default; do not leave an empty list.
The index above links directly to official parameter names, defaults, and
explanations. Built-in model strings use the optimization seed (42 by default)
for `random_state` / `seed`; custom estimators retain their configured values. Examples below show starting ranges,
not recommended settings for every dataset. See [training](training.md) for
refitting, [model selection](model_selection.md) for CV, and
[evaluation](evaluation.md) for metric definitions.

Each example lists the explicit constructor parameters for scikit-learn 1.6.1,
PySpark 4.0.2, LightGBM 4.6.0, and the installed SynapseML wrapper. Spark's
managed feature, label, prediction, probability, and raw-prediction columns and
SynapseML's internal `java_obj` are excluded: BrainModelKit configures those.
The SparkLightGBM example lists common native training parameters; its open-ended
native keyword arguments are documented in the official reference above.

Parameter lines come first, with defaults and explanations on the next comment
line. Optional entries remain commented out.
Uncomment only settings you need. `[value]` fixes a value; `[low, high]` tunes a
numeric interval. `[False, True]` samples either boolean (also accepted in reverse
order); `[False]` or `[True]` fixes it. `None` is an explicit value, not a request
for the default.
Structured values use `lambda trial: ...` because a list normally describes a
search interval. Some settings require compatible solvers, objectives, or input
columns; the full catalog is not a configuration to enable all at once.
For `train_model(model_params=...)`, use scalar values rather than intervals.
Early stopping needs estimator-specific validation support; `oot_df` is not
automatically passed to the estimator as an early-stopping dataset.

## Logistic regression

A linear classifier. Scale numeric inputs inside a pandas pipeline so that each
CV fold learns its own scaling. Smaller `C` means stronger regularization;
larger Spark `regParam` means stronger regularization. They are not interchangeable.

### Spark

```python
# Spark: built-in standardization is enabled by default.
model = "logistic_regression"
search_space = {
    "maxIter": [100],  # Default if omitted: 100; max number of iterations (>= 0)
    "regParam": [
        0.001,
        1.0,
    ],  # Default if omitted: 0.0; regularization parameter (>= 0)
    "fitIntercept": [
        False,
        True,
    ],  # Default if omitted: True; whether to fit an intercept term
    # "elasticNetParam": [0.0], # Default if omitted: 0.0; the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty
    # "tol": [1e-06], # Default if omitted: 1e-06; the convergence tolerance for iterative algorithms (>= 0)
    # "threshold": [0.5], # Default if omitted: 0.5; Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g....
    # "thresholds": [None], # Default if omitted: None; Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to...
    # "standardization": [False, True], # Default if omitted: True; whether to standardize the training features before fitting the model
    # "weightCol": [None], # Default if omitted: None; weight column name. If this is not set or empty, we treat all instance weights as 1.0
    # "aggregationDepth": [2], # Default if omitted: 2; suggested depth for treeAggregate (>= 2)
    # "family": ['auto'], # Default if omitted: 'auto'; The name of family which is a description of the label distribution to be used in the model. Supported options: auto,...
    # "lowerBoundsOnCoefficients": [None], # Default if omitted: None; The lower bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the...
    # "upperBoundsOnCoefficients": [None], # Default if omitted: None; The upper bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the...
    # "lowerBoundsOnIntercepts": [None], # Default if omitted: None; The lower bounds on intercepts if fitting under bound constrained optimization. The bounds vector size must beequal with 1 for...
    # "upperBoundsOnIntercepts": [None], # Default if omitted: None; The upper bounds on intercepts if fitting under bound constrained optimization. The bound vector size must be equal with 1 for...
    # "maxBlockSizeInMB": [0.0], # Default if omitted: 0.0; maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size...
}
```

### Pandas

```python
# pandas
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search_space = {
    "logisticregression__C": [
        0.001,
        10.0,
    ],  # Default if omitted: 1.0; Inverse of regularization strength; must be a positive float
    "logisticregression__fit_intercept": [
        False,
        True,
    ],  # Default if omitted: True; Specifies if a constant (a.k.a
    # "logisticregression__penalty": ['l2'], # Default if omitted: 'l2'; Specify the norm of the penalty:
    # "logisticregression__dual": [False, True], # Default if omitted: False; Dual (constrained) or primal (regularized, see also :ref:`this equation <regularized-logistic-loss>`) formulation
    # "logisticregression__tol": [0.0001], # Default if omitted: 0.0001; Tolerance for stopping criteria
    # "logisticregression__intercept_scaling": [1], # Default if omitted: 1; Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True
    # "logisticregression__class_weight": [None], # Default if omitted: None; Weights associated with classes in the form ``{class_label: weight}``
    # "logisticregression__random_state": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the data
    # "logisticregression__solver": ['lbfgs'], # Default if omitted: 'lbfgs'
    # "logisticregression__max_iter": [1000], # Configured here: 1000; upstream default: 100; Maximum number of iterations taken for the solvers to converge
    # "logisticregression__multi_class": ['deprecated'], # Default if omitted: 'deprecated'; If the option chosen is 'ovr', then a binary problem is fit for each label
    # "logisticregression__verbose": [0], # Default if omitted: 0; For the liblinear and lbfgs solvers set verbose to any positive number for verbosity
    # "logisticregression__warm_start": [False, True], # Default if omitted: False; When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    # "logisticregression__n_jobs": [None], # Default if omitted: None; Number of CPU cores used when parallelizing over classes if multi_class='ovr'"
    # "logisticregression__l1_ratio": [None], # Default if omitted: None; The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``
}
```

## Decision tree

A single tree. Depth limits complexity; minimum leaf size limits small leaves.
Use an estimator object in pandas so the same `model` also works with
`train_model` (the string `"decision_tree"` works in pandas optimization/CV only).

### Spark

```python
# Spark
model = "decision_tree"
search_space = {
    "maxDepth": [
        3,
        8,
    ],  # Default if omitted: 5; Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in...
    "minInstancesPerNode": [
        10,
    ],  # Default if omitted: 1; Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than...
    # "maxBins": [32], # Default if omitted: 32; Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature
    # "minInfoGain": [0.0], # Default if omitted: 0.0; Minimum information gain for a split to be considered at a tree node
    # "maxMemoryInMB": [256], # Default if omitted: 256; Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its...
    # "cacheNodeIds": [False, True], # Default if omitted: False; If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs...
    # "checkpointInterval": [10], # Default if omitted: 10; set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10...
    # "impurity": ['gini'], # Default if omitted: 'gini'; Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini
    # "seed": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); random seed
    # "weightCol": [None], # Default if omitted: None; weight column name. If this is not set or empty, we treat all instance weights as 1.0
    # "leafCol": [''], # Default if omitted: ''; Leaf indices column name. Predicted leaf index of each instance in each tree by preorder
    # "minWeightFractionPerNode": [0.0], # Default if omitted: 0.0; Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the...
}
```

### Pandas

```python
# pandas
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
search_space = {
    "max_depth": [
        3,
        8,
    ],  # Default if omitted: None (unlimited); The maximum depth of the tree
    "min_samples_leaf": [
        10,
        50,
    ],  # Default if omitted: 1; The minimum number of samples required to be at a leaf node
    # "criterion": ['gini'], # Default if omitted: 'gini'; The function to measure the quality of a split
    # "splitter": ['best'], # Default if omitted: 'best'; The strategy used to choose the split at each node
    # "min_samples_split": [2], # Default if omitted: 2; The minimum number of samples required to split an internal node:
    # "min_weight_fraction_leaf": [0.0], # Default if omitted: 0.0; The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
    # "max_features": [None], # Default if omitted: None; The number of features to consider when looking for the best split:
    # "random_state": [42], # Configured here: 42; upstream default: None; Controls the randomness of the estimator
    # "max_leaf_nodes": [None], # Default if omitted: None; Grow a tree with ``max_leaf_nodes`` in best-first fashion
    # "min_impurity_decrease": [0.0], # Default if omitted: 0.0; A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    # "class_weight": [None], # Default if omitted: None; Weights associated with classes in the form ``{class_label: weight}``
    # "ccp_alpha": [0.0], # Default if omitted: 0.0; Complexity parameter used for Minimal Cost-Complexity Pruning
    # "monotonic_cst": [None], # Default if omitted: None; Indicates the monotonicity constraint to enforce on each feature
}
```

## Random forest

An ensemble of randomized trees. Tree count affects fit cost; depth and minimum
leaf size control each tree's complexity.

### Spark

```python
# Spark
model = "random_forest"
search_space = {
    "maxDepth": [
        5,
        10,
    ],  # Default if omitted: 5; Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in...
    "minInstancesPerNode": [
        10,
        50,
    ],  # Default if omitted: 1; Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than...
    "numTrees": [100, 200],  # Default if omitted: 20; Number of trees to train (>= 1)
    # "maxBins": [32], # Default if omitted: 32; Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature
    # "minInfoGain": [0.0], # Default if omitted: 0.0; Minimum information gain for a split to be considered at a tree node
    # "maxMemoryInMB": [256], # Default if omitted: 256; Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its...
    # "cacheNodeIds": [False, True], # Default if omitted: False; If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs...
    # "checkpointInterval": [10], # Default if omitted: 10; set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10...
    # "impurity": ['gini'], # Default if omitted: 'gini'; Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini
    # "featureSubsetStrategy": ['auto'], # Default if omitted: 'auto'; The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If...
    # "seed": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); random seed
    # "subsamplingRate": [1.0], # Default if omitted: 1.0; Fraction of the training data used for learning each decision tree, in range (0, 1]
    # "leafCol": [''], # Default if omitted: ''; Leaf indices column name. Predicted leaf index of each instance in each tree by preorder
    # "minWeightFractionPerNode": [0.0], # Default if omitted: 0.0; Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the...
    # "weightCol": [None], # Default if omitted: None; weight column name. If this is not set or empty, we treat all instance weights as 1.0
    # "bootstrap": [False, True], # Default if omitted: True; Whether bootstrap samples are used when building trees
}
```

### Pandas

```python
# pandas
model = "random_forest"
search_space = {
    "n_estimators": [
        100,
        200,
    ],  # Default if omitted: 100; The number of trees in the forest
    "max_depth": [
        5,
        10,
    ],  # Default if omitted: None (unlimited); The maximum depth of the tree
    "min_samples_leaf": [
        10,
        50,
    ],  # Default if omitted: 1; The minimum number of samples required to be at a leaf node
    # "criterion": ['gini'], # Default if omitted: 'gini'; The function to measure the quality of a split
    # "min_samples_split": [2], # Default if omitted: 2; The minimum number of samples required to split an internal node:
    # "min_weight_fraction_leaf": [0.0], # Default if omitted: 0.0; The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
    # "max_features": ['sqrt'], # Default if omitted: 'sqrt'; The number of features to consider when looking for the best split:
    # "max_leaf_nodes": [None], # Default if omitted: None; Grow trees with ``max_leaf_nodes`` in best-first fashion
    # "min_impurity_decrease": [0.0], # Default if omitted: 0.0; A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    # "bootstrap": [False, True], # Default if omitted: True; Whether bootstrap samples are used when building trees
    # "oob_score": [False, True], # Default if omitted: False; Whether to use out-of-bag samples to estimate the generalization score
    # "n_jobs": [None], # Default if omitted: None; The number of jobs to run in parallel
    # "random_state": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); Controls both the randomness of the bootstrapping of the samples used when building trees (if ``bootstrap=True``) and the...
    # "verbose": [0], # Default if omitted: 0; Controls the verbosity when fitting and predicting
    # "warm_start": [False, True], # Default if omitted: False; When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just...
    # "class_weight": [None], # Default if omitted: None; Weights associated with classes in the form ``{class_label: weight}``
    # "ccp_alpha": [0.0], # Default if omitted: 0.0; Complexity parameter used for Minimal Cost-Complexity Pruning
    # "max_samples": [None], # Default if omitted: None; If bootstrap is True, the number of samples to draw from X to train each base estimator
    # "monotonic_cst": [None], # Default if omitted: None; Indicates the monotonicity constraint to enforce on each feature
}
```

## Gradient boosting

Sequential boosted trees. Tune the learning rate together with the number of
iterations; smaller learning rates often need more iterations.

### Spark

```python
# Spark
model = "gradient_boosting"
search_space = {
    "maxDepth": [
        2,
        5,
    ],  # Default if omitted: 5; Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in...
    "maxIter": [100, 200],  # Default if omitted: 20; max number of iterations (>= 0)
    "stepSize": [
        0.03,
        0.1,
    ],  # Default if omitted: 0.1; Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator
    # "maxBins": [32], # Default if omitted: 32; Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature
    # "minInstancesPerNode": [1], # Default if omitted: 1; Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than...
    # "minInfoGain": [0.0], # Default if omitted: 0.0; Minimum information gain for a split to be considered at a tree node
    # "maxMemoryInMB": [256], # Default if omitted: 256; Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its...
    # "cacheNodeIds": [False, True], # Default if omitted: False; If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs...
    # "checkpointInterval": [10], # Default if omitted: 10; set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10...
    # "lossType": ['logistic'], # Default if omitted: 'logistic'; Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic
    # "seed": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); random seed
    # "subsamplingRate": [1.0], # Default if omitted: 1.0; Fraction of the training data used for learning each decision tree, in range (0, 1]
    # "impurity": ['variance'], # Default if omitted: 'variance'; Criterion used for information gain calculation (case-insensitive). Supported options: variance
    # "featureSubsetStrategy": ['all'], # Default if omitted: 'all'; The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If...
    # "validationTol": [0.01], # Default if omitted: 0.01; Threshold for stopping early when fit with validation is used. If the error rate on the validation input changes by less than...
    # "validationIndicatorCol": [None], # Default if omitted: None; name of the column that indicates whether each row is for training or for validation. False indicates training; true indicates...
    # "leafCol": [''], # Default if omitted: ''; Leaf indices column name. Predicted leaf index of each instance in each tree by preorder
    # "minWeightFractionPerNode": [0.0], # Default if omitted: 0.0; Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the...
    # "weightCol": [None], # Default if omitted: None; weight column name. If this is not set or empty, we treat all instance weights as 1.0
}
```

### Pandas

```python
# pandas
model = "gradient_boosting"
search_space = {
    "learning_rate": [
        0.03,
        0.1,
    ],  # Default if omitted: 0.1; Learning rate shrinks the contribution of each tree by `learning_rate`
    "n_estimators": [
        100,
        200,
    ],  # Default if omitted: 100; The number of boosting stages to perform
    "max_depth": [
        2,
        5,
    ],  # Default if omitted: 3; Maximum depth of the individual regression estimators
    # "loss": ['log_loss'], # Default if omitted: 'log_loss'; The loss function to be optimized
    # "subsample": [1.0], # Default if omitted: 1.0; The fraction of samples to be used for fitting the individual base learners
    # "criterion": ['friedman_mse'], # Default if omitted: 'friedman_mse'; The function to measure the quality of a split
    # "min_samples_split": [2], # Default if omitted: 2; The minimum number of samples required to split an internal node:
    # "min_samples_leaf": [1], # Default if omitted: 1; The minimum number of samples required to be at a leaf node
    # "min_weight_fraction_leaf": [0.0], # Default if omitted: 0.0; The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
    # "min_impurity_decrease": [0.0], # Default if omitted: 0.0; A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    # "init": [None], # Default if omitted: None; An estimator object that is used to compute the initial predictions
    # "random_state": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); Controls the random seed given to each Tree estimator at each boosting iteration
    # "max_features": [None], # Default if omitted: None; The number of features to consider when looking for the best split:
    # "verbose": [0], # Default if omitted: 0; Enable verbose output
    # "max_leaf_nodes": [None], # Default if omitted: None; Grow trees with ``max_leaf_nodes`` in best-first fashion
    # "warm_start": [False, True], # Default if omitted: False; When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just...
    # "validation_fraction": [0.1], # Default if omitted: 0.1; The proportion of training data to set aside as validation set for early stopping
    # "n_iter_no_change": [None], # Default if omitted: None; ``n_iter_no_change`` is used to decide if early stopping will be used to terminate training when validation score is not improving
    # "tol": [0.0001], # Default if omitted: 0.0001; Tolerance for the early stopping
    # "ccp_alpha": [0.0], # Default if omitted: 0.0; Complexity parameter used for Minimal Cost-Complexity Pruning
}
```

## LightGBM

Leaf-wise boosted trees. Leaf count and minimum data per leaf control complexity.
The example leaf sizes below need sufficiently large training folds; reduce them
for small samples. These are illustrative search spaces, not universal defaults.

### Spark (SparkLightGBM)

```python
# SparkLightGBM: default Spark backend, native parameter names
model = "lightgbm"
search_space = {
    "num_iterations": [100, 200],  # Default if omitted: 100
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "num_leaves": [15, 63],  # Default if omitted: 31
    "min_data_in_leaf": [200, 1000],  # Default if omitted: 20
    # "max_depth": [-1], # Default if omitted: -1 (unlimited)
    # "min_sum_hessian_in_leaf": [0.001], # Default if omitted: 0.001
    # "min_gain_to_split": [0.0], # Default if omitted: 0.0
    # "lambda_l1": [0.0], # Default if omitted: 0.0
    # "lambda_l2": [0.0], # Default if omitted: 0.0
    # "feature_fraction": [1.0], # Default if omitted: 1.0
    # "bagging_fraction": [1.0], # Default if omitted: 1.0
    # "bagging_freq": [0], # Default if omitted: 0 (disabled)
    # "max_bin": [255], # Default if omitted: 255
    # "is_unbalance": [False, True], # Default if omitted: False
    # "scale_pos_weight": [1.0], # Default if omitted: 1.0
    # "seed": [42], # Default if omitted: optimization seed (42)
}
```

### Spark (SynapseML)

```python
# SynapseML: pass lgbm_backend="synapseml" to optimize AND train_model.
model = "lightgbm"
search_space = {
    "learningRate": [
        0.03,
        0.1,
    ],  # Default if omitted: 0.1; Learning rate or shrinkage rate
    "minDataInLeaf": [
        200,
        1000,
    ],  # Default if omitted: 20; Minimal number of data in one leaf. Can be used to deal with over-fitting
    "numIterations": [
        100,
        200,
    ],  # Default if omitted: 100; Number of iterations, LightGBM constructs num_class * num_iterations trees
    "numLeaves": [15, 63],  # Default if omitted: 31; Number of leaves
    # "baggingFraction": [1.0], # Default if omitted: 1.0; Bagging fraction
    # "baggingFreq": [0], # Default if omitted: 0; Bagging frequency
    # "baggingSeed": [3], # Default if omitted: 3; Bagging seed
    # "binSampleCount": [200000], # Default if omitted: 200000; Number of samples considered at computing histogram bins
    # "boostFromAverage": [False, True], # Default if omitted: True; Adjusts initial score to the mean of labels for faster convergence
    # "boostingType": ['gbdt'], # Default if omitted: 'gbdt'; Default gbdt = traditional Gradient Boosting Decision Tree. Options are: gbdt, gbrt, rf (Random Forest), random_forest, dart...
    # "catSmooth": [10.0], # Default if omitted: 10.0; this can reduce the effect of noises in categorical features, especially for categories with few data
    # "categoricalSlotIndexes": lambda trial: [], # Default if omitted: []; structured value; List of categorical column indexes, the slot index in the features column
    # "categoricalSlotNames": lambda trial: [], # Default if omitted: []; structured value; List of categorical column slot names, the slot name in the features column
    # "catl2": [10.0], # Default if omitted: 10.0; L2 regularization in categorical split
    # "chunkSize": [10000], # Default if omitted: 10000; Advanced parameter to specify the chunk size for copying Java data to native. If set too high, memory may be wasted, but if...
    # "dataRandomSeed": [1], # Default if omitted: 1; Random seed for sampling data to construct histogram bins
    # "dataTransferMode": ['streaming'], # Default if omitted: 'streaming'; Specify how SynapseML transfers data from Spark to LightGBM. Values can be streaming, bulk. Default is streaming
    # "defaultListenPort": [12400], # Default if omitted: 12400; The default listen port on executors, used for testing
    # "deterministic": [False, True], # Default if omitted: False; Used only with cpu devide type. Setting this to true should ensure stable results when using the same data and the same...
    # "driverListenPort": [0], # Default if omitted: 0; The listen port on a driver. Default value is 0 (random)
    # "dropRate": [0.1], # Default if omitted: 0.1; Dropout rate: a fraction of previous trees to drop during the dropout
    # "dropSeed": [4], # Default if omitted: 4; Random seed to choose dropping models. Only used in dart
    # "earlyStoppingRound": [0], # Default if omitted: 0; Early stopping round
    # "executionMode": [None], # Default if omitted: None; Deprecated. Please use dataTransferMode
    # "extraSeed": [6], # Default if omitted: 6; Random seed for selecting threshold when extra_trees is true
    # "featureFraction": [1.0], # Default if omitted: 1.0; Feature fraction
    # "featureFractionByNode": [None], # Default if omitted: None; Feature fraction by node
    # "featureFractionSeed": [2], # Default if omitted: 2; Feature fraction seed
    # "featuresShapCol": [''], # Default if omitted: ''; Output SHAP vector column name after prediction containing the feature contribution values
    # "fobj": [None], # Default if omitted: None; Customized objective function. Should accept two parameters: preds, train_data, and return (grad, hess)
    # "improvementTolerance": [0.0], # Default if omitted: 0.0; Tolerance to consider improvement in metric
    # "initScoreCol": [None], # Default if omitted: None; The name of the initial score column, used for continued training
    # "isEnableSparse": [False, True], # Default if omitted: True; Used to enable/disable sparse optimization
    # "isProvideTrainingMetric": [False, True], # Default if omitted: False; Whether output metric result over training dataset
    # "isUnbalance": [False, True], # Default if omitted: False; Set to true if training data is unbalanced in binary classification scenario
    # "lambdaL1": [0.0], # Default if omitted: 0.0; L1 regularization
    # "lambdaL2": [0.0], # Default if omitted: 0.0; L2 regularization
    # "leafPredictionCol": [''], # Default if omitted: ''; Predicted leaf indices's column name
    # "matrixType": ['auto'], # Default if omitted: 'auto'; Advanced parameter to specify whether the native lightgbm matrix constructed should be sparse or dense. Values can be auto,...
    # "maxBin": [255], # Default if omitted: 255; Max bin
    # "maxBinByFeature": lambda trial: [], # Default if omitted: []; structured value; Max number of bins for each feature
    # "maxCatThreshold": [32], # Default if omitted: 32; limit number of split points considered for categorical features
    # "maxCatToOnehot": [4], # Default if omitted: 4; when number of categories of one feature smaller than or equal to this, one-vs-other split algorithm will be used
    # "maxDeltaStep": [0.0], # Default if omitted: 0.0; Used to limit the max output of tree leaves
    # "maxDepth": [-1], # Default if omitted: -1; Max depth
    # "maxDrop": [50], # Default if omitted: 50; Max number of dropped trees during one boosting iteration
    # "maxNumClasses": [100], # Default if omitted: 100; Number of max classes to infer numClass in multi-class classification
    # "maxStreamingOMPThreads": [16], # Default if omitted: 16; Maximum number of OpenMP threads used by a LightGBM thread. Used only for thread-safe buffer allocation. Use -1 to use OpenMP...
    # "metric": [''], # Default if omitted: ''; Metrics to be evaluated on the evaluation data. Options are: empty string or not specified means that metric corresponding to...
    # "microBatchSize": [100], # Default if omitted: 100; Specify how many elements are sent in a streaming micro-batch
    # "minDataPerBin": [3], # Default if omitted: 3; Minimal number of data inside one bin
    # "minDataPerGroup": [100], # Default if omitted: 100; minimal number of data per categorical group
    # "minGainToSplit": [0.0], # Default if omitted: 0.0; The minimal gain to perform split
    # "minSumHessianInLeaf": [0.001], # Default if omitted: 0.001; Minimal sum hessian in one leaf
    # "modelString": [''], # Default if omitted: ''; LightGBM model to retrain
    # "monotoneConstraints": lambda trial: [], # Default if omitted: []; structured value; used for constraints of monotonic features. 1 means increasing, -1 means decreasing, 0 means non-constraint. Specify all...
    # "monotoneConstraintsMethod": ['basic'], # Default if omitted: 'basic'; Monotone constraints method. basic, intermediate, or advanced
    # "monotonePenalty": [0.0], # Default if omitted: 0.0; A penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree
    # "negBaggingFraction": [1.0], # Default if omitted: 1.0; Negative Bagging fraction
    # "numBatches": [0], # Default if omitted: 0; If greater than 0, splits data into separate batches during training
    # "numTasks": [0], # Default if omitted: 0; Advanced parameter to specify the number of tasks. SynapseML tries to guess this based on cluster configuration, but this...
    # "numThreads": [0], # Default if omitted: 0; Number of threads per executor for LightGBM. For the best speed, set this to the number of real CPU cores
    # "objective": ['binary'], # Default if omitted: 'binary'; The Objective. For regression applications, this can be: regression_l2, regression_l1, huber, fair, poisson, quantile, mape,...
    # "objectiveSeed": [5], # Default if omitted: 5; Random seed for objectives, if random process is needed. Currently used only for rank_xendcg objective
    # "otherRate": [0.1], # Default if omitted: 0.1; The retain ratio of small gradient data. Only used in goss
    # "parallelism": ['data_parallel'], # Default if omitted: 'data_parallel'; Tree learner parallelism, can be set to data_parallel or voting_parallel
    # "passThroughArgs": [''], # Default if omitted: ''; Direct string to pass through to LightGBM library (appended with other explicitly set params). Will override any parameters...
    # "posBaggingFraction": [1.0], # Default if omitted: 1.0; Positive Bagging fraction
    # "predictDisableShapeCheck": [False, True], # Default if omitted: False; control whether or not LightGBM raises an error when you try to predict on data with a different number of features than the...
    # "referenceDataset": [None], # Default if omitted: None; The reference Dataset that was used for the fit. If using samplingMode=custom, this must be set before fit()
    # "repartitionByGroupingColumn": [False, True], # Default if omitted: True; Repartition training data according to grouping column, on by default
    # "samplingMode": ['subset'], # Default if omitted: 'subset'; Data sampling for streaming mode. Sampled data is used to define bins. 'global': sample from all data, 'subset': sample from...
    # "samplingSubsetSize": [1000000], # Default if omitted: 1000000; Specify subset size N for the sampling mode 'subset'. 'binSampleCount' rows will be chosen from the first N values of the...
    # "seed": [None], # Default if omitted: None; built-in model string: optimization seed (default 42); Main seed, used to generate other seeds
    # "skipDrop": [0.5], # Default if omitted: 0.5; Probability of skipping the dropout procedure during a boosting iteration
    # "slotNames": lambda trial: [], # Default if omitted: []; structured value; List of slot names in the features column
    # "thresholds": [None], # Default if omitted: None; Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to...
    # "timeout": [1200.0], # Default if omitted: 1200.0; Timeout in seconds
    # "topK": [20], # Default if omitted: 20; The top_k value used in Voting parallel, set this to larger value for more accurate result, but it will slow down the training...
    # "topRate": [0.2], # Default if omitted: 0.2; The retain ratio of large gradient data. Only used in goss
    # "uniformDrop": [False, True], # Default if omitted: False; Set this to true to use uniform drop in dart mode
    # "useBarrierExecutionMode": [False, True], # Default if omitted: False; Barrier execution mode which uses a barrier stage, off by default
    # "useMissing": [False, True], # Default if omitted: True; Set this to false to disable the special handle of missing value
    # "useSingleDatasetMode": [False, True], # Default if omitted: True; Use single dataset execution mode to create a single native dataset per executor (singleton) to reduce memory and...
    # "validationIndicatorCol": [None], # Default if omitted: None; Indicates whether the row is for training or validation
    # "verbosity": [-1], # Default if omitted: -1; Verbosity where lt 0 is Fatal, eq 0 is Error, eq 1 is Info, gt 1 is Debug
    # "weightCol": [None], # Default if omitted: None; The name of the weight column
    # "xGBoostDartMode": [False, True], # Default if omitted: False; Set this to true to use xgboost dart mode
    # "zeroAsMissing": [False, True], # Default if omitted: False; Set to true to treat all zero as missing values (including the unshown values in LibSVM / sparse matrices). Set to false to...
}
```

`lgbm_backend` is a Spark execution option, not a hyperparameter. Spark also
accepts `model="lgbm"`. See [LightGBM setup and execution](optimization.md#spark-lightgbm)
for dependencies, executor requirements, and local/distributed behavior.

### Pandas

```python
# pandas: native LightGBM sklearn wrapper
model = "lightgbm"
search_space = {
    "num_leaves": [15, 63],  # Default if omitted: 31
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "n_estimators": [100, 200],  # Default if omitted: 100
    "min_child_samples": [200, 1000],  # Default if omitted: 20
    # "boosting_type": ['gbdt'], # Default if omitted: 'gbdt'
    # "max_depth": [-1], # Default if omitted: -1
    # "subsample_for_bin": [200000], # Default if omitted: 200000
    # "objective": [None], # Default if omitted: None
    # "class_weight": [None], # Default if omitted: None
    # "min_split_gain": [0.0], # Default if omitted: 0.0
    # "min_child_weight": [0.001], # Default if omitted: 0.001
    # "subsample": [1.0], # Default if omitted: 1.0
    # "subsample_freq": [0], # Default if omitted: 0
    # "colsample_bytree": [1.0], # Default if omitted: 1.0
    # "reg_alpha": [0.0], # Default if omitted: 0.0
    # "reg_lambda": [0.0], # Default if omitted: 0.0
    # "random_state": [None], # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "n_jobs": [None], # Default if omitted: None
    # "importance_type": ['split'], # Default if omitted: 'split'
}
```

`min_child_samples` is the sklearn wrapper's parameter for minimum data per leaf.
Use that name in pandas to avoid supplying it alongside its native alias.

## Custom classifiers and pipelines

Spark accepts compatible Spark ML estimators exposing feature, label, prediction,
probability, and raw-prediction parameters. Supply native Spark parameter names;
the library assembles raw feature columns into its pipeline. See the
[training contract](training.md#training-parameter-reference) for details.

Pandas accepts cloneable sklearn classifiers with `predict_proba`. Use their
parameter names directly, or `step_name__parameter` for pipeline steps, as in the
logistic-regression example. Keep learned preprocessing inside the pipeline.
