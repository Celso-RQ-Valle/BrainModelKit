# Models and hyperparameter optimization

[Main workflow](../../README.md) | [Optimization contract](../optimization.md)

All built-in models below are binary classifiers: targets are 0/1 and predictions
provide a class-1 probability. Logistic regression is a classifier here; the
optimization API does not currently support continuous-target regression metrics.

Use `[minimum, maximum]` for intervals: `[200, 1000]` samples integers throughout
that interval, including its bounds. If either bound is a float, sampling is
continuous; `[100]` fixes a parameter. Trials explore combinations within
the budget rather than exhaustively testing every possible value. See
[search-space rules](../optimization.md#search-spaces) for categorical choices
and logarithmic sampling.

Choose one model and its search space below, then pass them to `optimize`:

```python
from brainmodelkit.optimization.pandas import optimize
# For Spark: from brainmodelkit.optimization.pyspark import optimize

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
and [trial comparison](../optimization.md#save-and-compare-trials) to choose a
configuration yourself from the saved DataFrame.

Omit a hyperparameter entry to keep its default; do not leave an empty list.
Each model links to a full constructor catalog with default comments and upstream
documentation explaining the settings. Examples below show starting ranges,
not recommended settings for every dataset. See [training](../training.md) for
refitting, [model selection](../model_selection.md) for CV, and
[evaluation](../evaluation.md) for metric definitions.

## Logistic regression

[All parameters, defaults, and detailed explanations](logistic-regression.md).

A linear classifier. Scale numeric inputs inside a pandas pipeline so that each
CV fold learns its own scaling. Smaller `C` means stronger regularization;
larger Spark `regParam` means stronger regularization. They are not interchangeable.

```python
# pandas
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search_space = {
    "logisticregression__C": [0.001, 10.0],  # Default if omitted: 1.0
}
```

```python
# Spark: built-in standardization is enabled by default.
model = "logistic_regression"
search_space = {
    "regParam": [0.001, 1.0],  # Default if omitted: 0.0
    "maxIter": [100],  # Default if omitted: 100
}
```

## Decision tree

[All parameters, defaults, and detailed explanations](decision-tree.md).

A single tree. Depth limits complexity; minimum leaf size limits small leaves.
Use an estimator object in pandas so the same `model` also works with
`train_model` (the string `"decision_tree"` works in pandas optimization/CV only).

```python
# pandas
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
search_space = {
    "max_depth": [3, 8],  # Default if omitted: None (unlimited)
    "min_samples_leaf": [10, 50],  # Default if omitted: 1
}
```

```python
# Spark
model = "decision_tree"
search_space = {
    "maxDepth": [3, 8],  # Default if omitted: 5
    "minInstancesPerNode": [10, 50],  # Default if omitted: 1
}
```

## Random forest

[All parameters, defaults, and detailed explanations](random-forest.md).

An ensemble of randomized trees. Tree count affects fit cost; depth and minimum
leaf size control each tree's complexity.

```python
# pandas
model = "random_forest"
search_space = {
    "n_estimators": [100, 200],  # Default if omitted: 100
    "max_depth": [5, 10],  # Default if omitted: None (unlimited)
    "min_samples_leaf": [10, 50],  # Default if omitted: 1
}
```

```python
# Spark
model = "random_forest"
search_space = {
    "numTrees": [100, 200],  # Default if omitted: 20
    "maxDepth": [5, 10],  # Default if omitted: 5
    "minInstancesPerNode": [10, 50],  # Default if omitted: 1
}
```

## Gradient boosting

[All parameters, defaults, and detailed explanations](gradient-boosting.md).

Sequential boosted trees. Tune the learning rate together with the number of
iterations; smaller learning rates often need more iterations.

```python
# pandas
model = "gradient_boosting"
search_space = {
    "n_estimators": [100, 200],  # Default if omitted: 100
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "max_depth": [2, 5],  # Default if omitted: 3
}
```

```python
# Spark
model = "gradient_boosting"
search_space = {
    "maxIter": [100, 200],  # Default if omitted: 20
    "stepSize": [0.03, 0.1],  # Default if omitted: 0.1
    "maxDepth": [2, 5],  # Default if omitted: 5
}
```

## LightGBM

[All parameters, defaults, and detailed explanations](lightgbm.md).

Leaf-wise boosted trees. Leaf count and minimum data per leaf control complexity.
The example leaf sizes below need sufficiently large training folds; reduce them
for small samples. These are illustrative search spaces, not universal defaults.

```python
# pandas: native LightGBM sklearn wrapper
model = "lightgbm"
search_space = {
    "n_estimators": [100, 200],  # Default if omitted: 100
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "num_leaves": [15, 63],  # Default if omitted: 31
    "min_child_samples": [200, 1000],  # Default if omitted: 20
}
```

`min_child_samples` is the sklearn wrapper's parameter for minimum data per leaf.
Use that name in pandas to avoid supplying it alongside its native alias.

```python
# SparkLightGBM: default Spark backend, native parameter names
model = "lightgbm"
search_space = {
    "num_iterations": [100, 200],  # Default if omitted: 100
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "num_leaves": [15, 63],  # Default if omitted: 31
    "min_data_in_leaf": [200, 1000],  # Default if omitted: 20
}
```

```python
# SynapseML: pass lgbm_backend="synapseml" to optimize AND train_model.
model = "lightgbm"
search_space = {
    "numIterations": [100, 200],  # Default if omitted: 100
    "learningRate": [0.03, 0.1],  # Default if omitted: 0.1
    "numLeaves": [15, 63],  # Default if omitted: 31
    "minDataInLeaf": [200, 1000],  # Default if omitted: 20
}
```

`lgbm_backend` is a Spark execution option, not a hyperparameter. Spark also
accepts `model="lgbm"`. See [LightGBM setup and execution](../optimization.md#spark-lightgbm)
for dependencies, executor requirements, and local/distributed behavior.

## Custom classifiers and pipelines

Pandas accepts cloneable sklearn classifiers with `predict_proba`. Use their
parameter names directly, or `step_name__parameter` for pipeline steps, as in the
logistic-regression example. Keep learned preprocessing inside the pipeline.

Spark accepts compatible Spark ML estimators exposing feature, label, prediction,
probability, and raw-prediction parameters. Supply native Spark parameter names;
the library assembles raw feature columns into its pipeline. See the
[training contract](../training.md#training-parameter-reference) for details.
