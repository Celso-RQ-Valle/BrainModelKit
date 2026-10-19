# Validation and optimization without consuming final OOT

[Documentation home](README.md) ? [CV contract](model_selection.md) ? [Optimization reference](optimization.md)

The main [runnable journeys](../examples/journey/README.md) use explicit April
training, May validation, and June OOT with two manual configurations. They keep
selection on April and compare only on May. That temporal holdout is the default
story; the following are **alternative development branches**, starting after data
preparation/building, not after making choices using the final OOT metrics.

CV refits a model on each development fold. Hyperparameter optimization repeats
that process for parameter candidates and chooses a configuration. Selection
itself is fitting: CV on a subset selected with all fold labels is not an unbiased
assessment of the selector + model pipeline. The examples below deliberately use
**two predetermined features**, not the main workflow's learned `selected_features`.
Fixed zero filling was chosen in advance and estimates no statistics.

## Spark API alternatives

Continue only the Spark data preparation/building cells, which define `train_df`,
`validation_df`, and `oot_df`. Do not run the final evaluation first.

```python
from brainmodelkit.model_selection.pyspark import cross_validate
from brainmodelkit.optimization.pyspark import optimize
from brainmodelkit.training.pyspark import train_model

fixed_features = ["revenue__current", "balance__current"]
cv_result = cross_validate(
    train_df,
    "target",
    fixed_features,
    "logistic_regression",
    model_params={"maxIter": 10},
    cv=2,
    metrics=["auc"],
)
print(cv_result.fold_metrics)
print(cv_result.summary)

# Requires: python -m pip install "brainmodelkit[optimization]"
tuned = optimize(
    train_df,
    validation_df,
    "target",
    fixed_features,
    "logistic_regression",
    {
        "regParam": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        "maxIter": {"type": "int", "low": 10, "high": 10},
    },
    objective_metric="auc",
    n_trials=2,
    cv=2,
    metrics=["auc"],
)
print(tuned.best_params, tuned.best_value)
print(tuned.trials)
# Freeze configuration; only now refit on development and evaluate final OOT.
result = train_model(
    "tuned_final",
    "target",
    fixed_features,
    train_df.unionByName(validation_df),
    oot_df,
    model="logistic_regression",
    model_params=tuned.best_params,
    save_model_to="none",
)
```

Stratified CV here estimates performance on exchangeable development accounts;
it is not temporal validation. With temporal dependence, use explicit forward
splits and fit transforms/selection on each training slice. Spark centralized
`date_col`/`strategy="time_series"` creates ordered blocks but uses all other
blocks to train. `fold_col` likewise means complementary folds, not forward-only
training. A Spark estimator must expose the feature/label/probability parameters
expected by the factory; the API does not accept an arbitrary preprocessing callback.

## pandas API alternatives

Continue only the pandas data preparation/building cells. This branch also fixes
its feature recipe before CV and does not reuse globally selected features.

```python
import pandas as pd
from brainmodelkit.model_selection.pandas import cross_validate
from brainmodelkit.optimization.pandas import optimize
from brainmodelkit.training.pandas import train_model

fixed_features = ["revenue__current", "balance__current"]
cv_result = cross_validate(
    train_df,
    "target",
    fixed_features,
    "logistic_regression",
    model_params={"max_iter": 1000},
    cv=2,
    metrics=["auc"],
)
print(cv_result.fold_metrics)
print(cv_result.summary)

tuned = optimize(
    train_df,
    validation_df,
    "target",
    fixed_features,
    "logistic_regression",
    {
        "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
        "max_iter": {"type": "int", "low": 1000, "high": 1000},
    },
    objective_metric="auc",
    n_trials=2,
    cv=2,
    metrics=["auc"],
)
print(tuned.best_params, tuned.best_value)
print(tuned.trials)
result = train_model(
    "tuned_final",
    "target",
    fixed_features,
    pd.concat([train_df, validation_df], ignore_index=True),
    oot_df,
    model="logistic_regression",
    model_params=tuned.best_params,
    save_model_to="none",
)
```

For learned preprocessing and supervised selection, use a cloneable pipeline
fitted anew inside each fold. **Standalone alternative**, using the same prepared
training frame but not the preceding global selection:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SelectKBest(f_classif, k=1),
    LogisticRegression(max_iter=1000),
)
cv_result = cross_validate(
    train_df,
    "target",
    fixed_features,
    pipeline,
    cv=2,
    metrics=["auc"],
)
```

This demonstrates sklearn fold-fitted selection, not an automatic BrainModelKit
selector pipeline. For BrainModelKit selectors requiring DataFrames, use explicit
fold iteration: fit preprocessing on fold-training rows, transform both slices,
select on transformed training only, train, score validation, and aggregate.
Pandas `date_col` uses expanding `TimeSeriesSplit`; design entity/time boundaries
and target maturation gaps before splitting.

## Configuration and result contracts

`cross_validate` returns `CrossValidationResult`: fold metrics, metric mean/std
summary, optional train metrics and fitted models. Pandas tables are DataFrames;
Spark fold records and summaries are Python lists of dictionaries. The detailed
CV guide explains `strategy`, `group_col`, `date_col`, `fold_col` (Spark), metrics,
and optional models. No model persistence occurs.

`optimize` returns a study, `best_params`, `best_value`, `trials`, `best_model`,
selected trial number, and optional feasibility result. `best_model` is refitted
on `train_df`, not train + validation; the explicit final call above defines that
refit policy. `oot_df=validation_df` supplies development diagnostics to trials.
The default/unqualified objective (`auc` here) uses CV; never pass final OOT even
if its metrics would be diagnostic. If an objective/constraint names `oot_*`,
that frame is actively used for tuning and must be treated as validation.

Optimization currently uses stratified CV and exposes no strategy/group/date/fold
or arbitrary fold-preprocessing callback arguments. It is not a substitute for
custom temporal search. A pandas estimator pipeline can still encapsulate learned
transforms. Read [search spaces, metric aliases, constraints and feasibility](optimization.md)
before using advanced objectives. Spark's `lgbm_backend` is a top-level execution
choice, never a tunable estimator parameter. `save_as` belongs to optimization;
`save_model_to` belongs to training. They are not interchangeable keywords.
