# Feature Importance: feature_importance_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def feature_importance_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    model="random_forest",
    model_params=None,
    threshold=None,
    top_k=None,
    random_state=42,
): ...
```

### PySpark

```python
def feature_importance_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    n_tiles=10,
    model="random_forest",
    model_params=None,
    threshold=None,
    top_k=None,
    random_state=42,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.
`None` placeholders for train/target/features do not make those inputs optional
at call time. Unsupported backend keywords raise TypeError.

## Parameters

| Name | Meaning |
| --- | --- |
| `train_df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `target_col` | Target column name; exclude it from feature_cols. Required for supervised methods. Spark requires both 0 and 1; Pandas class support is described below. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `df` | Alias for train_df except permutation, where this is the held-out development DataFrame. Supplying distinct frames under both aliases raises ValueError. |
| `oot_df` | Optional labeled OOT DataFrame for supported importance/wrapper final evaluation; required by historical RFE. It never chooses the feature subset. |
| `df_oot` | Alias for oot_df. Supplying distinct frames under both spellings raises ValueError. |
| `df_scoring` | Optional unlabeled frame to score with the final selected-feature classifier; requires OOT evaluation. RFE scores it only at the final fit. |
| `run_name` | Label used for final training outputs or RFE iteration reports. |
| `model` | Named or compatible native estimator; permutation requires an already fitted model. Model restrictions and return schemas are explained below. |
| `model_params` | Optional dictionary of native estimator parameters: Pandas snake_case, Spark camelCase. Custom estimators are copied/cloned. |
| `threshold` | Inclusive score cutoff; None disables it. For correlation, reject at absolute correlation >= threshold in [0,1]. |
| `top_k` | Optional positive integer rank limit no larger than candidate count. Intersects other gates; ties favor input order. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `n_tiles` | Spark training/RFE metric approximation resolution; integer >=2. |

## Returns and errors

Returns `SelectionResult`. Read `selected_features`, `rejected_features`, and
`feature_table`; method-specific fields are explained below. An empty selection
is valid and must be handled before fitting another model.

Invalid columns, duplicate names, empty inputs, incompatible targets, nonfinite
numeric inputs, or out-of-range controls raise validation errors. Quality methods
and IV have their documented missing-value exceptions. Optional model dependencies
raise installation errors when unavailable. Backend estimator errors, such as
invalid model parameters or inadequate CV class counts, are propagated.


## Example setup

Run the [shared example setup](setup.md) before the snippet below, or execute
`python examples/feature_selection/method_example.py --method feature_importance_selection`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised; Pandas and native Spark.** `feature_importance_selection` fits a
fresh estimator and ranks its native tree importance or absolute linear
coefficient. The signed binary coefficient remains in `importance`; coefficient
arrays are also in `coefficients`. For Pandas multiclass models, the coefficient
L2 norm across classes determines importance and all signed coefficients remain
in diagnostics. Training/RFE's importance record convention is reused.

```python
from brainmodelkit.feature_selection.pandas import feature_importance_selection

result = feature_importance_selection(
    df,
    target_col="target",
    feature_cols=features,
    model="random_forest",
    model_params={"n_estimators": 100},
    top_k=5,
)
```

Supported names are `logistic_regression`, `decision_tree`, `random_forest`,
`gradient_boosting`, and `lightgbm`, plus compatible estimator instances.
Pandas estimators are cloned; Spark estimators are copied. Use native parameter
names in `model_params` (e.g. Spark `numTrees`). Named stochastic estimators use
`random_state=42`/Spark seed by default; custom estimator randomness remains the
caller's responsibility. Pandas LightGBM needs its optional extra. Spark LightGBM
requires SynapseML and matching JVM packages configured outside BrainModelKit.

Outputs include `feature`, `importance`, `ranking`, and `selected`, plus `model`.
With no oot_df, no OOT scoring is performed. When oot_df is supplied, selected
features are refitted for final evaluation; see [evaluation outputs](../feature_selection.md#evaluation-outputs).
Without OOT evaluation Spark returns a native classifier expecting an assembled
`features` vector; after OOT evaluation `model` is the final raw-column training
pipeline and `selection_model` retains the original native selector model.
Models must expose one finite native importance per feature; arbitrary pipelines
without a compatible importance attribute are not implicitly unwrapped.

Use this when importance should reflect a model family. Tree impurity importance
can favor high-cardinality features, and correlated predictors can share or mask
importance. Coefficient magnitudes depend on scaling. An importance is not a
causal effect and does not guarantee held-out benefit. Use permutation or temporal
evaluation where those limitations matter.


