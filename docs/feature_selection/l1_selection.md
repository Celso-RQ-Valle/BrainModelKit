# L1 Selection: l1_selection

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def l1_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name="feature_selection",
    C=1.0,
    coefficient_tolerance=1e-08,
    random_state=42,
    max_iter=1000,
): ...
```

### PySpark

```python
def l1_selection(
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
    reg_param=0.1,
    coefficient_tolerance=1e-08,
    max_iter=100,
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
| `C` | Pandas positive inverse L1 regularization strength. Smaller C means stronger regularization. |
| `coefficient_tolerance` | Nonnegative strict absolute coefficient cutoff; coefficients at or below it are rejected. |
| `random_state` | Seed for supported random estimators/resampling. Spark partition/order changes may still change results. |
| `max_iter` | Positive optimization or Boruta fit budget, depending on method. Inspect convergence or tentative decisions before interpreting output. |
| `n_tiles` | Spark training/RFE metric approximation resolution; integer >=2. |
| `reg_param` | Spark nonnegative L1 regularization strength. Larger values mean stronger regularization; not numerically equivalent to 1/C. |

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
`python examples/feature_selection/method_example.py --method l1_selection`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised binary classification; Pandas and native Spark.** L1 logistic
regression penalizes `sum(abs(coefficient))`, encouraging exact zeros. Selection
requires `abs(coefficient) > coefficient_tolerance` (default `1e-8`).

```python
from brainmodelkit.feature_selection.pandas import l1_selection

result = l1_selection(df, target_col="target", feature_cols=features, C=1.0)
```

Pandas uses L1/liblinear with `C > 0`, an inverse regularization strength, and
`max_iter=1000`. Spark uses native logistic regression with `elasticNetParam=1`,
`reg_param=0.1`, and `max_iter=100`. Spark's regularization parameter is **not**
an alias for sklearn C. Diagnostics are the same as feature importance.

Scale predictors and inspect convergence before interpreting coefficients. Spark
ML standardizes internally by default, while liblinear does not; reported
coefficients still depend on input units. L1 is useful for compact linear models,
but it may arbitrarily retain one of several correlated predictors and can miss
nonlinear interactions. Strong regularization can validly select no features.



