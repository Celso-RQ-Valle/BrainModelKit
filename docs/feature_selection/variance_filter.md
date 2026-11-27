# Variance: variance_filter

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pyspark as fs` for Spark,
or `from brainmodelkit.feature_selection import pandas as fs` for pandas.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### PySpark

```python
def variance_filter(df, feature_cols=None, *, min_variance=0.0): ...
```

### Pandas

```python
def variance_filter(df, feature_cols=None, *, min_variance=0.0): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.

## Parameters

| Name | Meaning |
| --- | --- |
| `df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `min_variance` | Nonnegative strict population-variance cutoff, calculated over non-missing observations. |

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

Use the [Spark setup](spark.md#example-data) or the [pandas setup](setup.md). From the repository root, run `python examples/feature_selection/method_example.py --backend spark --method variance_filter`. Use `--backend pandas` for the pandas example.

## How it works

**Unsupervised; native Spark and pandas.** Population variance is
`sum((x - mean(x))**2) / n`, calculated over non-missing observations (`ddof=0`).
Features pass only when `variance > min_variance`; the default zero removes
constants. A single observed value has zero variance; all-missing features have
undefined variance and are rejected.

**Spark** | [Example data](spark.md#example-data)

```python
result = fs.variance_filter(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_variance=0.0,  # Keep columns whose population variance exceeds this cutoff.
)
print(result.selected_features)
```

**pandas** | [Example data](setup.md)

```python
from brainmodelkit.feature_selection.pandas import variance_filter

result = variance_filter(
    df=df,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_variance=0.0,  # Keep columns whose population variance exceeds this cutoff.
)
```

Outputs are `feature`, `variance`, and `selected`. Numeric features are required.
This is useful for eliminating constants before expensive fitting or correlation.
Avoid a common nonzero cutoff across incomparable units: variance depends on
scale, and low-variance rare indicators can still be predictive. Standardization
before this filter makes a nonzero variance threshold largely uninformative.
Spark uses native `var_pop` aggregation.

## Computational cost and practical limits

Typical relative cost: **low**, with overlapping tiers across methods.
This is an implementation-based estimate, not a benchmark or a universal ranking.
Spark: native var_pop aggregates. Pandas: column moments.
Rows, feature count, cardinality, iterations, folds, and the chosen estimator can
change which method dominates. Use this method for before fitting when justified;
do not run every selector as a fixed pipeline.

See the [backend comparison](../selection-costs.md) for an ordered shortlist
and [input contracts](../feature_selection.md) before comparing scores.
