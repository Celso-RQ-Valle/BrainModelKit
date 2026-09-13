# Variance: variance_filter

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def variance_filter(train_df=None, feature_cols=None, *, df=None, min_variance=0.0): ...
```

### PySpark

```python
def variance_filter(train_df=None, feature_cols=None, *, df=None, min_variance=0.0): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.
`None` placeholders for train/target/features do not make those inputs optional
at call time. Unsupported backend keywords raise TypeError.

## Parameters

| Name | Meaning |
| --- | --- |
| `train_df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `df` | Alias for train_df except permutation, where this is the held-out development DataFrame. Supplying distinct frames under both aliases raises ValueError. |
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

Run the [shared example setup](setup.md) before the snippet below, or execute
`python examples/feature_selection/method_example.py --method variance_filter`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Unsupervised; Pandas and native Spark.** Population variance is
`sum((x - mean(x))**2) / n`, calculated over non-missing observations (`ddof=0`).
Features pass only when `variance > min_variance`; the default zero removes
constants. A single observed value has zero variance; all-missing features have
undefined variance and are rejected.

```python
from brainmodelkit.feature_selection.pandas import variance_filter

result = variance_filter(df, feature_cols=features, min_variance=0.0)
```

Outputs are `feature`, `variance`, and `selected`. Numeric features are required.
This is useful for eliminating constants before expensive fitting or correlation.
Avoid a common nonzero cutoff across incomparable units: variance depends on
scale, and low-variance rare indicators can still be predictive. Standardization
before this filter makes a nonzero variance threshold largely uninformative.
Spark uses native `var_pop` aggregation.



