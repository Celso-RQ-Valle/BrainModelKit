# Chi-Square: chi_square

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def chi_square(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    max_p_value=0.05,
    top_k=None,
): ...
```

### PySpark

```python
def chi_square(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    max_p_value=0.05,
    top_k=None,
    max_categories=1000,
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
| `max_p_value` | Inclusive unadjusted p-value cutoff in [0,1], or None to disable. Undefined tests are rejected. |
| `top_k` | Optional positive integer rank limit no larger than candidate count. Intersects other gates; ties favor input order. |
| `max_categories` | Spark discrete cardinality guard, integer >=2. Encode/bin high-cardinality data explicitly. |

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
`python examples/feature_selection/method_example.py --method chi_square`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised; Pandas and native Spark, with different input semantics.** The
statistic sums `(observed - expected)**2 / expected` under independence. Small
p-values indicate evidence against independence, subject to adequate expected
cell counts and independent observations.

```python
from brainmodelkit.feature_selection.pandas import chi_square

result = chi_square(
    counts_df, target_col="target", feature_cols=count_features, max_p_value=0.05
)
```

Pandas uses sklearn `chi2` on **nonnegative integer counts or indicators**. Spark
uses native `ChiSquareTest` on **nonnegative integer category codes**, treating
each distinct value as a categorical level. These statistics are not numerically
interchangeable. Both reject negative or fractional features. Discretize or encode
explicitly; never pass arbitrary continuous values and interpret them as counts.
Spark's `max_categories=1000` guards categorical contingency-table size; Spark
targets are binary 0/1. Pandas supports multiclass classification.

Outputs include `feature`, `statistic`, `p_value`, `ranking`, and `selected`.
Selection requires `p_value <= max_p_value` (default 0.05), optionally intersected
with `top_k`; use `max_p_value=None` to disable that gate. Constant/undefined
features are rejected. Useful for count, indicator, or explicitly categorical
screening, but sparse expected counts and many simultaneous tests weaken p-value
interpretation. See [sklearn chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)
and [Spark ChiSquareTest](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/stat/ChiSquareTest.html).



