# Correlation: correlation_filter

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def correlation_filter(
    train_df=None, feature_cols=None, *, df=None, threshold=0.9, method="pearson"
): ...
```

### PySpark

```python
def correlation_filter(
    train_df=None,
    feature_cols=None,
    *,
    df=None,
    threshold=0.9,
    method="pearson",
    max_features=256,
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
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `df` | Alias for train_df except permutation, where this is the held-out development DataFrame. Supplying distinct frames under both aliases raises ValueError. |
| `threshold` | Inclusive score cutoff; None disables it. For correlation, reject at absolute correlation >= threshold in [0,1]. |
| `method` | Algorithm variant: correlation accepts pearson/spearman; stability accepts feature_importance/l1. |
| `max_features` | Spark correlation's positive feature-count budget for the dense driver-side correlation matrix. |

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
`python examples/feature_selection/method_example.py --method correlation_filter`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Unsupervised; Pandas and native Spark.** Pearson correlation is normalized
covariance; Spearman correlation is Pearson correlation of ranks and captures
monotonic rather than only linear relationships. `method` accepts `"pearson"`
or `"spearman"`; both use absolute correlation for filtering.

```python
from brainmodelkit.feature_selection.pandas import correlation_filter

result = correlation_filter(
    df, feature_cols=features, threshold=0.90, method="spearman"
)
print(result.correlation_pairs)
```

The rule is deterministic: traverse features in input order and reject a feature
if its absolute correlation is at least the threshold with any **previously
retained** feature. Earlier retained features win. A rejected feature cannot
subsequently reject another feature. All high-correlation pairs are reported,
including pairs involving rejected features. Diagnostics include `rejected_by`.
Constant-feature correlations are undefined and do not trigger rejection; apply
variance filtering first. No target is used in deciding which representative wins.

Use this to reduce redundancy and coefficient instability, not to measure target
association. It misses non-monotonic dependencies and can discard variables that
are useful jointly. At least two complete finite numeric rows are required.
Spark uses native `Correlation.corr`; its matrix needs O(p²) driver memory.
`max_features=256` prevents an unexpectedly large matrix; prefilter or explicitly
raise that budget. Spearman requires distributed ranking and is more expensive.
See the [Spark correlation API](https://spark.apache.org/docs/3.5.7/api/python/reference/api/pyspark.ml.stat.Correlation.html).



