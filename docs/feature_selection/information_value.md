# Information Value: information_value

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def information_value(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    n_bins=10,
    min_iv=None,
    smoothing=0.5,
    binning="quantile",
): ...
```

### PySpark

```python
def information_value(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    n_bins=10,
    min_iv=None,
    smoothing=0.5,
    binning="quantile",
    relative_error=0.001,
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
| `n_bins` | Number of requested numeric bins, integer >=2. Duplicate boundaries can reduce effective bins. |
| `min_iv` | Optional inclusive IV cutoff; None reports/selects all defined IV values. |
| `smoothing` | Strictly positive additive per-observed-bin count smoothing for both classes. |
| `binning` | Numeric IV binning: quantile or uniform. Categorical values use their observed levels. |
| `relative_error` | Spark approximate quantile rank error in [0,1]; 0 requests exact quantiles at higher cost. |

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
`python examples/feature_selection/method_example.py --method information_value`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised binary classification; Pandas and native Spark.** IV compares the
event and non-event distributions across a feature's bins. Event means target 1.
For observed bin b, additive smoothing alpha gives:

```text
de[b] = (events[b] + alpha) / (total_events + alpha * observed_bin_count)
dn[b] = (non_events[b] + alpha) / (total_non_events + alpha * observed_bin_count)
WoE[b] = log(de[b] / dn[b])
IV = sum((de[b] - dn[b]) * WoE[b])
```

```python
from brainmodelkit.feature_selection.pandas import information_value

result = information_value(
    df,
    target_col="target",
    feature_cols=features,
    n_bins=10,
    min_iv=0.02,
    smoothing=0.5,
    binning="quantile",
)
print(result.bin_details)
```

Numeric variables use `binning="quantile"` (default) or `"uniform"` equal-width
bins. `n_bins >= 2`; duplicate boundaries can reduce effective bins. Categorical
variables use observed categories directly. Missing values have a separate bin,
not a reserved string that could collide with a real category. Both classes must
exist globally. `smoothing > 0` prevents infinite WoE for bins with only one class;
empty bins are excluded. An all-missing or constant feature has zero IV.

`feature_table` contains `feature`, `iv`, `ranking`, and `selected`. `min_iv=None`
reports/selects every defined IV; otherwise the cutoff is inclusive. `bin_details`
contains event/non-event counts, both smoothed distributions, `woe`,
`iv_component`, `bin`, and `is_missing`. Pandas also provides category labels.
Numeric cut points are in `metadata["bin_edges"]`.

Both backends put equality at a cut point in the lower bin.
Spark uses approximate quantiles (`relative_error=0.001`) and native SQL bin comparisons
with upper-inclusive intervals; Pandas uses exact sample quantiles. Their bins
and IV can differ at ties or approximate boundaries. Use the diagnostics rather
than assuming bitwise backend equality. Spark collects only bin edges and a few
scalar totals; all categorical counts and WoE rows stay distributed.

IV is useful for transparent binary univariate screening and bin-level risk
diagnostics. High IV can reveal leakage, rare-category overfit, or excessive
binning, not necessarily a good feature. Compare on development data and use
independent validation. This selector does not return a reusable WoE transformer;
do not independently re-bin OOT data and treat it as a fitted transformation.
It is unsuitable for multiclass targets without a separately designed extension.



