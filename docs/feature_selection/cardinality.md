# Cardinality: cardinality

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def cardinality(
    train_df=None,
    feature_cols=None,
    *,
    df=None,
    min_unique=2,
    max_unique=None,
    max_unique_ratio=None,
): ...
```

### PySpark

```python
def cardinality(
    train_df=None,
    feature_cols=None,
    *,
    df=None,
    min_unique=2,
    max_unique=None,
    max_unique_ratio=None,
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
| `min_unique` | Inclusive minimum exact non-missing distinct count (integer >=0). |
| `max_unique` | Optional inclusive maximum distinct count; must be at least min_unique. |
| `max_unique_ratio` | Optional inclusive unique_count/total_rows bound in [0,1]. Missing rows remain in the denominator. |

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
`python examples/feature_selection/method_example.py --method cardinality`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Unsupervised; Pandas and native Spark.** Cardinality counts distinct non-missing
values. `unique_ratio = unique_count / total_rows`, including missing rows in the
denominator. Defaults are `min_unique=2`, `max_unique=None`, and
`max_unique_ratio=None`; specified bounds are inclusive.

```python
from brainmodelkit.feature_selection.pandas import cardinality

result = cardinality(
    df, feature_cols=features, min_unique=2, max_unique=100, max_unique_ratio=0.50
)
```

Outputs are `feature`, `unique_count`, `unique_ratio`, and `selected`. Use it to
identify constants, identifier-like fields, or costly categorical encodings.
Do not automatically discard continuous numeric features because their unique
ratio is high. Counts are exact, not approximate; Spark distinct aggregation can
require substantial shuffle work. Consider the dataset's sample size and use
completeness separately rather than treating the ratio as coverage.



