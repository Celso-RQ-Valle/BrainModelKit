# Completeness: completeness

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### Pandas

```python
def completeness(
    train_df=None,
    feature_cols=None,
    *,
    df=None,
    min_completeness=0.7,
    group_by=None,
    require_all_groups=False,
): ...
```

### PySpark

```python
def completeness(
    train_df=None,
    feature_cols=None,
    *,
    df=None,
    min_completeness=0.7,
    group_by=None,
    require_all_groups=False,
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
| `min_completeness` | Inclusive non-missing fraction cutoff in [0,1]. |
| `group_by` | Optional grouping column name or list, including observed null groups. Controls grouped diagnostics or separate stability fits. |
| `require_all_groups` | When True, require global and every observed group's completeness to pass; requires group_by. |

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
`python examples/feature_selection/method_example.py --method completeness`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Unsupervised; Pandas and native Spark.** Completeness measures data availability,
not predictive power:

```text
completeness = non-null observations / total observations
missing_rate = 1 - completeness
```

Pandas missing values and Spark floating NaN count as missing, as do Spark SQL
nulls. Empty strings and sentinel values such as `-999` are not automatically
missing; normalize them first if that is their business meaning. Selection uses
`completeness >= min_completeness` (default 0.70). An all-missing feature therefore
passes only at an explicitly chosen zero threshold.

```python
from brainmodelkit.feature_selection.pandas import completeness

result = completeness(
    df, feature_cols=features, min_completeness=0.70, group_by="reference_date"
)
print(result.selected_features)
print(result.grouped_table)
```

`feature_table` contains `feature`, `completeness`, `missing_rate`, and `selected`.
Use `group_by=["reference_date", "segment"]` for multi-column analysis. Grouped
diagnostics include missing group keys. A trajectory such as Jan 99%, Feb 98%,
Mar 76%, Apr 42% can expose collection failures hidden by a high global average.
Group rows have their own threshold flags, but by default only global completeness
chooses features. Set `require_all_groups=True` to require every observed group
to pass as well. This option requires `group_by`; it can be sensitive to tiny
groups. Group column names cannot collide with diagnostic output names.

Use completeness early for operational reliability. Do not use it as a substitute
for predictive evaluation: missingness can itself carry information, and a rare
but valuable field may merit a dedicated model. Spark computes means of native
non-null indicators; group tables stay distributed.



