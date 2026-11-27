# Completeness: completeness

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pyspark as fs` for Spark,
or `from brainmodelkit.feature_selection import pandas as fs` for pandas.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

## Signatures

### PySpark

```python
def completeness(
    df,
    feature_cols=None,
    *,
    min_completeness=0.7,
    group_by=None,
    require_all_groups=False,
): ...
```

### Pandas

```python
def completeness(
    df,
    feature_cols=None,
    *,
    min_completeness=0.7,
    group_by=None,
    require_all_groups=False,
): ...
```

The listings show accepted arguments and defaults. Pass optional controls by
keyword; permutation requires a fitted model first and RFE requires run_name first.

## Parameters

| Name | Meaning |
| --- | --- |
| `df` | Training DataFrame for the selected backend. Fit selection on development/training rows only. |
| `feature_cols` | Explicit nonempty ordered list of unique existing feature names. RFE alone can infer names starting feat_. Selection outputs preserve this order. |
| `min_completeness` | Inclusive non-missing fraction cutoff in [0,1]. |
| `group_by` | List of grouping columns, including for one column. Use None for the whole frame. Observed null groups are retained. |
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

Use the [Spark setup](spark.md#example-data) or the [pandas setup](setup.md). From the repository root, run `python examples/feature_selection/method_example.py --backend spark --method completeness`. Use `--backend pandas` for the pandas example.

## How it works

**Unsupervised; native Spark and pandas.** Completeness measures data availability,
not predictive power:

```text
completeness = non-null observations / total observations
missing_rate = 1 - completeness
```

Spark SQL nulls and floating NaN count as missing, as do pandas missing values. Empty strings and sentinel values such as `-999` are not automatically
missing; normalize them first if that is their business meaning. Selection uses
`completeness >= min_completeness` (default 0.70). An all-missing feature therefore
passes only at an explicitly chosen zero threshold.

**Spark** | [Example data](spark.md#example-data)

```python
result = fs.completeness(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_completeness=0.95,  # Keep columns with observed values in at least this fraction of rows.
    group_by=["fold"],  # Report completeness for each observed group.
    require_all_groups=False,  # Require the completeness cutoff in every group when enabled.
)
print(result.selected_features)
```

**pandas** | [Example data](setup.md)

```python
from brainmodelkit.feature_selection.pandas import completeness

result = completeness(
    df=df,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_completeness=0.7,  # Keep columns with observed values in at least this fraction of rows.
    group_by=["reference_date"],  # Report completeness for each observed group.
    require_all_groups=False,  # Require the completeness cutoff in every group when enabled.
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

## Computational cost and practical limits

Typical relative cost: **low**, with overlapping tiers across methods.
This is an implementation-based estimate, not a benchmark or a universal ranking.
Spark: non-null aggregates; grouped shuffles. Pandas: non-null scans; optional grouped aggregates.
Rows, feature count, cardinality, iterations, folds, and the chosen estimator can
change which method dominates. Use this method for missing-input screening when justified;
do not run every selector as a fixed pipeline.

See the [backend comparison](../selection-costs.md) for an ordered shortlist
and [input contracts](../feature_selection.md) before comparing scores.
