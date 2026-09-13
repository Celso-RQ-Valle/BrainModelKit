# ANOVA / F-test: anova

[Method index](../feature_selection.md#api-quick-reference) · [README quickstart](https://github.com/Celso-RQ-Valle/BrainModelKit#feature-selection)

Use `from brainmodelkit.feature_selection import pandas as fs` for Pandas,
or `from brainmodelkit.feature_selection import pyspark as fs` for Spark.
Examples use preprocessed training data unless explicitly labeled held-out.
See the [complete runnable workflows](../../examples/feature_selection/README.md).

**Spark implementation:** read the [Spark algorithm and execution contract](spark.md#anova).
The explanations below describe shared concepts and the Pandas behavior. Consult
the signatures for backend-specific arguments; they are not interchangeable.

## Signatures

### Pandas

```python
def anova(
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
def anova(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    max_p_value=0.05,
    top_k=None,
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
`python examples/feature_selection/method_example.py --method anova`
from the repository root for a self-contained Pandas example. Add `--backend spark`
for the Spark implementation. See the method-specific parameter tables above.

## How it works

**Supervised classification; Pandas behavior (`anova`).** One-way ANOVA compares
between-class and within-class variation: `F = MS_between / MS_within`. It tests
whether class means differ, not arbitrary dependence. The classical p-value
assumes independent observations, approximately normal within-class errors, and
comparable class variances.

```python
from brainmodelkit.feature_selection.pandas import anova

result = anova(df, target_col="target", feature_cols=features, max_p_value=0.05)
```

This wraps sklearn `f_classif`. Outputs are `feature`, `f_statistic`, `p_value`,
`ranking`, and `selected`; `top_k` and `max_p_value` work as for Chi-Square.
There must be more observations than classes. Undefined scores are rejected;
a perfect class separation can legitimately produce an infinite F-statistic
with a zero p-value. Useful for inexpensive numerical screening. Avoid relying
on its p-values under severe heteroskedasticity or dependent observations, and
do not expect it to find symmetric/nonlinear relationships with equal means.
Spark computes distributed class statistics; read the [Spark ANOVA contract](spark.md#anova).

## Spark usage

See [the complete Spark setup and anova example](spark.md#anova) for the
distributed algorithm, parameters, result fields, limits, and interpretation.
